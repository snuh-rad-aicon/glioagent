"""Co-pilot ReAct agent: reads the autonomous trace and lets a reader apply corrections and redraft the report without re-running GPU tools.

Grounding, in three layers:
  1. tools return each value together with its source (GlioMT head + probability, VQA, seg/atlas features);
  2. the redraft is rendered by the pipeline's own deterministic report renderer, not written by the LLM;
  3. every chat reply is checked against the case record before it is shown -- a contradiction or an
     unknown number gets one rewrite, then a record-only reply.
"""
from __future__ import annotations

import contextvars
import copy
import json
import re
from pathlib import Path
from typing import Any, Optional

from agents.llm_client import (LLMClient, parse_tool_calls, tool_schema,
                               ascii_fold, get_client)  # get_client re-exported
from agents.prompts import COPILOT_SYSTEM as SYSTEM_PROMPT
from agents.report_assembly import _render_structured_report_text
from protocol.who2021_rules import derive_who2021

REPO = Path(__file__).resolve().parent.parent

# Fields a reader may override (mapped onto structured_findings).
CORRECTABLE_FIELDS = {
    "t2_flair_mismatch": "T2-FLAIR mismatch sign (yes/no)",
    "enhancement":       "enhancement present (yes/no)",
    "necrosis":          "central necrosis (yes/no)",
    "edema":             "peritumoral edema extent (minimal/moderate/extensive)",
    "mass_effect":       "mass effect / midline shift (yes/no)",
    "idh":               "IDH status (wildtype/mutant)",
    "1p19q":             "1p/19q codeletion (codeleted/non-codeleted)",
    "grade":             "CNS WHO grade (2/3/4)",
    "who_2021":          "WHO 2021 integrated diagnosis",
    "tumor_location":    "tumour location (lobe)",
    "laterality":        "left / right / midline / bilateral",
}


class CopilotCaseState:
    def __init__(self, case_dir: str, trace: dict):
        self.case_dir = case_dir
        self.trace = trace
        self.corrections: dict[str, Any] = {}
        self.correction_log: list[dict[str, Any]] = []


_STATES: dict[str, CopilotCaseState] = {}
_CURRENT_CASE: contextvars.ContextVar[str] = contextvars.ContextVar("copilot_current_case", default="")
_ACTIVE_CLIENT: Optional[LLMClient] = None
_KB = None


def _active_case() -> str:
    case_dir = _CURRENT_CASE.get()
    if not case_dir:
        raise RuntimeError("No active case bound -- run_turn() must set it first.")
    return case_dir


def _resolve_trace_path(case_dir: str) -> Path:
    """Locate glioagent_trace.json for a case. `case_dir` may be the
    autonomous out-dir itself, or a sample id under outputs/runs/."""
    candidates = [
        Path(case_dir) / "glioagent_trace.json",
        REPO / "outputs" / "runs" / case_dir / "glioagent_trace.json",
        REPO / case_dir / "glioagent_trace.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No glioagent_trace.json for '{case_dir}'. Run the autonomous "
        f"pipeline first: bash scripts/inference_auto.sh ... "
        f"(looked in: {', '.join(str(candidate) for candidate in candidates)})")


def load_case(case_dir: str) -> CopilotCaseState:
    if case_dir in _STATES:
        return _STATES[case_dir]
    trace = json.loads(_resolve_trace_path(case_dir).read_text())
    state = CopilotCaseState(case_dir, trace)
    _STATES[case_dir] = state
    return state


def reset_case(case_dir: str) -> None:
    _STATES.pop(case_dir, None)


def _bool_to_yes_no(value):
    return "yes" if value is True else ("no" if value is False else None)


# fields a reader answers yes/no; readers say present/absent just as often
_YES_NO_FIELDS = {"t2_flair_mismatch", "enhancement", "necrosis", "mass_effect"}
_YES_WORDS = {"yes", "y", "present", "positive", "true", "seen",
              "partial", "complete", "minimal", "mild", "moderate", "marked", "extensive"}
_NO_WORDS = {"no", "n", "absent", "negative", "false", "none", "not seen"}


def _as_yes_no(value) -> str:
    # canonical yes/no, so the recall rules fire however the reader phrased it
    text = str(value if value is not None else "").strip().lower()
    if text in _YES_WORDS:
        return "yes"
    if text in _NO_WORDS:
        return "no"
    return text


def _canonical(field: str, value) -> str:
    """Reader phrasing -> the vocabulary the rules and the renderer compare against."""
    text = str(value if value is not None else "").strip().lower()
    if field in _YES_NO_FIELDS:
        return _as_yes_no(text)
    if field == "idh":
        if "wild" in text or text in ("wt", "idh-wt", "idh wt"):
            return "wildtype"
        return "mutant" if "mut" in text else text
    if field == "1p19q":
        if re.search(r"\bnon|intact|\bnot\b|absent|\bno\b", text):
            return "non-codeleted"
        return "codeleted" if re.search(r"co-?del|\byes\b|present", text) else text
    if field == "grade":
        digits = re.findall(r"[234]", text)
        return digits[0] if len(digits) == 1 else text
    return str(value if value is not None else "").strip()


def _summarise_trace(trace: dict) -> dict:
    structured_findings = trace.get("structured_findings") or {}
    findings = structured_findings.get("findings") or {}
    impression = structured_findings.get("impression") or {}
    location = findings.get("location") or {}
    volumes = findings.get("volumes") or {}
    enhancement = findings.get("enhancement") or {}
    necrosis = findings.get("necrosis") or {}
    mismatch = findings.get("mismatch") or {}
    edema = findings.get("edema") or {}
    mass_effect = findings.get("mass_effect") or {}
    cortical = findings.get("cortical") or {}
    idh = impression.get("idh") or {}
    codeletion = impression.get("1p19q") or {}
    diameters = trace.get("measure_diameters") or {}
    checklist_items = (trace.get("checklist") or {}).get("items") or {}
    return {
        "tumor_location": location.get("lobe"),
        "region": location.get("region"),
        "laterality": location.get("laterality"),
        "et_cm3": volumes.get("et_cm3"), "ne_cm3": volumes.get("ne_cm3"),
        "wt_cm3": volumes.get("wt_cm3"),
        "max_diameters_mm": diameters.get("max_perpendicular_diameters_mm"),
        "enhancing_diameters_mm": diameters.get("enhancing_diameters_mm"),
        "measurable": _bool_to_yes_no(diameters.get("measurable")),
        "idh": idh.get("prediction"), "idh_p": idh.get("probability"),
        "1p19q": codeletion.get("prediction"),           # absent for IDH-wt
        "grade": impression.get("grade"), "who_2021": impression.get("who_2021"),
        "t2_flair_mismatch": _bool_to_yes_no(mismatch.get("present")),
        "enhancement": _bool_to_yes_no(enhancement.get("present")),
        "enhancement_proportion": enhancement.get("proportion"),
        "necrosis": _bool_to_yes_no(necrosis.get("present")),
        "edema": _bool_to_yes_no(edema.get("present")),
        "mass_effect": _bool_to_yes_no(mass_effect.get("present")),
        "cortical": _bool_to_yes_no(cortical.get("involved")),
        "confidence_tier": trace.get("confidence_tier"),
        "checklist_n_pass": sum(1 for passed in checklist_items.values() if passed),
        "checklist_n_total": len(checklist_items),
        "warnings_n": len(trace.get("warnings") or []),
    }


_FIELD_ALIASES = {
    "mismatch": "t2_flair_mismatch", "t2flair_mismatch": "t2_flair_mismatch",
    "t2_flair": "t2_flair_mismatch", "flair_mismatch": "t2_flair_mismatch",
    "idh_call": "idh", "idh_status": "idh", "idh_prediction": "idh",
    "enhancement_pattern": "enhancement", "enhancing": "enhancement",
    "codeletion": "1p19q", "1p_19q": "1p19q", "1p/19q": "1p19q",
    "location": "tumor_location", "lobe": "tumor_location", "side": "laterality",
    "who": "who_2021", "who2021": "who_2021", "diagnosis": "who_2021",
    "molecular_diagnosis": "who_2021", "integrated_diagnosis": "who_2021", "final_diagnosis": "who_2021",
    "confidence": "confidence_tier", "confidence_level": "confidence_tier", "tier": "confidence_tier",
    "molecular_confidence": "confidence_tier", "diagnostic_confidence": "confidence_tier",
    "diagnosis_confidence": "confidence_tier",
    "midline_shift": "mass_effect", "perilesional_edema": "edema",
    "grade_prediction": "grade",
    # size: volumes and bidimensional diameters together, or one of them
    "tumor_size": "size", "dimensions": "size", "extent": "size", "measurements": "size",
    "volume": "wt_cm3", "tumor_volume": "wt_cm3", "whole_tumor_volume": "wt_cm3",
    "enhancing_volume": "et_cm3", "non_enhancing_volume": "ne_cm3",
    "diameter": "max_diameters_mm", "diameters": "max_diameters_mm", "max_diameter": "max_diameters_mm",
    "bidimensional_diameters": "max_diameters_mm", "max_bidimensional_mm": "max_diameters_mm",
}
_SIZE_FIELDS = ("wt_cm3", "et_cm3", "ne_cm3", "max_diameters_mm", "enhancing_diameters_mm")


def _norm_field(field: str) -> str:
    """Map a model-guessed field name to a canonical summary key."""
    if not field:
        return field
    key = field.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return _FIELD_ALIASES.get(key, key)


# ---- case record: every value with its source ----

_GLIOMT_SOURCE = "GlioMT deep-learning model (T1/T1CE/T2/FLAIR + tumor mask, age/sex)"
# readable names for the trace's provenance tags
_SOURCE_NAMES = {
    "vqa_qwen_t2fm": "T2-FLAIR VQA (Qwen vision model on 3 axial T2/FLAIR slices)",
    "agent_reconciled": "IDH reconciliation by the agent (GlioMT >= 0.95 guardrail)",
    "agent_mismatch_reconciled": "astro-oligo gate (T2-FLAIR mismatch + uncertain 1p/19q codeletion)",
}
# field -> (structured_findings.findings block, key holding the call)
_FINDING_BLOCKS = {"t2_flair_mismatch": ("mismatch", "present"), "enhancement": ("enhancement", "present"),
                   "necrosis": ("necrosis", "present"), "edema": ("edema", "present"),
                   "mass_effect": ("mass_effect", "present"), "cortical": ("cortical", "involved")}
_MOLECULAR_FIELDS = ("idh", "1p19q", "grade", "who_2021")
_SUMMARY_FIELDS = ("age", "sex", "tumor_location", "laterality", "wt_cm3", "et_cm3", "ne_cm3", "max_diameters_mm",
                   "t2_flair_mismatch", "enhancement", "necrosis", "edema", "mass_effect", "cortical", "idh",
                   "1p19q", "grade", "who_2021")


def _rounded(value):
    return round(value, 3) if isinstance(value, float) else value


def _correction_reason(state: CopilotCaseState, field: str) -> str:
    return next((entry["reason"] for entry in reversed(state.correction_log) if entry["field"] == field), "")


def _pipeline_diagnosis(trace: dict) -> dict:
    """The autonomous molecular call exactly as the report states it, per-field sources included."""
    impression = (trace.get("structured_findings") or {}).get("impression") or {}
    molecular = trace.get("molecular") or {}
    idh, codeletion = impression.get("idh") or {}, impression.get("1p19q") or {}

    def head_source(head: str, tag: Optional[str]) -> str:
        return f"{_GLIOMT_SOURCE}, {head} head" + (f" -> {_SOURCE_NAMES[tag]}" if tag in _SOURCE_NAMES else "")

    return {
        "idh": idh.get("prediction"), "1p19q": codeletion.get("prediction"),
        "grade": impression.get("grade"), "who_2021": impression.get("who_2021"),
        "probability": {"idh": idh.get("probability"), "1p19q": codeletion.get("probability"),
                        "grade": impression.get("grade_probability")},
        "source": {"idh": head_source("IDH", (molecular.get("idh") or {}).get("source")),
                   "1p19q": head_source("1p/19q", (molecular.get("1p19q") or {}).get("source")),
                   "grade": head_source("grade", None),
                   "who_2021": "WHO 2021 rule on the IDH + 1p/19q calls"},
        "changed": False,
    }


def _imaging_rule(value, reader_fields: set) -> Optional[dict]:
    """Radiology-only override, fired only by findings the READER supplied: the pipeline's own
    findings were already weighed by the orchestrator (IDH guardrail, astro-oligo gate)."""
    if value("t2_flair_mismatch") == "yes" and "t2_flair_mismatch" in reader_fields:
        return {"idh": "mutant", "1p19q": "non-codeleted", "grade": "2 or 3",
                "who_2021": derive_who2021("mutant", "non-codeleted")[0],
                "basis": "reader-confirmed T2-FLAIR mismatch -> IDH-mutant, 1p/19q non-codeleted "
                         "astrocytoma (Patel 2017)"}
    if value("enhancement") == "yes" and value("necrosis") == "yes" and reader_fields & {"enhancement", "necrosis"}:
        return {"idh": "wildtype", "1p19q": None, "grade": 4,
                "who_2021": derive_who2021("wildtype", "non-codeleted")[0],
                "basis": "reader-confirmed enhancement + necrosis -> IDH-wildtype glioblastoma"}
    return None


def _current_diagnosis(state: CopilotCaseState, hypothetical: Optional[dict] = None) -> dict:
    """The molecular call the co-pilot stands behind now: the pipeline's, unless reader-supplied
    imaging findings fire a rule, with direct molecular corrections applied last.
    `hypothetical` = imaging values to try without storing them (recall_molecular arguments)."""
    hypothetical = {field: _as_yes_no(value) for field, value in (hypothetical or {}).items() if value}
    reader_fields = set(state.corrections) | set(hypothetical)

    def value(field):
        return hypothetical.get(field) or _as_yes_no(_effective(state, field))

    diagnosis = _pipeline_diagnosis(state.trace)
    rule = _imaging_rule(value, reader_fields)
    if rule:
        basis = rule.pop("basis")
        diagnosis.update(rule, changed=True,
                         probability={"idh": None, "1p19q": None, "grade": None},
                         source={field: basis for field in _MOLECULAR_FIELDS})

    direct = {field: state.corrections[field] for field in _MOLECULAR_FIELDS if field in state.corrections}
    for field, corrected in direct.items():
        diagnosis[field] = corrected
        diagnosis["probability"][field] = None
        diagnosis["source"][field] = f"reader correction ({_correction_reason(state, field)})"
    if ("idh" in direct or "1p19q" in direct) and "who_2021" not in direct \
            and diagnosis["idh"] in ("mutant", "wildtype"):
        if diagnosis["idh"] == "wildtype":            # 1p/19q is not evaluated for IDH-wildtype
            diagnosis.update({"1p19q": None, "grade": 4})
        who_diagnosis = derive_who2021(diagnosis["idh"], diagnosis["1p19q"] or "non-codeleted")[0]
        diagnosis["who_2021"] = who_diagnosis
        diagnosis["source"]["who_2021"] = "WHO 2021 rule on the reader-corrected IDH / 1p/19q"
    if direct:
        diagnosis["changed"] = True
    return diagnosis


def _effective(state: CopilotCaseState, field: str):
    field = _norm_field(field)
    if field in _MOLECULAR_FIELDS:
        return _current_diagnosis(state)[field]
    if field in state.corrections:
        return state.corrections[field]
    return _summarise_trace(state.trace).get(field)


def _evidence(state: CopilotCaseState, field: str) -> dict:
    """One field's current value plus where it came from -- all the co-pilot may cite."""
    field = _norm_field(field)
    if field in _MOLECULAR_FIELDS:
        diagnosis = _current_diagnosis(state)
        evidence = {"field": field, "value": diagnosis[field]}
        probability = diagnosis["probability"].get(field)
        if probability is not None:
            evidence["probability"] = _rounded(probability)
        evidence["source"] = diagnosis["source"][field]
        if field == "who_2021":                     # the integrated diagnosis carries the run's confidence tier
            evidence["confidence_tier"] = state.trace.get("confidence_tier")
        return evidence
    if field in state.corrections:
        return {"field": field, "value": state.corrections[field],
                "source": f"reader correction ({_correction_reason(state, field)})"}
    findings = (state.trace.get("structured_findings") or {}).get("findings") or {}
    if field in _FINDING_BLOCKS:
        block_name, key = _FINDING_BLOCKS[field]
        block = findings.get(block_name) or {}
        tag = block.get("provenance") or block.get("source") or "pipeline trace"
        evidence = {"field": field, "value": _bool_to_yes_no(block.get(key))}
        evidence.update({name: _rounded(value) for name, value in block.items()
                         if name not in (key, "provenance", "source", "note", "overrides_vasari")})
        evidence["source"] = _SOURCE_NAMES.get(tag, tag)
        return evidence
    clinical = (state.trace.get("structured_findings") or {}).get("clinical") or {}
    if field in ("age", "sex"):
        return {"field": field, "value": clinical.get(field), "source": "clinical input (samples.csv / --age --sex)"}
    summary = _summarise_trace(state.trace)
    if field == "size":
        return {"field": field, "value": {name: _rounded(summary[name]) for name in _SIZE_FIELDS},
                "source": "tumor volumes (seg_features) + bidimensional diameters (measure_diameters)"}
    value = summary.get(field)
    block = (findings.get("location") if field in ("tumor_location", "region", "laterality")
             else findings.get("volumes") if field.endswith("_cm3")
             else state.trace.get("measure_diameters") if field.endswith("diameters_mm") or field == "measurable"
             else None) or {}
    source = block.get("provenance") or ("pipeline trace" if value is not None else "not in the case record")
    return {"field": field, "value": _rounded(value), "source": source}


def _evidence_text(evidence: dict) -> str:
    probability = f" (p={evidence['probability']:.2f})" if "probability" in evidence else ""
    return f"{evidence['field']} = {evidence['value']}{probability}  [{evidence['source']}]"


# ---- tools the LLM calls ----

def inspect_trace(field: Optional[str] = None) -> str:
    """Read the autonomous result (corrections applied), each value with its source."""
    state = load_case(_active_case())
    if field:
        evidence = _evidence(state, field)
        evidence["corrected_by_reader"] = evidence["field"] in state.corrections
        return json.dumps(evidence)
    summary = {name: {key: value for key, value in _evidence(state, name).items() if key != "field"}
               for name in _SUMMARY_FIELDS}
    summary["confidence_tier"] = state.trace.get("confidence_tier")
    summary["_corrections_applied"] = list(state.corrections.keys())
    return json.dumps(summary)


def apply_correction(field: str, value: str, reason: str) -> str:
    field = _norm_field(field)
    if field not in CORRECTABLE_FIELDS:
        return json.dumps({"ok": False, "error": f"'{field}' not correctable",
                           "allowed": sorted(CORRECTABLE_FIELDS)})
    state = load_case(_active_case())
    value = _canonical(field, value)
    prior = _effective(state, field)
    state.corrections[field] = value
    state.correction_log.append({"field": field, "prior_value": prior,
                                 "new_value": value, "reason": reason})
    return json.dumps({"ok": True, "field": field, "prior_value": prior,
                       "new_value": value, "diagnosis_now": _current_diagnosis(state)["who_2021"],
                       "note": "Applied. Call redraft_report() to regenerate the report."})


def list_corrections() -> str:
    state = load_case(_active_case())
    return json.dumps({"corrections": state.correction_log})


def recall_molecular(prior_t2_flair_mismatch: Optional[str] = None,
                     prior_enhancement: Optional[str] = None,
                     prior_necrosis: Optional[str] = None) -> str:
    """Re-derive the molecular call from the current findings (no GlioMT re-run). Rules, fired
    only by reader-supplied findings (a correction, or the prior_* what-if arguments):
      * T2-FLAIR mismatch = yes -> IDH-mutant, 1p/19q non-codeleted astrocytoma (Patel 2017).
      * enhancement = yes AND necrosis = yes -> IDH-wildtype glioblastoma.
      * otherwise -> the GlioMT autonomous call stands.
    """
    state = load_case(_active_case())
    hypothetical = {"t2_flair_mismatch": prior_t2_flair_mismatch, "enhancement": prior_enhancement,
                    "necrosis": prior_necrosis}
    diagnosis = _current_diagnosis(state, hypothetical)
    evidence = {field: _as_yes_no(hypothetical[field] or _effective(state, field)) for field in hypothetical}
    return json.dumps({"derived": {field: diagnosis[field] for field in _MOLECULAR_FIELDS},
                       "basis": diagnosis["source"]["who_2021"], "evidence": evidence,
                       "hypothetical": any(hypothetical.values())})


def get_warnings() -> str:
    state = load_case(_active_case())
    items = (state.trace.get("checklist") or {}).get("items") or {}
    return json.dumps({"warnings": state.trace.get("warnings") or [],
                       "checklist_items": items,
                       "n_pass": sum(1 for passed in items.values() if passed),
                       "n_total": len(items)})


def query_literature(question: str) -> str:
    """Short citation / definition from the glioma-imaging knowledge base.

    Keyword scan over the KB chunks (no embedding model): rank chunks by how many
    question words appear in the title + body, return the best two.
    """
    global _KB
    try:
        if _KB is None:
            from tools.tool_rag_verifier import kb
            _KB = kb()
        words = [word for word in re.findall(r"[a-z0-9]+", question.lower()) if len(word) > 3]
        scored = sorted(
            ((sum((chunk["title"] + " " + chunk["chunk"]).lower().count(word) for word in words), chunk)
             for chunk in _KB),
            key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored[:2] if score > 0]
        if top_chunks:
            return json.dumps({"answer": " ".join(chunk.get("chunk", "") for chunk in top_chunks),
                               "sources": [chunk.get("source", "") for chunk in top_chunks]})
    except Exception:
        pass
    canned = {
        "patel": "Patel 2017 -- T2-FLAIR mismatch sign: near-complete T2 "
                 "hyperintensity with FLAIR suppression except a thin rim; high "
                 "PPV for IDH-mutant 1p/19q-non-codeleted astrocytoma.",
        "necros": "Ring enhancement + central necrosis on adult diffuse glioma "
                  "carries high PPV for IDH-wildtype glioblastoma.",
        "grade": "CNS WHO 2021 grades adult diffuse gliomas 2-4 by molecular + "
                 "histologic criteria; IDH-wildtype astrocytic tumours meeting GBM "
                 "criteria are grade 4.",
    }
    question_lower = question.lower()
    matched_answers = ([answer for keyword, answer in canned.items() if keyword in question_lower]
                       or list(canned.values()))
    return json.dumps({"answer": " ".join(matched_answers[:2]), "source": "canned"})


# measurements that described the pipeline's call and no longer hold once the reader overrides it
_STALE_ON_CORRECTION = {"t2_flair_mismatch": ("extent",), "enhancement": ("proportion", "quality"),
                        "necrosis": ("necrosis_cm3",), "mass_effect": ("midline_shift_mm",)}


def _corrected_structured_findings(state: CopilotCaseState) -> tuple[dict, dict]:
    """structured_findings with the reader's corrections and the current diagnosis written in."""
    structured_findings = copy.deepcopy(state.trace.get("structured_findings") or {})
    findings = structured_findings.setdefault("findings", {})
    for field, value in state.corrections.items():
        if field in _FINDING_BLOCKS:
            block_name, key = _FINDING_BLOCKS[field]
            block = findings.setdefault(block_name, {})
            block[key] = _as_yes_no(value) != "no" if field == "edema" else value == "yes"
            for stale in _STALE_ON_CORRECTION.get(field, ()):
                block[stale] = None
            if field == "enhancement":
                findings.setdefault("volumes", {})["et_cm3"] = None
        elif field == "tumor_location":
            findings.setdefault("location", {}).update({"lobe": value, "region": None, "junction_lobes": []})
        elif field == "laterality":
            findings.setdefault("location", {})["laterality"] = value
    diagnosis = _current_diagnosis(state)
    if diagnosis["changed"]:
        impression = structured_findings.setdefault("impression", {})
        probability = diagnosis["probability"]
        impression["idh"] = {"prediction": diagnosis["idh"], "probability": probability["idh"]}
        impression["1p19q"] = ({"prediction": diagnosis["1p19q"], "probability": probability["1p19q"]}
                               if diagnosis["1p19q"] else {})
        impression["grade"], impression["grade_probability"] = diagnosis["grade"], probability["grade"]
        impression["who_2021"] = diagnosis["who_2021"]
    return structured_findings, diagnosis


def redraft_report(style_instruction: Optional[str] = None) -> str:
    """Regenerate the structured report from the corrected case record with the pipeline's own
    renderer; a style instruction is an LLM rewrite that must pass the grounding check."""
    state = load_case(_active_case())
    structured_findings, diagnosis = _corrected_structured_findings(state)
    # a reader-changed diagnosis no longer carries the pipeline's confidence tier
    tier = None if diagnosis["changed"] else state.trace.get("confidence_tier")
    report_text = _render_structured_report_text(structured_findings, corrected_fields=state.corrections,
                                                 confidence_tier=tier)
    if state.correction_log:
        changes = "; ".join(f"{entry['field']} {entry['prior_value']} -> {entry['new_value']} ({entry['reason']})"
                            for entry in state.correction_log)
        report_text += f"\n\nREADER CORRECTIONS: {changes}."
        if diagnosis["changed"]:
            report_text += f" Diagnosis re-derived: {diagnosis['source']['who_2021']}."
    note = "rendered from the corrected case record (deterministic)"

    if style_instruction:
        prompt = ("Rewrite this brain-MRI glioma report in the requested style. Keep every finding, "
                  "every number and the FINAL DIAGNOSIS exactly; add nothing. ASCII only (use '-', '->'). "
                  f"Output ONLY the report.\n\nStyle: {style_instruction}\n\nReport:\n{report_text}\n\n"
                  "Rewritten report:")
        restyled = ascii_fold(_client_chat(prompt, max_new_tokens=700), strip_unmapped=False)
        problems = ["no output"] if (not restyled or restyled.startswith("(LLM")) else \
            _grounding_problems(restyled, state, record_text=report_text)
        if problems:
            note = f"style rewrite discarded ({'; '.join(problems)}); deterministic report returned"
        else:
            report_text, note = restyled, "restyled by the LLM; checked against the case record"
    return json.dumps({"narrative": report_text, "diagnosis": diagnosis["who_2021"],
                       "corrections": state.correction_log, "note": note})


# LLM prose can re-introduce unicode; normalise co-pilot redrafts to the same
# ASCII rules as the deterministic report for consistency.
def _client_chat(prompt: str, max_new_tokens: int = 600) -> str:
    if _ACTIVE_CLIENT is None:
        return "(LLM client not initialised)"
    try:
        return _ACTIVE_CLIENT.chat([{"role": "user", "content": prompt}],
                                   max_new_tokens=max_new_tokens,
                                   enable_thinking=False).strip()
    except Exception as exc:
        return f"(LLM error: {type(exc).__name__})"


TOOL_FUNCTIONS = {
    "inspect_trace": inspect_trace, "apply_correction": apply_correction,
    "list_corrections": list_corrections, "recall_molecular": recall_molecular,
    "get_warnings": get_warnings, "query_literature": query_literature,
    "redraft_report": redraft_report,
}


TOOL_SCHEMAS = [
    tool_schema("inspect_trace", "Read the case record: each value with its source and probability. "
            "Omit 'field' for a full summary, or pass a field name.",
            {"field": {"type": "string"}}),
    tool_schema("apply_correction", "Apply a radiologist correction to one field.",
            {"field": {"type": "string"}, "value": {"type": "string"},
             "reason": {"type": "string"}}, ["field", "value", "reason"]),
    tool_schema("list_corrections", "List corrections applied so far."),
    tool_schema("recall_molecular", "Re-derive the molecular call from the current findings. "
            "prior_* arguments are what-if values that are not stored.",
            {"prior_t2_flair_mismatch": {"type": "string"},
             "prior_enhancement": {"type": "string"},
             "prior_necrosis": {"type": "string"}}),
    tool_schema("get_warnings", "Return warnings + the 7-item checklist status."),
    tool_schema("query_literature", "Look up a short citation/definition.",
            {"question": {"type": "string"}}, ["question"]),
    tool_schema("redraft_report", "Regenerate the structured report reflecting "
            "corrections.", {"style_instruction": {"type": "string"}}),
]


# ---- grounding check: a reply may not contradict the case record or invent numbers ----

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|,\s*which\b|\b(?:but|however|whereas|while|although)\b", re.I)
# speculative, comparative or textbook clauses are not assertions about this case
_HEDGE = re.compile(r"\b(?:if|would|could|might|whether|differential|likely|likelihood|unlikely|versus|vs|"
                    r"rather than|instead of|over|alternative|rule out|r/o|excluded?|consider\w*|"
                    r"typical\w*|common\w*|uncommon|frequent\w*|characteristic\w*|associated|ppv|"
                    r"specific(?:ity)?|predictive|counterparts?|generally|usually|against|rul\w* out|"
                    r"less probable|possibility|overlap\w*|intersection|distinguish\w*|misclassif\w*|"
                    r"inconsistent|entity|entities|etiology|hallmarks?|originally|initially|previously|"
                    r"astrocytomas|oligodendrogliomas|glioblastomas|gliomas|tumou?rs)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|absent|absence|without|negative|lacks?|lacking|none|neither|nor)\b", re.I)
# the negation window before a finding stops at these (so 'no X and the Y is present' reads Y as present)
_WINDOW_STOP = re.compile(r"\bwith\b|(?:\band|,)(?=\s+(?:the|there|a|an|it|its|this)\b)", re.I)
_TUMOUR_PART = r"(?!\s+(?:volume|component|portion|part|signal|tissue))"
_FEATURE_PATTERNS = {"t2_flair_mismatch": r"mismatch(?!\s+(?:with|between))","enhancement": r"enhanc\w*\b(?!\s+(?:component|foc|volume|portion|status|pattern|quality|\d))",
                     "necrosis": r"necro\w*"}
_SUBTYPES = {"oligodendroglioma": "oligodendroglioma", "astrocytoma": "astrocytoma",
             "glioblastoma": "glioblastoma", "gbm": "glioblastoma"}
_NUMBER = re.compile(r"(?<![\w/.])\d+(?:\.\d+)?(?![\w/])")


def _stated_presence(clause: str, match: re.Match) -> Optional[str]:
    """yes/no the clause asserts for the finding at `match`, None if it cannot tell."""
    after = clause[match.end():match.end() + 60]
    # 'updated to present', 'changed from no to yes', 'no -> yes' (but not 'mild to minimal')
    changed_to = re.search(r"(?:\b(?:from\s+\S+|updated|changed|corrected|revised|set)\s+(?:\w+\s+)?to\b|->)"
                           r"\s*\W*(\w+)", after, re.I)
    if changed_to and _as_yes_no(changed_to.group(1)) in ("yes", "no"):
        return _as_yes_no(changed_to.group(1))
    if clause[max(0, match.start() - 4):match.start()].lower() == "non-":
        # 'non-enhancing mass' says no enhancement; 'non-enhancing signal / 7 cm3' names a tumour part
        lesion = re.match(rf"[\s,]*(?:[\w-]+\s+){{0,2}}?(?:mass|tumou?r|lesion|glioma)\b{_TUMOUR_PART}|\W*$",
                          after, re.I)
        return "no" if lesion else None
    before = clause[:match.start()]
    stops = list(_WINDOW_STOP.finditer(before))
    before = before[stops[-1].end():] if stops else before
    if _NEGATION.search(" ".join(before.split()[-6:])):
        # 'no avid enhancement' denies a quality, not the finding
        return None if re.search(r"\b(?:avid|strong|marked|intense|robust|ring|thick|nodular|solid)\W*$",
                                 before, re.I) else "no"
    if re.match(r"\W*(?:\w+\W+){0,3}?(?:absent|not|negative|no|none)\b", after, re.I):
        return "no"
    return "yes"


def _negated_mention(clause: str, match: re.Match) -> bool:
    # 'no / absence of / without' shortly before a diagnosis word: not a claim that it is present
    return bool(_NEGATION.search(" ".join(clause[:match.start()].split()[-4:])))


def _subtype_of(who: Optional[str]) -> Optional[str]:
    text = (who or "").lower()
    return next((subtype for word, subtype in _SUBTYPES.items() if word in text), None)


def _record_facts(state: CopilotCaseState, tool_calls: list[dict]) -> dict[str, set]:
    """Values a reply may assert: the current record, plus any what-if recall from this turn."""
    diagnoses = [_current_diagnosis(state)]
    facts = {field: {_as_yes_no(_effective(state, field))} for field in _FEATURE_PATTERNS}
    for call in tool_calls:
        if call["tool"] == "recall_molecular" and any(call["input"].values()):
            output = json.loads(call["output"])
            diagnoses.append(output["derived"])
            for field, value in output["evidence"].items():
                facts[field].add(value)
    facts["idh"] = {diagnosis["idh"] for diagnosis in diagnoses}
    facts["1p19q"] = {diagnosis["1p19q"] for diagnosis in diagnoses}
    facts["subtype"] = {_subtype_of(diagnosis["who_2021"]) for diagnosis in diagnoses}
    facts["grade"] = {digit for diagnosis in diagnoses for digit in re.findall(r"\d", str(diagnosis["grade"]))}
    return facts


def _stated_facts(text: str) -> list[tuple[str, str]]:
    """(field, value) assertions in the text, hedged clauses skipped."""
    stated = []
    text = re.sub(r",\s*which\b[^,.;]*,", ",", text or "")    # drop embedded ', which ...,' asides
    for clause in _CLAUSE_SPLIT.split(text):
        if not clause or _HEDGE.search(clause):
            continue
        for field, pattern in _FEATURE_PATTERNS.items():
            for match in re.finditer(pattern, clause, re.I):
                presence = _stated_presence(clause, match)
                if presence:
                    stated.append((field, presence))
        for match in re.finditer(r"\bIDH\b", clause):
            if _negated_mention(clause, match):
                continue
            window = clause[match.end():match.end() + 60]
            statuses = list(re.finditer(r"(\bto\s+)?\W*(wild[- ]?type|\bwt\b|mutant|mutated|\bmut\b)", window, re.I))
            if statuses:                                        # 'reconciled from wildtype to mutant' -> mutant
                status = next((found for found in statuses if found.group(1)), statuses[0]).group(2).lower()
                stated.append(("idh", "wildtype" if status.startswith(("wild", "wt")) else "mutant"))
        if re.search(r"non[- ]?co[- ]?del\w*|1p/19q[- ]intact|intact 1p/19q|(?:without|no) (?:1p/19q )?co[- ]?deletion",
                     clause, re.I):
            stated.append(("1p19q", "non-codeleted"))
        elif re.search(r"co[- ]?delet\w*", clause, re.I):
            stated.append(("1p19q", "codeleted"))
        for word, subtype in _SUBTYPES.items():
            for match in re.finditer(rf"\b{word}\b", clause, re.I):
                if not _negated_mention(clause, match):         # 'absence of glioblastoma features'
                    stated.append(("subtype", subtype))
        for match in re.finditer(r"\bgrade\s*(\d)\b(?!\s*(?:-|to|or|/|and)\s*\d)", clause, re.I):
            stated.append(("grade", match.group(1)))
    return stated


def _unknown_numbers(text: str, record_text: str) -> list[str]:
    """Numbers in the text that no record value explains (rounded, as %, or mm<->cm)."""
    known = [float(token) for token in _NUMBER.findall(record_text)]
    unknown = []
    for token in _NUMBER.findall(text or ""):
        value = float(token)
        if value <= 10 and "." not in token or 1900 <= value <= 2100:
            continue                                            # grades, counts, years
        decimals = len(token.split(".")[1]) if "." in token else 0
        if not any(round(scaled, decimals) == value
                   for number in known for scaled in (number, number * 100, number / 10, number * 10)):
            unknown.append(token)
    return unknown


_MODEL_WORDS = re.compile(r"\b(?:GlioMT|molecular model|deep[- ]learning|classifier|the model)\b", re.I)


def _attribution_problems(text: str, state: CopilotCaseState) -> list[str]:
    """The call credited to GlioMT when the record says a reader correction / rule made it."""
    sources = _current_diagnosis(state)["source"]
    if "GlioMT" in sources["idh"] and "GlioMT" in sources["1p19q"]:
        return []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        if (_MODEL_WORDS.search(sentence)
                and re.search(r"diagnos|IDH|astrocytoma|oligodendroglioma|glioblastoma", sentence, re.I)
                and not re.search(r"\b(?:if|would|could|originally|initially|previously|autonomous|pipeline)\b",
                                  sentence, re.I)):
            return [f"credited the call to the model, case record source is: {sources['who_2021']}"]
    return []


def _grounding_problems(text: str, state: CopilotCaseState, tool_calls: Optional[list[dict]] = None,
                        record_text: str = "") -> list[str]:
    """Contradictions with the case record, a wrongly credited source, and numbers the record
    does not contain."""
    tool_calls = tool_calls or []
    facts = _record_facts(state, tool_calls)
    problems = []
    for field, claimed in dict.fromkeys(_stated_facts(text)):
        allowed = {value for value in facts.get(field, ()) if value}
        if allowed and claimed not in allowed:
            record = "/".join(sorted(allowed))
            problems.append(f"said {field} = {claimed}, case record says {record}")
    problems += _attribution_problems(text, state)
    record_text =" ".join([record_text, json.dumps(state.trace), json.dumps(state.corrections)]
                           + [call["output"] for call in tool_calls])
    problems += [f"number {token} is not in the case record" for token in _unknown_numbers(text, record_text)]
    return problems


def _record_reply(state: CopilotCaseState, tool_calls: list[dict]) -> str:
    """Record-only reply for the fields this turn touched, used when the LLM reply fails the check."""
    fields, warnings = [], []
    for call in tool_calls:
        if call["tool"] == "get_warnings":                  # the warnings question: answer with the warnings
            data = json.loads(call["output"])
            listed = "; ".join(str(warning) for warning in data["warnings"])
            warnings.append(f"{len(data['warnings'])} active warnings" + (f" ({listed})" if listed else "")
                            + f", checklist {data['n_pass']}/{data['n_total']} passed  [get_warnings]")
            continue
        field = _norm_field(call["input"].get("field") or "")
        fields += [field] if field else []
        if call["tool"] in ("apply_correction", "recall_molecular", "redraft_report") or not field:
            fields.append("who_2021")
    fields = list(dict.fromkeys(fields)) or ([] if warnings else ["who_2021"])
    return "From the case record: " + "; ".join(
        [_evidence_text(_evidence(state, field)) for field in fields] + warnings) + "."


def _tool_brief(name: str, output: str) -> str:
    """One console line per tool call: what it returned."""
    try:
        data = json.loads(output)
    except ValueError:
        return ""
    if "error" in data:
        return f"error: {data['error']}"
    if name == "inspect_trace":
        return _evidence_text(data) if "field" in data else f"summary of {len(_SUMMARY_FIELDS)} fields"
    if name == "apply_correction":
        return f"{data['field']}: {data['prior_value']} -> {data['new_value']}   diagnosis now: {data['diagnosis_now']}"
    if name == "recall_molecular":
        return f"{data['derived']['who_2021']}   [{data['basis']}]" + ("  (what-if)" if data["hypothetical"] else "")
    if name == "redraft_report":
        return data["note"]
    if name == "get_warnings":
        return f"{len(data['warnings'])} warnings, checklist {data['n_pass']}/{data['n_total']}"
    if name == "query_literature":
        return ", ".join(data.get("sources") or [data.get("source", "")])
    if name == "list_corrections":
        return f"{len(data['corrections'])} corrections"
    return ""


def _strip(text: str) -> str:
    """Co-pilot reply cleanup. Deliberately NOT llm_client.strip_tool_xml: this
    truncates at the FIRST tool_call tag and also drops <think> blocks."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<tool_call\b[^>]*>.*", "", text, flags=re.I | re.S)
    text = re.sub(r"</?(?:parameter|function|tool_call)\b[^>]*>", "", text, flags=re.I)
    return ascii_fold(text.strip(), strip_unmapped=False)


def run_turn(case_dir: str, user_message: str,
             history: Optional[list[dict]] = None,
             client: Optional[LLMClient] = None,
             max_iters: int = 6) -> dict:
    """One chat turn: a bounded in-process tool-call loop, then the grounding check.
    Returns {reply, tool_calls, report, grounding, corrections}."""
    global _ACTIVE_CLIENT
    state = load_case(case_dir)               # raises if autonomous trace missing
    chat_client = client or get_client()
    _ACTIVE_CLIENT = chat_client

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_message in (history or [])[-6:]:
        if past_message.get("role") in ("user", "assistant") and (past_message.get("content") or "").strip():
            messages.append({"role": past_message["role"], "content": past_message["content"]})
    messages.append({"role": "user", "content": user_message})

    tool_calls: list[dict] = []
    reply = ""
    grounding: list[str] = []
    case_token = _CURRENT_CASE.set(case_dir)
    try:
        for _ in range(max_iters):
            text = chat_client.chat(messages, tools=TOOL_SCHEMAS, enable_thinking=False)
            calls = parse_tool_calls(text)
            messages.append({"role": "assistant", "content": text})
            if not calls:
                reply = _strip(text)
                break
            for call in calls:
                name, args = call["name"], call.get("arguments", {}) or {}
                tool_function = TOOL_FUNCTIONS.get(name)
                try:
                    output = tool_function(**args) if tool_function else json.dumps({"error": f"unknown tool {name}"})
                except Exception as exc:
                    output = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                tool_calls.append({"tool": name, "input": args, "output": output,
                                   "brief": _tool_brief(name, output)})
                messages.append({"role": "tool", "name": name, "content": output})

        # grounding check: one rewrite with the problems spelled out, then a record-only reply
        problems = _grounding_problems(reply, state, tool_calls, record_text=user_message)
        if problems:
            messages.append({"role": "user", "content": (
                "[grounding check] Your reply does not match the case record:\n- " + "\n- ".join(problems)
                + "\nRewrite the reply using only values from the tool outputs above. Add no new claims.")})
            rewrite = _strip(chat_client.chat(messages, enable_thinking=False))
            if rewrite and not _grounding_problems(rewrite, state, tool_calls, record_text=user_message):
                reply = rewrite
                grounding.append("reply rewritten -- " + "; ".join(problems))
            else:
                reply = _record_reply(state, tool_calls)
                grounding.append("LLM reply withheld, record shown instead -- " + "; ".join(problems))
        elif not reply and tool_calls:
            reply = _record_reply(state, tool_calls)
    finally:
        _CURRENT_CASE.reset(case_token)

    redrafts = [json.loads(call["output"]) for call in tool_calls
                if call["tool"] == "redraft_report" and '"narrative"' in call["output"]]
    return {"reply": reply or "(no reply)", "tool_calls": tool_calls,
            "report": redrafts[-1]["narrative"] if redrafts else None,
            "grounding": grounding, "corrections": state.correction_log}


__all__ = ["run_turn", "load_case", "reset_case", "get_client", "CORRECTABLE_FIELDS"]
