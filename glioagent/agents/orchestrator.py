"""LLM-orchestrated (ReAct) autonomous pipeline for a single adult glioma MRI case."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Optional

import nibabel as nib

from agents.llm_client import (LLMClient, parse_tool_calls, get_client,
                               ascii_fold, strip_tool_xml, tool_schema)
from agents.react_trace import ReActTrace, save_trace_md
from agents.prompts import (ORCHESTRATOR_SYSTEM as SYSTEM_PROMPT,
                            ASSESSMENT_SYSTEM as _ASSESSMENT_SYSTEM,
                            IDH_RECONCILE_SYSTEM as _IDH_RECONCILE_SYSTEM,
                            NECROSIS_CAVEAT as _NECROSIS_CAVEAT)
from agents.report_assembly import (
    GlioAgentTrace, _collect_subresults, _assemble_report,
    format_quantitative_summary, _confidence_tier, _render_structured_report_text,
)
from protocol.checklist import verify_report
from agents.discordance import check_discordance, retract_unreliable
from tools.tool_tumor_segment import predict_hdglio, summarise_seg
from tools.seg_cleanup import cleanup_seg_file
from tools.tool_molecular_subtype import molecular_subtype
from tools import tool_seg_features as seg_features, tool_atlas_features as atlas_features
from tools.tool_measure_diameters import measure_diameters
from tools.tool_vasari_feature import run_vasari
from agents.knowledge_select import select_chunks

TRACE_MD_FILE = "trace.md"

MANDATORY_BASE = ["segment_tumor", "predict_idh", "predict_grade",
                  "compute_seg_features", "compute_atlas_features", "measure_diameters",
                  "run_vasari", "run_t2fm"]

# A proposed IDH flip is discarded when GlioMT assigns >= this probability to its predicted class.
GLIOMT_CONFIDENT_P = 0.95

# Below this codeletion confidence, a confident T2-FLAIR mismatch favours astrocytoma
# (mismatch is specific for non-codeletion; Lasocki 2018).
CODEL_CONFIDENT_P = 0.90

# A wildtype -> mutant flip is discarded unless the tumour is (near) non-enhancing. An IDH-mutant
# diffuse glioma is characteristically non- or minimally-enhancing, so enhancement in a tumour the
# classifier called wildtype is a glioblastoma hallmark and not something to reconcile away. Over
# the four cohorts the proposals the LLM makes are 80% correct at an enhancing fraction of 0 and
# 14% correct at 0.15 or more; precision collapses at this cutoff.
MAX_ENHANCING_FRACTION_FOR_MUTANT = 0.05

# A mutant -> wildtype flip is never accepted. The only imaging cue the LLM has for that direction
# is the necrosis finding, and HD-GLIO labels no necrosis class: the reported value is
# non-enhancing tissue enclosed by enhancing tumour, which is anti-predictive here (of the flips
# proposed with that finding present, 1 of 11 was right).
ALLOW_MUTANT_TO_WILDTYPE = False

# Smallest whole-tumour volume that can carry a diagnosis; matches the checklist's [ALERT] cutoff.
MIN_TUMOUR_CM3 = 0.5


class CaseContext:
    # everything one case needs, shared by the tool wrappers
    def __init__(self, patient_id, mri_paths, age, sex, out_dir, cohort, gpu,
                 verbose: Optional[bool] = None):
        self.patient_id = patient_id
        self.age = age
        self.sex = sex
        self.out_dir = out_dir
        self.cohort = cohort
        self.gpu = gpu
        self.mri = mri_paths
        self.seg_path = os.path.join(out_dir, "hdglio_seg.nii.gz")
        self.results: dict[str, Any] = {}          # full tool output, read by the report
        self.tool_outputs: dict[str, Any] = {}     # what the LLM saw, per tool
        self.molecular_cache: Optional[dict] = None
        self.client = None
        self.called: list[str] = []                # every tool that ran, guard included
        self.llm_called: list[str] = []            # tools the LLM picked itself
        self.seconds: dict[str, float] = {}        # wall clock per tool
        self.llm_seconds = 0.0                     # wall clock spent in the backbone
        self.react_log = ReActTrace(verbose)       # the ReAct trace written to trace.md


# ---- tool wrappers: run a tool, keep the full output, return a summary to the LLM ----

def _tool_segment(context: CaseContext, **_) -> dict:
    # always segment; never reuse a mask found on disk
    predict_hdglio(t1=context.mri["t1"], t1ce=context.mri["t1ce"], t2=context.mri["t2"],
                   flair=context.mri["flair"], output_path=context.seg_path, gpu=context.gpu)
    # clean the raw mask IN PLACE (drop out-of-brain voxels and <1 cm3 islands, keep
    # genuine multifocal lesions) so every downstream tool sees tumour only
    context.results["seg_cleanup"] = cleanup_seg_file(context.seg_path, context.mri.get("t1"), context.seg_path)
    seg = summarise_seg(context.seg_path)
    # below this the mask cannot support a diagnosis: GlioMT's slice stack, the atlas location and the
    # diameters would all be computed on brain rather than tumour. Same value the checklist already
    # flags as an ALERT, so the two thresholds cannot drift apart.
    if (seg.get("wt_cm3") or 0.0) < MIN_TUMOUR_CM3:
        raise RuntimeError(f"segmentation too small to diagnose: whole-tumour {seg.get('wt_cm3')} cm3 "
                           f"< {MIN_TUMOUR_CM3} cm3 -- no diagnosis is issued")
    context.results["seg"] = seg
    return {"et_cm3": seg["et_cm3"], "ne_cm3": seg["ne_cm3"], "wt_cm3": seg["wt_cm3"],
            "laterality": seg.get("laterality"),
            "shape": "x".join(str(n) for n in nib.load(context.seg_path).shape)}


def _ensure_molecular(context: CaseContext) -> dict:
    # GlioMT runs once; the three heads read this cache
    if context.molecular_cache is None:
        context.molecular_cache = molecular_subtype(
            t1=context.mri["t1"], t1ce=context.mri["t1ce"], t2=context.mri["t2"],
            flair=context.mri["flair"], age=context.age, sex=context.sex, seg_mask=context.seg_path)
        context.results["molecular"] = context.molecular_cache
    return context.molecular_cache


def _tool_idh(context: CaseContext, **_) -> dict:
    from tools.gliomt.gliomt_inference import INPUT_SIZE

    molecular = _ensure_molecular(context)
    return {"idh": molecular["idh"]["prediction"], "probability": molecular["idh"]["probability"],
            "input": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]} slice stack"}


def _tool_grade(context: CaseContext, **_) -> dict:
    return {"grade": _ensure_molecular(context).get("grade_prediction")}


def _tool_1p19q(context: CaseContext, **_) -> dict:
    molecular = _ensure_molecular(context)
    if "1p19q" in molecular:
        return {"applicable": True, "1p19q": molecular["1p19q"]["prediction"],
                "probability": molecular["1p19q"]["probability"]}
    return {"applicable": False,
            "note": "IDH-wildtype -> 1p/19q not applicable (glioblastoma; WHO 2021)."}


def _tool_seg_features(context: CaseContext, **_) -> dict:
    features = seg_features.compute_all(context.seg_path, context.mri)
    context.results["seg_features"] = features
    edema_mm = (features.get("edema") or {}).get("radial_mm")
    foci = features.get("additional") or {}
    return {"necrosis": (features.get("necrosis") or {}).get("present"),
            "edema_present": bool(isinstance(edema_mm, (int, float)) and edema_mm > 0),
            "enhancement": (features.get("enhancement") or {}).get("present"),
            "multifocal": foci.get("multifocal"),
            "enhancing_foci_cm3": foci.get("multifocal_foci_cm3")}


def _tool_atlas_features(context: CaseContext, **_) -> dict:
    features = atlas_features.compute_all(
        t1_path=context.mri["t1"], seg_path=context.seg_path,
        xfm_dir=os.path.join(context.out_dir, "xfm"),
        seg_mni_out_path=os.path.join(context.out_dir, "seg_in_mni.nii.gz"))
    context.results["atlas_features"] = features
    location = features.get("location") or {}
    mni_path = os.path.join(context.out_dir, "seg_in_mni.nii.gz")
    return {"lobe": location.get("lobe"), "region": location.get("region"),
            "laterality": (features.get("laterality") or {}).get("dominant_side"),
            "midline_shift_mm": (features.get("mass_effect") or {}).get("midline_shift_mm"),
            "mni_shape": ("x".join(str(n) for n in nib.load(mni_path).shape)
                          if os.path.isfile(mni_path) else "-")}


def _tool_diameters(context: CaseContext, **_) -> dict:
    diameters = measure_diameters(context.seg_path, enhancing_label=2)
    context.results["diameters"] = diameters
    return {"measurable": diameters.get("measurable"),
            "diameters_mm": diameters.get("max_perpendicular_diameters_mm")}


def _tool_vasari(context: CaseContext, **_) -> dict:
    # reuse the T1->MNI transform compute_atlas_features wrote for THIS T1, instead of a second SyN
    cached = atlas_features.cached_transform(context.mri["t1"], os.path.join(context.out_dir, "xfm"))
    reuse = ({"moving": context.mri["t1"], "fwdtransforms": cached["fwdtransforms"],
              "invtransforms": cached["invtransforms"]} if cached else None)

    vasari = run_vasari(context.seg_path, anat_path=context.mri.get("t1ce"), reuse_transform=reuse)
    context.results["vasari"] = vasari
    # VASARI's own F9 multifocal is not used as a finding (seg_features.enhancing_foci is)
    return {"quality": vasari.get("enhancement_quality"),
            "ependymal_extension": vasari.get("ependymal_extension"),
            "registration": "reused" if reuse else "new"}


def _tool_t2fm(context: CaseContext, **_) -> dict:
    # vision VQA over 3 z-thirds slices, reusing the orchestrator's own backbone
    from tools import tool_t2fm

    result = tool_t2fm.predict_t2fm(context.seg_path, context.mri["t2"], context.mri["flair"],
                                    model=getattr(context.client, "_model", None),
                                    proc=getattr(context.client, "_processor", None),
                                    save_panel_dir=context.out_dir)
    context.results["t2fm"] = result
    return {"present": result["t2_flair_mismatch"], "extent": result["extent"],
            "confidence": result.get("confidence"), "panel_pngs": result.get("panel_pngs"),
            "panels": len(result.get("panel_pngs") or [])}


_TOOL_FUNCTIONS = {
    "segment_tumor": _tool_segment, "predict_idh": _tool_idh, "predict_grade": _tool_grade,
    "predict_1p19q": _tool_1p19q, "compute_seg_features": _tool_seg_features,
    "compute_atlas_features": _tool_atlas_features, "measure_diameters": _tool_diameters,
    "run_vasari": _tool_vasari, "run_t2fm": _tool_t2fm,
}

# what each tool reads, printed on the trace's start line
_TOOL_INPUTS = {
    "segment_tumor": "t1 t1ce t2 flair",
    "predict_idh": "t1 t1ce t2 flair + seg + age/sex",
    "predict_grade": "GlioMT cache",
    "predict_1p19q": "GlioMT cache",
    "compute_seg_features": "seg + t1ce t2 flair",
    "compute_atlas_features": "t1 + seg -> MNI",
    "measure_diameters": "seg",
    "run_vasari": "seg + t1ce",
    "run_t2fm": "seg + t2 flair (3 slices)",
}


def _format_mm(value) -> str:
    # trace-only rounding; report.json keeps full precision
    return f"{value:.2f} mm" if isinstance(value, (int, float)) else str(value)


_DONE_SUMMARY = {
    "segment_tumor": lambda out: (f"shape={out.get('shape')} ET={out.get('et_cm3'):.1f} "
                                  f"NE={out.get('ne_cm3'):.1f} WT={out.get('wt_cm3'):.1f} cm3"),
    "predict_idh": lambda out: (f"IDH={out.get('idh')} (p={out.get('probability'):.2f}) "
                                f"input={out.get('input')}"),
    "predict_grade": lambda out: f"grade={out.get('grade')}",
    "predict_1p19q": lambda out: (f"1p19q={out.get('1p19q')}" if out.get("applicable")
                                  else "SKIPPED-worthy (IDH-wt, n/a)"),
    "compute_seg_features": lambda out: (f"necrosis={out.get('necrosis')} "
                                         f"edema={out.get('edema_present')} "
                                         f"enh={out.get('enhancement')} "
                                         f"multifocal={out.get('multifocal')}"),
    "compute_atlas_features": lambda out: (f"loc={out.get('laterality')} {out.get('lobe')} "
                                           f"midline={_format_mm(out.get('midline_shift_mm'))} "
                                           f"mni={out.get('mni_shape')}"),
    "measure_diameters": lambda out: (f"measurable={out.get('measurable')} "
                                 f"diameters={out.get('diameters_mm')}"),
    "run_vasari": lambda out: (f"quality={out.get('quality')} "
                               f"reg={out.get('registration')}"),
    "run_t2fm": lambda out: (f"t2fm={out.get('present')} extent={out.get('extent')} "
                             f"panels={out.get('panels')}"),
}


TOOL_SCHEMAS = [
    tool_schema("segment_tumor", "Run HD-GLIO segmentation -> ET/NE/WT volumes + laterality. MANDATORY, run first."),
    tool_schema("predict_idh", "GlioMT IDH status + probability. MANDATORY."),
    tool_schema("predict_grade", "GlioMT CNS WHO grade. MANDATORY (always, IDH-independent)."),
    tool_schema("predict_1p19q", "GlioMT 1p/19q codeletion. Call ONLY if IDH is mutant; SKIP for IDH-wildtype (glioblastoma; not applicable per WHO 2021)."),
    tool_schema("compute_seg_features", "Deterministic seg features: necrosis, edema, enhancement (T2-FLAIR mismatch comes from run_t2fm). MANDATORY (findings, run regardless of IDH)."),
    tool_schema("compute_atlas_features", "MNI atlas features: lobe location, laterality, midline shift, cortical involvement. MANDATORY."),
    tool_schema("measure_diameters", "Bidimensional tumour size (whole-tumour + enhancing diameters). MANDATORY."),
    tool_schema("run_vasari", "VASARI features (enhancement quality, ependymal extension). MANDATORY."),
    tool_schema("run_t2fm", "Qwen vision T2-FLAIR mismatch VQA over 3 z-thirds slices. MANDATORY "
                "(imaging finding, run regardless of IDH). Returns present + extent (partial/complete); "
                "partial supports IDH-mutant, complete supports astrocytoma (weigh against GlioMT)."),
]


# ---- LLM-backed steps ----

def _generate_assessment(client, structured_findings: dict, molecular: dict,
                         knowledge_chunks: list, warnings: list, report: dict) -> str:
    # LLM review of the rule-based report; never invents values, "" on failure
    if client is None:
        return ""
    knowledge = "\n\n".join(f"[{chunk.get('title','')}] {chunk.get('chunk','')}"
                            for chunk in knowledge_chunks)
    idh = molecular.get("idh") or {}
    retracted = [f"- {name}: {value['retracted']}"
                 for name, value in (structured_findings.get("findings") or {}).items()
                 if isinstance(value, dict) and value.get("retracted")]
    user_message = (
        "RULE-BASED REPORT (deterministic; values are final):\n"
        + report.get("structured_report_text", "") + "\n\n"
        f"MOLECULAR CALL: {molecular.get('who_2021','?')} "
        f"(IDH {idh.get('prediction','?')} p={idh.get('probability','?')}).\n\n"
        "CONCORDANCE FLAGS RAISED BY THE PIPELINE:\n"
        + ("\n".join(warnings) if warnings else "(none)") + "\n\n"
        "FINDINGS WITHDRAWN BY THE PIPELINE:\n"
        + ("\n".join(retracted) if retracted else "(none)") + "\n\n"
        "REFERENCE KNOWLEDGE:\n" + (knowledge or "(none)") + "\n\n"
        + _NECROSIS_CAVEAT + "\n\n"
        "Write the assessment now."
    )
    try:
        answer = client.chat([{"role": "system", "content": _ASSESSMENT_SYSTEM},
                              {"role": "user", "content": user_message}],
                             max_new_tokens=1024, enable_thinking=False)
        return ascii_fold(strip_tool_xml(answer)).strip()
    except Exception:
        return ""



def _fmt_fraction(value) -> str:
    """The enhancing fraction for the reconciliation prompt; the guardrail reads the same number."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "not measured"


def _reconcile_idh(client, molecular: dict, structured_findings: dict, age,
                   knowledge_chunks: list, warnings: list, vqa_mismatch: dict | None = None):
    # final IDH from GlioMT + age + imaging + knowledge; returns (call, reasoning)
    idh = molecular.get("idh", {})
    prediction = idh.get("prediction")
    probability = idh.get("probability")
    p_mutant = probability if prediction == "mutant" else (
        1 - probability if probability is not None else 0.5)
    findings = structured_findings.get("findings", {})
    necrosis = findings.get("necrosis", {}).get("present")
    enhancement = findings.get("enhancement", {})
    location = findings.get("location", {})

    if vqa_mismatch and vqa_mismatch.get("extent") in ("complete", "partial"):
        prominent_enhancement = (bool(enhancement.get("present"))
                                 and (enhancement.get("proportion") or 0) >= 0.15)
        reliable = not (necrosis or prominent_enhancement)
        mismatch_line = (f"T2-FLAIR mismatch (vision model) = {vqa_mismatch['extent']} "
                         f"(confidence {vqa_mismatch.get('confidence')}); reliability = "
                         + ("OK" if reliable else "LOW -- necrosis/prominent enhancement present, "
                            "do NOT flip on this mismatch (Do 2022)"))
    else:
        mismatch_line = f"T2-FLAIR mismatch sign={findings.get('mismatch', {}).get('present')}"

    knowledge = "\n".join(f"[{chunk.get('title','')}] {chunk.get('chunk','')}"
                          for chunk in knowledge_chunks[:3])
    user_message = (
        f"GlioMT IDH prediction: {prediction} (P(mutant)={p_mutant:.2f}).\n"
        f"Patient age: {age}.\n"
        f"Imaging: {mismatch_line}; central necrosis={necrosis}; "
        f"enhancement present={enhancement.get('present')} quality={enhancement.get('quality')} "
        f"fraction={_fmt_fraction(enhancement.get('proportion'))}; "
        f"location={location.get('laterality')} {location.get('lobe')}.\n\n"
        f"CONCORDANCE FLAGS:\n{chr(10).join(warnings) if warnings else '(none)'}\n\n"
        f"REFERENCE:\n{knowledge}\n\n{_NECROSIS_CAVEAT}\n\nFinal IDH call:"
    )
    try:
        answer = ascii_fold(strip_tool_xml(client.chat(
            [{"role": "system", "content": _IDH_RECONCILE_SYSTEM},
             {"role": "user", "content": user_message}], enable_thinking=False))).strip()
    except Exception:
        return prediction, ""
    first_line = (answer.splitlines()[0] if answer else "").lower()
    final = "mutant" if "mutant" in first_line else (
        "wildtype" if "wildtype" in first_line else prediction)
    return final, answer


def _apply_idh_reconciliation(context: CaseContext, molecular: dict, new_idh: str) -> dict:
    # re-derive the WHO 2021 diagnosis; flipping to mutant needs the 1p/19q head
    from protocol.who2021_rules import derive_who2021

    original = molecular["idh"]["prediction"]
    molecular = dict(molecular)
    molecular["idh"] = {**molecular["idh"], "prediction": new_idh,
                        "reconciled_from": original, "source": "agent_reconciled"}
    if new_idh == "mutant":
        codeletion = "non-codeleted"
        try:
            from tools.gliomt.gliomt_inference import predict as gliomt_predict
            raw = gliomt_predict(t1=context.mri["t1"], t1ce=context.mri["t1ce"], t2=context.mri["t2"],
                                 flair=context.mri["flair"], seg=context.seg_path, age=context.age,
                                 sex=context.sex, conditional_1p19q=False)
            codeletion = raw.get("p1_19q_prediction", codeletion)
            probability = raw.get("p1_19q_prob_codel", 0.5)
            molecular["1p19q"] = {
                "prediction": codeletion,
                "probability": float(probability if codeletion == "codeleted" else 1 - probability)}
        except Exception:
            molecular.pop("1p19q", None)
        who_diagnosis, rule_grade = derive_who2021("mutant", codeletion)
        molecular["who_2021"] = who_diagnosis
        grade = molecular.get("grade_prediction") or rule_grade
        # oligodendroglioma is CNS WHO grade 2-3 only; clip an invalid grade-4 head output
        if codeletion == "codeleted" and grade == 4:
            context.react_log.note("[GRADE-CLIP] grade head predicted 4 for a codeleted tumour "
                        "(oligodendroglioma is grade 2-3) -> clipped to 3")
            grade = 3
        molecular["grade_prediction"] = grade
    else:
        molecular["who_2021"] = "Glioblastoma, IDH-wildtype, CNS WHO Grade 4"
        molecular["grade_prediction"] = 4
        molecular.pop("1p19q", None)
    context.results["molecular"] = molecular
    context.molecular_cache = molecular
    return molecular


# ---- tool dispatch ----

def _mandatory_for(context: CaseContext) -> list[str]:
    # 1p/19q only matters for IDH-mutant tumours
    required = list(MANDATORY_BASE)
    if (context.molecular_cache or {}).get("idh", {}).get("prediction") == "mutant":
        required.append("predict_1p19q")
    return required


def _dispatch(context: CaseContext, name: str, args: dict, forced: bool = False) -> str:
    # single gate for every tool call: run, record, summarise
    tool_function = _TOOL_FUNCTIONS.get(name)
    if tool_function is None:
        return json.dumps({"error": f"unknown tool {name}"})
    # every other tool reads THIS run's mask; a stale hdglio_seg.nii.gz in the out-dir must not be used
    if name != "segment_tumor" and "segment_tumor" not in context.called:
        context.react_log.note(f"[GUARD] segment_tumor must run before {name} -> forcing")
        _dispatch(context, "segment_tumor", {}, forced=True)
    tag = "GUARD-forced " if forced else ""
    if not forced:
        context.react_log.action(name)
    context.react_log.start(name, f"{tag}in: {_TOOL_INPUTS.get(name, '-')}")

    started = time.time()
    try:
        output = tool_function(context, **(args or {}))
    except Exception as error:
        output = {"error": f"{type(error).__name__}: {error}"}
    context.seconds[name] = time.time() - started
    context.tool_outputs[name] = output
    if name not in context.called:
        context.called.append(name)

    if isinstance(output, dict) and "error" in output:
        context.react_log.failed(name, f"{tag}{output['error']}"[:240])
    else:
        summary = (_DONE_SUMMARY.get(name, lambda r: "")(output)
                   if isinstance(output, dict) else str(output)[:80])
        context.react_log.done(name, f"{tag}out: {summary}   [{context.seconds[name]:.0f}s]")
    return json.dumps(output, default=str)


def _failed_mandatory(context: CaseContext) -> list[str]:
    # a mandatory tool that errored makes every downstream number meaningless
    return [name for name in _mandatory_for(context)
            if isinstance(context.tool_outputs.get(name), dict) and "error" in context.tool_outputs[name]]


# ---- pipeline ----

def run_pipeline(patient_id, mri_paths, age, sex, out_dir,
                 cohort=None, client: Optional[LLMClient] = None,
                 hdglio_gpu=os.environ.get("GLIOAGENT_GPU", "0"),
                 max_iters=16, verbose: Optional[bool] = None) -> dict:

    """LLM-orchestrated pipeline for one case. Writes glioagent_trace.json, report.json
    and trace.md into `out_dir` and returns the trace dict."""

    # absolute paths up front: VASARI chdirs into a temp dir, which breaks relative inputs
    out_dir = os.path.abspath(out_dir)
    mri_paths = {name: (os.path.abspath(path) if path else path)
                 for name, path in mri_paths.items()}
    os.makedirs(out_dir, exist_ok=True)
    if client is None:
        client = get_client()

    context = CaseContext(patient_id, mri_paths, age, sex, out_dir, cohort, hdglio_gpu, verbose)
    context.client = client
    sequence_qc = {"sequences_present": [name.upper() for name, path in mri_paths.items()
                                         if path and os.path.isfile(path)]}
    context.react_log.header(cohort, patient_id, age, sex, sequence_qc["sequences_present"])
    context.react_log.note("[MODE] LLM-orchestrated (agentic ReAct)")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Case: {age}-year-old {sex}, cohort {cohort or '?'}. Sequences: "
                    f"T1, T1CE, T2, FLAIR available. Segmentation to be run. "
                    f"Orchestrate the workup."}]

    # 1. ReAct loop: the LLM picks tools, their outputs feed back into the conversation
    context.react_log.console("--- tool selection + execution (ReAct) ---")
    for _ in range(max_iters):
        llm_started = time.time()
        answer = client.chat(messages, tools=TOOL_SCHEMAS, enable_thinking=False)
        context.llm_seconds += time.time() - llm_started
        calls = parse_tool_calls(answer)
        thought = strip_tool_xml(answer)
        if thought and thought.upper() != "DONE":
            context.react_log.thought(thought[:180])
        messages.append({"role": "assistant", "content": answer})
        if not calls:
            break
        for call in calls:
            output = _dispatch(context, call["name"], call.get("arguments", {}) or {})
            if call["name"] in _TOOL_FUNCTIONS and call["name"] not in context.llm_called:
                context.llm_called.append(call["name"])   # first occurrence only
            messages.append({"role": "tool", "name": call["name"], "content": output})
        if all(tool in context.called for tool in _mandatory_for(context)):
            break
    else:
        context.react_log.note(f"[ORCHESTRATOR] max_iters={max_iters} reached before all "
                    "mandatory tools ran -> the guard below will force the rest")

    # 2. guard: predict_idh first, because it decides whether 1p/19q is mandatory
    if "predict_idh" not in context.called:
        context.react_log.note("[GUARD] predict_idh not called by LLM -> forcing")
        _dispatch(context, "predict_idh", {}, forced=True)
    for tool in _mandatory_for(context):
        if tool not in context.called:
            context.react_log.note(f"[GUARD] mandatory tool '{tool}' not called -> forcing")
            _dispatch(context, tool, {}, forced=True)

    failed = _failed_mandatory(context)
    if failed:
        raise RuntimeError("mandatory tool(s) failed: " + "; ".join(
            f"{name}: {context.tool_outputs[name]['error']}" for name in failed))

    # 3. findings: collect -> concordance check -> evidence-based retraction
    context.react_log.console("--- findings, concordance check, IDH reconciliation ---")
    seg = context.results.get("seg") or {}
    molecular = context.molecular_cache or {}
    diameters = context.results.get("diameters") or {}
    vasari = context.results.get("vasari") or {}
    deterministic_features = {**(context.results.get("seg_features") or {}),
                              **(context.results.get("atlas_features") or {})}
    t2fm = context.results.get("t2fm")

    def findings_pass(molecular):
        # warnings see the raw findings, then unreliable ones are withdrawn (present=None);
        # IDH reconciliation is told an unreliable mismatch must not drive a flip (prompt)
        findings = _collect_subresults(seg, vasari, diameters, molecular,
                                       det_features=deterministic_features,
                                       qc=sequence_qc, age=age, sex=sex)
        # The T2-FLAIR mismatch sign comes from the vision VQA tool and nowhere else;
        # the deterministic rim-core rule this used to override was removed from
        # tool_seg_features, so without the VQA result there is no value to fall back on
        # and the finding is recorded as not assessed rather than left absent.
        if t2fm is not None:
            present = bool(t2fm.get("t2_flair_mismatch"))
            findings.setdefault("findings", {})["mismatch"] = {
                "present": present,
                "extent": (t2fm.get("extent") if present else "none"),
                "confidence": t2fm.get("confidence"),
                "source": "vqa_qwen_t2fm",
            }
        else:
            findings.setdefault("findings", {})["mismatch"] = {
                "present": None, "extent": None, "confidence": None,
                "source": "not_run",
            }
        warnings = check_discordance(findings, molecular, age)
        findings, retract_notes = retract_unreliable(findings, molecular)
        warnings += retract_notes
        if not (findings.get("findings", {}).get("location") or {}).get("reliable"):
            warnings.append("[ALERT] atlas registration unavailable -> location is coarse "
                            "(HD-GLIO centroid), not canonical")
        return findings, warnings

    structured_findings, warnings = findings_pass(molecular)
    context.react_log.note(f"[COLLECT] structured_findings assembled "
                f"({len(structured_findings.get('findings', {}))} fields)")
    for warning in warnings:
        context.react_log.note(warning)

    knowledge_chunks = select_chunks(molecular, structured_findings, warnings)

    # 4. IDH reconciliation, with a code guardrail over the LLM's proposal
    gliomt_idh = molecular.get("idh", {}).get("prediction")
    gliomt_confidence = molecular.get("idh", {}).get("probability")
    vqa_mismatch = ({"extent": t2fm.get("extent"), "confidence": t2fm.get("confidence")}
                    if (t2fm or {}).get("t2_flair_mismatch") else None)
    llm_started = time.time()
    reconciled_idh, reconciliation_reason = _reconcile_idh(
        client, molecular, structured_findings, age, knowledge_chunks, warnings,
        vqa_mismatch=vqa_mismatch)
    context.llm_seconds += time.time() - llm_started

    guardrail_fired = False
    if (reconciled_idh and reconciled_idh != gliomt_idh
            and isinstance(gliomt_confidence, (int, float))
            and gliomt_confidence >= GLIOMT_CONFIDENT_P):
        warnings.append(f"[INFO] reconciliation proposed {gliomt_idh}->{reconciled_idh}, but GlioMT "
                        f"confidence {gliomt_confidence:.2f} >= {GLIOMT_CONFIDENT_P}; "
                        "kept GlioMT call (guardrail)")
        context.react_log.note(f"[GUARDRAIL] kept GlioMT={gliomt_idh} (conf {gliomt_confidence:.2f}); "
                    f"discarded proposed {reconciled_idh}")
        guardrail_fired = True
        reconciled_idh = gliomt_idh

    enhancing_fraction = (structured_findings.get("findings", {}).get("enhancement") or {}).get("proportion")
    if reconciled_idh and reconciled_idh != gliomt_idh:
        if reconciled_idh == "wildtype" and not ALLOW_MUTANT_TO_WILDTYPE:
            discarded = ("mutant->wildtype reconciliation is not accepted: the necrosis finding is "
                         "enclosed non-enhancing tissue, not a necrosis label")
        elif reconciled_idh == "mutant" and not isinstance(enhancing_fraction, (int, float)):
            discarded = "the enhancing fraction could not be measured"
        elif (reconciled_idh == "mutant"
              and enhancing_fraction >= MAX_ENHANCING_FRACTION_FOR_MUTANT):
            discarded = (f"enhancing fraction {enhancing_fraction:.2f} >= "
                         f"{MAX_ENHANCING_FRACTION_FOR_MUTANT}; an enhancing tumour is not "
                         "reconciled to IDH-mutant")
        else:
            discarded = None
        if discarded:
            warnings.append(f"[INFO] reconciliation proposed {gliomt_idh}->{reconciled_idh}, but "
                            f"{discarded}; kept GlioMT call (imaging guardrail)")
            context.react_log.note(f"[GUARDRAIL] kept GlioMT={gliomt_idh}; discarded proposed "
                        f"{reconciled_idh} ({discarded})")
            imaging_guardrail_fired = True
            reconciled_idh = gliomt_idh
        else:
            imaging_guardrail_fired = False
    else:
        imaging_guardrail_fired = False

    if reconciled_idh and reconciled_idh != gliomt_idh:
        molecular = _apply_idh_reconciliation(context, molecular, reconciled_idh)
        context.react_log.note(f"[IDH-RECONCILE] GlioMT={gliomt_idh} -> agent={reconciled_idh} "
                    f"({molecular.get('who_2021','?')})")
        structured_findings, warnings = findings_pass(molecular)
        for warning in warnings:
            context.react_log.note(warning)
        knowledge_chunks = select_chunks(molecular, structured_findings, warnings)
    context.results["idh_reconciliation"] = {
        "gliomt": gliomt_idh, "final": reconciled_idh,
        "changed": reconciled_idh != gliomt_idh, "reason": reconciliation_reason,
        "guardrail_fired": guardrail_fired, "gliomt_confidence": gliomt_confidence,
        "imaging_guardrail_fired": imaging_guardrail_fired,
        "enhancing_fraction": enhancing_fraction}

    # 5. astro-oligo gate: uncertain codeletion + confident mismatch -> astrocytoma
    astrooligo_fired = False
    codeletion = molecular.get("1p19q") or {}
    codeletion_probability = codeletion.get("probability")
    mismatch_present = (structured_findings.get("findings", {}).get("mismatch") or {}).get("present")
    if (molecular.get("idh", {}).get("prediction") == "mutant"
            and codeletion.get("prediction") == "codeleted"
            and isinstance(codeletion_probability, (int, float))
            and codeletion_probability < CODEL_CONFIDENT_P
            and mismatch_present):
        from protocol.who2021_rules import derive_who2021

        molecular = dict(molecular)
        molecular["1p19q"] = {**codeletion, "prediction": "non-codeleted",
                              "reconciled_from": "codeleted",
                              "source": "agent_mismatch_reconciled"}
        molecular["who_2021"] = derive_who2021("mutant", "non-codeleted")[0]
        context.results["molecular"] = molecular
        context.molecular_cache = molecular
        gate_note = (f"[INFO] confident T2-FLAIR mismatch with uncertain codeletion "
                     f"(p={codeletion_probability:.2f} < {CODEL_CONFIDENT_P}) -> favouring "
                     "astrocytoma over oligodendroglioma (mismatch specific for non-codeletion)")
        context.react_log.note(f"[ASTRO-OLIGO] codeleted (p={codeletion_probability:.2f}) + confident "
                    "mismatch -> non-codeleted (astrocytoma)")
        astrooligo_fired = True
        # structured_findings copies who_2021/1p19q/grade, so rebuild it against the new call
        structured_findings, warnings = findings_pass(molecular)
        warnings.append(gate_note)
        knowledge_chunks = select_chunks(molecular, structured_findings, warnings)
        for warning in warnings:
            context.react_log.note(warning)
    context.results["idh_reconciliation"]["astrooligo_fired"] = astrooligo_fired

    # 6. rule-based report, LLM assessment, checklist, confidence tier
    context.react_log.console("--- report assembly + checklist ---")
    rag_hits = [{"title": chunk.get("title") or chunk.get("source", ""),
                 "chunk": chunk.get("chunk", "")} for chunk in knowledge_chunks]
    report = _assemble_report(sequence_qc, seg, vasari, diameters, molecular, rag_hits, warnings,
                              structured_findings=structured_findings)
    llm_started = time.time()
    assessment = _generate_assessment(client, structured_findings, molecular,
                                      knowledge_chunks, warnings, report)
    context.llm_seconds += time.time() - llm_started
    if assessment:
        report["clinical_assessment"] = assessment
        context.react_log.note(f"[ASSESSMENT] knowledge-grounded review ({len(knowledge_chunks)} chunks) generated")
    context.react_log.note("[REPORT] structured report generated")

    checklist = verify_report(report=report, seg=seg, molecular=molecular,
                              vasari=vasari, diameters=diameters)
    case_trace = GlioAgentTrace(patient_id=patient_id)
    case_trace.tumor_segment = seg
    case_trace.molecular = molecular
    case_trace.vasari = vasari
    case_trace.measure_diameters = diameters
    case_trace.structured_findings = structured_findings
    case_trace.report = report
    case_trace.warnings = warnings          # _confidence_tier reads hard-discordance flags
    case_trace.checklist = (asdict(checklist) if hasattr(checklist, "__dataclass_fields__")
                            else checklist)
    case_trace.confidence_tier = _confidence_tier(molecular, case_trace, checklist.passed)
    # re-render WITH the post-checklist tier, byte-identical to the co-pilot redraft path
    if isinstance(report.get("structured_report_text"), str):
        report["structured_report_text"] = _render_structured_report_text(
            structured_findings, confidence_tier=case_trace.confidence_tier)
    case_trace.tool_decisions["agentic_plan"] = list(context.llm_called)   # guard-forced excluded
    case_trace.tool_decisions["mode"] = "agentic"
    if context.results.get("seg_cleanup"):
        case_trace.tool_decisions["seg_cleanup"] = context.results["seg_cleanup"]
    case_trace.tool_decisions["idh_reconciliation"] = context.results["idh_reconciliation"]
    context.react_log.done("checklist", f"{checklist.score} pass")
    context.react_log.done("confidence", case_trace.confidence_tier)
    context.react_log.note("[TIMING] " + "  ".join(
        f"{name}={seconds:.0f}s" for name, seconds
        in sorted(context.seconds.items(), key=lambda item: -item[1]))
        + f"  |  tools={sum(context.seconds.values()):.0f}s backbone={context.llm_seconds:.0f}s")

    # 7. persist everything beside the case, so re-runs never overwrite another run
    with open(os.path.join(out_dir, "glioagent_trace.json"), "w") as handle:
        json.dump(asdict(case_trace), handle, indent=2, default=str)
    with open(os.path.join(out_dir, "report.json"), "w") as handle:
        json.dump({**report, "confidence_tier": case_trace.confidence_tier}, handle, indent=2)
    report_text = report.get("structured_report_text", "")
    if report.get("clinical_assessment"):
        report_text += "\n\nASSESSMENT (knowledge-grounded review):\n  " + report["clinical_assessment"]
    quantitative_summary = format_quantitative_summary(
        structured_findings, patient_id, case_trace.confidence_tier)
    trace_md_path = os.path.join(out_dir, TRACE_MD_FILE)
    context.react_log.note(f"[DONE] {patient_id} -> {TRACE_MD_FILE}")
    save_trace_md(trace_md_path, context.react_log, report_text, quantitative_summary, patient_id)
    return asdict(case_trace)


__all__ = ["run_pipeline"]
