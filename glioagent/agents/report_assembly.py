"""Report assembly: format tool outputs into structured findings and report text.

Values come verbatim from the tools; these functions never recompute imaging values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GlioAgentTrace:
    patient_id: str
    input_qc: dict = field(default_factory=dict)
    tumor_segment: dict = field(default_factory=dict)
    vasari: dict = field(default_factory=dict)
    measure_diameters: dict = field(default_factory=dict)
    molecular: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    structured_findings: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    checklist: dict = field(default_factory=dict)
    confidence_tier: str = "MODERATE"
    tool_decisions: dict = field(default_factory=dict)


_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOW_FREQUENCY_FINDINGS = ("multifocal", "ependymal_extension")

# Edema has no qualitative tier: HD-GLIO's NE label combines oedema + non-enhancing
# tumor (not separable) and the radial-mm proxy tracks tumor extent, not oedema, so
# it is reported binary present/absent only; radial_mm is retained as a [PROXY] datum.


def _is_hard_discordance(warning: str) -> bool:
    """Hard-discordance check used by the confidence tier. `[INFO]` warnings are
    never hard; `[ALERT]` warnings are hard unless the text softens the claim."""
    if warning.startswith("[INFO]"):
        return False
    softening_terms = ("partial", "equivocal", "borderline", "supporting")
    if warning.startswith("[ALERT]"):
        return not any(term in warning.lower() for term in softening_terms)
    warning_lower = warning.lower()
    if not ("discord" in warning_lower or "contradict" in warning_lower):
        return False
    return not any(term in warning_lower for term in softening_terms)


def _confidence_tier(molecular: dict, trace: GlioAgentTrace, checklist_pass: bool) -> str:
    """Tier the case as HIGH / MODERATE / LOW.

    IDH probability is the primary signal; 1p/19q (IDH-mutant only) and grade are
    secondary. HIGH additionally requires grade_probability >= 0.6 -- an uncertain
    grade prediction (which FINAL DIAGNOSIS still states as a definite WHO grade)
    caps the tier at MODERATE. A failed checklist or a hard-discordance warning
    forces LOW.
    """
    if not checklist_pass:
        return "LOW"
    if any(_is_hard_discordance(warning) for warning in trace.warnings):
        return "LOW"
    if molecular.get("stub"):
        return "MODERATE"

    idh_block = molecular.get("idh") or {}
    idh_probability = idh_block.get("probability")
    idh_prediction = idh_block.get("prediction")
    if idh_probability is None:
        return "LOW"

    secondary_probabilities: list[float] = []
    codeletion_block = molecular.get("1p19q") or {}
    if idh_prediction == "mutant" and codeletion_block.get("probability") is not None:
        secondary_probabilities.append(codeletion_block["probability"])

    grade_probability = molecular.get("grade_probability")
    grade_ok = (not isinstance(grade_probability, (int, float))) or grade_probability >= 0.6

    if idh_probability >= 0.8 and all(probability >= 0.6 for probability in secondary_probabilities) and grade_ok:
        return "HIGH"
    if idh_probability >= 0.6:
        return "MODERATE"
    return "LOW"


def _collect_subresults(
    seg: dict,
    vasari: dict,
    diameters: dict,
    molecular: dict,
    *,
    det_features: Optional[dict] = None,
    qc: Optional[dict] = None,
    age: Optional[int] = None,
    sex: Optional[str] = None,
) -> dict:
    """Normalise every tool's sub-result into one `structured_findings` dict.

    `det_features` numerics override VASARI's unreliable fields. Missing
    sub-results degrade gracefully (fields become None); never raises.
    """
    seg = seg or {}
    vasari = vasari or {}
    diameters = diameters or {}
    molecular = molecular or {}
    deterministic_features = det_features or {}
    has_atlas = bool(deterministic_features.get("location") or deterministic_features.get("mass_effect")
                     or deterministic_features.get("laterality"))
    unreliable_fields = set(vasari.get("_unreliable_fields") or [])
    # VASARI omits F5 proportion_enhancing from its own unreliable list; Ruffle 2024
    # reports F5 accuracy 49.42%, so we add it explicitly.
    unreliable_fields.add("proportion_enhancing")

    feature_volumes = deterministic_features.get("volumes") or {}
    et = feature_volumes.get("et_cm3", seg.get("et_cm3"))
    ne = feature_volumes.get("ne_cm3", seg.get("ne_cm3"))
    wt = feature_volumes.get("wt_cm3", seg.get("wt_cm3"))
    volumes = {
        "et_cm3": et, "ne_cm3": ne, "wt_cm3": wt,
        "provenance": "seg_features" if feature_volumes else "HD-GLIO summarise_seg",
    }

    feature_enhancement = deterministic_features.get("enhancement") or {}
    enhancement_present = feature_enhancement.get("present")
    enhancement_proportion = feature_enhancement.get("proportion")
    if enhancement_present is None and et is not None and wt:
        enhancement_present = bool(et > 0.10)
        enhancement_proportion = (et / wt) if wt else None
    enhancement = {
        "present": enhancement_present,
        "proportion": enhancement_proportion,
        "enhancement_z": feature_enhancement.get("enhancement_z"),
        "quality": feature_enhancement.get("quality"),
        "provenance": "seg_features (present/proportion/quality)",
    }

    # Necrosis = non-enhancing tissue ENCLOSED by enhancing tumor (proxy for central
    # necrosis). HD-GLIO has no necrosis label, so this underestimates true necrosis.
    feature_necrosis = deterministic_features.get("necrosis") or {}
    necrosis_cm3 = feature_necrosis.get("necrosis_cm3")
    necrosis = {
        "present": feature_necrosis.get("present"),
        "necrosis_cm3": necrosis_cm3,
        "necrosis_over_et": (necrosis_cm3 / et if isinstance(necrosis_cm3, (int, float))
                             and isinstance(et, (int, float)) and et > 0 else None),
        "provenance": "seg_features.necrosis (enclosed-NE; underestimates true necrosis)",
        "overrides_vasari": "proportion_necrosis" in unreliable_fields,
    }

    # The T2-FLAIR mismatch block is filled in by the orchestrator from the VQA tool
    # (run_t2fm); there is no deterministic fallback.
    mismatch = {"present": None, "extent": None, "confidence": None,
                "source": "run_t2fm not yet applied",
                "overrides_vasari": "t2_flair_mismatch" in unreliable_fields}

    feature_edema = deterministic_features.get("edema") or {}
    radial_mm = feature_edema.get("radial_mm")
    edema = {
        "present": bool(isinstance(radial_mm, (int, float)) and radial_mm > 0),
        "radial_mm": radial_mm,   # [PROXY] datum only -- not a tiered extent
        "provenance": "seg_features.edema_extent (radial mm from ET centroid) -- PROXY",
        "note": "binary present/absent only -- NE combines oedema + non-enhancing tumor "
                "and radial_mm reflects tumor spatial extent, not oedema volume",
        "overrides_vasari": "proportion_edema" in unreliable_fields,
    }

    feature_mass_effect = deterministic_features.get("mass_effect") or {}
    mass_effect = {
        "present": feature_mass_effect.get("present"),
        "midline_shift_mm": feature_mass_effect.get("midline_shift_mm"),
        "provenance": "atlas_features.midline_shift (MNI)" if has_atlas
                       else "unavailable (no atlas registration)",
    }

    feature_cortical = deterministic_features.get("cortical") or {}
    if feature_cortical.get("involved") is not None:
        cortical = {"involved": feature_cortical.get("involved"),
                     "overlap_frac": feature_cortical.get("overlap_frac"),
                     "provenance": "atlas_features.cortical_involvement (MNI GM ribbon)"}
    else:
        cortical = {"involved": vasari.get("cortical_involvement"),
                     "overlap_frac": None,
                     "provenance": "VASARI F20 (fallback -- atlas unavailable)"}

    feature_location = deterministic_features.get("location") or {}
    feature_laterality = deterministic_features.get("laterality") or deterministic_features.get("laterality_seg_only") or {}
    if has_atlas and feature_location:
        location = {
            "lobe": feature_location.get("lobe"),
            "region": feature_location.get("region"),
            "junction_lobes": feature_location.get("junction_lobes") or [],
            "laterality": feature_laterality.get("dominant_side"),
            "midline_crossing": feature_laterality.get("midline_crossing"),
            "reliable": True,
            "provenance": "atlas_features (MNI AAL116 + laterality) -- canonical",
        }
    else:
        location = {
            "lobe": seg.get("tumor_location") or vasari.get("tumor_location"),
            "region": vasari.get("tumor_location"),
            "laterality": seg.get("laterality") or feature_laterality.get("dominant_side")
                           or vasari.get("side"),
            "midline_crossing": feature_laterality.get("midline_crossing"),
            "reliable": False,
            "provenance": "HD-GLIO summarise + VASARI (atlas registration unavailable)",
        }

    # Multifocality comes from seg_features.enhancing_foci (>= 2 ET components >= 1 cm3 in ONE
    # whole-tumour component), not VASARI F9 (> 10 components on WT, which oedema fuses into one).
    feature_foci = deterministic_features.get("additional") or {}
    additional = {
        "multifocal": feature_foci.get("multifocal", vasari.get("multifocal")),
        "multifocal_foci_cm3": feature_foci.get("multifocal_foci_cm3") or [],
        "multicentric_foci_cm3": feature_foci.get("multicentric_foci_cm3") or [],
        "ependymal_extension": vasari.get("ependymal_extension"),
        "provenance": ("seg_features.enhancing_foci (>= 2 enhancing foci >= 1 cm3, same whole-tumour "
                       "component) + VASARI 2.0 F19 ependymal (kappa 0.33, lowest inter-rater agreement)"),
        "overrides_vasari": True,
    }

    # WHO 2021 gating: for IDH-wildtype the molecular dict has no 1p19q key, so
    # structured_findings omits it entirely (absent, not None).
    impression = {
        "who_2021": molecular.get("who_2021"),
        "idh": molecular.get("idh") or {},
        "grade": molecular.get("grade_prediction"),
        "stub": molecular.get("stub", False),
        "provenance": "GlioMT (external, public)"
                       + (" [stub]" if molecular.get("stub") else ""),
    }
    if "1p19q" in molecular:
        impression["1p19q"] = molecular["1p19q"]

    diameters_block = {
        "measurable": diameters.get("measurable"),
        "diameters_mm": diameters.get("max_perpendicular_diameters_mm"),  # whole tumor
        "enhancing_diameters_mm": diameters.get("enhancing_diameters_mm"),
        "enhancing_present": diameters.get("enhancing_present"),
        "product_mm2": diameters.get("product_mm2"),
        "provenance": "measure_diameters (whole-tumor bidimensional; enhancing secondary)",
    }

    not_assessable = list(deterministic_features.get("not_assessable") or
                          ["calcification", "diffusion", "perfusion"])
    for extra_item in ("hemorrhage/susceptibility (no SWI/GRE)",):
        if extra_item not in not_assessable:
            not_assessable.append(extra_item)

    sequences = (qc or {}).get("sequences_present") if qc else None
    sequences = ([str(sequence).upper() for sequence in sequences] if sequences
                 else [sequence.upper() for sequence in ("t1", "t1ce", "t2", "flair")])

    return {
        "clinical": {"age": age, "sex": sex},
        "technique": {"sequences": sequences},
        "findings": {
            "location": location,
            "volumes": volumes,
            "enhancement": enhancement,
            "necrosis": necrosis,
            "mismatch": mismatch,
            "edema": edema,
            "mass_effect": mass_effect,
            "cortical": cortical,
            # deep structures the lobar atlas cannot express (brainstem, corpus callosum) or buries
            # under the cortical bulk (thalamus, basal ganglia); trace-only, the prose is unchanged
            "deep_structures": deterministic_features.get("deep_structures") or {},
            "additional": additional,
        },
        "impression": impression,
        "diameters": diameters_block,
        "not_assessable": not_assessable,
        "_meta": {
            "atlas_available": has_atlas,
            "vasari_unreliable_fields": sorted(unreliable_fields),
        },
    }


def format_quantitative_summary(
    structured_findings: dict, sample_id: str = "?", confidence_tier: Optional[str] = None
) -> str:
    """Render the internal QUANTITATIVE SUMMARY block from structured_findings.

    This is the INTERNAL view -- it may include values deliberately excluded
    from the report prose (negative T2/FLAIR mismatch, confidence tier).
    """
    findings = structured_findings.get("findings", {})
    volumes = findings.get("volumes", {})
    mismatch = findings.get("mismatch", {})
    necrosis = findings.get("necrosis", {})
    edema = findings.get("edema", {})
    mass_effect = findings.get("mass_effect", {})
    impression = structured_findings.get("impression", {})
    idh = impression.get("idh", {})
    codeletion = impression.get("1p19q", {})

    def format_number(value, spec="{:.1f}"):
        try:
            return spec.format(float(value))
        except (TypeError, ValueError):
            return "n/a"

    lines = [f"=== QUANTITATIVE SUMMARY ({sample_id}) ==="]
    lines.append(f"Volumes:     ET {format_number(volumes.get('et_cm3'))} | NE {format_number(volumes.get('ne_cm3'))} "
                 f"| WT {format_number(volumes.get('wt_cm3'))} cm3")
    codeletion_text = (f"1p19q {codeletion.get('prediction')} (probability={format_number(codeletion.get('probability'),'{:.2f}')})"
                       if codeletion.get("prediction") else "1p19q n/a (IDH-wt)")
    lines.append(f"Molecular:   IDH {idh.get('prediction','?')} (probability={format_number(idh.get('probability'),'{:.2f}')}) "
                 f"| {codeletion_text} | grade {impression.get('grade','?')}")
    lines.append(f"Mismatch:    {'present' if mismatch.get('present') else 'absent'} "
                 f"(extent={mismatch.get('extent') or 'none'}, confidence={mismatch.get('confidence') or 'n/a'}, "
                 f"{mismatch.get('source') or 'n/a'})   [trace-only; report shows positive only]")
    necrosis_cm3 = necrosis.get("necrosis_cm3")
    et = volumes.get("et_cm3")
    if not necrosis.get("present"):
        necrosis_sanity = "n/a (no necrosis)"
    elif isinstance(necrosis_cm3, (int, float)) and isinstance(et, (int, float)):
        necrosis_sanity = "pass" if necrosis_cm3 < et else "FAIL"
    else:
        necrosis_sanity = "n/a"
    lines.append(f"Necrosis:    {'present' if necrosis.get('present') else 'absent'} "
                 f"({format_number(necrosis_cm3)} cm3, <ET: {necrosis_sanity})")
    lines.append(f"Edema:       {edema.get('tier','n/a')} (radial {format_number(edema.get('radial_mm'),'{:.0f}')} mm)")
    lines.append(f"Mass effect: midline {format_number(mass_effect.get('midline_shift_mm'),'{:.1f}')} mm "
                 f"({'present' if mass_effect.get('present') else 'absent'})")
    if confidence_tier:
        lines.append(f"Confidence:  {confidence_tier}   [internal triage; not in report]")
    meta = structured_findings.get("_meta", {})
    lines.append(f"Provenance:  atlas={meta.get('atlas_available')} "
                 f"| molecular -> {impression.get('provenance','?')}")
    lines.append("=" * 40)
    return "\n".join(lines)


def _fmt_cm3(value: Any) -> str:
    try:
        return f"{float(value):.1f} cm3"
    except (TypeError, ValueError):
        return "n/a"


def _dx_confident(molecular: dict, warnings: list) -> bool:
    """Diagnostic confidence for the CONCLUSION hedge: high IDH probability and no
    hard-discordance flag. Distinct from `_confidence_tier` (which also gates on
    the checklist and is computed after the report)."""
    idh_probability = (molecular.get("idh") or {}).get("probability")
    if not isinstance(idh_probability, (int, float)) or idh_probability < 0.8:
        return False
    return not any(_is_hard_discordance(warning) for warning in (warnings or []))


def _dx_line(impression: dict) -> str:
    """The ': diagnosis' line -- definitive when the molecular call is confident,
    otherwise an 'r/o ... favouring ...' hedge in radiologist grammar."""
    who = impression.get("who_2021") or "diffuse glioma"
    if impression.get("confident"):
        return f"  : {who}"
    if "Glioblastoma" in who:
        return "  : r/o high-grade glioma, favouring glioblastoma, IDH-wildtype"
    if "Oligodendroglioma" in who:
        return "  : r/o diffuse glioma, favouring oligodendroglioma, IDH-mutant, 1p/19q-codeleted"
    if "Astrocytoma" in who:
        return "  : r/o diffuse glioma, favouring astrocytoma, IDH-mutant"
    return f"  : r/o {who}"


def _english_join(items: list) -> str:
    """Oxford-comma join for narrative prose: [a] -> 'a'; [a,b] -> 'a and b';
    [a,b,c] -> 'a, b, and c'."""
    items = [str(item) for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


# Minimum enclosed-necrosis volume (cm3) to call the mass "centrally necrotic" in the
# IMPRESSION; below this a sub-cm3 focus in a large tumour does not warrant the claim.
_NECROSIS_MIN_CM3 = 1.0


def _lesion_character(enhancement_present, enhancement_quality, necrosis_significant,
                      edema_present, cortical, mass_effect_present):
    """A DESCRIPTIVE imaging characterisation of the lesion -- morphology only, with
    NO grade determination. IMPRESSION must not independently grade the tumour (that
    would contradict the molecular FINAL DIAGNOSIS); it describes what the mass looks
    like and defers the subtype/grade to GlioMT. `necrosis_significant` gates the
    'centrally necrotic' claim on a minimum necrosis volume (see _NECROSIS_MIN_CM3)."""
    quality = (enhancement_quality or "").lower()
    adjectives = []
    if enhancement_present:
        adjectives.append("avidly enhancing" if quality == "avid" else "minimally enhancing")
    else:
        adjectives.append("non-enhancing")
    if necrosis_significant:
        adjectives.append("centrally necrotic")
    core = ", ".join(adjectives) + " infiltrative mass"
    article = "An" if core[:1].lower() in "aeiou" else "A"
    associated_findings = []
    if edema_present:
        associated_findings.append("perilesional edema")
    if mass_effect_present:
        associated_findings.append("mass effect")
    if cortical:
        associated_findings.append("cortical involvement")
    sentence = f"{article} {core}"
    if associated_findings:
        sentence += " with " + _english_join(associated_findings)
    return sentence + "."


_EMIT_SUBTYPE_NAMES = True
# Multifocal gate: additional.multifocal is recomputed in finalize_report.py as >=2
# connected components on the ENHANCING (ET) mask each >=1 cm3 (not the old VASARI F9
# on WT, which fused everything through tumour-wide edema, 0/13 sensitivity).
_EMIT_MULTIFOCAL = True
# Head-prediction certainty cutoff: predicted-class prob must reach this to be
# "certain" (single candidate); below it a second candidate is named.
_SUBTYPE_CERTAIN_P = 0.70


def _subtype_candidates(impression: dict) -> list:
    """Ordered subtype-NAME candidates built from the head PREDICTIONS only (NO
    fabricated joint probability). Rules:
      - primary = the integrated argmax subtype (== FINAL DIAGNOSIS / who_2021 call);
      - a SECOND candidate is named only when the deciding head is uncertain
        (predicted-class prob < _SUBTYPE_CERTAIN_P):
          * IDH uncertain  -> add the other IDH branch
            (wildtype primary -> 'an IDH-mutant glioma', since 1p/19q is not evaluated
             for IDH-wildtype and so astro/oligo cannot be distinguished);
          * IDH certain-mutant but 1p/19q uncertain -> add the other 1p/19q branch;
      - oligodendroglioma appears only on the IDH-mutant branch.
    Returns [] when IDH prediction is unavailable."""
    idh = impression.get("idh") or {}
    codeletion = impression.get("1p19q") or {}
    idh_prediction = idh.get("prediction")
    idh_probability = idh.get("probability")
    if not idh_prediction:
        return []
    idh_certain = isinstance(idh_probability, (int, float)) and idh_probability >= _SUBTYPE_CERTAIN_P
    if idh_prediction == "wildtype":
        primary, idh_alternative, codeletion_alternative, codeletion_certain = "IDH-wildtype glioblastoma", "an IDH-mutant glioma", None, True
    else:  # mutant
        codeletion_prediction = codeletion.get("prediction"); codeletion_probability = codeletion.get("probability")
        codeletion_certain = isinstance(codeletion_probability, (int, float)) and codeletion_probability >= _SUBTYPE_CERTAIN_P
        if codeletion_prediction == "codeleted":
            primary, codeletion_alternative = "IDH-mutant, 1p/19q-codeleted oligodendroglioma", "IDH-mutant astrocytoma"
        else:
            primary, codeletion_alternative = "IDH-mutant astrocytoma", "IDH-mutant, 1p/19q-codeleted oligodendroglioma"
        idh_alternative = "IDH-wildtype glioblastoma"
    candidates = [primary]
    if not idh_certain:
        candidates.append(idh_alternative)                       # IDH borderline -> other IDH branch
    elif idh_prediction == "mutant" and not codeletion_certain and codeletion_alternative:
        candidates.append(codeletion_alternative)                # IDH certain-mutant, 1p/19q borderline
    return candidates


def _marginal_probs(impression: dict) -> str:
    """GlioMT MARGINAL head probabilities for the IMPRESSION molecular sentence:
    'IDH <pred> p=X · 1p/19q <pred> p=Y · grade <G> p=Z'. These are the model's actual
    per-head outputs (NOT a fabricated joint subtype product). 1p/19q is shown only on the
    IDH-mutant branch (WHO-2021 does not evaluate 1p/19q for IDH-wildtype). grade_probability
    lives on molecular and is backfilled into structured_findings['impression'] at
    assembly/finalize time."""
    idh = impression.get("idh") or {}
    parts = []
    idh_prediction, idh_probability = idh.get("prediction"), idh.get("probability")
    if idh_prediction and isinstance(idh_probability, (int, float)):
        parts.append(f"IDH {idh_prediction} p={idh_probability:.2f}")
    if idh_prediction == "mutant":
        codeletion = impression.get("1p19q") or {}
        codeletion_prediction, codeletion_probability = codeletion.get("prediction"), codeletion.get("probability")
        if codeletion_prediction and isinstance(codeletion_probability, (int, float)):
            parts.append(f"1p/19q {codeletion_prediction} p={codeletion_probability:.2f}")
    grade, grade_probability = impression.get("grade"), impression.get("grade_probability")
    if grade is not None and isinstance(grade_probability, (int, float)):
        parts.append(f"grade {grade} p={grade_probability:.2f}")
    return " · ".join(parts)


_SUBTYPE_LABELS = {
    "GBM": "IDH-wildtype glioblastoma",
    "A": "IDH-mutant astrocytoma",
    "O": "IDH-mutant, 1p/19q-codeleted oligodendroglioma",
}


def _subtype_ranking(impression: dict) -> list:
    """Ordered differential of the three diffuse-glioma subtypes (GPT-style). The PRIMARY
    is the integrated argmax call (== FINAL DIAGNOSIS / _integrated_dx_name); the other two
    are ranked as differentials by a marginal-PRODUCT score used for ORDER ONLY -- never
    shown as a fabricated joint probability (that was rejected: oligo=0 for IDH-wildtype is
    a 1p/19q-not-evaluated artifact). For IDH-wildtype the 1p/19q head is not evaluated, so
    oligodendroglioma carries a low codeletion prior so astrocytoma ranks as the principal
    differential."""
    idh = impression.get("idh") or {}
    idh_prediction, idh_probability = idh.get("prediction"), idh.get("probability")
    if not idh_prediction or not isinstance(idh_probability, (int, float)):
        return []
    mutant_probability = idh_probability if idh_prediction == "mutant" else (1 - idh_probability)
    wildtype_probability = 1 - mutant_probability
    codeletion = impression.get("1p19q") or {}
    codeletion_prediction, codeletion_probability = codeletion.get("prediction"), codeletion.get("probability")
    if idh_prediction == "mutant" and codeletion_prediction and isinstance(codeletion_probability, (int, float)):
        codeleted_probability = codeletion_probability if codeletion_prediction == "codeleted" else (1 - codeletion_probability)
    else:
        codeleted_probability = 0.15   # 1p/19q not evaluated for IDH-wildtype -> low oligodendroglioma prior
    score = {"GBM": wildtype_probability, "A": mutant_probability * (1 - codeleted_probability),
             "O": mutant_probability * codeleted_probability}
    if idh_prediction == "wildtype":
        primary = "GBM"
    elif codeletion_prediction == "codeleted":
        primary = "O"
    else:
        primary = "A"
    others = sorted((subtype_key for subtype_key in ("GBM", "A", "O") if subtype_key != primary),
                    key=lambda subtype_key: score[subtype_key], reverse=True)
    return [_SUBTYPE_LABELS[subtype_key] for subtype_key in ([primary] + others)]


def _subtype_name_sentence(impression: dict, confidence_tier=None) -> str:
    """IMPRESSION molecular sentence: the integrated subtype call + a RANKED differential of
    the other two subtypes (GPT-style: 'principal differential' / 'less likely'), plus the
    GlioMT marginal head probabilities. The lead-in matches the confidence tier so it never
    contradicts the 'Diagnostic confidence' line (HIGH stated definitively; MODERATE/LOW
    hedged with 'favors')."""
    ranked = _subtype_ranking(impression)
    if not ranked:
        return ""
    marginal_probs_text = _marginal_probs(impression)
    tail = f" ({marginal_probs_text})" if marginal_probs_text else ""
    lead = ("Molecular subtype prediction:" if confidence_tier == "HIGH"
            else "Molecular subtype prediction favors")
    if len(ranked) >= 3:
        return (f"{lead} {ranked[0]}{tail}; {ranked[1]} is the principal differential, "
                f"and {ranked[2]} is less likely.")
    if len(ranked) == 2:
        return f"{lead} {ranked[0]}{tail}; {ranked[1]} is the principal differential."
    return f"{lead} {ranked[0]}{tail}."


def _integrated_dx_name(impression: dict) -> str:
    """Top integrated subtype NAME from the argmax head predictions (== GlioMT's
    who_2021 call), formatted 'IDH-<status> <subtype>'. Probability-free."""
    idh = impression.get("idh") or {}
    codeletion = impression.get("1p19q") or {}
    idh_prediction = idh.get("prediction")
    if idh_prediction == "wildtype":
        return "IDH-wildtype glioblastoma"
    if idh_prediction == "mutant":
        if codeletion.get("prediction") == "codeleted":
            return "IDH-mutant, 1p/19q-codeleted oligodendroglioma"
        return "IDH-mutant astrocytoma"
    return impression.get("who_2021") or "diffuse glioma, molecular subtype indeterminate"


def _final_diagnosis_line(impression: dict, confidence_tier=None) -> str:
    """FINAL DIAGNOSIS: the top integrated subtype + WHO grade on ONE line, form
    INDEPENDENT of probability (uncertainty is carried by Diagnostic confidence)."""
    name = _integrated_dx_name(impression)
    grade = impression.get("grade")
    grade_clause = f", CNS WHO grade {grade}" if grade is not None else ""
    confidence_clause = f" Diagnostic confidence: {confidence_tier}." if confidence_tier else ""
    return f"{name}{grade_clause}.{confidence_clause}"


def _render_structured_report_text(structured_findings: dict, corrected_fields=None,
                                   confidence_tier=None) -> str:
    """Render `structured_findings` in GPT-5.6 report STYLE -- bold inline section
    labels, flowing prose, no bullet lists, no separate conclusion:

      **Findings:**     -- positive imaging findings as one NARRATIVE prose
                          paragraph: technique, the dominant mass, the observed
                          findings, the T2-FLAIR-mismatch status, an optional
                          multifocal sentence, and the tumour volumes.
      **Impression:**   -- a DESCRIPTIVE imaging characterisation of the lesion +
                          the tumour-type read ("diffuse glioma"). NO grade call:
                          IMPRESSION must not independently grade (that would
                          contradict FINAL DIAGNOSIS). Grading/subtyping is deferred.
      FINAL DIAGNOSIS:  -- the GlioMT 3-class subtype probabilities (glioblastoma /
                          astrocytoma / oligodendroglioma), a definitive call when
                          the top class >= threshold, else a probability listing.

    No sentence is duplicated across sections; the molecular layer (Impression vs
    Final diagnosis) is kept strictly separate. Values come verbatim from the
    findings; decision-rule internals (z-scores, SD thresholds, necrosis %) stay in
    the trace, not the prose.

    `corrected_fields` (redraft only): findings the reader just edited. A finding
    corrected TO absent is stated explicitly so the edit is visible; UNcorrected
    absent findings stay omitted.
    `confidence_tier` (LOW/MODERATE/HIGH): rendered as the final FINAL-DIAGNOSIS line
    when provided (generation passes the post-checklist tier; redraft passes the
    trace's tier) so both paths emit an identical report.
    """
    corrected_fields_set = set(corrected_fields or ())
    findings = structured_findings.get("findings", {})
    location = findings.get("location", {}); volumes = findings.get("volumes", {})
    enhancement = findings.get("enhancement", {}); necrosis = findings.get("necrosis", {})
    mismatch = findings.get("mismatch", {}); edema = findings.get("edema", {})
    mass_effect = findings.get("mass_effect", {}); cortical = findings.get("cortical", {})
    additional = findings.get("additional", {}); diameters_block = structured_findings.get("diameters", {})
    impression = structured_findings.get("impression", {})

    def _format_volume(value):
        # integer at/above 10 cm3, one decimal below 10; None if n/a
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return f"{value:.0f}" if value >= 10 else f"{value:.1f}"

    diameters_mm = diameters_block.get("diameters_mm") or [None, None]
    wt_volume_text, et_volume_text = _format_volume(volumes.get("wt_cm3")), _format_volume(volumes.get("et_cm3"))
    enhancing_component_clause = f" (enhancing component {et_volume_text} cm3)" if et_volume_text is not None else ""
    _has_size = isinstance(diameters_mm[0], (int, float)) and isinstance(diameters_mm[1], (int, float)) and diameters_mm[0] > 0
    # The 2D measure is WHOLE-TUMOR bidimensional (T2/FLAIR extent incl. non-enhancing
    # signal), stated explicitly so it is not read as over-measurement of the solid mass.
    _size_clause = (f"; the whole-tumor extent (T2/FLAIR) measures {diameters_mm[0] / 10:.1f} x {diameters_mm[1] / 10:.1f} cm"
                    if _has_size else "")
    lobe = location.get("lobe") or "indeterminate location"
    laterality = location.get("laterality") or ""
    junction_lobes = location.get("junction_lobes") or []
    _LOBE_DISPLAY = {
        "deep_or_extracortical": "deep/extra-cortical region",
        "insular": "insula",
        "cerebellum": "cerebellum",
        "brainstem": "brainstem",
    }
    if junction_lobes:
        lobe_display = f"{lobe}-{junction_lobes[0]} junction"
    elif lobe in ("frontal", "temporal", "parietal", "occipital"):
        lobe_display = f"{lobe} lobe"
    else:
        lobe_display = _LOBE_DISPLAY.get(lobe, str(lobe).replace("_", " "))
    region = f" ({location.get('region')})" if location.get("region") else ""
    midline_crossing_clause = ", crossing the midline" if location.get("midline_crossing") else ""
    location_text = f"{laterality} {lobe_display}".strip()

    enhancement_present = enhancement.get("present")
    enhancement_quality = (enhancement.get("quality") or "").lower()
    enhancement_proportion = enhancement.get("proportion")
    necrosis_present = necrosis.get("present")
    mismatch_present = mismatch.get("present")
    mismatch_extent = mismatch.get("extent")

    sentences = [f"There is a mass in the {location_text}{region}{midline_crossing_clause}{_size_clause}."]

    features = []
    if enhancement_present:
        quality_prefix = (enhancement_quality + " ") if enhancement_quality else ""
        if isinstance(enhancement_proportion, (int, float)) and enhancement_proportion > 0:
            enhancing_percent = enhancement_proportion * 100
            if round(enhancing_percent) < 1:
                # a sub-1% enhancing fraction reads cleaner qualitatively than "<1%"
                features.append(f"a small focus of {quality_prefix}enhancement")
            else:
                features.append(f"{quality_prefix}enhancement involving approximately {enhancing_percent:.0f}% of the tumor volume")
        else:
            features.append(f"{quality_prefix}enhancement")
    elif enhancement_present is False and "enhancement" in corrected_fields_set:
        features.append("no appreciable enhancement")
    if necrosis_present:
        necrosis_volume_cm3 = necrosis.get("necrosis_cm3")
        necrosis_volume_clause = (f" (approximately {necrosis_volume_cm3:.1f} cm3)"
                                  if isinstance(necrosis_volume_cm3, (int, float)) and "necrosis" not in corrected_fields_set
                                  else "")
        features.append(f"an enclosed non-enhancing focus{necrosis_volume_clause}")
    elif necrosis_present is False and "necrosis" in corrected_fields_set:
        features.append("no enclosed non-enhancing focus")
    if edema.get("present"):
        features.append("perilesional non-enhancing signal")
    if cortical.get("involved"):
        features.append("cortical involvement")
    if additional.get("ependymal_extension"):
        features.append("ependymal/subependymal extension")
    if mass_effect.get("present"):
        midline_shift_mm = mass_effect.get("midline_shift_mm")
        features.append(f"midline shift of approximately {midline_shift_mm:.1f} mm" if isinstance(midline_shift_mm, (int, float))
                        else "mass effect with midline shift")
    if features:
        sentences.append(f"The mass shows {_english_join(features)}.")

    # T2-FLAIR mismatch reported ONLY when present or not evaluable; the "absent"
    # negative is omitted for brevity. `None` = not evaluable, noted so the omission
    # is not mistaken for a silent negative.
    if mismatch_present is True:
        sentences.append("The T2-FLAIR mismatch sign is present"
                         + (f" ({mismatch_extent})." if mismatch_extent in ("partial", "complete") else "."))
    elif mismatch_present is None:
        sentences.append("The T2-FLAIR mismatch sign could not be assessed on this study.")

    # Multifocality: >=2 discrete ENHANCING foci (additional.multifocal_foci_cm3,
    # descending, dominant first). Up to three foci listed individually; beyond that
    # the two largest + summed remainder.
    multifocal_foci = additional.get("multifocal_foci_cm3") or []
    if _EMIT_MULTIFOCAL and len(multifocal_foci) >= 2:
        secondary_foci = multifocal_foci[1:]   # secondary foci (dominant mass = multifocal_foci[0])
        secondary_count = len(secondary_foci)
        if secondary_count == 1:
            sentences.append(f"The tumor is multifocal, with a second discrete enhancing focus "
                             f"(approximately {secondary_foci[0]:.1f} cm3) separate from the dominant enhancing mass.")
        elif secondary_count <= 3:
            count_word = {2: "two", 3: "three"}[secondary_count]
            foci_volumes_text = _english_join([f"{focus_cm3:.1f}" for focus_cm3 in secondary_foci])
            sentences.append(f"The tumor is multifocal, with {count_word} additional discrete enhancing foci "
                             f"(approximately {foci_volumes_text} cm3) separate from the dominant enhancing mass.")
        else:
            remaining_cm3 = sum(secondary_foci[2:])
            sentences.append(
                f"The tumor is multifocal, with {secondary_count} additional discrete enhancing foci separate from "
                f"the dominant enhancing mass (largest two approximately {secondary_foci[0]:.1f} and "
                f"{secondary_foci[1]:.1f} cm3; the remaining {secondary_count - 2} together approximately {remaining_cm3:.1f} cm3).")

    # Multicentric: separate enhancing focus/foci in a DIFFERENT whole-tumor component
    # (not contiguous even through non-enhancing signal). 0 cohort cases -- wording is
    # a proposal, unexercised.
    multicentric_foci = additional.get("multicentric_foci_cm3") or []
    if _EMIT_MULTIFOCAL and multicentric_foci:
        if len(multicentric_foci) == 1:
            sentences.append(f"There is also a multicentric component: a separate enhancing focus "
                             f"(approximately {multicentric_foci[0]:.1f} cm3) in a distinct region, not contiguous with "
                             f"the dominant enhancing mass on T2/FLAIR.")
        else:
            multicentric_volumes_text = _english_join([f"{focus_cm3:.1f}" for focus_cm3 in multicentric_foci])
            sentences.append(f"There is also a multicentric component: separate enhancing foci "
                             f"(approximately {multicentric_volumes_text} cm3) in distinct regions, not contiguous with the "
                             f"dominant enhancing mass on T2/FLAIR.")

    if wt_volume_text is not None:
        sentences.append(f"Tumor volume: approx. {wt_volume_text} cm3{enhancing_component_clause}.")

    findings_paragraph = " ".join(sentences)

    necrosis_cm3 = necrosis.get("necrosis_cm3")
    necrosis_significant = bool(necrosis_present and isinstance(necrosis_cm3, (int, float))
                                and necrosis_cm3 >= _NECROSIS_MIN_CM3)
    lesion_description = _lesion_character(enhancement_present, enhancement_quality, necrosis_significant,
                                           edema.get("present"), cortical.get("involved"),
                                           mass_effect.get("present"))
    impression_paragraph = f"{lesion_description} The findings are consistent with a diffuse glioma."
    if _EMIT_SUBTYPE_NAMES:
        subtype_sentence = _subtype_name_sentence(impression, confidence_tier)
        if subtype_sentence:
            impression_paragraph += " " + subtype_sentence

    final_paragraph = _final_diagnosis_line(impression, confidence_tier)

    return (f"**Findings:** {findings_paragraph}\n\n"
            f"**Impression:** {impression_paragraph}\n\n"
            f"FINAL DIAGNOSIS: {final_paragraph}")


def _assemble_report(
    qc: dict,
    seg: dict,
    vasari: dict,
    diameters: dict,
    molecular: dict,
    rag: list,
    warnings: list[str],
    structured_findings: Optional[dict] = None,
) -> dict[str, str]:
    """Assemble the report dict: legacy flat keys + `structured_report`
    (sectioned dict) + `structured_report_text` (rendered report)."""
    et = seg.get("et_cm3"); ne = seg.get("ne_cm3"); wt = seg.get("wt_cm3")

    diameters_mm = diameters.get("max_perpendicular_diameters_mm") or []
    first_diameter_mm, second_diameter_mm = (list(diameters_mm) + [None, None])[:2]
    if isinstance(first_diameter_mm, (int, float)) and isinstance(second_diameter_mm, (int, float)) and first_diameter_mm > 0:
        size_text = f"Whole-tumor bidimensional {first_diameter_mm / 10:.1f} x {second_diameter_mm / 10:.1f} cm."
    else:
        size_text = "Bidimensional size not assessed."

    idh_block = molecular.get("idh") or {}
    idh = idh_block.get("prediction")
    idh_probability = idh_block.get("probability")
    who = molecular.get("who_2021") or "integrated diagnosis indeterminate"
    codeletion_block = molecular.get("1p19q") or {}   # absent for IDH-wildtype (WHO 2021 gating)
    codeletion_prediction = codeletion_block.get("prediction")
    codeletion_probability = codeletion_block.get("probability")

    resolved_structured_findings = structured_findings
    if resolved_structured_findings is None:
        resolved_structured_findings = _collect_subresults(seg, vasari, diameters, molecular)
    resolved_structured_findings.setdefault("impression", {})["confident"] = _dx_confident(molecular, warnings)
    # grade_probability lives on `molecular`; surface it into impression so the render
    # (which only receives the structured findings) can show the grade marginal in IMPRESSION.
    resolved_structured_findings["impression"]["grade_probability"] = molecular.get("grade_probability")

    findings = resolved_structured_findings.get("findings", {})
    enhancement = findings.get("enhancement", {})
    necrosis = findings.get("necrosis", {})
    edema = findings.get("edema", {})
    mismatch = findings.get("mismatch", {})
    mass_effect = findings.get("mass_effect", {})
    cortical = findings.get("cortical", {})
    additional = findings.get("additional", {})
    low_frequency_present = [name.replace("_", " ") for name in LOW_FREQUENCY_FINDINGS if additional.get(name)]
    enhancement_proportion = enhancement.get("proportion")
    enhancement_proportion_text = f", ~{enhancement_proportion*100:.0f}% of tumor" if isinstance(enhancement_proportion, (int, float)) else ""
    vasari_text = (
        f"Location {vasari.get('tumor_location','?')} / side {vasari.get('side','?')}; "
        f"enhancement {enhancement.get('quality','?')}{enhancement_proportion_text}; "
        f"necrosis {'present' if necrosis.get('present') else 'absent'}"
        + (f" (~{necrosis['necrosis_cm3']:.1f} cm3)" if isinstance(necrosis.get('necrosis_cm3'), (int, float)) and necrosis.get('present') else "")
        + f"; edema {'present' if edema.get('present') else 'absent'}; "
        f"T2/FLAIR mismatch {'present' if mismatch.get('present') else 'absent'}; "
        f"mass effect {'present' if mass_effect.get('present') else 'absent'}; "
        f"cortical {'yes' if cortical.get('involved') else 'no'}"
        + (f"; {', '.join(low_frequency_present)}" if low_frequency_present else "")
        + " [necrosis/edema from seg_features; mismatch from the T2-FLAIR VQA]."
    )

    warnings_text = "; ".join(warnings) if warnings else "no warnings."
    structured_report_text = _render_structured_report_text(resolved_structured_findings)

    location = findings.get("location", {})
    location_laterality = location.get("laterality") or seg.get("laterality", "?")
    location_lobe = location.get("lobe") or seg.get("tumor_location", "?")
    location_region_clause = f", {location['region']}" if location.get("region") else ""

    idh_probability_text = f" probability={idh_probability:.2f}" if isinstance(idh_probability, (int, float)) else ""
    codeletion_probability_text = f" probability={codeletion_probability:.2f}" if isinstance(codeletion_probability, (int, float)) else ""
    return {
        "tumor_location_laterality": f"{location_laterality} {location_lobe}{location_region_clause}",
        "volume_measurability": (f"ET {_fmt_cm3(et)}, NE {_fmt_cm3(ne)}, "
                                      f"WT {_fmt_cm3(wt)}. {size_text}"),
        "vasari_key_features": vasari_text,
        "who_2021_integrated_diagnosis": (
            f"{who} (IDH {idh}{idh_probability_text}"
            + (f", 1p/19q {codeletion_prediction}{codeletion_probability_text}" if codeletion_prediction is not None else "")
            + ")."
        ),
        "size_assessment": size_text,
        "warnings_and_recommendations": warnings_text,
        "rag_evidence": [hit["title"] + ": " + hit["chunk"][:200] for hit in rag[:3]],
        "structured_report": resolved_structured_findings,
        "structured_report_text": structured_report_text,
    }
