"""Concordance checks and evidence-based retraction of imaging findings vs the molecular call."""
from __future__ import annotations


def check_discordance(structured_findings: dict, molecular: dict, age) -> list[str]:
    """Return warnings from the findings vs the molecular call. Does not modify structured_findings."""
    warnings: list[str] = []
    idh = (molecular.get("idh") or {}).get("prediction")
    findings = structured_findings.get("findings", {})
    mismatch_present = (findings.get("mismatch") or {}).get("present")
    necrosis_present = (findings.get("necrosis") or {}).get("present")
    enhancement_present = (findings.get("enhancement") or {}).get("present")
    volumes = findings.get("volumes") or {}
    et, wt = volumes.get("et_cm3"), volumes.get("wt_cm3")

    if idh == "wildtype" and mismatch_present:
        warnings.append("[ALERT] IDH-wildtype with T2-FLAIR mismatch sign -> discordant "
                        "(the sign is essentially absent in glioblastoma; Patel 2017, Do 2022)")
    if necrosis_present and mismatch_present:
        warnings.append("[ALERT] central necrosis with T2-FLAIR mismatch sign -> "
                        "the sign requires homogeneous solid T2 hyperintensity and is not "
                        "assessable with a necrotic cavity (Do 2022 confounder list)")
    if age is not None and age > 55 and idh == "mutant":
        warnings.append("[INFO] age>55 with IDH-mutant prediction -> unusual; IDH-mutant "
                        "gliomas are typically younger (median ~40; Maynard 2020)")
    if idh == "wildtype" and enhancement_present is False:
        warnings.append("[INFO] IDH-wildtype without appreciable enhancement -> supporting "
                        "evidence is partial; IDH-wildtype glioblastoma may lack the classical "
                        "enhancing phenotype (Foltyn-Dumitru 2025: 7.4% of GBM non-enhancing)")
    if isinstance(et, (int, float)) and isinstance(wt, (int, float)) and wt > 0 and et / wt > 0.8:
        warnings.append("[INFO] enhancing-to-whole-tumor ratio > 0.8 -> unusual ET dominance")
    if isinstance(wt, (int, float)) and wt < 0.5:
        warnings.append("[ALERT] whole-tumor volume < 0.5 cm3 -> segmentation suspicious")

    # A reconciled (single-model-overturned) call is inherently less certain.
    idh_call = molecular.get("idh") or {}
    if idh_call.get("source") == "agent_reconciled":
        warnings.append(f"[INFO] IDH call reconciled from GlioMT {idh_call.get('reconciled_from')} "
                        f"to {idh_call.get('prediction')} on joint age and imaging evidence")
    return warnings


def retract_unreliable(structured_findings: dict, molecular: dict) -> tuple[dict, list[str]]:
    """Withdraw findings the evidence base says are not assessable in this context.
    Sets `present` to None (not assessable), never to False; the reason is kept on
    the finding under `retracted`. Returns the amended structured_findings and the
    retraction notes."""
    notes: list[str] = []
    findings = structured_findings.get("findings", {})
    mismatch = findings.get("mismatch") or {}
    necrosis = findings.get("necrosis") or {}
    idh = (molecular.get("idh") or {}).get("prediction")

    if mismatch.get("present"):
        if necrosis.get("present"):
            mismatch["present"] = None
            mismatch["retracted"] = ("not assessable with central necrosis; the sign requires "
                                     "homogeneous solid T2 hyperintensity (Do 2022)")
            notes.append("[RETRACT] T2-FLAIR mismatch withdrawn: central necrosis present")
        elif idh == "wildtype":
            mismatch["present"] = None
            mismatch["retracted"] = ("the sign is essentially absent in IDH-wildtype "
                                     "glioblastoma (Patel 2017; Do 2022)")
            notes.append("[RETRACT] T2-FLAIR mismatch withdrawn: IDH-wildtype")
    return structured_findings, notes
