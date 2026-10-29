"""GlioMT wrapper for molecular subtyping (IDH / 1p-19q / grade)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path as _Path
from typing import Any, Optional

from protocol.who2021_rules import derive_who2021


_TOOLS = _Path(__file__).resolve().parent
GLIOMT_CKPT_DIR = os.environ.get("GLIOMT_CKPT_DIR", str(_TOOLS / "ckpts" / "gliomt"))


def _checkpoints_present() -> bool:
    needed = ("idh_vitb.pth", "1p19q_swins.pth", "grade_swins.pth")
    return all(os.path.isfile(os.path.join(GLIOMT_CKPT_DIR, filename)) for filename in needed)


def molecular_subtype(
    t1: str,
    t1ce: str,
    t2: str,
    flair: str,
    age: int,
    sex: str,
    seg_mask: Optional[str] = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """IDH / 1p-19q / grade dict from GlioMT (stub output if weights absent)."""
    if not _checkpoints_present() or seg_mask is None:
        # Without this the pipeline returns a plausible IDH-wildtype glioblastoma with no
        # sign that no model ran. The `stub` flag in the output is kept for the trace; the
        # warning is what a deployment missing its weights actually needs to see.
        reason = ("GlioMT checkpoints not found in " + GLIOMT_CKPT_DIR
                  if not _checkpoints_present() else "no segmentation mask supplied")
        warnings.warn(f"molecular_subtype: {reason}; returning the heuristic STUB output "
                      "(IDH-wildtype, p=0.60) -- this is not a model prediction",
                      RuntimeWarning, stacklevel=2)
        return _stub_output(t1ce)
    return _real_predict(t1, t1ce, t2, flair, age=age, sex=sex, seg_mask=seg_mask, device=device)


def _stub_output(t1ce_path: str) -> dict[str, Any]:
    """Heuristic stub used until GlioMT checkpoints are available; defaults to IDH-wildtype GBM."""
    who_diagnosis, _grade = derive_who2021("wildtype", "non-codeleted")
    return {
        "idh": {"prediction": "wildtype", "probability": 0.60},
        # No "1p19q" key: not applicable to IDH-wildtype (WHO 2021).
        "who_2021": who_diagnosis,
        "grade_prediction": 4,
        "grade_probability": 0.60,
        "p1_19q_evaluated": False,
        "stub": True,
    }


def _real_predict(t1, t1ce, t2, flair, age, sex, seg_mask, device) -> dict[str, Any]:
    """Run GlioMT inference and reshape into the orchestrator schema (no MGMT head)."""
    from tools.gliomt.gliomt_inference import predict as _gliomt_predict

    # conditional_1p19q=True: the 1p/19q head runs only for IDH-mutant.
    gliomt_output = _gliomt_predict(
        t1=t1, t1ce=t1ce, t2=t2, flair=flair, seg=seg_mask,
        age=age, sex=sex, device=device, conditional_1p19q=True,
    )
    idh = gliomt_output["idh_prediction"]
    grade_pred = gliomt_output["grade_prediction"]
    idh_prob = (gliomt_output["idh_prob_mutant"] if idh == "mutant"
                else 1 - gliomt_output["idh_prob_mutant"])

    result: dict[str, Any] = {
        "idh": {"prediction": idh, "probability": float(idh_prob)},
        "grade_probability": float(gliomt_output["grade_probability"]),
        "p1_19q_evaluated": bool(gliomt_output.get("p1_19q_evaluated",
                                                   "p1_19q_prediction" in gliomt_output)),
        "stub": False,
        "raw": gliomt_output,
    }

    if idh == "wildtype":
        result["who_2021"] = "Glioblastoma, IDH-wildtype, CNS WHO Grade 4"
        result["grade_prediction"] = 4
        return result

    codeletion_status = gliomt_output["p1_19q_prediction"]
    # GlioMT's grade head is 3-class (2/3/4); oligo only exists at grades 2-3.
    if codeletion_status == "codeleted" and grade_pred == 4:
        grade_pred = 3
    if codeletion_status == "codeleted":
        result["who_2021"] = f"Oligodendroglioma, IDH-mutant, 1p/19q-codeleted (Grade {grade_pred})"
    else:
        result["who_2021"] = f"Astrocytoma, IDH-mutant (Grade {grade_pred})"
    result["grade_prediction"] = int(grade_pred)
    p1_19q_prob = (gliomt_output["p1_19q_prob_codel"] if codeletion_status == "codeleted"
                   else 1 - gliomt_output["p1_19q_prob_codel"])
    result["1p19q"] = {"prediction": codeletion_status, "probability": float(p1_19q_prob)}
    return result
