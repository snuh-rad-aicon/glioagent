"""Thin wrapper around the HD-GLIO segmentation predictor.

HD-GLIO labels: 0=background, 1=non-enhancing T2/FLAIR, 2=enhancing tumor;
whole tumor = (label > 0).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from typing import Any

import nibabel as nib
import numpy as np

from .tool_measure_diameters import measure_diameters

DEFAULT_GPU = "2"


@contextlib.contextmanager
def _quiet():
    # nnU-Net and its worker processes print citations, path warnings and per-tile
    # progress; swallow it at fd level and replay only when something fails
    sys.stdout.flush()
    sys.stderr.flush()
    captured = None
    with tempfile.TemporaryFile(mode="w+") as sink:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        except Exception:
            sink.seek(0)
            captured = sink.read()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            if captured:
                sys.stdout.write(captured)


def _resolve_device_index(gpu_or_device: str) -> int:
    """Accept either a `cuda:N` string or a bare GPU index string."""
    device_str = str(gpu_or_device)
    if device_str.startswith("cuda:"):
        return int(device_str.split(":")[1])
    return int(device_str)


def _patch_torch_load() -> None:
    # nnU-Net pickles need weights_only=False on torch >= 2.6
    import torch

    if getattr(torch.load, "__hdglio_patched__", False):
        return
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    patched.__hdglio_patched__ = True   # type: ignore[attr-defined]
    torch.load = patched


def warm_hdglio() -> None:
    """Import nnU-Net and check the weights up front, so the first case pays no import
    cost. nnU-Net still loads the fold parameters inside every predict call."""
    _patch_torch_load()
    with _quiet():
        from hd_glio.setup_hd_glio import maybe_download_weights
        from nnunet.inference.predict import predict_cases  # noqa: F401

        maybe_download_weights()


def predict_hdglio(
    t1: str,
    t1ce: str,
    t2: str,
    flair: str,
    output_path: str,
    gpu: str = DEFAULT_GPU,
) -> str:
    """Run HD-GLIO inference and write the seg mask to `output_path`, routing
    CUDA to `gpu` so it does not contest the GPU running the LLM."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import torch

    _patch_torch_load()
    if torch.cuda.is_available():
        torch.cuda.set_device(_resolve_device_index(gpu))

    with _quiet():
        from hd_glio.paths import folder_with_parameter_files
        from hd_glio.setup_hd_glio import maybe_download_weights
        from nnunet.inference.predict import predict_cases

        maybe_download_weights()
        predict_cases(
            folder_with_parameter_files,
            [[t1, t1ce, t2, flair]],
            [output_path],
            (0,),
            False,
            1,
            1,
            None,
            True,
            None,
            True,
            checkpoint_name="model_final_checkpoint",
        )
    return output_path


_LATERALITY_MIDLINE_FRAC = 0.55


def _laterality_from_mask(seg: np.ndarray, affine: np.ndarray | None = None) -> str:
    """Return patient-anatomical left / right / midline / bilateral.

    Patient left/right is decided from the sign of `affine[0, 0]`, which fixes
    whether increasing voxel-i moves toward +x (patient-right) in RAS+. Without
    an affine we fall back to the (unsigned) image-i side, the wrong convention
    for UCSF (HD-GLIO image-i = +R) that caused disagreement with VASARI.
    """
    if not seg.any():
        return "bilateral"
    i_coords = np.where(seg)[0]
    i_size = seg.shape[0]
    high_i_frac = float((i_coords >= i_size / 2).mean())
    low_i_frac = 1.0 - high_i_frac

    if affine is None or affine[0, 0] >= 0:
        # i=0 -> patient-left; high-i fraction is patient-right.
        right_frac = high_i_frac
    else:
        # i=0 -> patient-right (radiological storage); high-i is patient-left.
        right_frac = low_i_frac
    left_frac = 1.0 - right_frac

    if left_frac >= _LATERALITY_MIDLINE_FRAC:
        return "left"
    if right_frac >= _LATERALITY_MIDLINE_FRAC:
        return "right"
    if abs(left_frac - right_frac) < 0.05:
        return "midline"
    return "bilateral"


def _coarse_location(seg: np.ndarray) -> str:
    """Best-effort lobe label without atlas registration: split the bounding
    box into thirds along the y axis (rough A->P). Fallback only; atlas_features
    gives the real value."""
    if not seg.any():
        return "no tumor detected"
    y_coords = np.where(seg)[1]
    y_size = seg.shape[1]
    centre = y_coords.mean() / y_size
    if centre < 0.33:
        return "anterior (likely frontal)"
    if centre < 0.66:
        return "middle (likely parietal/temporal)"
    return "posterior (likely occipital)"


def summarise_seg(seg_mask_path: str) -> dict[str, Any]:
    seg_img = nib.load(seg_mask_path)
    labels = seg_img.get_fdata().astype(np.int16)
    zooms = seg_img.header.get_zooms()[:3]
    voxel_cm3 = float(np.prod(zooms)) / 1000.0  # mm3 -> cm3

    et = int((labels == 2).sum())
    ne = int((labels == 1).sum())
    wt = int((labels > 0).sum())

    diameters = measure_diameters(seg_mask_path, enhancing_label=2)
    max_bidimensional_mm = float(max(diameters["max_perpendicular_diameters_mm"]))

    wt_mask = labels > 0

    return {
        "seg_mask_path": seg_mask_path,
        "et_cm3": round(et * voxel_cm3, 2),
        "ne_cm3": round(ne * voxel_cm3, 2),
        "wt_cm3": round(wt * voxel_cm3, 2),
        "tumor_location": _coarse_location(wt_mask),
        "laterality": _laterality_from_mask(wt_mask, affine=seg_img.affine),
        "measurable": bool(diameters["measurable"]),
        "max_bidimensional_mm": round(max_bidimensional_mm, 2),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("seg_mask")
    args = ap.parse_args()
    print(json.dumps(summarise_seg(args.seg_mask), indent=2))
