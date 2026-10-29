"""Deterministic HD-GLIO segmentation cleanup: out-of-brain and small-island removal."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as cc_label

ISLAND_CM3 = 1.0                      # min connected-component volume to keep
_CONN = np.ones((3, 3, 3), dtype=int)  # 26-connectivity


def _voxvol_cm3(zooms) -> float:
    return float(np.prod(np.asarray(zooms)[:3])) / 1000.0


def _vols(seg: np.ndarray, voxel_cm3: float) -> dict[str, float]:
    return {
        "et_cm3": float((seg == 2).sum()) * voxel_cm3,
        "ne_cm3": float((seg == 1).sum()) * voxel_cm3,
        "wt_cm3": float((seg > 0).sum()) * voxel_cm3,
    }


def cleanup_seg_array(
    seg: np.ndarray,
    brain: np.ndarray | None,
    zooms,
    island_cm3: float = ISLAND_CM3,
    remove_out_of_brain: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (cleaned_seg, report). `brain` is a boolean brain mask (T1>0); if
    None or `remove_out_of_brain=False`, the out-of-brain step is skipped but its
    volume is still reported."""
    voxel_cm3 = _voxvol_cm3(zooms)
    seg = np.asarray(seg).astype(np.int16).copy()
    before = _vols(seg, voxel_cm3)
    wt_mask_before = seg > 0
    labels_before, n_components_before = cc_label(wt_mask_before, structure=_CONN)

    out_of_brain_cm3 = 0.0
    out_of_brain_removed = False
    if brain is not None and brain.shape == seg.shape:
        out_of_brain_mask = (seg > 0) & (~brain.astype(bool))
        out_of_brain_cm3 = float(out_of_brain_mask.sum()) * voxel_cm3
        if remove_out_of_brain:
            seg[out_of_brain_mask] = 0
            out_of_brain_removed = True

    wt = seg > 0
    removed_islands = 0
    removed_island_cm3 = 0.0
    largest_frac_after = 0.0
    if wt.any():
        component_labels, n_components = cc_label(wt, structure=_CONN)
        sizes = np.bincount(component_labels.ravel())[1:] * voxel_cm3
        keep_labels = {i for i, size_cm3 in enumerate(sizes, start=1) if size_cm3 >= island_cm3}
        removed_islands = int(n_components - len(keep_labels))
        removed_island_cm3 = float(sizes[sizes < island_cm3].sum())
        if keep_labels:
            keep_mask = np.isin(component_labels, list(keep_labels))
            seg[~keep_mask] = 0
            kept_sizes = sizes[[i - 1 for i in sorted(keep_labels)]]
            wt_after = float(kept_sizes.sum())
            largest_frac_after = float(kept_sizes.max() / wt_after) if wt_after > 0 else 0.0
        else:
            # all components below threshold: keep the single largest so we never wipe the whole tumour
            largest_label = int(np.argmax(sizes)) + 1
            seg[component_labels != largest_label] = 0
            removed_islands = int(n_components - 1)
            largest_frac_after = 1.0

    after = _vols(seg, voxel_cm3)
    wt_before = before["wt_cm3"]
    wt_drop_frac = ((wt_before - after["wt_cm3"]) / wt_before) if wt_before > 0 else 0.0

    report = {
        "voxvol_cm3": voxel_cm3,
        "before": before,
        "after": after,
        "n_components_before": int(n_components_before),
        "out_of_brain_cm3": out_of_brain_cm3,
        "out_of_brain_removed": out_of_brain_removed,
        "removed_out_of_brain_cm3": out_of_brain_cm3 if out_of_brain_removed else 0.0,
        "removed_islands": removed_islands,
        "removed_island_cm3": removed_island_cm3,
        "wt_drop_frac": wt_drop_frac,
        "largest_frac_after": largest_frac_after,
        "island_cm3": island_cm3,
    }
    return seg, report


def cleanup_seg_file(
    seg_path: str, t1_path: str | None, out_path: str,
    island_cm3: float = ISLAND_CM3,
) -> dict[str, Any]:
    """Read seg (+ optional T1 for the brain mask), clean, write to out_path.
    Returns the cleanup report."""
    import nibabel as nib

    seg_img = nib.load(seg_path)
    seg = np.asanyarray(seg_img.dataobj)
    zooms = seg_img.header.get_zooms()
    brain = None
    if t1_path:
        try:
            t1 = np.asanyarray(nib.load(t1_path).dataobj)
            if t1.shape == seg.shape:
                brain = t1 > 0
        except Exception:
            brain = None
    cleaned, report = cleanup_seg_array(seg, brain, zooms, island_cm3=island_cm3)
    nib.save(nib.Nifti1Image(cleaned.astype(np.int16), seg_img.affine, seg_img.header),
             out_path)
    report["out_path"] = out_path
    return report
