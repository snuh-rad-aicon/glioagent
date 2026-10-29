"""Bidimensional diameters of the tumour, and whether both reach 10 mm, from a seg mask."""

from __future__ import annotations

import json
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, binary_fill_holes

from protocol.measurement_reference import MEASURABLE_THRESHOLD_MM   # 10 mm (Wen 2023)
from tools.tool_seg_features import ENHANCE_CM3                     # 0.10 cm3, one definition of "enhances"


def _max_in_plane_diameter_mm(blob: np.ndarray, spacing_axis0: float, spacing_axis1: float) -> tuple[float, float, tuple]:
    """Longest in-plane chord of the blob and its perpendicular extent, in mm.

    `spacing_axis0`/`spacing_axis1` are the spacings of the blob's first/second axes (np.where's order).
    Returns (long_mm, short_mm, (p1, p2)) with p1/p2 the long axis' voxel endpoints.
    """
    idx_axis0, idx_axis1 = np.where(blob)
    if len(idx_axis0) < 2:
        return 0.0, 0.0, ((0, 0), (0, 0))

    points_mm = np.stack([idx_axis0 * spacing_axis0, idx_axis1 * spacing_axis1], axis=1)
    diffs = points_mm[:, None, :] - points_mm[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    long_mm = float(dists[i, j])

    long_axis_dir = points_mm[j] - points_mm[i]
    if np.linalg.norm(long_axis_dir) < 1e-6:
        return long_mm, 0.0, ((idx_axis0[i], idx_axis1[i]), (idx_axis0[j], idx_axis1[j]))
    long_axis_dir /= np.linalg.norm(long_axis_dir)
    perpendicular_dir = np.array([-long_axis_dir[1], long_axis_dir[0]])

    perpendicular_proj_mm = points_mm @ perpendicular_dir
    short_mm = float(perpendicular_proj_mm.max() - perpendicular_proj_mm.min())
    return long_mm, short_mm, ((int(idx_axis0[i]), int(idx_axis1[i])), (int(idx_axis0[j]), int(idx_axis1[j])))


def _best_bidim(mask, spacing_axis0: float, spacing_axis1: float) -> tuple[float, float, int]:
    """Largest-area axial slice's two perpendicular in-plane diameters (mm)."""
    best_long = best_short = 0.0
    best_z = -1
    for z in range(mask.shape[2]):
        axial_slice = mask[:, :, z]
        if not axial_slice.any():
            continue
        labelled, n_components = cc_label(axial_slice)
        if n_components == 0:
            continue
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        blob = labelled == int(np.argmax(sizes))
        long_mm, short_mm, _ = _max_in_plane_diameter_mm(blob, spacing_axis0, spacing_axis1)
        if long_mm * short_mm > best_long * best_short:
            best_long, best_short, best_z = long_mm, short_mm, z
    return best_long, best_short, best_z


def _axis_extents_mm(mask: np.ndarray, zooms) -> list[float]:
    """Bounding-box extent along each image axis, in mm (axis-aligned cohorts)."""
    if not mask.any():
        return [0.0, 0.0, 0.0]
    idx = np.array(np.where(mask))
    return [float((idx[axis].max() - idx[axis].min() + 1) * zooms[axis]) for axis in range(3)]


def _enhancing_mass(seg: np.ndarray, enhancing_label: int) -> np.ndarray:
    """The enhancing component with its enclosed interior filled -- a broken rim would otherwise be
    measured as its largest fragment."""
    et = seg == enhancing_label
    if not et.any():
        return et
    idx = np.array(np.where(et))
    lo = np.maximum(idx.min(axis=1) - 2, 0); hi = np.minimum(idx.max(axis=1) + 3, et.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    filled = np.zeros_like(et)
    filled[box] = binary_fill_holes(et[box])
    return filled


def measure_diameters(
    seg_mask_path: str,
    enhancing_label: int = 2,
    min_diameter_mm: float = MEASURABLE_THRESHOLD_MM,
) -> dict[str, Any]:
    """Bidimensional (two perpendicular in-plane diameters) size of the tumour.

    Primary size is the WHOLE tumour (label > 0), so that non/minimally-enhancing
    gliomas -- which have little or no enhancing component -- still get a size.
    The enhancing-only diameters are reported separately when an
    enhancing component is present. Measurability = both diameters >= 10 mm; note
    this applies only the size half of the published criterion (Wen et al., J Clin Oncol
    2023), not the <=5 mm slice-thickness requirement, so it is reported as a diameter,
    not as a measurability verdict.
    """
    seg_img = nib.load(seg_mask_path)
    seg = seg_img.get_fdata()
    dx, dy, dz = [float(spacing) for spacing in seg_img.header.get_zooms()[:3]]

    wt_long_mm, wt_short_mm, wt_best_z = _best_bidim(seg > 0, dx, dy)
    et_long_mm, et_short_mm, _ = _best_bidim(seg == enhancing_label, dx, dy)

    # Conditional size, which is what a radiologist reports: the enhancing mass for an enhancing
    # tumour, the T2/FLAIR lesion otherwise. Measured as the longest of the three axis extents.
    # Chosen on the dev split (MAE 1.22 cm vs 1.99 for whole-lesion bidimensional and 2.29 for
    # whole-lesion axis extent, against the largest dimension stated in the radiology report).
    voxel_cm3 = float(dx * dy * dz) / 1000.0
    et_cm3 = float((seg == enhancing_label).sum()) * voxel_cm3
    enhancing = et_cm3 > ENHANCE_CM3
    conditional_mask = _enhancing_mass(seg, enhancing_label) if enhancing else (seg > 0)
    conditional_axes_mm = _axis_extents_mm(conditional_mask, (dx, dy, dz))
    whole_axes_mm = _axis_extents_mm(seg > 0, (dx, dy, dz))
    measurable = wt_long_mm >= min_diameter_mm and wt_short_mm >= min_diameter_mm
    return {
        "measurable": bool(measurable),
        "max_perpendicular_diameters_mm": (round(wt_long_mm, 2), round(wt_short_mm, 2)),  # whole tumour
        "product_mm2": round(wt_long_mm * wt_short_mm, 2),
        "enhancing_diameters_mm": (round(et_long_mm, 2), round(et_short_mm, 2)),
        "enhancing_present": bool((seg == enhancing_label).any()),
        "best_slice_z": int(wt_best_z),
        "conditional_axes_mm": [round(v, 2) for v in conditional_axes_mm],
        "conditional_max_mm": round(max(conditional_axes_mm), 2),
        "conditional_basis": "enhancing mass (filled)" if enhancing else "whole lesion (T2/FLAIR)",
        "whole_tumour_axes_mm": [round(v, 2) for v in whole_axes_mm],
        "provenance": "whole-tumour bidimensional (label>0); enhancing (label==2) secondary; "
                      "conditional_max_mm = longest axis extent of the enhancing mass when the tumour "
                      "enhances (ET > 0.10 cm3), of the whole lesion otherwise",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("seg_mask")
    parser.add_argument("--enhancing-label", type=int, default=2)
    parser.add_argument("--min-mm", type=float, default=MEASURABLE_THRESHOLD_MM)
    args = parser.parse_args()
    out = measure_diameters(args.seg_mask, args.enhancing_label, args.min_mm)
    print(json.dumps(out, indent=2))
