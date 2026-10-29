"""Deterministic seg-only imaging features (Section A).

Seg label convention: 0 = background, 1 = non-enhancing (NE), 2 = enhancing (ET);
whole tumour WT = (seg > 0). Intensity metrics z-score against normal-appearing
brain (brain_mask & ~WT) before any pooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import (binary_erosion, binary_dilation, binary_fill_holes,
                           label as connected_components, generate_binary_structure)


EDEMA_MM      = 20.0    # extensive-edema radial threshold (mm)
NECROSIS_CM3  = 0.25    # min enclosed-NE volume (cm3, ~8 mm sphere); preserves all reader-study retraction cases (min 0.49 cm3)
ENHANCE_CM3   = 0.10    # minimum ET volume to call enhancement present (cm3)
NECROSIS_CLOSE_MM = 3.0 # ET-shell gap-closing radius before hole-fill (mm)
FOCUS_MIN_CM3 = 0.1     # an ET component counts as a discrete enhancing focus at this size.
                        # 0.1 cm3 is a ~6 mm nodule, which is what a report means by a
                        # "focus"; 1.0 cm3 (~12 mm) missed most of them. Against the 227
                        # reports that commit, positive agreement is 56.3% at 0.1 and 31.0%
                        # at 1.0, and negative agreement 68.6% and 92.9% (Fig 5b).
T_ENH_AVID_Z  = 2.0     # enhancement_z >= this -> VASARI 2.0 F4 "Avid" (2 SD above brain), else "Minimal"


def _load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI; return (data, affine, zooms)."""
    nifti_img = nib.load(str(path))
    return nifti_img.get_fdata().astype(np.float32), nifti_img.affine, np.asarray(
        nifti_img.header.get_zooms()[:3], dtype=float)


def _voxel_volume_cm3(zooms: np.ndarray) -> float:
    return float(np.prod(zooms)) / 1000.0


def _brain_mask(intensities: dict[str, np.ndarray]) -> np.ndarray:
    """Skull-stripped input assumed; use non-zero union across sequences."""
    reference_image = intensities.get("t1") if "t1" in intensities else next(iter(intensities.values()))
    nonzero_union = reference_image > 0
    for image in intensities.values():
        nonzero_union = nonzero_union | (image > 0)
    return nonzero_union


def _zscore(image: np.ndarray, normal_mask: np.ndarray) -> np.ndarray:
    """Z-score `image` against normal-appearing brain."""
    normal_values = image[normal_mask & np.isfinite(image)]
    if normal_values.size < 100:
        return np.zeros_like(image)
    normal_mean = float(normal_values.mean()); normal_std = float(normal_values.std())
    if normal_std < 1e-6: normal_std = 1.0
    return (image - normal_mean) / normal_std


def _mean_safe(image: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.nanmean(image[mask]))


def _std_safe(image: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 8:
        return float("nan")
    return float(np.nanstd(image[mask]))


def _phys_ball(zooms: np.ndarray, radius_mm: float) -> np.ndarray:
    """Ellipsoidal structuring element approximating a physical `radius_mm` ball.

    Per-axis voxel radii come from `zooms`, so on anisotropic voxels (e.g.
    1x1x4 mm) the element reaches ~radius_mm in every PHYSICAL direction. This
    replaces `iterations = round(radius_mm / min_spacing)` applied on all axes,
    which over-reaches the coarse (through-plane) axis by the anisotropy factor
    (12 mm instead of 3 mm on 1x1x4). On isotropic voxels it is a plain ball."""
    radii_vox = [max(1, int(round(radius_mm / float(spacing)))) for spacing in zooms]
    axis_grids = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii_vox)]
    dist_sq = sum((grid.astype(np.float64) / radius) ** 2 for grid, radius in zip(axis_grids, radii_vox))
    return dist_sq <= 1.0


def volumes(seg: np.ndarray, zooms: np.ndarray) -> dict[str, float]:
    voxel_cm3 = _voxel_volume_cm3(zooms)
    return {
        "et_cm3": float((seg == 2).sum()) * voxel_cm3,
        "ne_cm3": float((seg == 1).sum()) * voxel_cm3,
        "wt_cm3": float((seg  > 0).sum()) * voxel_cm3,
    }


def enhancement(seg: np.ndarray, t1_z: np.ndarray, t1ce_z: np.ndarray,
                 tumour_volumes: dict) -> dict[str, Any]:
    et = seg == 2
    et_cm3 = tumour_volumes["et_cm3"]
    wt_cm3 = tumour_volumes["wt_cm3"]
    proportion = et_cm3 / wt_cm3 if wt_cm3 > 0 else 0.0
    present = bool(et_cm3 > ENHANCE_CM3)
    if et.any():
        enhancement_z = _mean_safe(t1ce_z, et) - _mean_safe(t1_z, et)
    else:
        enhancement_z = float("nan")
    enhancement_z_or_none = float(enhancement_z) if enhancement_z == enhancement_z else None
    # VASARI 2.0 F4 quality from differential-enhancement z (T1CE-T1 within ET, in brain SD)
    if not present:
        quality = "Absent"
    elif enhancement_z_or_none is None:
        quality = None
    elif enhancement_z_or_none >= T_ENH_AVID_Z:
        quality = "Avid"
    else:
        quality = "Minimal"
    return {
        "present":       present,
        "proportion":    float(proportion),
        "enhancement_z": enhancement_z_or_none,
        "quality":       quality,
    }


def necrosis(seg: np.ndarray, zooms: np.ndarray) -> dict[str, Any]:
    """A necrotic pocket is an NE component enclosed by ET.

    Robust definition: close small gaps in the ET shell (which HD-GLIO
    often predicts as an incomplete ring rather than a closed sphere)
    with a spherical structuring element of radius NECROSIS_CLOSE_MM,
    then fill topological holes.  Voxels that became "ET" only after
    fill-holes but were originally NE are enclosed NE, i.e. necrosis.
    """
    et = seg == 2
    ne = seg == 1
    if not et.any() or not ne.any():
        return {"present": False, "necrosis_cm3": 0.0}
    closing_ball = _phys_ball(zooms, NECROSIS_CLOSE_MM)
    et_closed = binary_dilation(et, structure=closing_ball)
    et_closed_filled = binary_fill_holes(et_closed)
    et_filled_shell = binary_erosion(et_closed_filled, structure=closing_ball)
    enclosed_ne = et_filled_shell & ~et & ne
    voxel_cm3 = _voxel_volume_cm3(zooms)
    enclosed_ne_cm3 = float(enclosed_ne.sum()) * voxel_cm3
    return {
        "present":     bool(enclosed_ne_cm3 > NECROSIS_CM3),
        "necrosis_cm3": float(enclosed_ne_cm3),
    }


def enhancing_foci(seg: np.ndarray, zooms: np.ndarray) -> dict[str, Any]:
    """Discrete ENHANCING foci >= FOCUS_MIN_CM3, split into multifocal and multicentric.

    multifocal   = foci sharing the DOMINANT focus' whole-tumour component, i.e. joined to it
                   through non-enhancing tumour/oedema -- the medical definition.
    multicentric = foci in a DIFFERENT whole-tumour component (not contiguous even through NE).
    Computed on ET, not WT: oedema fuses separate lesions into one WT component. 6-connectivity.
    """
    foci_structure = generate_binary_structure(3, 1)
    voxel_cm3 = _voxel_volume_cm3(zooms)
    et_labels, n_components = connected_components(seg == 2, foci_structure)
    empty = {"multifocal": False, "multifocal_foci_cm3": [], "multicentric_foci_cm3": [],
             "focus_min_cm3": FOCUS_MIN_CM3}
    if n_components == 0:
        return empty
    component_sizes = np.bincount(et_labels.ravel()); component_sizes[0] = 0
    foci = [(label_code, float(component_sizes[label_code]) * voxel_cm3)
            for label_code in range(1, len(component_sizes))
            if component_sizes[label_code] * voxel_cm3 >= FOCUS_MIN_CM3]
    if not foci:
        return empty
    foci.sort(key=lambda label_and_volume: -label_and_volume[1])     # dominant focus first
    if len(foci) == 1:
        return {**empty, "multifocal_foci_cm3": [round(foci[0][1], 1)]}

    whole_tumour_labels, _ = connected_components(seg > 0, foci_structure)

    def whole_tumour_of(et_label: int) -> int:
        labels_here = whole_tumour_labels[et_labels == et_label]
        labels_here = labels_here[labels_here > 0]
        return int(np.bincount(labels_here).argmax()) if labels_here.size else 0

    dominant_whole_tumour = whole_tumour_of(foci[0][0])
    connected, separate = [], []
    for et_label, volume_cm3 in foci:
        same_component = dominant_whole_tumour > 0 and whole_tumour_of(et_label) == dominant_whole_tumour
        (connected if same_component else separate).append(round(volume_cm3, 1))
    return {"multifocal": len(connected) >= 2,
            "multifocal_foci_cm3": sorted(connected, reverse=True),
            "multicentric_foci_cm3": sorted(separate, reverse=True),
            "focus_min_cm3": FOCUS_MIN_CM3}


def edema_extent(seg: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Radial extent (mm) of NE from the ET centroid (or WT centroid when no ET).

    HD-GLIO's NE label combines oedema and non-enhancing tumour, so radial_mm is
    a coarse spatial-spread proxy, not calibrated oedema. Qualitative tiering is
    applied downstream; here only radial_mm and `extensive` (> EDEMA_MM) return.
    """
    ne = seg == 1
    et = seg == 2
    wt = seg > 0
    if not wt.any():
        return {"radial_mm": 0.0, "extensive": False}

    centroid_mask = et if et.any() else wt
    centroid_mask_ijk = np.array(np.where(centroid_mask))
    centroid_ijk = centroid_mask_ijk.mean(axis=1)
    centroid_world = affine[:3, :3] @ centroid_ijk + affine[:3, 3]

    if not ne.any():
        return {"radial_mm": 0.0, "extensive": False}
    ne_ijk = np.array(np.where(ne))
    ne_world = affine[:3, :3] @ ne_ijk + affine[:3, 3:4]
    ne_dist_mm = np.linalg.norm(ne_world - centroid_world[:, None], axis=0)
    radial_mm = float(np.max(ne_dist_mm))
    return {
        "radial_mm": radial_mm,
        "extensive": bool(radial_mm > EDEMA_MM),
    }


def laterality_seg_only(seg: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Direction-aware laterality from the affine (RAS: +x = Right)."""
    wt = seg > 0
    if not wt.any():
        return {"dominant_side": "unknown", "signed_x_mm": 0.0}
    wt_ijk = np.array(np.where(wt))
    wt_world = affine[:3, :3] @ wt_ijk + affine[:3, 3:4]
    signed_x = float(wt_world[0].mean())     # world x in mm; +x = Right in RAS
    side = "right" if signed_x > 0 else "left"
    return {
        "dominant_side": side,
        "signed_x_mm":   signed_x,
    }


def compute_all(seg_path: str, mri_paths: dict[str, str]) -> dict[str, Any]:
    """Compute all Section A features.

    `mri_paths` maps 't1' / 't1ce' / 't2' / 'flair' to NIfTI paths.  Any
    missing sequence is tolerated; metrics that need it return None.
    """
    seg, seg_affine, seg_zooms = _load(seg_path)
    seg = seg.astype(np.int16)

    intensities: dict[str, np.ndarray] = {}
    for seq in ("t1", "t1ce", "t2", "flair"):
        seq_path = mri_paths.get(seq)
        if seq_path and Path(seq_path).is_file():
            image, _, _ = _load(seq_path)
            intensities[seq] = image

    brain = _brain_mask(intensities) if intensities else (seg >= 0)
    normal_brain = brain & (seg == 0)

    def _zscore_or_zeros(seq: str) -> np.ndarray:
        image = intensities.get(seq)
        if image is None:
            return np.zeros_like(seg, dtype=np.float32)
        return _zscore(image, normal_brain)

    t1_z    = _zscore_or_zeros("t1")
    t1ce_z  = _zscore_or_zeros("t1ce")

    tumour_volumes       = volumes(seg, seg_zooms)
    enhancement_features = enhancement(seg, t1_z, t1ce_z, tumour_volumes) if "t1ce" in intensities else {
        "present": None, "proportion": None, "enhancement_z": None}
    necrosis_features    = necrosis(seg, seg_zooms)
    foci_features        = enhancing_foci(seg, seg_zooms)
    edema_features       = edema_extent(seg, seg_affine)
    laterality_features  = laterality_seg_only(seg, seg_affine)

    return {
        "volumes":     tumour_volumes,
        "enhancement": enhancement_features,
        "necrosis":    necrosis_features,
        "edema":       edema_features,
        "additional":  foci_features,
        "laterality_seg_only": laterality_features,
        # Every threshold that decides a reported finding. FOCUS_MIN_CM3 is coupled to
        # seg_cleanup.ISLAND_CM3: cleanup drops whole-tumour components below that size
        # before this runs, so a focus smaller than ISLAND_CM3 can never be counted here.
        "thresholds": {
            "EDEMA_MM":          EDEMA_MM,
            "NECROSIS_CM3":      NECROSIS_CM3,
            "ENHANCE_CM3":       ENHANCE_CM3,
            "NECROSIS_CLOSE_MM": NECROSIS_CLOSE_MM,
            "FOCUS_MIN_CM3":     FOCUS_MIN_CM3,
            "T_ENH_AVID_Z":      T_ENH_AVID_Z,
        },
    }


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t1ce", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--flair", required=True)
    args = parser.parse_args()
    out = compute_all(args.seg, {"t1": args.t1, "t1ce": args.t1ce,
                                    "t2": args.t2, "flair": args.flair})
    json.dump(out, sys.stdout, indent=2)
    print()
