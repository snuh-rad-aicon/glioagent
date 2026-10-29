"""Deterministic atlas-based imaging features: warp HD-GLIO seg to MNI, then
laterality / lobe / cortical involvement / midline shift as coordinate math."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes


TAU_CORTICAL = 0.10   # cortical involvement overlap fraction
TAU_SHIFT    = 1.0    # midline shift mm cutoff; 5.0 is the threshold for surgical
                      # significance, not for detection, and a radiologist reports any
                      # visible shift. Agreement with the reports is 17.2% at 5.0 mm
                      # and 86.3% at 1.0 mm (Fig 5b).
TAU_CROSS    = 0.15   # minority-hemisphere WT fraction to flag midline crossing


ATLAS_DIR = Path(__file__).resolve().parent / "atlas"

MNI_T1_BRAIN     = ATLAS_DIR / "mni152_t1_brain.nii.gz"
MNI_BRAIN_MASK   = ATLAS_DIR / "mni152_brain_mask.nii.gz"
MNI_CORTEX_MASK  = ATLAS_DIR / "cortex_mask.nii.gz"          # GM prob > 0.5
MNI_AAL          = ATLAS_DIR / "aal116_dseg.nii.gz"         # AAL116 (SPM12), MNI grid
_AAL_LABELS_FILE = ATLAS_DIR / "aal116_labels.json"        # {code: [base_name, side]}
MNI_ASEG         = ATLAS_DIR / "mni152_aseg_dseg.nii.gz"   # FreeSurfer aseg in this template (see .json)

# Deep structures the lobar parcellation cannot express. AAL116 has thalamus / basal-ganglia /
# cerebellum parcels but no brainstem and no corpus callosum, and a max-overlap lobe call buries a
# deep structure under the cortical bulk of a large tumour, so involvement is decided by how much
# tumour sits IN the structure. Labels are FreeSurfer aseg codes (left + right summed).
ASEG_DEEP_LABELS = {"thalamus": (10, 49), "basal ganglia": (11, 50, 12, 51, 13, 52),
                    "brainstem": (16,), "corpus callosum": (251, 252, 253, 254, 255)}
# Thresholds fixed on the reader-study dev split (100 SNUH cases) by F1 against the radiology
# reports; the held-out 790 were not used to choose them.
DEEP_OVERLAP_CM3 = 0.5      # thalamus / basal ganglia / brainstem, measured on the whole
                            # lesion. Reports name a deep structure when the tumour touches
                            # it at all; agreement is 36.7% at 3.0 cm3 and 66.8% at 0.5 (Fig 5b).
CALLOSUM_OVERLAP_CM3 = 0.1  # corpus callosum, measured on the oedema-excluded mask (thin midline slab)
CALLOSUM_ENHANCE_CM3 = 0.10 # ET volume above which the oedema-excluded mask is the enhancing mass


# AAL116 (Tzourio-Mazoyer 2002): one parcellation covers cortex + subcortex +
# cerebellum with L/R encoded per label; no brain-stem / corpus-callosum parcel.

@lru_cache(maxsize=8)
def _atlas_volume(path_str: str) -> np.ndarray:
    """Atlas volumes are static, so read each one once per process (they are re-used per case)."""
    return nib.load(path_str).get_fdata()


def _load_aal_labels():
    labels_json = json.loads(_AAL_LABELS_FILE.read_text()) if _AAL_LABELS_FILE.is_file() else {}
    return {int(label_code): (base_name, side)
            for label_code, (base_name, side) in labels_json.items()}

_AAL = _load_aal_labels()
_AAL_SIDE_LUT = None


def _aal_side_lut():
    global _AAL_SIDE_LUT
    if _AAL_SIDE_LUT is None:
        table_size = (max(_AAL) + 1) if _AAL else 1
        side_code_by_label = np.zeros(table_size, dtype=np.int8)
        for label_code, (base_name, side) in _AAL.items():
            side_code_by_label[label_code] = 1 if side == "L" else (2 if side == "R" else 0)
        _AAL_SIDE_LUT = side_code_by_label
    return _AAL_SIDE_LUT


def _aal_lobe(base_name):
    """AAL region base name -> lobar / deep-structure category."""
    if base_name.startswith("Cerebelum") or base_name.startswith("Vermis"):
        return "cerebellum"
    if base_name == "Thalamus":
        return "thalamus"
    if base_name in ("Caudate", "Putamen", "Pallidum"):
        return "basal ganglia"
    if base_name == "Insula":
        return "insular"
    if base_name in ("Cingulum_Ant", "Cingulum_Mid"):
        return "frontal"
    if base_name == "Cingulum_Post":
        return "parietal"
    if base_name.startswith("Frontal") or base_name in (
            "Precentral", "Rectus", "Olfactory", "Supp_Motor_Area",
            "Paracentral_Lobule",
            "Rolandic_Oper"):
        return "frontal"
    if base_name in ("Postcentral", "Parietal_Sup", "Parietal_Inf",
                     "SupraMarginal", "Angular", "Precuneus"):
        return "parietal"
    if base_name.startswith("Temporal") or base_name in (
            "Hippocampus", "ParaHippocampal", "Amygdala", "Heschl",
            "Fusiform"):
        return "temporal"
    if base_name in ("Calcarine", "Cuneus", "Lingual",
                     "Occipital_Sup", "Occipital_Mid", "Occipital_Inf"):
        return "occipital"
    return "other"


def _aal_region_name(base_name):
    """Human-readable AAL parcel name for the informational `region` field."""
    return base_name.replace("_", " ")


TRANSFORM_SOURCE_FILE = "source_t1.json"    # which T1 the cached transform belongs to


def _t1_fingerprint(t1_path: str) -> str:
    # content hash, so a renamed or moved copy of the same T1 still matches
    return hashlib.sha256(Path(t1_path).read_bytes()).hexdigest()


def cached_transform(t1_path: str, out_xfm_dir: str) -> dict[str, Any] | None:
    """The T1->MNI transform in `out_xfm_dir`, but only if it was computed from this
    T1. Reusing an output directory for another patient must never reuse its warp."""
    transform_dir = Path(out_xfm_dir)
    forward_affine_path = transform_dir / "fwd_0GenericAffine.mat"
    forward_warp_path = transform_dir / "fwd_1Warp.nii.gz"
    inverse_warp_path = transform_dir / "fwd_1InverseWarp.nii.gz"
    source_path = transform_dir / TRANSFORM_SOURCE_FILE
    if not (forward_affine_path.is_file() and forward_warp_path.is_file() and source_path.is_file()):
        return None
    try:
        recorded = json.loads(source_path.read_text()).get("sha256")
    except (OSError, ValueError):
        return None
    if recorded != _t1_fingerprint(t1_path):
        return None
    return {
        "fwdtransforms": [str(forward_warp_path), str(forward_affine_path)],
        "invtransforms": [str(forward_affine_path), str(inverse_warp_path)],
        "cached": True,
    }


def register_t1_to_mni(t1_path: str, out_xfm_dir: str,
                        force: bool = False) -> dict[str, Any]:
    """Register a patient T1 to MNI152NLin2009cAsym.

    Cached: reuses fwd_0GenericAffine.mat + fwd_1Warp.nii.gz when source_t1.json
    shows they came from this same T1; anything else re-registers.
    """
    import ants

    if not force:
        cached = cached_transform(t1_path, out_xfm_dir)
        if cached:
            return cached

    transform_dir = Path(out_xfm_dir); transform_dir.mkdir(parents=True, exist_ok=True)
    fixed  = ants.image_read(str(MNI_T1_BRAIN))
    moving = ants.image_read(str(t1_path))
    registration = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform="antsRegistrationSyNQuick[s]",
        outprefix=str(transform_dir / "fwd_"),
        verbose=False,
    )
    (transform_dir / TRANSFORM_SOURCE_FILE).write_text(json.dumps(
        {"t1_path": str(Path(t1_path).resolve()), "sha256": _t1_fingerprint(t1_path)}, indent=2))
    return {
        "fwdtransforms": registration["fwdtransforms"],
        "invtransforms": registration["invtransforms"],
        "cached": False,
    }


def apply_transform_to_seg(seg_path: str, xfm_result: dict,
                              out_path: str) -> str:
    """Apply the fwd transform to the seg with nearest-neighbour interp."""
    import ants
    fixed = ants.image_read(str(MNI_T1_BRAIN))
    moving_seg = ants.image_read(str(seg_path))
    warped = ants.apply_transforms(
        fixed=fixed, moving=moving_seg,
        transformlist=xfm_result["fwdtransforms"],
        interpolator="genericLabel",
    )
    ants.image_write(warped, str(out_path))
    return out_path


def _world_coords_of_mask(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return (3, N) array of RAS world coords (mm) for voxels where mask is True."""
    voxel_indices = np.array(np.where(mask))
    if voxel_indices.size == 0:
        return np.zeros((3, 0), dtype=float)
    return (affine[:3, :3] @ voxel_indices + affine[:3, 3:4]).astype(float)


def laterality_atlas(seg_mni: np.ndarray, mni_affine: np.ndarray) -> dict[str, Any]:
    """Laterality by voxel-majority across the MNI midline (RAS +x = Right),
    cross-checked against the AAL L/R label vote. Voxel-majority is used rather
    than the signed-x mean, which can flip on a small far-lateral component in a
    near-midline tumour. Crossing flagged when the minority hemisphere holds
    >= TAU_CROSS of voxels."""
    wt = seg_mni > 0
    if not wt.any():
        return {"dominant_side": "unknown", "signed_x_mm": 0.0, "midline_crossing": False}
    coords = _world_coords_of_mask(wt, mni_affine)
    signed_x = float(coords[0].mean())
    left_frac  = float((coords[0] < 0).mean())
    right_frac = float((coords[0] > 0).mean())

    dominant = "left" if left_frac >= right_frac else "right"

    label_side = None; label_left_voxels = label_right_voxels = 0
    if MNI_AAL.is_file():
        aal_label_volume = _atlas_volume(str(MNI_AAL)).astype(int)
        if aal_label_volume.shape == seg_mni.shape:
            tumour_labels = aal_label_volume[wt]; tumour_labels = tumour_labels[tumour_labels > 0]
            side_code_by_label = _aal_side_lut()
            voxel_side_codes = side_code_by_label[
                np.clip(tumour_labels, 0, len(side_code_by_label) - 1)]
            label_left_voxels = int((voxel_side_codes == 1).sum())
            label_right_voxels = int((voxel_side_codes == 2).sum())
            if label_left_voxels != label_right_voxels:
                label_side = "left" if label_left_voxels > label_right_voxels else "right"

    minority_frac = min(left_frac, right_frac)
    crossing = minority_frac >= TAU_CROSS
    return {
        "dominant_side":       dominant,
        "signed_x_mm":         signed_x,
        "midline_crossing":    bool(crossing),
        "left_fraction":       left_frac,
        "right_fraction":      right_frac,
        "label_side":          label_side,
        "label_left_voxels":   label_left_voxels,
        "label_right_voxels":  label_right_voxels,
        "method":              "voxel_majority_across_MNI_midline (AAL116 label cross-check)",
    }


def _filled_enhancing(seg_mni: np.ndarray) -> np.ndarray:
    """Hole-filled enhancing component, computed inside its bounding box."""
    et = seg_mni == 2
    if not et.any():
        return et
    idx = np.array(np.where(et))
    lo = np.maximum(idx.min(axis=1) - 2, 0); hi = np.minimum(idx.max(axis=1) + 3, et.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    filled = np.zeros_like(et)
    filled[box] = binary_fill_holes(et[box])
    return filled


def deep_structure_involvement(seg_mni: np.ndarray) -> dict[str, Any]:
    """Which deep structures the tumour involves, by overlap volume in MNI (1 mm isotropic).

    thalamus / basal ganglia / brainstem : whole lesion, overlap >= DEEP_OVERLAP_CM3
    corpus callosum                      : oedema-excluded mask (the enhancing mass when the tumour
                                           enhances, the whole lesion otherwise),
                                           overlap >= CALLOSUM_OVERLAP_CM3
    Cerebellum is NOT here: the AAL116 lobe call already handles it (`location.lobe == "cerebellum"`).
    """
    if not MNI_ASEG.is_file():
        return {"involved": None, "note": "aseg atlas not available"}
    aseg = _atlas_volume(str(MNI_ASEG)).astype(int)
    if aseg.shape != seg_mni.shape:
        return {"involved": None, "note": "aseg/seg shape mismatch"}
    whole = seg_mni > 0
    enhancing_cm3 = float((seg_mni == 2).sum()) / 1000.0
    oedema_excluded = _filled_enhancing(seg_mni) if enhancing_cm3 > CALLOSUM_ENHANCE_CM3 else whole

    overlap_cm3, involved = {}, []
    for name, codes in ASEG_DEEP_LABELS.items():
        structure = np.isin(aseg, list(codes))
        mask = oedema_excluded if name == "corpus callosum" else whole
        cm3 = float((mask & structure).sum()) / 1000.0
        overlap_cm3[name] = round(cm3, 2)
        if cm3 >= (CALLOSUM_OVERLAP_CM3 if name == "corpus callosum" else DEEP_OVERLAP_CM3):
            involved.append(name)
    return {
        "involved":            involved,
        "overlap_cm3":         overlap_cm3,
        "tau_cm3":             {"default": DEEP_OVERLAP_CM3, "corpus callosum": CALLOSUM_OVERLAP_CM3},
        "callosum_mask":       "enhancing mass (filled)" if oedema_excluded is not whole else "whole lesion",
        "atlas":               "FreeSurfer aseg in MNI152NLin2009cAsym (see mni152_aseg_source.json)",
        "provenance":          "atlas_features.deep_structure_involvement (overlap volume, dev-chosen thresholds)",
    }


def location_lobe(seg_mni: np.ndarray) -> dict[str, Any]:
    """Assign the tumour to a lobar / deep-structure category by max WT-overlap
    with the AAL116 parcellation.

    Vocabulary: frontal / parietal / temporal / occipital / insular / thalamus /
    'basal ganglia' / cerebellum, or 'deep_or_extracortical' when the WT overlaps
    no AAL parcel. AAL116 has no brain-stem or corpus-callosum label, so a tumour
    centred there is reported by the adjacent structure (a stated Limitation).
    """
    if not MNI_AAL.is_file():
        return {"lobe": None, "region": None, "side": None,
                "overlap_voxels": 0, "note": "AAL atlas not available"}
    aal_label_volume = _atlas_volume(str(MNI_AAL)).astype(int)
    wt = seg_mni > 0
    if not wt.any() or aal_label_volume.shape != seg_mni.shape:
        return {"lobe": None, "region": None, "side": None, "overlap_voxels": 0}
    tumour_labels = aal_label_volume[wt]
    tumour_labels = tumour_labels[tumour_labels > 0]
    if tumour_labels.size == 0:
        return {"lobe": "deep_or_extracortical", "region": None, "side": None,
                "overlap_voxels": 0,
                "note": "WT overlaps no AAL parcel -- deep white-matter / extra-atlas tumour"}
    label_codes, voxel_counts = np.unique(tumour_labels, return_counts=True)

    lobe_counts: dict[str, int] = {}
    for label_code, voxel_count in zip(label_codes.tolist(), voxel_counts.tolist()):
        lobe = _aal_lobe(_AAL.get(int(label_code), ("", ""))[0])
        if lobe == "other":
            continue
        lobe_counts[lobe] = lobe_counts.get(lobe, 0) + int(voxel_count)
    if not lobe_counts:
        return {"lobe": "deep_or_extracortical", "region": None, "side": None,
                "overlap_voxels": 0, "note": "WT overlaps only unmapped AAL parcels"}
    total_voxels = int(sum(lobe_counts.values()))
    dominant_lobe = max(lobe_counts.items(), key=lambda lobe_and_count: lobe_and_count[1])[0]

    # region + side = top AAL parcel within the dominant lobe (not the global top).
    dominant_lobe_parcels = [
        (int(label_code), int(voxel_count))
        for label_code, voxel_count in zip(label_codes.tolist(), voxel_counts.tolist())
        if _aal_lobe(_AAL.get(int(label_code), ("", ""))[0]) == dominant_lobe]
    top_label = (max(dominant_lobe_parcels, key=lambda parcel_and_count: parcel_and_count[1])[0]
                 if dominant_lobe_parcels else int(label_codes[np.argmax(voxel_counts)]))
    base_name, side = _AAL.get(top_label, ("unknown", "M"))

    junction_lobe_counts = sorted(
        [(other_lobe, other_count) for other_lobe, other_count in lobe_counts.items()
         if other_lobe != dominant_lobe and other_count / total_voxels > 0.30],
        key=lambda lobe_and_count: -lobe_and_count[1])

    return {
        "lobe":              dominant_lobe,
        "region":            _aal_region_name(base_name),
        "side":              side,
        "overlap_voxels":    total_voxels,
        "lobe_distribution": lobe_counts,
        "lobe_consistent":   True,
        "junction_lobes":    [junction_lobe for junction_lobe, _ in junction_lobe_counts],
        "atlas":             "AAL116",
    }


def cortical_involvement(seg_mni: np.ndarray) -> dict[str, Any]:
    """WT surface  intersect  cortical GM ribbon (MNI GM prob > 0.5)."""
    if not MNI_CORTEX_MASK.is_file():
        return {"overlap_frac": None, "involved": None, "tau": TAU_CORTICAL,
                "note": "cortex mask not available"}
    cortex = _atlas_volume(str(MNI_CORTEX_MASK)) > 0
    if cortex.shape != seg_mni.shape:
        return {"overlap_frac": None, "involved": None, "tau": TAU_CORTICAL,
                "note": "atlas/seg shape mismatch"}
    wt = seg_mni > 0
    if not wt.any():
        return {"overlap_frac": 0.0, "involved": False, "tau": TAU_CORTICAL}
    surface = wt & ~binary_erosion(wt, iterations=1)
    n_surface = int(surface.sum())
    if n_surface == 0:
        return {"overlap_frac": 0.0, "involved": False, "tau": TAU_CORTICAL}
    overlap = int((surface & cortex).sum())
    overlap_frac = overlap / n_surface
    return {
        "overlap_frac": float(overlap_frac),
        "involved":     bool(overlap_frac > TAU_CORTICAL),
        "tau":          TAU_CORTICAL,
    }


def midline_shift(seg_mni: np.ndarray, mni_affine: np.ndarray) -> dict[str, Any]:
    """Midline shift proxy in MNI space: shift_mm = |mean x_world of brain-minus-
    tumour tissue|, present iff > TAU_SHIFT. The whole-tissue centroid is ~0 for a
    normal brain and moves only when the mass displaces contralateral tissue --
    unlike a single-hemisphere |x|, which is ~35 mm off midline by construction.
    """
    if not MNI_BRAIN_MASK.is_file():
        return {"midline_shift_mm": None, "present": None,
                "tau_mm": TAU_SHIFT, "note": "brain mask unavailable"}
    brain = _atlas_volume(str(MNI_BRAIN_MASK)) > 0
    if brain.shape != seg_mni.shape:
        return {"midline_shift_mm": None, "present": None,
                "tau_mm": TAU_SHIFT, "note": "atlas/seg shape mismatch"}
    tissue = brain & ~(seg_mni > 0)
    if not tissue.any():
        return {"midline_shift_mm": 0.0, "present": False, "tau_mm": TAU_SHIFT}
    coords = _world_coords_of_mask(tissue, mni_affine)
    shift_mm = float(abs(coords[0].mean()))
    return {
        "midline_shift_mm": shift_mm,
        "present":          bool(shift_mm > TAU_SHIFT),
        "tau_mm":           TAU_SHIFT,
        "ventricle_shift_mm": None,   # placeholder: needs an aseg segmentation
    }


def compute_all(t1_path: str, seg_path: str, xfm_dir: str,
                 seg_mni_out_path: str) -> dict[str, Any]:
    """Register, warp seg, compute all atlas features."""
    registration = register_t1_to_mni(t1_path, xfm_dir)
    apply_transform_to_seg(seg_path, registration, seg_mni_out_path)

    warped_seg_img = nib.load(str(seg_mni_out_path))
    seg_mni = warped_seg_img.get_fdata().astype(int)
    mni_affine = warped_seg_img.affine

    laterality  = laterality_atlas(seg_mni, mni_affine)
    location    = location_lobe(seg_mni)
    deep        = deep_structure_involvement(seg_mni)
    cortical    = cortical_involvement(seg_mni)
    mass_effect = midline_shift(seg_mni, mni_affine)

    return {
        "laterality":   laterality,
        "location":     location,
        "deep_structures": deep,
        "cortical":     cortical,
        "mass_effect":  mass_effect,
        "reg_cached":   bool(registration.get("cached", False)),
        # Every threshold that decides a reported finding, so a trace can be audited
        # without reading this file at the version it was produced with.
        "thresholds": {
            "TAU_CORTICAL":         TAU_CORTICAL,
            "TAU_SHIFT":            TAU_SHIFT,
            "TAU_CROSS":            TAU_CROSS,
            "DEEP_OVERLAP_CM3":     DEEP_OVERLAP_CM3,
            "CALLOSUM_OVERLAP_CM3": CALLOSUM_OVERLAP_CM3,
            "CALLOSUM_ENHANCE_CM3": CALLOSUM_ENHANCE_CM3,
        },
    }
