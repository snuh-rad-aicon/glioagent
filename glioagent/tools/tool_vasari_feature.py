"""Wrap VASARI-auto for HD-GLIO-style segmentation masks.

HD-GLIO NE must NOT be injected into VASARI label 2 (oedema): that would
fabricate a perilesional-oedema signal HD-GLIO never labelled. Because of this
mapping three VASARI outputs (``proportion_edema``, ``t2_flair_mismatch``,
``proportion_necrosis``) are known-unreliable, listed under
``_unreliable_fields``, and must be overridden downstream
(``agents.report_assembly._collect_subresults``) with ``tools.tool_seg_features``.
VASARI is kept only for location / laterality / enhancement quality / cortical /
multifocal / ependymal extension. CPU-only.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


# set per run_vasari call: the T1->MNI transform atlas_features already computed
_REUSE_TRANSFORM: dict | None = None


def _same_grid(first_path: str, second_path: str) -> bool:
    # a transform may only be reused when both images share the sampling grid
    try:
        first_img, second_img = nib.load(first_path), nib.load(second_path)
        return (first_img.shape[:3] == second_img.shape[:3]
                and np.allclose(first_img.affine, second_img.affine, atol=1e-4))
    except Exception:
        return False


def _remap_hdglio_to_vasari(seg_path: str, out_path: str) -> str:
    seg_img = nib.load(seg_path)
    hdglio_labels = seg_img.get_fdata().astype(np.int16)
    vasari_labels = np.zeros_like(hdglio_labels)
    vasari_labels[hdglio_labels == 1] = 1  # non-enhancing -> VASARI nCET
    vasari_labels[hdglio_labels == 2] = 3  # enhancing -> VASARI enhancing
    nib.save(nib.Nifti1Image(vasari_labels, seg_img.affine, seg_img.header), out_path)
    return out_path


def _patch_vasari_register() -> None:
    """Replace `vasari_auto.utils.register_to_mni` with a path-aware version.

    The shipped implementation calls `ants.write_transform(t, '...h5')`, but on
    ANTsPy >=0.6 `t` is a path string, which breaks and forces a wrong suffix.
    This replacement copies the ANTs outputs verbatim and returns their paths.
    """
    import os
    import shutil

    import ants
    from vasari_auto import utils as vasari_utils

    if getattr(vasari_utils.register_to_mni, "__patched__", False):
        return

    def register_to_mni(anat_img_path, atlas_masks_folder, output_prefix):
        mni_path = os.path.join(atlas_masks_folder, "MNI152_T1_1mm_brain.nii.gz")
        if not os.path.isfile(mni_path):
            mni_path = os.path.join(atlas_masks_folder, "MNI152_in_SRI24_T1_1mm_brain.nii.gz")
        if not os.path.isfile(mni_path):
            raise FileNotFoundError(f"No reference brain in {atlas_masks_folder}")
        anat = ants.image_read(anat_img_path)
        mni = ants.image_read(mni_path)

        reuse = _REUSE_TRANSFORM
        if reuse and _same_grid(anat_img_path, reuse["moving"]):
            # atlas_features already warped this grid to MNI; apply it instead of
            # paying for a second SyN registration
            warped = ants.apply_transforms(fixed=mni, moving=anat,
                                           transformlist=list(reuse["fwdtransforms"]),
                                           interpolator="linear")
            warped.to_file(f"{output_prefix}_to_MNI.nii.gz")
            return {"warped_image": f"{output_prefix}_to_MNI.nii.gz",
                    "transform": list(reuse["fwdtransforms"]),
                    "inverse_transform": list(reuse["invtransforms"])}

        registration = ants.registration(fixed=mni, moving=anat, type_of_transform="SyN")
        registration["warpedmovout"].to_file(f"{output_prefix}_to_MNI.nii.gz")

        forward_paths, inverse_paths = [], []
        for i, transform in enumerate(registration["fwdtransforms"]):
            if isinstance(transform, str):
                extension = (".nii.gz" if transform.endswith(".nii.gz")
                             else (os.path.splitext(transform)[1] or ".mat"))
                destination_path = f"{output_prefix}_fwd_{i}{extension}"
                shutil.copy(transform, destination_path)
            else:
                destination_path = f"{output_prefix}_fwd_{i}.h5"
                ants.write_transform(transform, destination_path)
            forward_paths.append(destination_path)
        for i, transform in enumerate(registration["invtransforms"]):
            if isinstance(transform, str):
                extension = (".nii.gz" if transform.endswith(".nii.gz")
                             else (os.path.splitext(transform)[1] or ".mat"))
                destination_path = f"{output_prefix}_inv_{i}{extension}"
                shutil.copy(transform, destination_path)
            else:
                destination_path = f"{output_prefix}_inv_{i}.h5"
                ants.write_transform(transform, destination_path)
            inverse_paths.append(destination_path)
        return {
            "warped_image": f"{output_prefix}_to_MNI.nii.gz",
            "transform": forward_paths,
            "inverse_transform": inverse_paths,
        }

    register_to_mni.__patched__ = True  # type: ignore[attr-defined]
    vasari_utils.register_to_mni = register_to_mni
    import vasari_auto.vasari_auto as vasari_auto_module
    vasari_auto_module.register_to_mni = register_to_mni


def run_vasari(
    seg_mask_path: str,
    anat_path: str | None = None,
    template_space: str = "mni152",
    verbose: bool = False,
    reuse_transform: dict | None = None,
) -> dict[str, Any]:
    """Run VASARI-auto on a HD-GLIO-style mask, return feature dict. `reuse_transform`
    skips the SyN registration when the same grid was already warped to MNI."""
    global _REUSE_TRANSFORM

    _patch_vasari_register()
    from vasari_auto.vasari_auto import get_vasari_features

    _REUSE_TRANSFORM = reuse_transform

    with tempfile.TemporaryDirectory() as temp_dir:
        remapped_seg_path = str(Path(temp_dir) / "seg_vasari_labels.nii.gz")
        _remap_hdglio_to_vasari(seg_mask_path, remapped_seg_path)
        # register_to_mni writes transforms to CWD-relative prefixes; run inside
        # the per-case tempdir so they land there and get cleaned up (parallel-safe).
        previous_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            features = get_vasari_features(
                file=remapped_seg_path,
                anat_img=os.path.abspath(anat_path) if anat_path else anat_path,
                template_space=template_space,
                verbose=verbose,
            )
        except ZeroDivisionError:
            # Upstream raises this when total_lesion_burden = 0 (HD-GLIO seg with
            # only non-enhancing remapped: no enhancing, no oedema).
            return {
                "tumor_location": "", "side": "",
                "enhancement_quality": "", "proportion_enhancing": "0%",
                "proportion_necrosis": "0%", "proportion_edema": "0%",
                "t2_flair_mismatch": False,
                "cortical_involvement": False, "multifocal": False,
                "ependymal_extension": False,
                "_unreliable_fields": ["proportion_edema", "t2_flair_mismatch",
                                        "proportion_necrosis"],
                "raw": {"error": "VASARI total_lesion_burden = 0 (no enhancing or oedema voxels)"},
            }
        finally:
            os.chdir(previous_cwd)
            _REUSE_TRANSFORM = None

    # DataFrame.to_dict() gives {column name: {row index: value}}; VASARI-auto
    # returns one row per case
    if hasattr(features, "to_dict"):
        feature_columns = features.to_dict()
    else:
        feature_columns = dict(features)

    def _first_value(column):
        if isinstance(column, dict) and column:
            return next(iter(column.values()))
        return column

    raw = {name: _first_value(column) for name, column in feature_columns.items()}

    def _lookup_feature(prefix: str, default=None):
        # prefix is a VASARI feature code such as "F1"; column names are "F1 Tumour Location"
        for name, value in raw.items():
            if name.startswith(prefix + " ") or name == prefix:
                return value
        return default

    import math
    def _is_valid_number(value):
        try:
            return value is not None and not math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    LOCATION_CODES = {
        1: "frontal", 2: "temporal", 3: "insular", 4: "parietal",
        5: "occipital", 6: "brainstem", 7: "cerebellum", 8: "other",
    }
    SIDE_CODES = {
        1: "right", 2: "left", 3: "centre/midline", 4: "bilateral",
        5: "non-applicable",
    }
    QUALITY_CODES = {1: "none", 2: "mild/minimal", 3: "marked/avid"}
    # VASARI-auto emits the official 1..8 proportion scale (1 = n/a, 2 = 0%), not 0-based.
    PROPORTION_CODES = {
        1: "not-applicable", 2: "0%", 3: "<5%", 4: "6-33%", 5: "34-67%",
        6: "68-95%", 7: ">95%", 8: "~100%",
    }
    BINARY_YES = {2: True, 1: False, 3: True}

    location_code = _lookup_feature("F1")
    side_code = _lookup_feature("F2")
    enhancement_code = _lookup_feature("F4")
    enhancing_proportion_code = _lookup_feature("F5")
    necrosis_proportion_code = _lookup_feature("F7")
    edema_proportion_code = _lookup_feature("F14")
    ependymal_code = _lookup_feature("F19")
    cortical_code = _lookup_feature("F20")
    multifocal_code = _lookup_feature("F9")

    return {
        "tumor_location": (LOCATION_CODES.get(int(location_code), str(location_code))
                           if _is_valid_number(location_code) else ""),
        "side": (SIDE_CODES.get(int(side_code), str(side_code))
                 if _is_valid_number(side_code) else ""),
        "enhancement_quality": (QUALITY_CODES.get(int(enhancement_code), str(enhancement_code))
                                if _is_valid_number(enhancement_code) else ""),
        "proportion_enhancing": (PROPORTION_CODES.get(int(enhancing_proportion_code),
                                                      str(enhancing_proportion_code))
                                 if _is_valid_number(enhancing_proportion_code) else ""),
        "proportion_necrosis": (PROPORTION_CODES.get(int(necrosis_proportion_code),
                                                     str(necrosis_proportion_code))
                                if _is_valid_number(necrosis_proportion_code) else ""),
        "proportion_edema": (PROPORTION_CODES.get(int(edema_proportion_code),
                                                  str(edema_proportion_code))
                             if _is_valid_number(edema_proportion_code) else ""),
        "t2_flair_mismatch": False,  # VASARI does not derive F26 from masks alone
        "cortical_involvement": (BINARY_YES.get(int(cortical_code), False)
                                 if _is_valid_number(cortical_code) else False),
        "multifocal": (BINARY_YES.get(int(multifocal_code), False)
                       if _is_valid_number(multifocal_code) else False),
        "ependymal_extension": (BINARY_YES.get(int(ependymal_code), False)
                                if _is_valid_number(ependymal_code) else False),
        "_unreliable_fields": ["proportion_edema", "t2_flair_mismatch",
                                "proportion_necrosis"],
        "raw": {str(name): (value if isinstance(value, (str, int, float, bool, type(None)))
                            else str(value))
                for name, value in raw.items()},
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("seg_mask")
    ap.add_argument("--anat")
    ap.add_argument("--template-space", default="mni152")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_vasari(args.seg_mask, args.anat, args.template_space, args.verbose), indent=2, default=str))
