import ants
import os

def register_to_mni(anat_img_path, atlas_masks_folder, output_prefix):
    """
    Registers an anatomical image to the MNI152 atlas using nonlinear ANTs warping.

    Parameters:
        anat_img_path (str): Path to the anatomical image (e.g., T1w).
        atlas_masks_folder (str): Path to the folder containing the atlas masks.
            The folder must contain a reference brain NIfTI (either
            ``MNI152_T1_1mm_brain.nii.gz`` or
            ``MNI152_in_SRI24_T1_1mm_brain.nii.gz``).
        output_prefix (str): Prefix for output files.

    Returns:
        dict: Dictionary with keys 'warped_image', 'transform', and 'inverse_transform'.
    """
    # Find the reference brain — try MNI152 first, then SRI24
    mni_atlas_path = os.path.join(atlas_masks_folder, 'MNI152_T1_1mm_brain.nii.gz')
    if not os.path.isfile(mni_atlas_path):
        mni_atlas_path = os.path.join(atlas_masks_folder, 'MNI152_in_SRI24_T1_1mm_brain.nii.gz')
    if not os.path.isfile(mni_atlas_path):
        raise FileNotFoundError(
            f"No reference brain image found in {atlas_masks_folder}. "
            f"Expected MNI152_T1_1mm_brain.nii.gz or MNI152_in_SRI24_T1_1mm_brain.nii.gz"
        )
    anat = ants.image_read(anat_img_path)
    mni = ants.image_read(mni_atlas_path)

    reg = ants.registration(
        fixed=mni,
        moving=anat,
        type_of_transform='SyN'
    )

    warped_image = reg['warpedmovout']
    transform = reg['fwdtransforms']
    inverse_transform = reg['invtransforms']

    warped_image.to_file(f"{output_prefix}_to_MNI.nii.gz")
    # Save transforms
    for i, t in enumerate(transform):
        ants.write_transform(t, f"{output_prefix}_fwd_{i}.h5")
    for i, t in enumerate(inverse_transform):
        ants.write_transform(t, f"{output_prefix}_inv_{i}.h5")

    return {
        'warped_image': f"{output_prefix}_to_MNI.nii.gz",
        'transform': [f"{output_prefix}_fwd_{i}.h5" for i in range(len(transform))],
        'inverse_transform': [f"{output_prefix}_inv_{i}.h5" for i in range(len(inverse_transform))]
    }