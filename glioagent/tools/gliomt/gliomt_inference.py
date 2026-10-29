"""GlioMT inference: run the IDH / 1p19q / grade heads on a single case.

GlioMT models internally `.cuda()` the BERT embeddings, so the active CUDA
device must be visible at import time (we honour CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GLIOMT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GLIOMT_ROOT.parents[1]
CKPT_DIR = Path(os.environ.get("GLIOMT_CKPT_DIR", str(GLIOMT_ROOT.parent / "ckpts" / "gliomt")))

if str(GLIOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(GLIOMT_ROOT))

from src.utils import normalize_images  # noqa: E402
from src.models.multimodal_swin import create_swint_model  # noqa: E402
from src.models.multimodal_vit import create_model as create_vit_model  # noqa: E402

import SimpleITK as sitk  # noqa: E402
from monai.transforms import Resized  # noqa: E402


SLICE_PERCENTILE = 75  # matches configs/data/eval.yaml
INPUT_SIZE = (224, 224)


def _load_volume(path: str) -> np.ndarray:
    """sitk -> numpy [D, H, W]."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _build_slice_stack(
    t1: str, t1ce: str, t2: str, flair: str, seg: str,
    slice_percentile: int = SLICE_PERCENTILE,
) -> tuple[torch.Tensor, list[int]]:
    """Mirror Brain_Dataset.__getitem__ from the GlioMT eval script.

    Returns a (1, 4*num_slices, H, W) tensor and the axial slice indices used.
    """
    t1_arr = _load_volume(t1)
    t1ce_arr = _load_volume(t1ce)
    t2_arr = _load_volume(t2)
    flair_arr = _load_volume(flair)
    seg_arr = _load_volume(seg)
    seg_arr = (seg_arr != 0).astype(np.uint8)

    t1_norm, t1ce_norm, t2_norm, flair_norm = normalize_images(t1_arr, t1ce_arr, t2_arr, flair_arr)
    image_stack = np.stack([t1_norm, t1ce_norm, t2_norm, flair_norm], axis=0)  # (4, D, H, W)

    tumor_voxels_per_slice = seg_arr.sum(axis=(-1, -2))  # voxels per axial slice
    if not (tumor_voxels_per_slice > 0).any():
        roi_slice_indices = [seg_arr.shape[0] // 2]
    elif slice_percentile <= 1:
        roi_slice_indices = [int(np.argmax(tumor_voxels_per_slice))]
    else:
        voxel_count_cutoff = np.percentile(tumor_voxels_per_slice[tumor_voxels_per_slice > 0],
                                           100 - slice_percentile)
        roi_slice_indices = np.where(tumor_voxels_per_slice > voxel_count_cutoff)[0].tolist()
        if not roi_slice_indices:
            roi_slice_indices = [int(np.argmax(tumor_voxels_per_slice))]

    roi_images = image_stack[:, roi_slice_indices]    # (4, num_slices, H, W)
    roi_seg = seg_arr[roi_slice_indices]              # (num_slices, H, W)
    C, num_slices, H, W = roi_images.shape
    channel_stack = roi_images.reshape(C * num_slices, H, W)  # (4*num_slices, H, W)
    resize_input = {"image": channel_stack.astype(np.float32), "seg": roi_seg.astype(np.float32)}
    resize_transform = Resized(keys=["image", "seg"], spatial_size=INPUT_SIZE, mode=["bicubic", "nearest"])
    resized = resize_transform(resize_input)
    image_tensor = resized["image"]
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)
    image_tensor = image_tensor.float().unsqueeze(0)  # (1, 4*num_slices, 224, 224)
    return image_tensor, roi_slice_indices


def _slice_ensemble(inputs: torch.Tensor, clinical_feats, net, in_chans: int = 4) -> torch.Tensor:
    """Average logits across the num_slices axial slices (per GlioMT eval)."""
    B, channels_x_slices, H, W = inputs.shape
    assert B == 1
    num_slices = channels_x_slices // in_chans
    inputs = inputs.view(B, in_chans, num_slices, H, W).contiguous()
    logit_sum = None
    for i in range(num_slices):
        slice_input = inputs[:, :, i]
        logits = net(slice_input, clinical_feats).detach().cpu()
        logit_sum = logits if logit_sum is None else logit_sum + logits
    return logit_sum / num_slices


@dataclass
class GlioMTBundle:
    idh: torch.nn.Module
    p1_19q: torch.nn.Module
    grade: torch.nn.Module


def _build_models(device: str) -> GlioMTBundle:
    """Instantiate the three architectures with the eval-config kwargs.

    Grade head has num_classes=3 (WHO 2021 grades 2/3/4 collapsed to
    0/1/2 in the released checkpoint); IDH and 1p/19q are binary.
    """
    swin_kwargs_binary = dict(
        pretrained=False,
        pretrained_cfg="ms_in22k_ft_in1k",
        pretrained_strict=False,
        in_chans=4,
        num_classes=1,
        decoder_depth=2,
        clini_embed_token="word",
        age_cutoff=45,
    )
    swin_kwargs_grade = dict(swin_kwargs_binary, num_classes=3)
    vit_kwargs = dict(
        pretrained=False,
        pretrained_cfg="augreg_in21k_ft_in1k",
        pretrained_strict=False,
        in_chans=4,
        num_classes=1,
        decoder_depth=2,
        clini_embed_token="word",
        age_cutoff=45,
    )

    idh_net = create_vit_model("vit_base_patch16_224", **vit_kwargs).to(device)
    p1_19q_net = create_swint_model("swin_small_patch4_window7_224", **swin_kwargs_binary).to(device)
    grade_net = create_swint_model("swin_small_patch4_window7_224", **swin_kwargs_grade).to(device)
    return GlioMTBundle(idh=idh_net, p1_19q=p1_19q_net, grade=grade_net)


_BUNDLE_CACHE: dict[str, GlioMTBundle] = {}


def _load_ckpts(bundle: GlioMTBundle, ckpt_dir: str = str(CKPT_DIR)) -> None:
    ckpt_files = {
        "idh": "idh_vitb.pth",
        "p1_19q": "1p19q_swins.pth",
        "grade": "grade_swins.pth",
    }
    for head_attr, ckpt_name in ckpt_files.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("net", ckpt)
        net: torch.nn.Module = getattr(bundle, head_attr)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        # `missing` are pretrained timm keys skipped at build time; only
        # unexpected keys matter, and should be near-zero.
        if unexpected:
            print(f"[gliomt] {ckpt_name}: {len(unexpected)} unexpected keys (will be ignored)")
        net.eval()


def load_bundle(device: str = "cuda") -> GlioMTBundle:
    cache_key = str(device)
    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]
    bundle = _build_models(device=device)
    _load_ckpts(bundle)
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def predict(
    t1: str, t1ce: str, t2: str, flair: str, seg: str,
    age: int, sex: str,
    device: str = "cuda",
    conditional_1p19q: bool = True,
) -> dict:
    """Run the GlioMT heads on one case, in WHO 2021 order.

    `sex` accepts "male"/"female"/"m"/"f" or 0/1 ints. IDH and grade always
    run; with `conditional_1p19q=True` the 1p/19q head runs only for
    IDH-mutant cases (IDH-wildtype results omit the `p1_19q_*` keys).
    """
    sex_text = str(sex).strip().lower()
    if sex_text in {"0", "m", "male"}:
        sex_label = "male"
    elif sex_text in {"1", "f", "female"}:
        sex_label = "female"
    else:  # a silent default would shift every head's prediction
        raise ValueError(f"unrecognized sex {sex!r}: expected male/female, m/f or 0/1")

    bundle = load_bundle(device=device)
    image_tensor, _ = _build_slice_stack(t1, t1ce, t2, flair, seg)
    image_tensor = image_tensor.to(device, dtype=torch.float32, non_blocking=True)

    clinical_feats = [[str(int(age))], [sex_label]]  # B=1 batches

    # WHO 2021 order: IDH first, grade always; 1p/19q only for IDH-mutant.
    with torch.inference_mode():
        idh_logit = _slice_ensemble(image_tensor, clinical_feats, bundle.idh)
        grade_logits = _slice_ensemble(image_tensor, clinical_feats, bundle.grade)

    idh_prob_mutant = torch.sigmoid(idh_logit).flatten()[0].item()
    idh_pred = "mutant" if idh_prob_mutant >= 0.5 else "wildtype"

    # Grade head is 3-class (WHO grades 2/3/4 -> indices 0/1/2). Softmax
    # over the (1, 3) logits to recover P(grade==k).
    grade_probs = torch.softmax(grade_logits.flatten(), dim=0).tolist()
    grade_pred_idx = int(np.argmax(grade_probs))
    grade_pred = 2 + grade_pred_idx  # 0->2, 1->3, 2->4
    grade_prob = float(grade_probs[grade_pred_idx])

    result: dict = {
        "idh_prob_mutant": idh_prob_mutant,
        "idh_prediction": idh_pred,
        "grade_prediction": grade_pred,
        "grade_probability": grade_prob,
        "grade_probs": {f"grade_{2+i}": prob for i, prob in enumerate(grade_probs)},
        "raw_logits": {
            "idh": float(idh_logit.flatten()[0]),
            "grade": [float(logit) for logit in grade_logits.flatten()],
        },
        # Records whether the 1p/19q head was actually run (WHO 2021 gating).
        "p1_19q_evaluated": bool((not conditional_1p19q) or idh_pred == "mutant"),
    }

    # 1p/19q -- evaluated only for IDH-mutant (or when gating disabled).
    if result["p1_19q_evaluated"]:
        with torch.inference_mode():
            p1_19q_logit = _slice_ensemble(image_tensor, clinical_feats, bundle.p1_19q)
        p1_19q_prob_codel = torch.sigmoid(p1_19q_logit).flatten()[0].item()
        # Conventions: 1p/19q codeletion = 1 -> codeleted.
        result["p1_19q_prob_codel"] = p1_19q_prob_codel
        result["p1_19q_prediction"] = "codeleted" if p1_19q_prob_codel >= 0.5 else "non-codeleted"
        result["raw_logits"]["p1_19q"] = float(p1_19q_logit.flatten()[0])

    return result


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t1ce", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--flair", required=True)
    parser.add_argument("--seg", required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", default="male")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = predict(args.t1, args.t1ce, args.t2, args.flair, args.seg,
                     age=args.age, sex=args.sex, device=args.device)
    print(json.dumps(result, indent=2))
