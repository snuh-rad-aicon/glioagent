"""T2-FLAIR mismatch sign detection via few-shot VQA. Call `predict_t2fm(seg, t2, flair)`."""
from __future__ import annotations

import os
import re
from typing import Any, Optional

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_DIR = os.path.join(HERE, "t2fm_refs")
MODEL_ID = os.environ.get("GLIOAGENT_BACKBONE_HF", "Qwen/Qwen3.5-27B")
PANEL = 512

SYSTEM = (
    "You are a neuroradiologist scoring the T2-FLAIR mismatch sign on adult diffuse glioma. "
    "Each image is a TUMOUR-CROPPED axial view: LEFT = T2, RIGHT = FLAIR.\n\n"
    "HOW TO JUDGE (look at the tumour INTERIOR, not the whole brain):\n"
    " 1. On T2 the solid tumour is homogeneously HYPERINTENSE (uniformly bright).\n"
    " 2. On FLAIR the tumour INTERIOR is RELATIVELY LESS bright (suppressed) than its own "
    "peripheral RIM, which stays hyperintense. The interior need not be as dark as CSF; a clear "
    "relative drop of the centre vs a bright rim is the sign.\n"
    " - COMPLETE: near the whole interior is suppressed with a hyperintense rim.\n"
    " - PARTIAL: convincing but focal/incomplete interior suppression (still counts as present).\n"
    " - NONE: the interior is as bright as the rim on FLAIR (matched T2/FLAIR), OR the lesion is "
    "necrotic/cystic (a cyst/necrosis cavity is NOT the sign).\n\n"
    "PRIORS (tie-breakers, not to override what you see): the sign occurs in ~30-50% of "
    "IDH-mutant astrocytomas and is essentially ABSENT in IDH-wildtype glioma and in "
    "oligodendroglioma. If the interior clearly suppresses relative to the rim, call it present.\n\n"
    "Answer in EXACTLY this one line and nothing else:\n"
    "present: <yes|no> ; extent: <complete|partial|none> ; confidence: <high|moderate|low> ; reason: <one sentence>\n"
    "Rules: present=yes iff extent is complete or partial."
)
QUESTION = (
    "Score THIS tumour-cropped case (LEFT=T2, RIGHT=FLAIR). Compare the FLAIR tumour INTERIOR "
    "to its rim. Call partial/complete if the interior is relatively suppressed vs a hyperintense "
    "rim; call none if matched (interior as bright as rim) or necrotic/cystic. Exact one-line format."
)

EXEMPLARS = [
    ("non_01_IDHwt-LrGG_matchedPattern_Malik2024-fig4-d1d2.png", "none",
     "IDH-wildtype glioma: bright on T2 AND stays bright on FLAIR (matched) — NO mismatch."),
    ("non_02_oligo-1p19qCD_matchedPattern_Malik2024-fig4-c1c2.png", "none",
     "Oligodendroglioma: matched T2/FLAIR signal — NO mismatch (does not occur in oligo)."),
    ("non_03_astro-IDHmut-nonCanonical-ATRXretained_matchedPattern_Malik2024-fig3-b1b2.png", "none",
     "IDH-mut astrocytoma WITHOUT the sign: FLAIR interior not suppressed — NONE (sign is only ~30-50% sensitive)."),
    ("non_04_oligo-1p19qCD_minimalDiscordance_Malik2024-fig4-b1b2.png", "none",
     "Only minimal/patchy FLAIR discordance — not enough; NONE."),
    ("partial_01_oligo-IDHmut-1p19qCD_rTemporalPole_patchySuppr-noRim_BJR2021-fig8.png", "partial",
     "PARTIAL: only a portion of the interior shows central FLAIR suppression."),
    ("partial_02_astro-IDHmut-1p19qNC-G2_focalSupprOnly_BJR2021-fig10.png", "partial",
     "PARTIAL: focal-only interior FLAIR suppression in an IDH-mut astrocytoma."),
    ("fully_02_astro-IDHmut-1p19qNC-G2_inhomogSuppr-completeRim_BJR2021-fig1ab.png", "complete",
     "COMPLETE: near-entire interior suppressed on FLAIR with a hyperintense rim — IDH-mut astrocytoma."),
    ("fully_02_astro-IDHmut-1p19qNC-G3-anaplastic_uniformSuppr-incompleteRim_BJR2021-fig1cd.png", "complete",
     "COMPLETE: uniform interior FLAIR suppression across essentially the whole lesion."),
]

_PAT = {
    "present": re.compile(r"present\s*[:=]\s*(yes|no)", re.I),
    "extent": re.compile(r"extent\s*[:=]\s*(complete|partial|none)", re.I),
    "confidence": re.compile(r"confidence\s*[:=]\s*(high|moderate|low)", re.I),
    "reason": re.compile(r"reason\s*[:=]\s*(.+)$", re.I | re.M),
}


def parse(text: str) -> dict:
    fields = {field: (found.group(1).strip() if (found := pattern.search(text)) else None)
              for field, pattern in _PAT.items()}
    if fields.get("extent") == "none":
        fields["present"] = "no"
    elif fields.get("extent") in ("complete", "partial"):
        fields["present"] = "yes"
    return fields


def _orient(slice2d: np.ndarray) -> np.ndarray:
    return np.flipud(slice2d.T)


def _win_local(slice2d: np.ndarray, mask2d: np.ndarray, lo: int = 1, hi: int = 99) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    dilated_region = binary_dilation(mask2d, iterations=8)
    window_min, window_max = np.percentile(slice2d[dilated_region], [lo, hi])
    return np.clip((slice2d - window_min) / max(window_max - window_min, 1e-6), 0, 1)


def _to_panel(gray01: np.ndarray, size: int = PANEL) -> Image.Image:
    gray_u8 = (gray01 * 255).astype(np.uint8)
    h, w = gray_u8.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = gray_u8
    return Image.fromarray(canvas).resize((size, size), Image.LANCZOS)


def _z_thirds(mask3d: np.ndarray) -> list[int]:
    tumor_z_indices = np.where(mask3d.any(axis=(0, 1)))[0]
    if len(tumor_z_indices) == 0:
        return []
    max_area_z = []
    for third in np.array_split(np.arange(tumor_z_indices.min(), tumor_z_indices.max() + 1), 3):
        if len(third) == 0:
            continue
        areas = [mask3d[:, :, z].sum() for z in third]
        max_area_z.append(int(third[int(np.argmax(areas))]))
    seen, unique_z = set(), []
    for z in max_area_z:
        if z not in seen:
            seen.add(z)
            unique_z.append(z)
    return unique_z


def _panel_at(seg: np.ndarray, t2: np.ndarray, flair: np.ndarray, z: int,
              margin_frac: float = 0.35, min_margin: int = 16) -> Optional[Image.Image]:
    """Tumor-cropped T2|FLAIR query panel, padded to a square (no distortion)."""
    tumor_mask = ((seg == 1) | (seg == 2))[:, :, z]
    if tumor_mask.sum() < 10:
        return None
    ys, xs = np.where(tumor_mask)
    margin_y = max(min_margin, int((ys.max() - ys.min() + 1) * margin_frac))
    margin_x = max(min_margin, int((xs.max() - xs.min() + 1) * margin_frac))
    y0, y1 = max(0, ys.min() - margin_y), min(tumor_mask.shape[0], ys.max() + margin_y)
    x0, x1 = max(0, xs.min() - margin_x), min(tumor_mask.shape[1], xs.max() + margin_x)

    def crop_to_panel(vol):
        windowed_crop = _win_local(vol[:, :, z], tumor_mask)[y0:y1, x0:x1]
        return _to_panel(_orient(windowed_crop)).convert("RGB")

    composite = Image.new("RGB", (PANEL * 2 + 8, PANEL + 30), (0, 0, 0))
    composite.paste(crop_to_panel(t2), (0, 30))
    composite.paste(crop_to_panel(flair), (PANEL + 8, 30))
    draw = ImageDraw.Draw(composite)
    draw.text((6, 8), "T2", fill=(255, 255, 0))
    draw.text((PANEL + 14, 8), "FLAIR", fill=(255, 255, 0))
    return composite


_MODEL = None
_PROC = None


def _load_backbone():
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        _PROC = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
    return _MODEL, _PROC


def _load_exemplars():
    return [(Image.open(os.path.join(REFS_DIR, filename)).convert("RGB"), (filename, label, note))
            for filename, label, note in EXEMPLARS]


def _build_messages(exemplar_imgs, query_img):
    content = [{"type": "text", "text": "Reference examples (each lesion-cropped: LEFT=T2, RIGHT=FLAIR):"}]
    for exemplar_img, (_, label, note) in exemplar_imgs:
        content.append({"type": "image", "image": exemplar_img})
        content.append({"type": "text", "text": f"-> extent: {label}. {note}"})
    content.append({"type": "text", "text": "=== QUERY CASE ==="})
    content.append({"type": "image", "image": query_img})
    content.append({"type": "text", "text": QUESTION})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def _score_panel(panel, exemplar_imgs, model, processor) -> dict:
    import torch
    messages = _build_messages(exemplar_imgs, panel)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", enable_thinking=False)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=90, do_sample=False)
    answer = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0].strip()
    return parse(answer)


def predict_t2fm(seg_path: str, t2_path: str, flair_path: str,
                 model: Any = None, proc: Any = None,
                 save_panel_dir: Optional[str] = None) -> dict[str, Any]:
    """Detect the T2-FLAIR mismatch sign (3-slice z-thirds, any positive).

    Args:
        seg_path:   HD-GLIO segmentation NIfTI (1=non-enhancing, 2=enhancing).
        t2_path:    T2 NIfTI (skull-stripped, co-registered to seg).
        flair_path: FLAIR NIfTI (skull-stripped, co-registered to seg).
        model/proc: optional preloaded Qwen backbone (else lazy-loaded once).

    Returns dict:
        t2_flair_mismatch : bool   -- final present (any z-third present)
        extent            : str    -- extent of the firing slice, else 'none'
        confidence        : str    -- confidence of the firing slice, else that of max-area slice
        z_thirds          : list[int]
        slices            : list[{z, present, extent, confidence, reason}]
        method            : str
    """
    if model is None or proc is None:
        model, proc = _load_backbone()
    exemplar_imgs = _load_exemplars()

    seg = nib.load(seg_path).get_fdata().astype(np.int16)
    t2 = nib.load(t2_path).get_fdata().astype(np.float32)
    flair = nib.load(flair_path).get_fdata().astype(np.float32)
    if not (seg.shape == t2.shape == flair.shape):
        raise ValueError(f"shape mismatch: seg{seg.shape} t2{t2.shape} flair{flair.shape}")

    tumor_mask = (seg == 1) | (seg == 2)
    z_thirds = _z_thirds(tumor_mask)
    if save_panel_dir:
        os.makedirs(save_panel_dir, exist_ok=True)
    slices = []
    png_paths = []
    for i, z in enumerate(z_thirds, 1):
        panel = _panel_at(seg, t2, flair, z)
        if panel is None:
            slices.append({"z": z, "present": "no", "extent": "none",
                           "confidence": None, "reason": "no tumor on slice", "png": None})
            continue
        if save_panel_dir:
            png_path = os.path.join(save_panel_dir, f"t2fm_slice{i}.png")
            panel.save(png_path); png_paths.append(png_path)
        else:
            png_path = None
        prediction = _score_panel(panel, exemplar_imgs, model, proc)
        slices.append({"z": z, "present": prediction.get("present"), "extent": prediction.get("extent"),
                       "confidence": prediction.get("confidence"), "reason": prediction.get("reason"),
                       "png": png_path})

    fired = next((slice_result for slice_result in slices if slice_result["present"] == "yes"), None)
    mismatch_present = fired is not None
    reported_slice = fired or (slices[0] if slices else {})
    return {
        "t2_flair_mismatch": mismatch_present,
        "extent": (fired["extent"] if fired else "none"),
        "confidence": reported_slice.get("confidence"),
        "z_thirds": z_thirds,
        "slices": slices,
        "panel_pngs": png_paths,
        "method": "_to_panel query + 3-slice z-thirds + lesion-cropped refs (any-yes)",
    }


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="T2-FLAIR mismatch")
    ap.add_argument("seg"); ap.add_argument("t2"); ap.add_argument("flair")
    args = ap.parse_args()
    print(json.dumps(predict_t2fm(args.seg, args.t2, args.flair), indent=2))
