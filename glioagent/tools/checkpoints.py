"""Verify/download model checkpoints under tools/ckpts/{model}/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CKPTS = TOOLS / "ckpts"

# GlioMT (external, public) Google-Drive ids, per task head.
GLIOMT_FILES = {
    "idh_vitb.pth":    "1VtRU3W_Tl_ghrsa2Fjpmr8Y-yJqaV_sn",
    "1p19q_swins.pth": "1AkXwnEBg_M0f7eaF11TuSyqu6ktDCuPb",
    "grade_swins.pth": "1lnasGRbYbA5c3cPmZa0kBwJkixCrV9Sj",
}


def _ok(message): return          # per-item detail is summarised by maybe_checkpoints
def _info(message): print(f"  [..]   {message}", flush=True)
def _warn(message): print(f"  [!!]   {message}", flush=True)


def maybe_download_hdglio() -> bool:
    (CKPTS / "hdglio").mkdir(parents=True, exist_ok=True)
    try:
        from hd_glio.setup_hd_glio import maybe_download_weights
    except Exception as e:
        _warn(f"hdglio: hd_glio package not importable ({e}).")
        return False
    try:
        maybe_download_weights()
        _ok("hdglio: weights present.")
        return True
    except Exception as e:
        _warn(f"hdglio: download failed ({e}).")
        return False


def _backbone_id() -> str:
    """Backbone repo id -- single source of truth in agents/config.py."""
    from agents.config import BACKBONE_HF_ID
    return BACKBONE_HF_ID


def maybe_download_qwen(force: bool = False) -> bool:
    """The backbone loads from the HF cache (HF_HOME), never from ckpts/: report on
    that cache and, with `force`, pre-fetch into it (no second copy in the repo)."""
    from huggingface_hub import snapshot_download
    model_id = _backbone_id()
    hf_cache_dir = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    try:
        snapshot_download(model_id, local_files_only=True)
        _ok(f"qwen: {model_id} already in the HF cache ({hf_cache_dir}).")
        return True
    except Exception:
        pass
    if not force:
        _warn(f"qwen: {model_id} not in the HF cache ({hf_cache_dir}); transformers downloads it "
              "(~54 GB) on first use. Pass --with-qwen to fetch it now.")
        return False
    try:
        snapshot_download(model_id)
        _ok(f"qwen: {model_id} fetched into the HF cache ({hf_cache_dir}).")
        return True
    except Exception as e:
        _warn(f"qwen: pre-fetch failed ({e})."); return False


def _ensure_gdown() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except Exception:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
        try:
            import gdown  # noqa: F401
            return True
        except Exception:
            return False


def maybe_download_gliomt() -> bool:
    gliomt_dir = CKPTS / "gliomt"
    gliomt_dir.mkdir(parents=True, exist_ok=True)
    if ([filename for filename in GLIOMT_FILES if not (gliomt_dir / filename).is_file()]
            and not _ensure_gdown()):
        _warn("gliomt: gdown unavailable."); return False
    import gdown, torch
    all_ok = True
    for filename, drive_file_id in GLIOMT_FILES.items():
        ckpt_path = gliomt_dir / filename
        if not ckpt_path.is_file():
            _info(f"gliomt: downloading {filename}")
            try:
                gdown.download(id=drive_file_id, output=str(ckpt_path), quiet=False)
            except Exception as e:
                _warn(f"gliomt: {filename} failed ({e})"); all_ok = False; continue
        try:
            torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            _ok(f"gliomt: {filename} loads.")
        except Exception as e:
            _warn(f"gliomt: {filename} bad ({e})"); all_ok = False
    return all_ok and all((gliomt_dir / filename).is_file() for filename in GLIOMT_FILES)


def maybe_download_vasari() -> bool:
    """VASARI-auto is rule/atlas based (no weights); verify its bundled MNI atlas."""
    try:
        import vasari_auto
        from vasari_auto.vasari_auto import get_vasari_features  # noqa: F401
    except Exception as e:
        _warn(f"vasari: not importable ({e})."); return False
    atlas_path = Path(vasari_auto.__file__).parent / "atlas_masks" / "mni152" / "MNI152_T1_1mm_brain.nii.gz"
    if not atlas_path.is_file():
        _warn(f"vasari: bundled MNI atlas missing ({atlas_path})."); return False
    _ok("vasari: importable + MNI atlas present (rule/atlas based)."); return True


def maybe_checkpoints(with_qwen: bool = False) -> dict:
    # every entry is required: qwen is the orchestrator, the rest are the tools
    CKPTS.mkdir(parents=True, exist_ok=True)
    status = {"hdglio": maybe_download_hdglio(), "qwen": maybe_download_qwen(with_qwen),
              "gliomt": maybe_download_gliomt(), "vasari": maybe_download_vasari()}
    print(f"  {'':<14} " + "  ".join(f"{name}={'OK' if ready else 'MISSING'}"
                                     for name, ready in status.items()), flush=True)
    return status
