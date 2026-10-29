"""Deterministic tools. Each module is a pure function of the case inputs;
consumers import the submodule they need, e.g.:

    from tools import tool_seg_features as seg_features, tool_atlas_features as atlas_features
    from tools.tool_tumor_segment import predict_hdglio, summarise_seg
    from tools.tool_molecular_subtype import molecular_subtype

Vendored official model code lives in-repo under tools/ and is registered on
sys.path here so the wrappers import the IN-REPO copies (not a pip-installed one):
    tools/hdglio -> `hd_glio`     (HD-GLIO segmentation, CCI-Bonn)
    tools/vasari -> `vasari_auto`  (VASARI-auto, nikitas-k; MNI atlas bundled)
GlioMT (tools/gliomt/src -> `src`) is registered by tools/gliomt/gliomt_inference.py.
"""
import sys as _sys
from pathlib import Path as _Path

_TOOLS = _Path(__file__).resolve().parent
for _name in ("hdglio", "vasari"):
    _p = str(_TOOLS / _name)
    if (_TOOLS / _name).is_dir() and _p not in _sys.path:
        _sys.path.insert(0, _p)
