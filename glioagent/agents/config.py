"""Unified LLM backbone: every LLM component is pinned to one model.

BACKBONE_HF_ID is the HF repo id used by the in-process client
(agents.llm_client); BACKBONE_SERVED_NAME is the vLLM --served-model-name
alias used by the archived vLLM-backed co-pilot under previous/.
"""
from __future__ import annotations

import os

BACKBONE_HF_ID:       str = os.environ.get("GLIOAGENT_BACKBONE_HF",   "Qwen/Qwen3.5-27B")
BACKBONE_SERVED_NAME: str = os.environ.get("GLIOAGENT_BACKBONE_NAME", "qwen3.5-27b")
