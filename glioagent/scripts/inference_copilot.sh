#!/usr/bin/env bash
# GlioAgent co-pilot: the autonomous run on one case, then a chat to correct findings and redraft.
# Flags: README.md or --help.  Quick start: bash scripts/inference_copilot.sh --sample UCSF_00000236_00000001
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# model cache: $HF_HOME, else repo-local hf_cache
if [ -z "${HF_HOME:-}" ] && [ -d "$REPO_DIR/hf_cache" ]; then export HF_HOME="$REPO_DIR/hf_cache"; fi

# interpreter: $PYTHON, else the glioagent conda env, else python
GLIOAGENT_PY="$(conda env list 2>/dev/null | awk '$1=="glioagent"{print $NF"/bin/python"; exit}')" || true
exec "${PYTHON:-${GLIOAGENT_PY:-python}}" "$REPO_DIR/agents/main.py" "$@"
