"""In-process LLM orchestrator client with XML/JSON tool-call parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract `[{"name": str, "arguments": dict}, ...]` from an assistant turn.

    Handles two formats Qwen emits:
      * Hermes-JSON   : `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
      * Qwen3.5-XML   : `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
    """
    calls: list[dict] = []
    for tool_call in TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(tool_call.strip())
            if isinstance(payload, dict) and "name" in payload:
                calls.append({
                    "name": str(payload["name"]).strip(),
                    "arguments": payload.get("arguments", {}) or {},
                })
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        for function_name, function_body in FUNCTION_RE.findall(tool_call):
            arguments: dict[str, Any] = {}
            for parameter_name, parameter_value in PARAMETER_RE.findall(function_body):
                parameter_value = parameter_value.strip()
                try:
                    arguments[parameter_name] = json.loads(parameter_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[parameter_name] = parameter_value
            calls.append({"name": function_name.strip(), "arguments": arguments})
    return calls


def strip_tool_xml(text: str) -> str:
    """Drop tool-call markup from an assistant turn, leaving the prose."""
    text = re.sub(r"<tool_call\b.*?</tool_call>", "", text, flags=re.I | re.S)
    text = re.sub(r"<tool_call\b.*", "", text, flags=re.I | re.S)
    text = re.sub(r"<function\b.*", "", text, flags=re.I | re.S)
    return text.strip()


ASCII_MAP = {
    "cm\u00b3": "cm3", "mm\u00b3": "mm3", "cm\u00b2": "cm2", "mm\u00b2": "mm2",
    "\u00b3": "3", "\u00b2": "2", "\u00d7": "x", "\u2265": ">=", "\u2264": "<=",
    "\u2260": "!=", "\u2248": "~", "\u2212": "-", "\u00b5": "u", "\u00b0": " deg",
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->", "\u2014": "-", "\u2013": "-",
    "\u2026": "...", "\u00b7": "-", "\u2022": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00a0": " ",
}


def ascii_fold(text: str, strip_unmapped: bool = True) -> str:
    """ASCII-fold, mapping common medical/typographic unicode so meaning is kept
    (>=, <=, cm3, ->, u) rather than silently dropped. `strip_unmapped` drops any
    remaining non-ASCII (report text); the co-pilot leaves it in place."""
    text = text or ""
    for unicode_character, replacement in ASCII_MAP.items():
        text = text.replace(unicode_character, replacement)
    return text.encode("ascii", "ignore").decode("ascii") if strip_unmapped else text


def tool_schema(name, description, properties=None, required=None) -> dict:
    """OpenAI-style function schema for a tool exposed to the model."""
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": required or []}}}


def _default_backbone_hf() -> str:
    """Lazily fetch the backbone HF id from agents.config (avoids circular import)."""
    try:
        from agents.config import BACKBONE_HF_ID
        return BACKBONE_HF_ID
    except Exception:
        return "Qwen/Qwen3.5-27B"


@dataclass
class LLMClient:
    """Drives LLM with tool calls + a system prompt.

    Attributes
    ----------
    model_name : str
        HF id, defaults to the fixed backbone `Qwen/Qwen3.5-27B`.
        Override via env `GLIOAGENT_BACKBONE_HF` or by passing
        `LLMClient(model_name=...)` explicitly.
    device : str
        Where to put the model.
    """

    model_name: str = None  # type: ignore[assignment]  # filled in __post_init__
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512

    _model: Any = None
    _processor: Any = None

    last_logprob_confidence: Optional[float] = None

    def __post_init__(self):
        if self.model_name is None:
            object.__setattr__(self, "model_name", _default_backbone_hf())

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            device_map=self.device,
        )
        self._model.eval()

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             max_new_tokens: Optional[int] = None,
             enable_thinking: bool = False) -> str:
        """Return the assistant turn's raw text (tool calls included).

        `enable_thinking=False` strips Qwen3.5's default chain-of-thought
        preamble so the model emits the structured answer directly -- the
        few-shot prompt + multiple-choice tag we ask for shouldn't need
        a thinking block, and the budget was getting eaten by it.
        """
        self.load()
        # Qwen3-VL processor requires content as a list of typed parts.
        normalised: list[dict] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                message = {**message, "content": [{"type": "text", "text": content}]}
            normalised.append(message)
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": enable_thinking,
        }
        if tools:
            kwargs["tools"] = tools
        inputs = self._processor.apply_chat_template(normalised, **kwargs)
        on_device = {key: (value.to(self._model.device) if isinstance(value, torch.Tensor) else value)
                     for key, value in inputs.items()}
        with torch.inference_mode():
            generation = self._model.generate(
                **on_device,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequences = generation.sequences if hasattr(generation, "sequences") else generation
        generated_tokens = sequences[:, on_device["input_ids"].shape[1]:]
        text = self._processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # Mean log-prob of generated tokens: a reproducible confidence signal under greedy decode.
        self.last_logprob_confidence = None
        scores = getattr(generation, "scores", None)
        if scores is not None and len(scores) > 0:
            try:
                log_probs = []
                tokens = generated_tokens[0]
                pad_token_id = getattr(self._processor.tokenizer, "pad_token_id", None)
                for step, logits in enumerate(scores):
                    if step >= tokens.shape[0]:
                        break
                    token = int(tokens[step].item())
                    if pad_token_id is not None and token == pad_token_id:
                        continue
                    log_probs.append(torch.log_softmax(logits[0].float(), dim=-1)[token].item())
                if log_probs:
                    import math
                    self.last_logprob_confidence = float(
                        math.exp(sum(log_probs) / len(log_probs)))
            except Exception:
                self.last_logprob_confidence = None

        return text.strip()


_CLIENT: Optional[LLMClient] = None


def get_client(model: Optional[str] = None, device: str = "cuda") -> LLMClient:
    """Load (once) and return the process-wide in-process LLM client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient(model_name=model, device=device) if model else LLMClient(device=device)
        _CLIENT.load()
    return _CLIENT
