"""ReAct-style inference trace for the GlioAgent pipeline."""

from __future__ import annotations

import os
from pathlib import Path

_LABEL_WIDTH = 15  # pad tool label so `done` columns line up


class ReActTrace:
    def __init__(self, verbose: bool | None = None) -> None:
        """`verbose` streams the trace to stdout; None defers to GLIOAGENT_DEBUG=1."""
        self.lines: list[str] = []
        self.debug = (os.environ.get("GLIOAGENT_DEBUG", "") == "1"
                      if verbose is None else bool(verbose))

    def _emit(self, line: str) -> None:
        self.lines.append(line)
        if self.debug:
            print(line, flush=True)

    def header(self, cohort, sample_id, age, sex, sequences) -> None:
        self._emit("=== GlioAgent Inference Trace ===")
        self._emit(f"Dataset: {cohort or '?'}  |  Patient: {sample_id}")
        self._emit(f"Demographics: {age} y, {sex}  |  Sequences: "
                   f"{', '.join(sequences)}")
        self._emit("")

    def thought(self, message: str) -> None:
        self._emit(f"[Thought] {message}")

    def action(self, tool: str) -> None:
        self._emit(f"[Action] call {tool}")

    def start(self, tool: str, detail: str = "") -> None:
        self._emit(f"  [tool called: {tool:<{_LABEL_WIDTH}}] start  {detail}".rstrip())

    def done(self, tool: str, summary: str) -> None:
        self._emit(f"  [tool called: {tool:<{_LABEL_WIDTH}}] done  {summary}")

    def failed(self, tool: str, reason: str) -> None:
        self._emit(f"  [tool called: {tool:<{_LABEL_WIDTH}}] FAILED: {reason}")

    def note(self, message: str) -> None:
        self._emit(message)

    def console(self, message: str) -> None:
        """Terminal-only separator; not saved, so the trace .md format is unchanged."""
        if self.debug:
            print(f"\n{message}", flush=True)

    def render(self) -> str:
        return "\n".join(self.lines)


def save_trace_md(
    out_path: str,
    trace: ReActTrace,
    structured_report_text: str = "",
    quantitative_summary: str = "",
    sample_id: str = "",
) -> str:
    """Write the full trace + structured report + quant summary as markdown."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"# GlioAgent Inference Trace -- {sample_id}", ""]
    sections += ["## ReAct trace", "", "```", trace.render(), "```", ""]
    if structured_report_text:
        sections += ["## Structured report", "", "```", structured_report_text, "```", ""]
    if quantitative_summary:
        sections += ["## Quantitative summary", "", "```", quantitative_summary, "```", ""]
    path.write_text("\n".join(sections), encoding="utf-8")
    return str(path)
