"""Knowledge-base loader: parses `knowledge/*.md` into chunks (no embedding/query;
selection is deterministic, done by `agents/knowledge_select.select_chunks`)."""

from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge"


def _split_chunks(text: str, source: str) -> List[dict]:
    """Split a Markdown file on `### ` headings into {source, title, chunk}."""
    chunks: List[dict] = []
    current_title = "(preamble)"
    current_body_lines: list[str] = []

    def flush_chunk():
        chunk_body = "\n".join(current_body_lines).strip()
        if chunk_body:
            chunks.append({"source": source, "title": current_title, "chunk": chunk_body})

    for line in text.splitlines():
        if line.startswith("### "):
            flush_chunk()
            current_title = line[4:].strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)
    flush_chunk()
    return chunks


def kb() -> List[dict]:
    """All knowledge chunks from `knowledge/*.md`."""
    chunks: List[dict] = []
    for md_path in sorted(KB_DIR.glob("*.md")):
        chunks.extend(_split_chunks(md_path.read_text(), md_path.name))
    if not chunks:
        raise RuntimeError(f"No KB chunks found under {KB_DIR}")
    return chunks


if __name__ == "__main__":
    for chunk in kb():
        print(f"[{chunk['source']}] {chunk['title']}")
