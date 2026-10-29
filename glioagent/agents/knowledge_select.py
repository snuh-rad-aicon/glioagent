"""Rule-based knowledge chunk selection (replaces vector-search RAG)."""

from __future__ import annotations

from typing import List

from tools.tool_rag_verifier import kb

# Chunk titles must match `### ` headings in knowledge/*.md exactly.
T_CONCORDANCE = "Imaging feature concordance with molecular subtype (Louis 2021; Patel 2017; Do 2022; Maynard 2020)"
T_AGE = "IDH status prediction from age and morphology (Maynard 2020)"
T_WHO_RULE = "WHO 2021 -- molecular-to-diagnosis decision rule"
T_MISMATCH_PPV = "T2-FLAIR mismatch sign -- clinical prevalence and predictive value"
T_MISMATCH_CONF = "T2-FLAIR mismatch sign -- confounders (Do 2022)"
T_MISMATCH_VQA = "T2-FLAIR mismatch sign -- vision-model extent and integrated interpretation"
T_NECROSIS = "Necrosis -- clinical context (WHO 2021; cross-reference)"
T_ENHANCEMENT = "Enhancement phenotype by molecular subtype (Foltyn-Dumitru 2025; Setyawan 2024)"
T_ASTRO_OLIGO = "Distinguishing astrocytoma from oligodendroglioma (IDH-mutant)"

_KB_CACHE: List[dict] | None = None


def _index() -> dict:
    """title -> chunk dict, built once per process."""
    global _KB_CACHE
    if _KB_CACHE is None:
        _KB_CACHE = kb()
    return {chunk["title"]: chunk for chunk in _KB_CACHE}


def select_chunks(molecular: dict, structured_findings: dict, warnings: list | None = None) -> List[dict]:
    """Return the knowledge chunks grounding this case, deterministically."""
    chunks_by_title = _index()
    findings = (structured_findings or {}).get("findings", {})
    mismatch = findings.get("mismatch") or {}
    necrosis = findings.get("necrosis") or {}
    enhancement = findings.get("enhancement") or {}
    idh_prediction = (molecular.get("idh") or {}).get("prediction")
    codeletion_prediction = (molecular.get("1p19q") or {}).get("prediction")

    wanted: list[str] = [T_WHO_RULE, T_CONCORDANCE, T_AGE]
    if mismatch.get("present") or mismatch.get("retracted"):
        wanted += [T_MISMATCH_PPV, T_MISMATCH_CONF, T_MISMATCH_VQA]
    if necrosis.get("present"):
        wanted.append(T_NECROSIS)
    if idh_prediction == "mutant" and (mismatch.get("present") or codeletion_prediction == "codeleted"):
        wanted.append(T_ASTRO_OLIGO)
    # Enhancement grounding only when absent (non-enhancing-GBM exception).
    if enhancement.get("quality") == "Absent" or enhancement.get("present") is False:
        wanted.append(T_ENHANCEMENT)

    # A title that no longer matches a `### ` heading would otherwise drop its grounding
    # without a trace: the report would be written with one fewer reference and nothing
    # would say so. Editing a heading in knowledge/*.md is a packaging error, so it fails.
    missing = [title for title in wanted if title not in chunks_by_title]
    if missing:
        raise KeyError("knowledge chunk(s) not found in knowledge/*.md -- a `### ` heading "
                       f"was renamed or removed: {missing}")

    selected_chunks, seen = [], set()
    for title in wanted:
        if title not in seen:
            seen.add(title)
            selected_chunks.append(chunks_by_title[title])
    return selected_chunks
