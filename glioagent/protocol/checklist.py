"""Deterministic rule-based verification checklist for generated reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistResult:
    items: dict
    passed: bool
    missing: list

    @property
    def score(self) -> str:
        num_passed = sum(1 for item_passed in self.items.values() if item_passed)
        return f"{num_passed}/{len(self.items)}"


def _has(haystack: str, needles: list[str]) -> bool:
    haystack_lower = haystack.lower()
    return any(needle.lower() in haystack_lower for needle in needles)


def verify_report(
    report: dict,
    seg: dict,
    molecular: dict,
    vasari: dict | None = None,
    diameters: dict | None = None,
) -> ChecklistResult:
    report_text = "\n".join(str(section) for section in report.values())
    items: dict[str, bool] = {}

    items["1_location"] = (
        _has(report_text, [seg.get("laterality", ""), "lobe"])
        and bool(seg.get("tumor_location"))
    )

    items["2_size_volume"] = _has(report_text, ["cm3"]) and _has(report_text, ["bidimensional"])

    # For IDH-wildtype, 1p/19q codeletion is only meaningful in IDH-mut context (WHO 2021), so relax the rule.
    idh_pred = (molecular or {}).get("idh", {}).get("prediction", "") if molecular else ""
    idh_mentioned = _has(report_text, ["IDH"])
    codel_mentioned = _has(report_text, ["1p", "codel"])
    if idh_pred == "wildtype":
        items["3_idh_codel"] = idh_mentioned
    else:
        items["3_idh_codel"] = idh_mentioned and codel_mentioned

    items["4_who2021"] = _has(report_text, [
        "Glioblastoma, IDH-wildtype",
        "Oligodendroglioma, IDH-mutant",
        "Astrocytoma, IDH-mutant",
        "WHO Grade",
        "WHO 2021",
    ])

    # An IDH-wildtype call together with a positive T2-FLAIR mismatch sign is a
    # contradiction (the sign is near-absent in IDH-wildtype glioma). The sign comes from
    # the vision VQA tool: `vasari["t2_flair_mismatch"]` is hard-coded False and listed in
    # the tool's own `_unreliable_fields`, so reading it made this item pass for every case.
    mismatch_present = ((report.get("structured_report") or {})
                        .get("findings", {}).get("mismatch") or {}).get("present")
    if molecular is not None:
        items["5_vasari_consistency"] = not (
            molecular["idh"]["prediction"] == "wildtype" and mismatch_present is True
        )
    else:
        items["5_vasari_consistency"] = True

    # VASARI 'side' uses a radiological (L-R flipped) convention, so compare seg vs atlas laterality (both anatomical), not VASARI.
    atlas_location = (report.get("structured_report") or {}).get("findings", {}).get("location", {})
    atlas_laterality = (atlas_location.get("laterality") or "").lower()
    seg_laterality = (seg.get("laterality") or "").lower()
    neutral_lateralities = ("midline", "bilateral", "indeterminate")
    if (not atlas_laterality or not seg_laterality
            or seg_laterality in neutral_lateralities or atlas_laterality in neutral_lateralities):
        items["7_no_contradiction"] = True
    else:
        items["7_no_contradiction"] = (atlas_laterality.startswith(seg_laterality[:3])
                                       or seg_laterality.startswith(atlas_laterality[:3]))

    items["8_warnings_visible"] = ("warning" in report_text.lower()
                                   or "no warnings" in report_text.lower()
                                   or "warnings_and_recommendations" in report)

    missing = [item_key for item_key, item_passed in items.items() if not item_passed]
    return ChecklistResult(items=items, passed=len(missing) == 0, missing=missing)
