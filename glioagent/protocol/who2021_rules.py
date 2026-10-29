from __future__ import annotations

from typing import Literal

IDHClass = Literal["wildtype", "mutant"]
CodelClass = Literal["codeleted", "non-codeleted"]


def derive_who2021(idh: IDHClass, codel: CodelClass) -> tuple[str, int]:
    """Map IDH/1p19q to WHO 2021 diagnosis + grade (simplified per Louis et al.,
    Neuro-Oncol 2021; ATRX/H3, pediatric subtypes, grade 3 astrocytoma not differentiated)."""
    if idh == "wildtype":
        return "Glioblastoma, IDH-wildtype, CNS WHO Grade 4", 4
    if codel == "codeleted":
        return "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted", 2
    return "Astrocytoma, IDH-mutant", 2
