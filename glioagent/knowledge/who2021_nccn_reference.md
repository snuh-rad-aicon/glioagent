# WHO 2021 / NCCN reference corpus

Authoritative reference text for the knowledge-grounded review (classification
and differential diagnosis), paraphrased from the sources cited below. Each
`### ...` heading is one chunk, indexed for RAG by `tools/tool_rag_verifier.py`.
Used for review/verification, not clinical care.

## Sources

* **WHO 2021** -- Louis DN et al., *The 2021 WHO Classification of
  Tumors of the Central Nervous System: a summary*. Neuro-Oncology
  23(8):1231-1251, 2021.
* **Tumour measurement** -- Wen PY et al., update to the response
  assessment criteria for high- and low-grade gliomas in adults.
  J Clin Oncol 41(33):5187-5199, 2023.
* **NCCN Glioma v2.2024** -- National Comprehensive Cancer Network,
  Clinical Practice Guidelines in Oncology: CNS Cancers.

---

### WHO 2021 -- adult-type diffuse gliomas, top-level classes

Adult-type diffuse gliomas are partitioned into three families that are
**defined by IDH1/IDH2 mutation status and 1p/19q codeletion**, not by
histology alone:

1. *Astrocytoma, IDH-mutant* -- CNS WHO grade 2, 3, or 4.
2. *Oligodendroglioma, IDH-mutant and 1p/19q-codeleted* -- CNS WHO
   grade 2 or 3.
3. *Glioblastoma, IDH-wildtype* -- CNS WHO grade 4 (regardless of
   classical GBM histology, presence of microvascular proliferation
   or necrosis is sufficient when combined with TERT promoter
   mutation, EGFR amplification, or +7/-10).

### WHO 2021 -- molecular-to-diagnosis decision rule

* IDH-wildtype -> **Glioblastoma, IDH-wildtype, CNS WHO Grade 4**.
* IDH-mutant + 1p/19q codeletion -> **Oligodendroglioma, IDH-mutant
  and 1p/19q-codeleted**.
* IDH-mutant + 1p/19q intact -> **Astrocytoma, IDH-mutant** (grade
  refined by CDKN2A/B status: homozygous deletion of CDKN2A/B -> grade 4).

### WHO 2021 -- grade-defining molecular features

* CDKN2A/B homozygous deletion in IDH-mutant astrocytoma -> CNS WHO grade 4.
* TERT promoter mutation **or** EGFR amplification **or** +7/-10 chromosomal
  pattern in a histologically lower-grade IDH-wildtype diffuse astrocytic
  tumor -> diagnose as Glioblastoma, CNS WHO grade 4.

### Measurable disease -- enhancing component (Wen 2023)

A target lesion is **measurable** when both perpendicular diameters of
the enhancing component on a single axial slice are **>= 10 mm** and the
slice thickness is <= 5 mm (or twice that for 2.5 mm). The sum of
products of perpendicular diameters is the standard quantitative metric.

### Measurable disease -- non-enhancing component (Wen 2023)

Non-enhancing T2/FLAIR signal abnormality is measurable when the
hyperintense region exceeds **>= 10 mm in the longest axis** on a slice
with thickness <= 5 mm. Non-enhancing disease is used as the primary
response-assessment metric for IDH-mutant, non-enhancing tumors (the
unified high-/low-grade framework of Wen 2023).

### Response categories after therapy (Wen 2023)

* Complete Response (CR): no enhancing or non-enhancing measurable
  disease, off corticosteroids, no new lesions, sustained >= 4 weeks.
* Partial Response (PR): >= 50 % decrease in the sum of products of
  perpendicular diameters of enhancing disease, no new lesions, no
  increase in non-enhancing component.
* Stable Disease (SD): neither PR nor PD.
* Progressive Disease (PD): >= 25 % increase in sum of products of
  perpendicular diameters of enhancing disease, **or** >= 40 % increase
  in non-enhancing T2/FLAIR (IDH-mutant tumors), **or** any new lesion
  or clinical deterioration not attributable to non-tumor causes.

### NCCN -- initial workup of newly diagnosed adult glioma

Recommended baseline workup: brain MRI with gadolinium (T1, T1+contrast,
T2, FLAIR; DWI and perfusion when available), neurosurgical evaluation
for maximal safe resection, molecular profiling (at minimum IDH1/2 and
1p/19q; MGMT promoter methylation for high-grade tumors), and KPS
assessment. CT alone is not adequate for diagnosis or treatment
planning.

### NCCN -- first-line treatment, IDH-wildtype glioblastoma (KPS >= 60)

Maximal safe resection followed by the Stupp regimen: concurrent
radiotherapy 60 Gy in 30 fractions with daily temozolomide
75 mg/m2/day, then adjuvant temozolomide 150-200 mg/m2 days 1-5 every
28 days x 6 cycles. TTFields may be added for grade-4 tumors. MGMT-
methylated tumors derive larger benefit from temozolomide.

### NCCN -- first-line treatment, IDH-mutant astrocytoma

* Grade 2 + favorable risk: maximal safe resection followed by close
  imaging surveillance OR adjuvant RT + PCV (procarbazine, CCNU,
  vincristine).
* Grade 2 + unfavorable risk or grade 3: maximal safe resection
  followed by RT + adjuvant chemotherapy (PCV or temozolomide).
* Grade 4 IDH-mutant astrocytoma: managed similarly to glioblastoma
  but with vorasidenib considered in select grade 2/3 unresected disease.

### NCCN -- first-line treatment, oligodendroglioma (1p/19q-codeleted)

Maximal safe resection followed by RT + adjuvant PCV (preferred) or
temozolomide. Active surveillance only if completely resected,
asymptomatic, grade 2, and small.

### Clinical rules -- confidence tiers used by GlioAgent

* HIGH -- all tool confidences > 0.8 and no cross-tool contradictions.
* MODERATE -- 1-2 low-confidence items or one minor discordance.
* LOW -- any major contradiction, non-measurable size + missing T1CE,
  or downstream QC failure -> human review required.

### Anomaly heuristics

* age > 55 + IDH-mutant prediction -> unusual; flag for re-evaluation.
* IDH-wildtype + T2/FLAIR mismatch sign -> discordant; flag.
* Enhancing-to-whole-tumor ratio > 0.8 -> unusual ET dominance; flag.
* WT volume < 0.5 cm3 -> segmentation suspicious; call slice_vqa QC.
* Disconnected enhancing tumor regions > 1 -> multifocal disease.
