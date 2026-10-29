# EANO 2021 — Therapy: specific recommendations (co-pilot knowledge, INDEX)

> Source of record: Weller M et al. EANO guidelines on the diagnosis and treatment of diffuse gliomas of adulthood. Nat Rev Clin Oncol 2021;18:170-186, section 'Therapy - specific recommendations'; Stupp R et al. N Engl J Med 2005;352:987-996.

> Version pinned: **EANO 2021 (+ Stupp 2005 for protocol details)**. This file is the human-readable INDEX of `eano_2021_therapy.json`, which is what `recommend_treatment()` reads. Evidence quotes in the JSON were extracted from the source PDF and gated programmatically (verbatim=true only when the quote is a substring of the PDF text); all 47 evidence quotes pass, and the value fields were corrected against the guideline body (see each row's `corrections_applied`). Do NOT mix later (post-2021) guideline updates (e.g. vorasidenib).

> Predicted scope: glioblastoma:4, astrocytoma:2, astrocytoma:3, astrocytoma:4, oligodendroglioma:2, oligodendroglioma:3. H3-altered → out of scope.


## Glioblastoma, IDH-wildtype, CNS WHO grade 4  (`glioblastoma:4`)
- **Surgery first:** Maximal safe resection when performance status allows; stereotactic biopsy if the tumour is deep or in eloquent (language/motor) cortex.
- **First-line adjuvant (if confirmed):** Stupp protocol: radiotherapy 60 Gy in 30 fractions of 2 Gy over 6 weeks with concomitant temozolomide 75 mg/m2 daily during radiotherapy, then adjuvant temozolomide 150-200 mg/m2 on days 1-5 of 6 x 28-day cycles.
- **Observation:** Not standard.
- **Age branch:** Elderly or frail patients (older than about 65-70 years or poor KPS) receive reduced-intensity treatment chosen by KPS and MGMT status: hypofractionated radiotherapy (40 Gy in 15 fractions, ~2.67 Gy) with or without temozolomide, or temozolomide alone if MGMT-methylated.
- **At progression:** Re-resection, re-irradiation, nitrosourea (lomustine), temozolomide rechallenge, bevacizumab for symptom control, or clinical trials.
- **Pending determinants:** histologic confirmation, MGMT promoter status, KPS, extent of resection
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Glioblastoma, IDH wild type, WHO grade 4_: "Temozolomide chemoradiotherapy (54-60 Gy in 1.8-2 Gy fractions) (potentially without concomitant temozolomide) Nitrosourea; temozolomide rechallenge; bevacizumabf Per extrapolation from IDH-mutant anaplastic astrocytoma88 or from glioblastoma59 Glioblastoma, IDH wild type, WHO grade 4; giant cell glioblastoma; gliosarcoma; epithelioid glioblastoma Temozolomide chemoradiotherapy (54-60 Gy in 1.8-2" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Surgery for glioblastoma should involve gross total resection whenever feasible46." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Concomitant radiotherapy and chemotherapy with temozolomide (75 mg/m2 daily throughout radiotherapy, including at weekends) plus six cycles of maintenance temozolomide (150-200 mg/m2, 5 out of 28 days) is the standard of care for adults with newly diagnosed glioblastoma who are in good general and neurological condition and are aged <70 years59." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Patients with unfavourable prognostic factors (defined by age and/or KPS) can be treated with hypofractionated radiotherapy (such as 40 Gy in 15 fractions), which has similar activity to irradiation with 60 Gy in 30 fractions54." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The addition of temozolomide to hypofractionated radiotherapy54 has also been shown to improve OS in patients aged ≥60 years94." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "patients with tumours lacking MGMT promoter methylation or of unknown MGMT promoter methylation status should be treated with hypofractionated radiotherapy alone whereas those with tumours with MGMT promoter methylation status should receive temozolomide alone (5 out of 28 days until disease progression or for 12 months)97." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The standard of care for patients with IDH-wild-type glioblastoma aged <70 years and with a KPS >70 includes resection as feasible or biopsy followed by involved-field radiotherapy and concomitant radio therapy and six cycles of maintenance temozolomide chemotherapy (EORTC 26981-NCIC CE.3)59." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Temozolomide might only be active in patients with MGMT promoter-methylated tumours whereas its activity in patients with MGMT promoter- unmethylated tumours is probably marginal95." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Elderly patients not considered candidates for temozolomide chemoradiotherapy should be treated on the basis of MGMT promoter methylation status (NOA-08, Nordic Trial) with radiotherapy (such as 15 × 2.66 Gy) or temozolomide (5 out of 28 days) alone96,98,97." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "At recurrence, standards of care are less well defined." (verbatim=True)

## Astrocytoma, IDH-mutant, CNS WHO grade 4  (`astrocytoma:4`)
- **Surgery first:** Maximal safe resection; stereotactic biopsy if deep or eloquent.
- **First-line adjuvant (if confirmed):** Temozolomide-based chemoradiotherapy with radiotherapy 54-60 Gy; an alternative is radiotherapy followed by temozolomide without concomitant temozolomide. Evidence is extrapolated from grade 3 IDH-mutant astrocytoma and glioblastoma trials rather than dedicated trials - state this explicitly.
- **Observation:** Not standard.
- **Age branch:** No specific age rule; the same KPS-based intensity caveat applies.
- **At progression:** Nitrosourea, temozolomide rechallenge, bevacizumab for symptom control.
- **Pending determinants:** histologic confirmation, CDKN2A/B homozygous deletion (grading), MGMT promoter status, KPS, extent of resection
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Astrocytoma, IDH-mutant, WHO grade 4_: "Temozolomide chemoradiotherapy (54-60 Gy in 1.8-2 Gy fractions) (potentially without concomitant temozolomide) Nitrosourea; temozolomide rechallenge; bevacizumabf Per extrapolation from IDH-mutant anaplastic astrocytoma88 or from glioblastoma59 Glioblastoma, IDH wild type, WHO grade 4; giant cell glioblastoma; gliosarcoma; epithelioid glioblastoma Temozolomide chemoradiotherapy (54-60 Gy in 1.8-2" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Tumours formerly diagnosed as IDH-mutant glioblastoma are now referred to as IDH-mutant astrocytoma, WHO grade 4, and are managed either as IDH-wild-type glioblastoma or as IDH-mutant astrocytoma, WHO grade 3 (Table 2)." (verbatim=True)

## Astrocytoma, IDH-mutant, CNS WHO grade 3  (`astrocytoma:3`)
- **Surgery first:** Maximal safe resection or biopsy, then adjuvant therapy.
- **First-line adjuvant (if confirmed):** Radiotherapy 54-60 Gy followed by maintenance temozolomide for 12 cycles. The added benefit of concomitant temozolomide during radiotherapy is uncertain in this guideline - do not present concomitant temozolomide as standard here.
- **Observation:** Not standard except rare special situations.
- **Age branch:** No specific age rule.
- **At progression:** Re-resection, temozolomide rechallenge or nitrosourea; re-irradiation may be considered if more than about 12 months since prior radiotherapy and the lesion is small.
- **Pending determinants:** histologic confirmation, extent of resection, KPS
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Astrocytoma, IDH-mutant, WHO grade 3_: "Radiotherapy (54-60 Gy in 1.8-2 Gy fractions) followed by temozolomide (or wait-and-see) Nitrosourea; temozolomide rechallenge 88 Anaplastic astrocytoma, IDH wild type, WHO grade 3 Radiotherapy (54-60 Gy in 1.8-2 Gy fractions); temozolomide chemoradiotherapy, by MGMT promoter methylation status (?) Temozolomide rechallenge; nitrosourea; bevacizumabf Per extrapolation from IDH-wild-type glioblastom" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The standard of care for patients with this disease subtype is maximal surgical resection or biopsy followed by radiotherapy at 60 Gy in 1.8-2 Gy fractions (Table 1)." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The EORTC 26053 trial (CATNON) of radiotherapy alone, with concomitant or maintenance temozolomide or with both concomitant and maintenance temozolomide showed a significant prolongation of OS in patients receiving radiotherapy followed by 12 cycles of maintenance temozolomide and, thus, this approach should be considered standard of care; however, the role of concomitant temozolomide remains uncertain88." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "For those with disease relapse after radiotherapy, re-irradiation after a minimum interval of ~12 months following the first course of radiotherapy is an option, although tumour size and patterns of recurrence limit the option of re-irradiation and the overall efficacy of this strategy remains uncertain in the absence of data from RCTs." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The standard of care for IDH-mutant astrocytomas, WHO grade 3 includes resection as feasible or biopsy followed by involved field radiotherapy and maintenance temozolomide (CATNON)88." (verbatim=True)

## Astrocytoma, IDH-mutant, CNS WHO grade 2  (`astrocytoma:2`)
- **Surgery first:** Maximal safe resection; stereotactic biopsy if deep or eloquent.
- **First-line adjuvant (if confirmed):** If treatment is indicated, radiotherapy 50-54 Gy followed by PCV (procarbazine, lomustine, vincristine).
- **Observation:** Low-risk patients may be observed after surgery - typically younger than about 40-45 years, complete resection, and asymptomatic or seizures only.
- **Age branch:** Age 40 or older, or residual tumour after surgery -> radiotherapy followed by PCV is standard rather than observation. Age younger than 40 with complete resection -> observation is an option.
- **At progression:** Re-resection, temozolomide or nitrosourea, re-irradiation in selected cases.
- **Pending determinants:** histologic confirmation, extent of resection, symptoms/neurologic deficit, KPS
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Astrocytoma, IDH-mutant, WHO grade 2_: "Wait-and-see or radiotherapy (50-54 Gy in 1.8-2 Gy fractions) followed by PCV (or temozolomide chemoradiotherapy) Temozolomide (or nitrosourea) RTOG 9802 (ref.75) and per extrapolation from WHO grade 3 tumours88 Diffuse astrocytoma, IDH wild type, WHO grade 2a,e Wait-and-see (?); radiotherapy (50-54 Gy in 1.8-2 Gy fractions); radiotherapy followed by PCV or temozolomide chemoradiotherapy (by MGMT" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Maximal surgical resection, if safely feasible, is the best initial therapeutic approach84." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Younger patients (pragmatic cut-off ~40-45 years of age) who are asymptomatic or with seizures only, can be managed through observation alone after gross total resection." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Involved-field radiotherapy (50 Gy in 1.8 Gy fractions) should be considered for patients with incomplete resection and/or for patients aged >40 years." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The RTOG 9802 trial reported a major prolongation of OS with the addition of PCV polychemotherapy to radiotherapy (54 Gy), from 7.8 years to 13.3 years in patients with high-risk WHO grade 2 gliomas who were 18-39 years of age and had undergone a subtotal resection or biopsy or in those aged ≥40 years75." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "The standard of care for IDH-mutant astrocytomas, WHO grade 2 requiring further treatment includes resection as feasible or biopsy followed by involved field radiotherapy and maintenance PCV polychemotherapy (RTOG 9802)75." (verbatim=True)

## Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, CNS WHO grade 3  (`oligodendroglioma:3`)
- **Surgery first:** Maximal safe resection or biopsy.
- **First-line adjuvant (if confirmed):** Radiotherapy 54-60 Gy followed by PCV; the long-term survival benefit of adding PCV is established. Radiotherapy plus PCV is the rule over observation.
- **Observation:** Very selective: age younger than 40, complete resection, no neurologic deficit, and no CDKN2A/B homozygous deletion.
- **Age branch:** Age younger than 40 is one of the observation conditions but is not sufficient alone.
- **At progression:** Re-resection first; temozolomide; bevacizumab for symptom control if other options are exhausted.
- **Pending determinants:** histologic confirmation, 1p/19q codeletion confirmation, CDKN2A/B status, extent of resection, KPS
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, WHO grade 3_: "Radiotherapy (54-60 Gy in 1.8-2 Gy fractions) followed by PCV (or wait-and-see) Temozolomide 79,80 Anaplastic oligodendroglioma, NOS, WHO grade 3 See oligodendroglioma, IDH-mutant and 1p/19q-codeleted, WHO grade 3 See oligodendroglioma, IDH-mutant and 1p/19q-codeleted, WHO grade 3 Per extrapolation because most of these tumours carry IDH mutations Oligoastrocytoma, NOS, WHO grade 2 Wait-and-see; r" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "In this subtype, the extent of resection is a prognostic factor78." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "watch-and-wait strategies after complete resection can also be considered for younger patients (<40 years of age) with WHO grade 3 tumours, specifically for those without homozygous CDKN2A/B deletion, although only after gross total resection and in the absence of neurological deficits." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Two large RCTs showed that the addition of PCV, either prior to or after radiotherapy, in the first-line of treatment approximately doubled the OS79,80." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Second surgery should always be considered." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "bevacizumab can be used for symptom control; however, the antitumour efficacy of bevacizumab is unknown and no evidence supports its combination with cytotoxic agents in this setting." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Patients with IDH-mutant and 1p/19q-codeleted oligodendrogliomas, WHO grade 3 should be treated with radiotherapy followed by PCV polychemotherapy (EORTC 26951, RTOG 9402)79,80." (verbatim=True)

## Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, CNS WHO grade 2  (`oligodendroglioma:2`)
- **Surgery first:** Maximal safe resection; stereotactic biopsy if deep or eloquent.
- **First-line adjuvant (if confirmed):** If treatment is indicated, radiotherapy followed by PCV.
- **Observation:** Prognosis is comparatively favourable; low-risk patients (younger than about 40, complete resection, asymptomatic) may be observed after surgery.
- **Age branch:** Age 40 or older, or residual tumour -> radiotherapy followed by PCV rather than observation. Age younger than 40 with complete resection -> observation is an option.
- **At progression:** Re-resection, chemotherapy (PCV or temozolomide), re-irradiation in selected cases.
- **Pending determinants:** histologic confirmation, 1p/19q codeletion confirmation, extent of resection, KPS
- **Evidence:**
  - _EANO 2021 — Table 2 (p.179): Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, WHO grade 2_: "Wait-and-see; radiotherapy (50-54 Gy in 1.8-2 Gy fractions) followed by PCV Temozolomide Per extrapolation from WHO grade 3 tumours79,80 and RTOG 9802 (ref.75) Oligodendroglioma, NOS, WHO grade 2 See oligodendroglioma, IDH-mutant and 1p/19q-codeleted, WHO grade 2 See oligodendroglioma, IDH-mutant and 1p/19q- codeleted, WHO grade 2 Per extrapolation because most of these tumours carry IDH mutations" (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Surgery is the primary treatment modality for patients with gliomas of this subtype." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "watch-and-wait strategies are justified in those with gross total resection and potentially also in younger patients (<40 years of age) with incomplete resection if the tumour has not yet caused neurological deficits beyond symptomatic epilepsy." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "If further treatment beyond surgery is deemed necessary, the standard of care is radiotherapy followed by PCV75." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Patients with IDH-mutant and 1p/19q-codeleted oligodendrogliomas, WHO grade 2 requiring further treatment should be treated with radiotherapy followed by PCV polychemotherapy." (verbatim=True)
  - _EANO 2021 — Therapy - specific recommendations_: "Temozolomide chemotherapy is standard treatment at progression after surgery and radiotherapy for most patients with IDH-mutant gliomas, WHO grade 2 or 3." (verbatim=True)
