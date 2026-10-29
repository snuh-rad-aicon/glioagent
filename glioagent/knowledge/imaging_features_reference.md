# Glioma imaging-feature reference

Per-feature operational definitions, categorical bins, and prevalence /
diagnostic-performance numbers for the imaging features used in the report and
the knowledge-grounded review. Each chunk cites the source it is paraphrased /
quoted from so the assessment is reviewer-auditable.

Indexed for RAG alongside `who2021_nccn_reference.md` by
`tools/tool_rag_verifier.py` (each `### ...` heading becomes a chunk).

## Sources

* **VASARI 2.0** -- Negro A, Cerase A, Lavalle L, et al. *VASARI 2.0: a
  new updated MRI VASARI lexicon to predict grading and IDH status in
  brain glioma.* Front Oncol. 2024;14:1449982. doi:10.3389/fonc.2024.1449982.
  (open access; PMC11700831)
* **VASARI-auto (Ruffle 2024)** -- Ruffle JK, et al. *VASARI-auto: equitable,
  efficient, and economical featurisation of glioma MRI.* NeuroImage: Clinical.
  2024;44:103668. doi:10.1016/j.nicl.2024.103668. Reports per-feature agreement
  vs radiologist ground truth: F9 multifocality kappa 1.00 (highest), F19
  ependymal extension kappa 0.33 (lowest inter-rater); and per-feature accuracy
  F5 proportion-enhancing 49.42%, F6 nCET 20.69%, F14 oedema 32.18% -- the basis
  for adopting only F9 + F19 and treating the morphologic proportions as unreliable.
* **VASARI Research Project (TCIA)** -- The Cancer Imaging Archive,
  *VASARI Research Project*, Wiki, Public TCIA resource. The original
  VASARI feature set (24 observations) developed by an 8-neuroradiologist
  consensus panel under the NIH Cancer Imaging Program.
* **Patel 2017 (T2-FLAIR mismatch original)** -- Patel SH, Poisson LM,
  Brat DJ, et al. *T2-FLAIR Mismatch, an Imaging Biomarker for IDH and
  1p/19q Status in Lower-grade Gliomas: A TCGA/TCIA Project.* Clin
  Cancer Res. 2017;23(20):6078-6085. doi:10.1158/1078-0432.CCR-17-0560.
* **Lasocki 2018 (MRI features predict 1p/19q)** -- Lasocki A, Gaillard F,
  Gorelik A, Gonzales M. *MRI Features Can Predict 1p/19q Status in
  Intracranial Gliomas.* AJNR Am J Neuroradiol. 2018;39(4):687-692.
  doi:10.3174/ajnr.A5572. Findings quoted below are from the Results
  ("MR Imaging Assessment: Validation Cohort"; validation cohort n=59 of 69
  total, of which 21 codeleted / 38 non-codeleted): >50% T2-FLAIR mismatch in
  14 of the 59 tumours, all non-codeleted (100% specificity for non-codeletion);
  calcification in 7 tumours, 6 codeleted (97% specificity for codeletion).
* **Do 2022 (T2-FLAIR mismatch meta-analysis)** -- Do YA, Cho SJ,
  Choi BS, et al. *Predictive accuracy of T2-FLAIR mismatch sign for
  the IDH-mutant, 1p/19q noncodeleted low-grade glioma: An updated
  systematic review and meta-analysis.* Neuro-Oncology Advances.
  2022;4(1):vdac010. doi:10.1093/noajnl/vdac010.
* **Setyawan 2024** -- Setyawan NH, Choridah L, Nugroho HA, et al.
  *Beyond invasive biopsies: using VASARI MRI features to predict
  grade and molecular parameters in gliomas.* Cancer Imaging.
  2024;24:3. doi:10.1186/s40644-023-00638-8.
* **Louis 2021 (WHO CNS5)** -- Louis DN, Perry A, Wesseling P, et al.
  *The 2021 WHO Classification of Tumors of the Central Nervous
  System: a summary.* Neuro-Oncology. 2021;23(8):1231-1251.
  doi:10.1093/neuonc/noab106.
* **Maynard 2020 (age + morphology -> IDH)** -- Maynard J, Okuchi S,
  Wastling S, et al. *World Health Organization Grade II/III Glioma
  Molecular Status: Prediction by MRI Morphologic Features and Apparent
  Diffusion Coefficient.* Radiology. 2020;296(1):111-121.
  doi:10.1148/radiol.2020191832.
* **Foltyn-Dumitru 2025 (non-enhancing glioblastoma)** -- Foltyn-Dumitru
  M, et al. *Histopathological and molecular characteristics of
  IDH-wildtype glioblastoma without contrast enhancement: implications
  for clinical outcomes.* Neuro-Oncology. 2025;27(7):1878-1887.
  doi:10.1093/neuonc/noaf070.
* **Midline shift surgical threshold** -- Standard neurosurgical
  literature: midline shift > 5 mm on neuroimaging is conventionally
  cited as the threshold above which surgical decompression is
  considered (cf. Liao et al., *Brain midline shift measurement and
  its automation: a review of techniques and algorithms*, Int J Biomed
  Imaging 2018;2018:4303161, and the textbook references therein).

---

### T2-FLAIR mismatch sign -- operational definition (Patel 2017; Do 2022)

> Defined as the **presence of a complete or near-complete hyperintense
> signal of the tumour on T2-weighted imaging, in combination with a
> relative hypointense signal on FLAIR except for a hyperintense
> peripheral rim** (Patel et al. 2017; reused verbatim in the Do 2022
> meta-analysis). An alternative *objective* operationalisation used in
> some validation studies is **"over 50% area of signal nulling on
> FLAIR compared to T2-WI"** (Do 2022).

The sign is a property of the **non-enhancing tumour core**, not the
peritumoural edema, and is not assessable in lesions with extensive
necrosis (Do 2022, confounder list).

### T2-FLAIR mismatch sign -- clinical prevalence and predictive value

In Patel et al. 2017:
* Discovery cohort (TCGA/TCIA, n = 125): **15 cases positive (12.0 %)**.
  Positive predictive value (PPV) for IDH-mutant, 1p/19q-noncodeleted
  status = **100 %**. Negative predictive value (NPV) = 54 %.
* Validation cohort (n = 82): **10 cases positive (12.2 %)**. PPV = 100 %.
  NPV = 76 %.

Do et al. 2022 meta-analysis (12 studies, > 1500 patients):
* Pooled **sensitivity = 40 %** (95 % CI 28-53 %).
* Pooled **specificity = 100 %** (95 % CI 95-100 %) for the IDH-mutant,
  1p/19q non-codeleted molecular subtype.
* Per-study positive-call rate ranged from ~20 % to ~70 % across diverse
  cohort compositions.

**Operating point.** Because specificity is essentially 1.0, the sign
is a *rule-in* feature: a confidently called mismatch is strong
evidence for IDH-mutant astrocytoma. Because sensitivity is only ~40 %,
*absence* of the sign does **not** rule out IDH-mutant astrocytoma.

### T2-FLAIR mismatch sign -- confounders (Do 2022)

Sources of false-positive calls reported in the meta-analysis:
* Small microcystic areas within the tumour misinterpreted as central
  FLAIR hypointensity.
* Central necrosis (T1-hypointense, also T2-hyperintense): the mismatch
  sign requires homogeneous *solid* T2 hyperintensity, not a necrotic
  cavity. The sign is essentially absent in grade-4 / glioblastoma
  tumours.
* Cystic change with thin enhancing rim.
* Variable interpretation when only partial mismatch is present
  (the sign is a binary call; partial patterns should be reported as
  "no mismatch" in classical use).

### T2-FLAIR mismatch sign -- vision-model extent and integrated interpretation

GlioAgent scores the T2-FLAIR mismatch with an in-context vision VQA
(Qwen3.5-27B) over three tumor slices and reports an EXTENT when present:

* **partial** -- focal / incomplete interior FLAIR suppression. Rule-in for an
  **IDH-mutant** glioma along the IDH axis, but it is weaker evidence for the
  1p/19q status. Favor IDH-mutant; do not commit to astrocytoma vs
  oligodendroglioma on a partial mismatch alone.
* **complete** -- near-entire interior suppression with a hyperintense rim.
  Rule-in for **IDH-mutant ASTROCYTOMA (1p/19q non-codeleted)** -- the sign is
  ~100 % specific for this subtype (Patel 2017; Do 2022) and is essentially
  absent in oligodendroglioma and IDH-wildtype glioblastoma.

**The vision model can be wrong.** Heterogeneous interior signal, small
microcysts, central necrosis, or a bright cyst rim can be misread as
suppression (false positive), and subtle true signs can be missed (false
negative; sensitivity ~40 %). Therefore the orchestrator must NOT take the
mismatch extent as ground truth. It weighs the extent AGAINST the GlioMT
molecular prediction (IDH probability, 1p/19q codeletion probability) and the
rest of the imaging picture, and reads the whole trace before deciding:

* When the mismatch and GlioMT **agree**, state the diagnosis with added
  confidence.
* When they **disagree**, GlioMT's prediction takes precedence -- especially
  where GlioMT is confident (IDH probability >= 0.95, or 1p/19q codeletion
  probability >= 0.90). Only overturn GlioMT toward astrocytoma on a
  *complete* mismatch when GlioMT's codeletion call is uncertain
  (probability < 0.90) and the lesion is non-necrotic with no prominent
  enhancement (the reliability caveat, Do 2022).
* If necrosis or prominent enhancement is present, treat the mismatch as
  LOW reliability and do NOT flip the molecular call on it.

### VASARI feature F4 -- Enhancement Quality (Negro 2024)

> Categorical options (VASARI 2.0):
> 1. **Absent** -- no enhancing tissue is visible after contrast.
> 2. **Minimal** -- barely perceptible enhancement.
> 3. **Avid** -- clearly visible, marked enhancement of any portion of
>    the tumour.

Operational rule: enhancement is tissue that is dark on the pre-contrast
T1 sequence and becomes bright after gadolinium administration. Tissue
that is already bright on pre-contrast T1 (e.g., proteinaceous or
haemorrhagic content) is **not** counted as enhancement.

### VASARI feature F5 -- Enhanced Area (Negro 2024)

> Proportion of the entire tumour volume that enhances (VASARI 2.0
> categorical bins):
> 1. Absent / < 5 %
> 2. 6 - 25 %
> 3. 26 - 50 %
> 4. 51 - 75 %
> 5. 76 - 95 %
> 6. > 95 %

The original VASARI lexicon used finer bins (`<5 % / 6-33 % / 34-67 %
/ 68-95 % / 96-99 % / 100 %`); the 2024 update consolidates these to
six broader categories for higher inter-rater agreement (Negro 2024).

### VASARI feature F6 -- Non-enhanced Area (nCET; Negro 2024)

> Proportion of the tumour that is T2-hyperintense and T1-hypointense
> with mass effect and "blurring of the gray-white interface", but does
> **not** enhance after contrast. Same six-bin scale as F5
> (Absent / < 5 % ... > 95 %).

This is the substrate on which the T2-FLAIR mismatch sign is evaluated.

### VASARI feature F7 -- Necrosis Area (Negro 2024; Setyawan 2024)

> "Necrotic areas are defined as areas that are hypointense on T1-WI"
> with "irregular border, high signal on T2-weighted and proton density
> imaging" and **minimal or no enhancement of the central component**,
> typically surrounded by an enhancing rim (Setyawan 2024 paraphrasing
> VASARI; consistent with Negro 2024 VASARI 2.0 definition).
>
> VASARI 2.0 categorical bins:
> 1. Absent / < 5 %
> 2. 6 - 25 %
> 3. 26 - 50 %
> 4. 51 - 75 %
> 5. 76 - 95 %
> 6. > 95 %

Distinction from non-enhancing tumour: necrosis remains dark on T1
*and* on T1-post-contrast, and is typically surrounded by an enhancing
rim. Non-enhancing tumour is also dark on T1 but bulky and
infiltrative without a clear central cavity. Cystic change has a
smooth wall and very low T1 signal, often without solid surrounding
enhancement.

### Necrosis -- clinical context (WHO 2021; cross-reference)

The presence of central necrosis is one of the histologic criteria
that, in combination with the molecular profile, supports a CNS WHO
grade-4 designation. Imaging-detected necrosis correlates strongly with
IDH-wildtype glioblastoma; isolated necrosis is uncommon in CNS WHO
grade-2 / 3 IDH-mutant gliomas (Setyawan 2024 introduction; Negro 2024
discussion).

### VASARI feature F14 -- Edema Area (Negro 2024)

> Categorical bins (VASARI 2.0):
> 1. Absent / < 5 %
> 2. 6 - 25 %
> 3. 26 - 50 %
> 4. 51 - 75 %
> 5. 76 - 95 %

(VASARI 2.0 does not retain a > 95 % category for edema.)

Edema is identified on FLAIR (and T2) as hyperintense signal that
extends **beyond** the enhancing or non-enhancing tumour margin. It is
typically more diffuse than the tumour itself and follows the
white-matter tracts of the centrum semiovale and corona radiata.
Distinction from non-enhancing tumour is imperfect on FLAIR alone;
when the FLAIR-hyperintense rim is thick and mass-like, it should be
recorded conservatively (Negro 2024 methods).

### VASARI feature for cortical involvement / eloquent brain

The original VASARI feature set codes eloquent-brain / cortical
involvement as a binary or categorical observation: whether the tumour
*reaches or invades the cortical ribbon* (cortical) versus is *confined
to subcortical white matter* (subcortical) (TCIA VASARI Research
Project; the per-feature breakdown is identical between VASARI v1.1
and VASARI 2.0 for this observation).

Inter-rater kappa for "cortical involvement" is one of the lower-
agreement features in the literature (kappa = 0.167 in the original
multireader study reported on the TCIA VASARI Research Project page),
indicating intrinsic ambiguity at the gray-white junction. Cases with
cortical thinning or signal abnormality of the cortical ribbon
adjacent to the tumour should be called cortical; cases where the
ribbon is intact and only the white matter beneath is involved should
be called subcortical (Negro 2024 discussion).

### Mass effect / midline shift -- measurement and thresholds

> Midline shift is measured in millimetres as the perpendicular distance
> between a midline structure (most commonly the **septum pellucidum**)
> and a reference line drawn between the anterior and posterior
> attachments of the falx cerebri to the inner table of the skull.
> The third ventricle and the pineal gland are alternative midline
> reference structures, both centrally located and caudal to the
> septum (Wikipedia / standard neuroradiology references; primary
> reviews in Liao 2018).

> A midline shift **> 5 mm** is conventionally cited as the threshold
> above which surgical decompression should be considered in mass-
> producing lesions (cited in textbooks and clinical guidelines;
> see references in Liao 2018).

Reference structure rules:
* Use the **septum pellucidum**, *not* the tumour edge, as the moving
  landmark.
* Asymmetric ventricular size *without* septum displacement is **not**
  a midline shift.
* Local sulcal effacement adjacent to the tumour is **not** a midline
  shift.

### Prior probabilities (combined sources)

Coarse prior frequencies useful when calibrating positive calls:
* T2-FLAIR mismatch sign -- present in roughly **10-15 %** of adult
  diffuse gliomas across cohorts (Patel 2017 reports 12 % in two
  cohorts; Do 2022 meta-analysis shows wide between-study variation).
  Essentially confined to IDH-mutant, 1p/19q-non-codeleted
  astrocytomas; absent in 1p/19q-codeleted oligodendrogliomas and in
  IDH-wildtype glioblastomas (Patel 2017; Do 2022).
* Central necrosis -- common in IDH-wildtype glioblastoma; uncommon in
  CNS WHO grade-2 / 3 IDH-mutant gliomas (Setyawan 2024).
* Ring enhancement -- most characteristic of IDH-wildtype glioblastoma;
  less common in IDH-mutant astrocytomas; rare in oligodendrogliomas
  (Setyawan 2024 multivariate model: significant predictor of higher
  grade and IDH-wildtype status).
* Midline shift > 5 mm -- typically associated with bulky high-grade
  gliomas; small or low-grade gliomas frequently show no measurable
  shift (Liao 2018 review).

### IDH status prediction from age and morphology (Maynard 2020)

Patient age is an **independent, non-imaging predictor of IDH status** and is
one of the inputs to the molecular reconciliation step (used jointly with
imaging, never alone).

Maynard et al. 2020 (Radiology; n = 290 WHO grade II/III gliomas) built models
that predict the three-way molecular status (IDH-wildtype vs IDH-mutant/1p19q-
intact vs IDH-mutant/1p19q-codeleted) from **patient age + MRI morphologic
features + the ADCmean-to-ADCNAWM ratio**, reaching **82-86% test accuracy**.
Age entered the models as an independent contributor: within their cohort,
older age shifted the prediction toward IDH-wildtype and younger age toward
IDH-mutant.

Epidemiologic direction (WHO 2021 / Louis 2021 and cohort literature):
* **IDH-mutant** astrocytomas and oligodendrogliomas present in **younger
  adults** (typically 3rd-5th decade).
* **IDH-wildtype glioblastoma** presents in **older adults** (typically 6th-7th
  decade); it is uncommon, though not impossible, in young adults.

**Operating heuristic used by reconciliation** (age is corroborating, not
decisive): age <= 50 (especially < 45) with imaging lacking glioblastoma
hallmarks supports reconsidering a wildtype call toward mutant; age >= 60,
particularly with central necrosis and avid enhancement, supports wildtype.
Because IDH-wildtype glioblastoma is rare but real in young patients, age never
overrides imaging on its own -- it only tips a low-confidence single-model call
when imaging is concordant.

### Imaging feature concordance with molecular subtype (Louis 2021; Patel 2017; Do 2022; Maynard 2020)

Expected direction of each deterministic finding for the three WHO 2021 adult
diffuse-glioma subtypes. Used by the review layer to state whether the imaging
SUPPORTS or CONTRADICTS the molecular call. Each cell is a *typical* tendency,
not a rule; discordance lowers confidence rather than reclassifying.

| finding | Astrocytoma, IDH-mutant | Oligodendroglioma, IDH-mut 1p/19q-codel | Glioblastoma, IDH-wildtype |
|---|---|---|---|
| T2-FLAIR mismatch sign | **characteristic** (rule-in; spec ~100%, sens ~40%) | essentially absent | essentially absent |
| Central necrosis | uncommon (esp. grade 2-3) | uncommon | **common** |
| Enhancement | absent / minimal (grade 2-3) | variable, often patchy | **avid**, though ~7% non-enhancing |
| Typical age band | younger (3rd-5th decade) | younger-to-middle | older (6th-7th decade) |
| Cortical / calcification tendency | subcortical, infiltrative | cortical-based, calcification common | variable, often necrotic core |

Sourcing: T2-FLAIR mismatch specificity/sensitivity and its confinement to
IDH-mutant 1p/19q-non-codeleted astrocytoma -- Patel 2017; Do 2022; Lasocki 2018
(>50% mismatch 100% specific for non-codeletion). Calcification and frontal /
cortical-subcortical location favouring 1p/19q-codeleted oligodendroglioma --
Lasocki 2018 (calcification 97% specific for codeletion). Necrosis /
enhancement tendencies and the WHO 2021 subtype definitions -- Louis 2021 (with
imaging-phenotype correlation in Setyawan 2024). Age direction -- Maynard 2020.
The non-enhancing IDH-wildtype glioblastoma exception (~7.4%) -- Foltyn-Dumitru
2025.

**Concordance reading rules.**
* Mismatch present + IDH-mutant astrocytoma call -> strongly concordant.
* Mismatch present + IDH-wildtype or codeleted call -> discordant (the sign is
  essentially absent in both; re-examine the mismatch call and the molecular
  call).
* Central necrosis + avid enhancement + IDH-wildtype call -> concordant.
* No necrosis + no/minimal enhancement + young age + IDH-wildtype call ->
  partial discordance; consider IDH-mutant (Maynard 2020 age effect), but a
  non-enhancing IDH-wildtype glioblastoma remains possible (Foltyn-Dumitru 2025).

### Enhancement phenotype by molecular subtype (Foltyn-Dumitru 2025; Setyawan 2024)

Enhancement has two INDEPENDENT axes that must not be conflated:
* **Avidity (quality)** -- how bright the enhancing tissue is. GlioAgent measures
  this as `enhancement_z`: the T1CE-minus-T1 z-score computed WITHIN the enhancing
  (ET) region, against normal-appearing brain. It is an INTENSITY measure of the
  enhancing tissue, NOT the fraction of the tumour that enhances. "Avid" (VASARI
  2.0 F4) means the enhancing tissue is markedly bright, independent of how much of
  the tumour enhances.
* **Extent (proportion)** -- the fraction of the tumour that enhances (ET / whole
  tumour). A separate number, reported separately. A tumour can enhance over a large
  extent yet only minimally in avidity, or enhance avidly over a small extent.

Subtype tendency (Setyawan 2024, VASARI, n=107): enhancement quality is among the
VASARI predictors of glioma grade, and proportion-enhancing among the predictors of
IDH status -- greater enhancement burden favouring higher-grade, IDH-wildtype
tumours; IDH-mutant lower-grade gliomas typically show little or no enhancement.

**Non-enhancing glioblastoma exception.** Absence of enhancement does NOT exclude
IDH-wildtype glioblastoma: Foltyn-Dumitru 2025 found 98/1323 (7.4%) IDH-wildtype
glioblastomas without contrast enhancement (a less aggressive phenotype, longer
survival). This grounds the soft "IDH-wildtype without appreciable enhancement"
flag as INFO, not a contradiction.

### Distinguishing astrocytoma from oligodendroglioma (IDH-mutant)

Within IDH-mutant diffuse gliomas, the astrocytoma versus oligodendroglioma
distinction is defined by 1p/19q codeletion (oligodendroglioma = codeleted;
astrocytoma = intact).

Imaging discriminators:
* T2-FLAIR mismatch sign is essentially confined to IDH-mutant, 1p/19q-intact
  astrocytoma. In the Lasocki 2018 validation cohort (n=59; 21 codeleted, 38
  non-codeleted), a >50% mismatch was present in 14 tumours, all non-codeleted
  (100% specificity for a non-codeleted tumour), consistent with Patel 2017. A
  confident mismatch sign therefore argues against oligodendroglioma.
* Calcification favours oligodendroglioma (Lasocki 2018: present in 7 tumours,
  6 of them codeleted; 97% specificity for codeletion) but requires CT or SWI and
  is not assessable on a four-sequence MRI protocol.
* Oligodendroglioma predominates in a frontal, cortical-subcortical location;
  astrocytoma is more variable. Location is supportive, not decisive.
