# The tools: what each one computes, and what its output is

One row per field the agent can read. **Type** says how to treat the value:

| type | meaning |
|---|---|
| **binary** | `true` / `false`; a threshold on a continuous quantity turned it into a decision |
| **continuous** | a measured number with a unit; no threshold applied |
| **categorical** | one of a fixed, ordered or unordered set |
| **set** | zero or more labels at once |
| **probability** | 0 to 1 from a trained classifier |

Where a binary field is a thresholded continuous one, both are listed, because the continuous
value is what a reader should quote and the threshold is what the agent acted on.

---

## 1. `tool_tumor_segment` — HD-GLIO segmentation

Input: T1, T1CE, T2, FLAIR. Output: a label volume, then cleaned by `seg_cleanup`.

Labels: `0` background, `1` non-enhancing T2/FLAIR signal, `2` enhancing tumour.
**Whole tumour is `label > 0`** — it includes oedema, which HD-GLIO does not separate from
non-enhancing tumour.

| field | type | range | note |
|---|---|---|---|
| `et_cm3` | continuous | 0 to ~300 cm³ | enhancing tumour, `label == 2` |
| `ne_cm3` | continuous | 0 to ~400 cm³ | non-enhancing, `label == 1`; oedema and non-enhancing tumour combined |
| `wt_cm3` | continuous | 0 to ~500 cm³ | whole tumour, `label > 0` |
| `laterality` | categorical | `left` / `right` / `bilateral` | segmentation-only fallback; ≥ 55% of the volume on one side makes it that side |

`seg_cleanup` drops whole-tumour components below `ISLAND_CM3` before anything else runs, so a
focus smaller than that can never be counted later.

**No-call rule:** whole tumour < `MIN_TUMOUR_CM3 = 0.5` cm³ carries no diagnosis. Two patients of
2,126 fall here, and they count as wrong for every model.

---

## 2. `tool_seg_features` — findings measured on the segmentation

| field | type | range | threshold | note |
|---|---|---|---|---|
| `enhancement.proportion` | **continuous** | 0 to 1 | — | enhancing volume / whole-tumour volume; **the quantity the IDH guardrail acts on** |
| `enhancement.present` | binary | — | `ENHANCE_CM3` on `et_cm3` | |
| `enhancement.enhancement_z` | continuous | about −2 to +8 | — | enhancing signal in SD above normal brain |
| `enhancement.quality` | categorical | `Absent` / `Minimal` / `Avid` | `T_ENH_AVID_Z = 2.0` | VASARI 2.0 F4; Avid at ≥ 2 SD |
| `necrosis.necrosis_cm3` | continuous | 0 to ~100 cm³ | — | reported as **central necrosis**, the term the radiologist uses and the one the agreement is against, but **not a necrosis label.** HD-GLIO has no necrosis class: this is non-enhancing tissue *enclosed by* enhancing tumour, closed over a `NECROSIS_CLOSE_MM = 3.0` mm gap. It underestimates true necrosis, and absence does not exclude it |
| `necrosis.present` | binary | — | `NECROSIS_CM3` | anti-predictive for a mutant→wildtype flip (1 of 11 such proposals was right), which is why that direction is disabled |
| `necrosis.necrosis_over_et` | continuous | 0 to 1 | — | enclosed fraction of the enhancing core |
| `additional.multifocal` | binary | — | `FOCUS_MIN_CM3 = 0.1` cm³ | **two or more enhancing components inside one whole-tumour component.** 0.1 cm³ is about a 6 mm nodule, which is what a report means by a "focus" |
| `multifocal_foci_cm3` | set of continuous | each ≥ 0.1 cm³ | — | the foci joined to the dominant one |
| `multicentric_foci_cm3` | set of continuous | each ≥ 0.1 cm³ | — | foci in a *different* whole-tumour component; bounded below by `seg_cleanup.ISLAND_CM3` |
| `edema.radial_mm` | continuous | 0 to ~90 mm | `EDEMA_MM = 20.0` | **a proxy.** Radial extent of non-enhancing signal from the enhancing centroid; it reflects spatial extent, not oedema volume |
| `edema.present` | binary | — | as above | |
| `mismatch.flair_rim_minus_center` | continuous | about −5 to +5 | — | FLAIR rim minus core z-score; the quantitative form of the T2-FLAIR mismatch sign |
| `mismatch.t2_bright`, `t2_uniformity` | continuous | — | — | T2 interior brightness and uniformity |
| `mismatch.present` | binary | — | rim/core rule | **superseded in the run of record** by `tool_t2fm`, which stores only the three-level call; the quantitative fields survive in `pipeline_v11` |

---

## 3. `tool_atlas_features` — location, after registration to MNI

The tumour is warped to MNI152 with ANTs, then intersected with AAL116 (lobes), a cortex
probability mask, and the FreeSurfer aseg labels. The transform is cached on the T1's hash.

| field | type | range | threshold | note |
|---|---|---|---|---|
| `location.lobe` | categorical | frontal / temporal / parietal / occipital / insular / cerebellum / thalamus / basal ganglia / `deep_or_extracortical` | max overlap | the single lobe of greatest whole-tumour overlap. Figure 5 calls it **Tumor epicenter**: VASARI F1 is the tumour location by its epicentre, and VASARI-auto computes F1 exactly this way, as the lobe of largest overlap |
| `location.region` | categorical | one AAL116 parcel | — | e.g. `Frontal Mid`, `Postcentral` |
| `location.junction_lobes` | **set** | zero or more lobes | — | other lobes the tumour occupies; with `location.lobe`, the set Figure 5 calls **Lobar involvement** and compares, as recall, with the lobes the report names |
| `location.laterality` | categorical | `left` / `right` / `bilateral` | — | atlas-based, canonical |
| `location.midline_crossing` | binary | — | `TAU_CROSS = 0.15` | minority-hemisphere share of the whole tumour |
| `cortical.overlap_frac` | **continuous** | 0 to 1 | — | tumour surface overlapping the cortex mask |
| `cortical.involved` | binary | — | `TAU_CORTICAL = 0.10` | **saturated:** true in 543 of 544 IDH-mutant cases, and present in all 125 of the patients whose report commits on it. Positive agreement is therefore 100% by construction and chance-corrected agreement is **kappa 0.00**. It is off the agreement radar for that reason; quote `overlap_frac`, never this |
| `mass_effect.midline_shift_mm` | **continuous** | 0 to ~7 mm | — | shift of brain-minus-tumour tissue in MNI space. A *proxy*, not the septum-pellucidum measurement a radiologist makes |
| `mass_effect.present` | binary | — | `TAU_SHIFT = 1.0` mm | **1.0 mm is a detection threshold.** The familiar 5 mm is the surgical-decompression threshold; agreement with reports is 17.2% at 5.0 mm and 86.3% at 1.0 mm |
| `deep_structures.involved` | **set** | thalamus / basal ganglia / brainstem / corpus callosum | `DEEP_OVERLAP_CM3 = 0.5` cm³ (callosum `0.1`) | overlap volume with the aseg labels. **Absent from the run-of-record traces** — the reported agreement was recomputed from the stored MNI segmentations |
| `ventricle_shift_mm` | — | — | — | placeholder, needs an aseg segmentation; not produced |

---

## 4. `tool_measure_diameters` — bidimensional diameters

| field | type | range | threshold | note |
|---|---|---|---|---|
| `max_perpendicular_diameters_mm` | continuous pair | each 0 to ~100 mm | — | longest diameter and its perpendicular, on the **whole tumour** |
| `enhancing_diameters_mm` | continuous pair | each 0 to ~100 mm | — | the same on the **enhancing core**; the two pairs are not interchangeable |
| `measurable` | binary | — | both diameters ≥ `min_diameter_mm` (10 mm) | the size half of the published criterion (Wen 2023); slice thickness is not checked, so it is reported as a diameter |

---

## 5. `tool_t2fm` — T2-FLAIR mismatch, a vision model

Three slices through the tumour are shown to a vision-language model; the per-slice calls are
combined. This is the only non-deterministic imaging tool.

| field | type | range | note |
|---|---|---|---|
| `extent` | **categorical, ordered** | `none` / `partial` / `complete` | the sign is specific for IDH-mutant astrocytoma; a complete mismatch is stronger evidence than a partial one |
| `present` | binary | — | `extent` in (partial, complete) |
| `confidence` | categorical | `low` / `moderate` / `high` | the model's own statement |
| per-slice records | set | three entries | each with its own extent, confidence and reason |

**Reliability guardrail:** when central necrosis or avid enhancement is present the mismatch call
is unreliable (necrotic or cystic change mimics it) and must not drive a flip.

---

## 6. `tool_vasari_feature` — VASARI-auto

Derives VASARI 2.0 features from the segmentation mask alone. **Only two of its fields reach the
reported findings**, and both are flagged:

| field | type | range | note |
|---|---|---|---|
| F9 `multifocal` | binary | — | Ruffle 2024 inter-rater kappa **1.00**; superseded by the seg-based enhancing-foci count |
| F19 `ependymal_extension` | binary | — | kappa **0.33, the lowest inter-rater agreement in VASARI**; this is the only VASARI field still on the radar |
| F18 pial invasion, F13 margin definition, F3 eloquent brain | — | — | **always NaN**: VASARI-auto cannot derive them from a mask, so they carry no information |
| `proportion_edema`, `t2_flair_mismatch`, `proportion_necrosis` | — | — | marked `_unreliable_fields` and overridden by the deterministic tools |

---

## 7. `tool_molecular_subtype` — GlioMT

Input: the four sequences, the segmentation, age and sex.

| field | type | range | note |
|---|---|---|---|
| `idh.prediction` | categorical | `mutant` / `wildtype` | |
| `idh.probability` | **probability** | 0 to 1 | of the predicted class; `idh_prob_mutant` is the raw head output |
| `1p19q.prediction` | categorical | `codeleted` / `non-codeleted` | **only when IDH is mutant** — WHO 2021 makes it inapplicable otherwise |
| `p1_19q_prob_codel` | probability | 0 to 1 | accuracy on pathology-confirmed IDH-mutant cases is 75.3%, and the errors are symmetric, so no threshold shift repairs them |
| `grade_prediction` | categorical, ordered | 2 / 3 / 4 | |
| `grade_probability` | probability | 0 to 1 | |
| `who_2021` | categorical | the integrated diagnosis string | derived by `protocol/who2021_rules`, not predicted |

---

## 8. `tool_rag_verifier` — knowledge retrieval

Returns the reference chunks the reconciliation and the clinical assessment are allowed to cite
(EANO 2021, WHO 2021 / NCCN, imaging-feature references). Output is text, not a value.

---

## 9. The decisions made on top of the tools

These are in `agents/orchestrator.py`, not in any tool.

| constant | value | acts on | effect |
|---|---|---|---|
| `GLIOMT_CONFIDENT_P` | 0.95 | `idh.probability` | a proposed flip is discarded when the classifier is this sure |
| `MAX_ENHANCING_FRACTION_FOR_MUTANT` | 0.05 | `enhancement.proportion` | a wildtype→mutant flip is kept only below this |
| `ALLOW_MUTANT_TO_WILDTYPE` | `False` | — | that direction is never kept |
| `CODEL_CONFIDENT_P` | 0.90 | `p1_19q_prob_codel` | below this, a confident mismatch may favour astrocytoma |
| `MIN_TUMOUR_CM3` | 0.5 | `wt_cm3` | below this, no diagnosis is issued |

---

## 10. Which fields carry a measured number, and which are a judgement

Quote the continuous value, not the binary, wherever both exist:

| finding | the number to quote | the binary that was acted on |
|---|---|---|
| enhancement | `proportion` (0 to 1) | `present`, `quality` |
| central necrosis | `necrosis_cm3` (the enclosed non-enhancing proxy) | `necrosis.present` |
| multifocality | `multifocal_foci_cm3` (the list) | `multifocal` |
| cortical involvement | `overlap_frac` | `involved` (saturated — do not use) |
| midline crossing | minority-hemisphere fraction | `midline_crossing` |
| midline shift | `midline_shift_mm` | `mass_effect.present` |
| deep structures | overlap volume per structure | the involved set |
| oedema | `radial_mm` (a proxy) | `edema.present` |
| T2-FLAIR mismatch | `flair_rim_minus_center` (where stored) | `extent` |
| tumour size | the two diameter pairs | `measurable` |
