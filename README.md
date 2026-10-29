# GlioAgent

This is the official repository of the paper **"An auditable agentic system for molecular
characterization and radiologic report generation in adult-type diffuse glioma"**.

GlioAgent calls deterministic imaging tools and a molecular classifier, reconciles their output with a
language model under explicit code guardrails, and returns the WHO 2021 molecular characterization of the
tumor (IDH status, 1p/19q codeletion, grade) with a structured radiology report in which every statement
can be traced to the tool that produced it. A co-pilot mode answers a radiologist's questions about the
case, corrects findings and redrafts the report on request.

## Repository

```
glioagent/    the code and six sample patients
  agents/       orchestrator, co-pilot, report assembly, language-model client
  tools/        tumor segmentation, measurement, segmentation and atlas features, VASARI,
                T2-FLAIR mismatch, molecular subtype, knowledge retrieval
  protocol/     the rule-based checklist and the WHO 2021 decision rules
  knowledge/    the reference corpus the agent cites (WHO 2021, NCCN, EANO 2021, imaging features)
  scripts/      inference_auto.sh (autonomous mode), inference_copilot.sh (co-pilot mode)
  samples/      six UCSF-PDGM patients with their four sequences and pathology
  third_party/  HD-BET, for brain extraction of uploaded studies
  TOOL_REFERENCE.md   every field each tool returns: type, range and threshold
```

## Install

```bash
conda create -n glioagent python=3.10 && conda activate glioagent
pip install -r requirements.txt        # edit the cu130 tag to match your CUDA runtime
```

## Models

No model weights are distributed in this repository.

| model | role | where it comes from |
|---|---|---|
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | the language model that orchestrates the tools and writes the report | Hugging Face, loaded in-process on first use (about 54 GB in bf16) |
| [HD-GLIO](https://github.com/NeuroAI-HD/HD-GLIO) | tumor segmentation | its official parameters, fetched into `glioagent/tools/ckpts/hdglio/` on first use |
| [GlioMT](https://github.com/bys9595/GlioMT) | IDH, 1p/19q and grade | its official checkpoints, fetched into `glioagent/tools/ckpts/gliomt/` on first use |

`--with-qwen` fetches the language model before the first case; `HF_HOME` chooses where it is cached.

## Run

From the repository root:

```bash
bash glioagent/scripts/inference_auto.sh --list-samples
bash glioagent/scripts/inference_auto.sh    --sample UCSF_00000004_00000001
bash glioagent/scripts/inference_copilot.sh --sample UCSF_00000004_00000001
```

`inference_auto.sh` runs the autonomous mode and stops at the report; `inference_copilot.sh` continues into
the co-pilot. Each run writes the trace, the report and the segmentation into `--out-dir` (default
`glioagent/outputs/runs/<ID>`). For your own study, give the four sequences instead of `--sample`:

```bash
bash glioagent/scripts/inference_auto.sh --patient-id case1 \
    --t1 t1.nii.gz --t1ce t1ce.nii.gz --t2 t2.nii.gz --flair flair.nii.gz \
    --age 54 --sex female
```

A GPU is required. `--llm-gpu` and `--seg-gpu` place the language model and the imaging models.

## Sample data

`glioagent/samples/` holds six patients from the University of California San Francisco Preoperative
Diffuse Glioma MRI dataset (UCSF-PDGM), public through The Cancer Imaging Archive, so the pipeline can be
run end to end without a data-access application. `samples.csv` carries each patient's pathology. If you
use them, cite:

> Calabrese E, Villanueva-Meyer JE, Rudie JD, Rauschecker AM, Baid U, Bakas S, Cha S, Mongan JT, Hess CP.
> The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset. *Radiology: Artificial
> Intelligence*. 2022;4(6):e220058. <https://doi.org/10.1148/ryai.220058>

> Calabrese E, et al. The University of California San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM)
> [Data set]. The Cancer Imaging Archive. <https://doi.org/10.7937/tcia.bdgf-8v37>

Use of the data is governed by the terms of the [TCIA collection](https://doi.org/10.7937/tcia.bdgf-8v37).

## Acknowledgements

GlioAgent builds on the following work; we thank the authors for releasing it. Their code in this
repository keeps its own licence (Apache 2.0 for each).

- **HD-GLIO**, tumor segmentation — <https://github.com/NeuroAI-HD/HD-GLIO>, code in `glioagent/tools/hdglio/`.
  Kickingereder P, Isensee F, Tursunova I, et al. Automated quantitative tumor response assessment of MRI in
  neuro-oncology with artificial neural networks: a multicentre, retrospective study. *Lancet Oncol*.
  2019;20(5):728-740. <https://doi.org/10.1016/S1470-2045(19)30098-1>
- **GlioMT**, molecular subtype and grade — <https://github.com/bys9595/GlioMT>, code in `glioagent/tools/gliomt/`.
  Byeon Y, Park YW, Lee S, et al. Interpretable multimodal transformer for prediction of molecular subtypes
  and grades in adult-type diffuse gliomas. *npj Digit Med*. 2025;8:140.
  <https://doi.org/10.1038/s41746-025-01530-4>
- **HD-BET**, brain extraction — <https://github.com/MIC-DKFZ/HD-BET>, code in `glioagent/third_party/HD-BET/`.
  Isensee F, Schell M, Tursunova I, et al. Automated brain extraction of multi-sequence MRI using artificial
  neural networks. *Hum Brain Mapp*. 2019. <https://doi.org/10.1002/hbm.24750>
- **VASARI-auto**, VASARI features — <https://github.com/jamesruffle/vasari-auto>, code in `glioagent/tools/vasari/`.
  Ruffle JK, Mohinta S, Pegoretti Baruteau K, et al. VASARI-auto: equitable, efficient, and economical
  featurisation of glioma MRI. *NeuroImage: Clinical*. 2024;44:103668. <https://doi.org/10.1016/j.nicl.2024.103668>
- **Qwen3.5-27B**, the language model — <https://huggingface.co/Qwen/Qwen3.5-27B>.

## License

GlioAgent is released under the [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE)
license (CC BY-NC 4.0): it may be used, shared and adapted for non-commercial purposes with attribution.
The third-party code listed above, the model weights and the sample data keep their own terms.

GlioAgent is a research tool. It is not a medical device and must not be used for clinical decisions.
