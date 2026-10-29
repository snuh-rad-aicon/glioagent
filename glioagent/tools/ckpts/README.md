# Checkpoints

No weights are distributed in this repository. They are fetched from the official releases into this
folder on the first run, or can be placed here by hand:

| folder | model | official source |
|---|---|---|
| `hdglio/` | HD-GLIO tumour segmentation | <https://github.com/NeuroAI-HD/HD-GLIO> (parameters on Zenodo, record 4014850) |
| `gliomt/` | GlioMT IDH, 1p/19q and grade heads: `idh_vitb.pth`, `1p19q_swins.pth`, `grade_swins.pth` | <https://github.com/bys9595/GlioMT> |

The language model is not stored here; it is loaded from the Hugging Face cache
(<https://huggingface.co/Qwen/Qwen3.5-27B>).
