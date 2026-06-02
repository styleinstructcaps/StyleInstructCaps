# StyleInstructCaps Evaluation Toolkit

Official evaluation code for **instruction-following speaking-style captioning** with audio language models.

Evaluate your model's generated style captions against the [StyleInstructCapsDB](https://huggingface.co/datasets/StyleInstructCaps/StyleInstructCapsDB) benchmark across eight axes: metadata groundedness, hallucination, instruction following, speaker verification, speaker consistency, unseen-dataset generalization, out-of-domain prompts, and multi-model speaker clustering plots.

**License:** [CC BY-NC 4.0](LICENSE)

## Overview

This repository is **standalone**. All evaluation code lives under `evaluation/` and does not depend on any private project paths. You provide:

1. Model inference outputs as a JSON file (see [DATA_FORMAT.md](DATA_FORMAT.md))
2. Local copies of upstream audio corpora referenced by the dataset

The Qwen3-32B LLM-as-judge metrics (axes 1–3, 6–7) and e5-base-v2 speaker metrics (axes 4–5, 8) match the methodology in the StyleInstructCaps paper.

## Dataset

Metadata and evaluation splits are hosted on Hugging Face:

**[StyleInstructCaps/StyleInstructCapsDB](https://huggingface.co/datasets/StyleInstructCaps/StyleInstructCapsDB)**

```python
from datasets import load_dataset

REPO = "StyleInstructCaps/StyleInstructCapsDB"

metaset = load_dataset(REPO, "StyleInstructCaps-MetaSet", split="test")       # 434
vctk    = load_dataset(REPO, "VCTK_speaker_prompts", split="test")            # 320
ood     = load_dataset(REPO, "VCTK_unseen_prompts", split="test")             # 32
giga    = load_dataset(REPO, "Gigaspeech_SpeechCraft_captions", split="test") # 600
in_dom  = load_dataset(REPO, "In_domain_balanced_set", split="test")          # 400
```

Audio is **not** included in the dataset. Resolve each row's `relative_audio_path` against the upstream corpus listed in the [dataset card](hf_dataset/README.md).

## Input format

Every evaluation script consumes one JSON file: a list of per-utterance records with `wav_path`, `generated_captions`, and `metadata`.

**Full specification:** [DATA_FORMAT.md](DATA_FORMAT.md)  
**Example:** [examples/sample_input.json](examples/sample_input.json)  
**Template script:** [examples/manifest_to_input_template.py](examples/manifest_to_input_template.py)

Minimal record shape:

```json
{
  "wav_path": "/path/to/audio.wav",
  "status": "success",
  "generated_captions": {
    "holistic_creative_synthesis": {
      "used_instruction": "Combine all elements...",
      "generated_caption": "A calm female speaker..."
    }
  },
  "metadata": {
    "source": "ears",
    "relative_audio_path": "p026/emo_relief_freeform.wav",
    "speakerid": "p026",
    "transcription": "..."
  }
}
```

## Installation

Two conda environments are required (different dependency profiles):

```bash
# LLM-as-judge (Qwen3-32B): axes 1, 2, 3, 6, 7
conda env create -f envs/spk_style_eval_env.yaml
conda activate spk_style_eval_env

# Speaker metrics (e5-base-v2): axes 4, 5, 8
conda env create -f envs/spk_consistency.yaml
conda activate spk_consistency
```

**Hardware notes:**
- Qwen3-32B judge: multi-GPU recommended (~64 GB VRAM total). Set `CUDA_VISIBLE_DEVICES` as needed.
- e5-base-v2 metrics: GPU recommended; CPU works for small runs.
- First run downloads model weights from Hugging Face. Set `HF_HOME` to a scratch volume if needed.

## Quick start

Run a single axis (MetaSet metadata groundedness):

```bash
bash scripts/run_metadata_groundedness.sh \
    /path/to/my_model_metaset.json \
    /path/to/output_dir
```

Run all eight axes:

```bash
bash scripts/eval_all.sh \
    /path/to/metaset_results.json \
    /path/to/vctk_results.json \
    /path/to/gigaspeech_results.json \
    /path/to/ood_results.json \
    /path/to/models.csv \
    /path/to/output_root
```

`models.csv` format (one model per line, for axis 8):

```
My Model,/path/to/vctk_results.json
Baseline,/path/to/baseline_vctk_results.json
```

## Evaluation axes

| Axis | Metric | Dataset config | Script | Env |
| ---- | ------ | -------------- | ------ | --- |
| 1 | Metadata groundedness | `StyleInstructCaps-MetaSet` | `scripts/run_metadata_groundedness.sh` | `spk_style_eval_env` |
| 2 | Hallucination | `StyleInstructCaps-MetaSet` | `scripts/run_hallucination.sh` | `spk_style_eval_env` |
| 3 | Instruction following | `StyleInstructCaps-MetaSet` | `scripts/run_instruction_following.sh` | `spk_style_eval_env` |
| 4 | Speaker caption verification | `VCTK_speaker_prompts` | `scripts/run_speaker_verification.sh` | `spk_consistency` |
| 5 | Speaker consistency | `VCTK_speaker_prompts` | `scripts/run_speaker_consistency.sh` | `spk_consistency` |
| 6 | Unseen eval (groundedness + hallucination + IF) | `Gigaspeech_SpeechCraft_captions` | `scripts/run_unseen_eval.sh` | `spk_style_eval_env` |
| 7 | OOD prompt analysis (IF only) | `VCTK_unseen_prompts` | `scripts/run_ood_prompts.sh` | `spk_style_eval_env` |
| 8 | Speaker clustering plots | `VCTK_speaker_prompts` (multi-model) | `scripts/run_speaker_cluster_plots.sh` | `spk_consistency` |

### Output files

| Axis | Primary outputs |
| ---- | --------------- |
| 1 | `metadata_groundedness.json` — mean similarity score (1–10) |
| 2 | `hallucination.json` — mean hallucination count, severity distribution |
| 3 | `instruction_following.json` — mean accuracy (1–10), pass rate |
| 4 | `*_metrics.txt` — EER, minDCF, AUC |
| 5 | `clustering_metrics.txt`, `tsne_6speakers.pdf`, `tsne_all_speakers.pdf` |
| 6 | Same as 1–3 under `gigaspeech/` |
| 7 | `vctk_unseen_prompts/instruction_following.json` |
| 8 | `tsne_6speakers_singlecol.pdf`, `tsne_all_speakers_singlecol.pdf` |

## Repository layout

```
release/
├── README.md              ← you are here
├── DATA_FORMAT.md         ← input JSON specification
├── LICENSE
├── envs/                  ← conda environment definitions
├── evaluation/            ← Python evaluation modules
├── scripts/               ← per-axis launcher shell scripts
├── examples/              ← sample input + template script
└── hf_dataset/            ← Hugging Face dataset upload artifacts
    ├── README.md          ← dataset card
    ├── data/              ← parquet shards
    └── scripts/           ← prepare_release.py, upload_to_hf.py
```

## Hugging Face dataset upload

To regenerate or re-upload the StyleInstructCapsDB parquet release:

```bash
pip install -r hf_dataset/scripts/requirements.txt
python hf_dataset/scripts/prepare_release.py
python hf_dataset/scripts/upload_to_hf.py --org StyleInstructCaps --name StyleInstructCapsDB
```

## Citation

```bibtex
@article{styleinstructcaps2026,
  title   = {StyleInstructCaps: Instruction-Following Speaking Style Captioning},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## Acknowledgments

- Judge model: [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- Text embeddings: [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
