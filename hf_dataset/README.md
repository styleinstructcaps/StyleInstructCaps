---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-speech
configs:
- config_name: StyleInstructCaps
  data_files:
  - split: train
    path: data/StyleInstructCaps/train-*.parquet
  - split: eval
    path: data/StyleInstructCaps/eval-*.parquet
- config_name: StyleInstructCaps-MetaSet
  data_files:
  - split: test
    path: data/StyleInstructCaps-MetaSet/test-*.parquet
- config_name: In_domain_balanced_set
  data_files:
  - split: test
    path: data/In_domain_balanced_set/test-*.parquet
- config_name: VCTK_speaker_prompts
  data_files:
  - split: test
    path: data/VCTK_speaker_prompts/test-*.parquet
- config_name: VCTK_unseen_prompts
  data_files:
  - split: test
    path: data/VCTK_unseen_prompts/test-*.parquet
- config_name: Gigaspeech_SpeechCraft_captions
  data_files:
  - split: test
    path: data/Gigaspeech_SpeechCraft_captions/test-*.parquet
---

# StyleInstructCaps

Style-captioned speech metadata for speaking-style captioning and instruction-following evaluation.

## Overview

This dataset release contains speaking-style captions, caption-generation instructions (when available), and minimal utterance metadata for speech captioning research. Each config includes only:

- `source`, `relative_audio_path`, `transcription`, `speakerid` (when present in the source manifest)
- Speaking-style caption fields (when present)
- Instruction fields for caption generation (when present)

Configs included:

- **StyleInstructCaps** — large-scale training and evaluation splits
- **StyleInstructCaps-MetaSet** (434 utterances)
- **In-domain balanced set** — accent/gender-balanced in-distribution evaluation set
- **VCTK speaker prompts** — VCTK evaluation with 10 speaker-idiosyncratic style prompts
- **VCTK (unseen prompts)** — VCTK out-of-distribution prompt evaluation (32 utterances, 36 prompts per row)
- **Gigaspeech (with SpeechCraft captions)** — unseen GigaSpeech evaluation set with SpeechCraft reference captions

This is a **metadata-only** release. Audio is **not** redistributed. Each row includes a `relative_audio_path` that should be resolved against the corresponding upstream audio corpus listed below.

## License

All resources are released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Usage

```python
from datasets import load_dataset

REPO = "StyleInstructCaps/StyleInstructCapsDB"

train = load_dataset(REPO, "StyleInstructCaps", split="train")
eval_split = load_dataset(REPO, "StyleInstructCaps", split="eval")
metaset = load_dataset(REPO, "StyleInstructCaps-MetaSet", split="test")
in_domain = load_dataset(REPO, "In_domain_balanced_set", split="test")
vctk_speaker = load_dataset(REPO, "VCTK_speaker_prompts", split="test")
vctk_unseen = load_dataset(REPO, "VCTK_unseen_prompts", split="test")
gigaspeech = load_dataset(REPO, "Gigaspeech_SpeechCraft_captions", split="test")

print(train[0])
```

## Dataset Structure

Each config is stored as parquet files under `data/<config>/`. Tags, phonemes, SNR, gender/accent labels, and other non-caption metadata from the source manifests are **not** included in this release.

### Shared release columns

These columns appear in a config when present in the source manifest:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `source` | string | Upstream dataset identifier |
| `relative_audio_path` | string | Relative path to the audio file within the upstream corpus |
| `transcription` | string | Transcript of the utterance |
| `speakerid` | string | Speaker identifier |

Style caption and instruction columns are config-specific; see below.

### StyleInstructCaps

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `train` | 1,031,840 | Full-scale training set (11 parquet shards) |
| `eval` | 14,756 | Held-out evaluation set with instructions |

**Style caption columns (train and eval):**

| Column | Type | Description |
| ------ | ---- | ----------- |
| `speaker_idiosyncratic_style` | string | Speaker identity style caption |
| `situational_contextual_style` | string | Situational/contextual style caption |
| `expressive_emotional_style` | string | Expressive/emotional style caption |
| `linguistic_pragmatic_style` | string | Linguistic/pragmatic style caption |
| `perceptual_listener_centric_style` | string | Listener-centric style caption |
| `holistic_creative_synthesis` | string | Holistic creative synthesis caption |

**Instruction columns (eval only):**

| Column | Type | Description |
| ------ | ---- | ----------- |
| `speaker_idiosyncratic_style_instruction` | string | Instruction for speaker style caption |
| `situational_contextual_style_instruction` | string | Instruction for situational style |
| `expressive_emotional_style_instruction` | string | Instruction for expressive style |
| `linguistic_pragmatic_style_instruction` | string | Instruction for linguistic style |
| `perceptual_listener_centric_style_instruction` | string | Instruction for listener-centric style |
| `holistic_creative_synthesis_instruction` | string | Instruction for holistic synthesis |

### StyleInstructCaps-MetaSet

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `test` | 434 | StyleInstructCaps-MetaSet evaluation subset |

Schema matches **StyleInstructCaps/eval** (base columns, all style caption fields, and all six `*_instruction` columns).

### In-domain balanced set

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `test` | 400 | Accent/gender-balanced in-distribution evaluation set |

**Columns:** base metadata (`source`, `relative_audio_path`, `speakerid`) plus `prompt1_speaker_idiosyncratic_style_instruction` through `prompt10_speaker_idiosyncratic_style_instruction`.

### VCTK speaker prompts

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `test` | 320 | VCTK evaluation with 10 speaker-idiosyncratic style prompts |

**Columns:** base metadata (`source`, `relative_audio_path`, `speakerid`) plus `prompt1_speaker_idiosyncratic_style_instruction` through `prompt10_speaker_idiosyncratic_style_instruction`.

### VCTK (unseen prompts)

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `test` | 32 | VCTK OOD prompt evaluation (36 instruction prompts per utterance) |

**Columns:** base metadata (`source`, `relative_audio_path`, `speakerid`) plus `prompt1_speaker_idiosyncratic_style_instruction` through `prompt36_holistic_creative_synthesis_style_instruction`.

### Gigaspeech (with SpeechCraft captions)

| Split | Rows | Description |
| ----- | ---- | ----------- |
| `test` | 600 | Unseen GigaSpeech evaluation set |

**Columns:**

| Column | Type | Description |
| ------ | ---- | ----------- |
| `source` | string | Upstream dataset identifier (`eval_giga`) |
| `relative_audio_path` | string | Relative path within the evaluation GigaSpeech subset |
| `ground_truth_caption` | string | SpeechCraft reference style caption |
| `holistic_creative_synthesis_instruction` | string | Instruction for holistic caption generation |

## Audio Sources

Map each row's `source` and `relative_audio_path` to the corresponding upstream corpus:

| `source` value | Upstream audio corpus |
| -------------- | --------------------- |
| `emilia` | [Emilia-Dataset (EN)](https://huggingface.co/datasets/amphion/Emilia-Dataset) |
| `voxceleb` | [VoxCeleb / VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| `VCTK` | [VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/2950) |
| `ears` | [EARS Dataset](https://github.com/facebookresearch/ears_dataset) |
| `expresso` | [Expresso](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset) |
| `eval_giga` | [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) (evaluation subset) |
