# Input JSON Format for StyleInstructCaps Evaluation

Every evaluation script in this release consumes a **single JSON file** whose top level is a **list of per-utterance records**. Each record describes one audio clip, the model's generated speaking-style captions, and the reference metadata from the dataset.

See [`examples/sample_input.json`](examples/sample_input.json) for a concrete example.

## Top-level structure

```json
[
  {
    "wav_path": "/absolute/path/to/audio.wav",
    "status": "success",
    "generated_captions": { ... },
    "metadata": { ... }
  }
]
```

## Required fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `wav_path` | string | Absolute path to the audio file your model consumed during inference. Used as a stable utterance identifier in verification metrics. |
| `generated_captions` | object | Map from caption-task name → `{used_instruction, generated_caption}`. See below. |
| `metadata` | object | Reference row from the dataset (ground truth). Required for metadata groundedness, hallucination, and speaker metrics. |

## Optional fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `status` | string | `"success"` or an error code. Failed rows may be skipped by some metrics. |
| `ground_truth_caption` | string | SpeechCraft reference caption for Gigaspeech rows (also copied into `metadata` if missing). |

## `generated_captions` schema

Each key names a caption task. The value is always:

```json
{
  "used_instruction": "<instruction string passed to the model>",
  "generated_caption": "<model output>"
}
```

### Standard six-task keys (MetaSet, Gigaspeech)

Used when evaluating the six canonical style dimensions:

- `speaker_idiosyncratic_style`
- `situational_contextual_style`
- `expressive_emotional_style`
- `linguistic_pragmatic_style`
- `perceptual_listener_centric_style`
- `holistic_creative_synthesis`

### Multi-prompt keys (VCTK, OOD)

For configs with multiple prompts per utterance, keys follow:

```
prompt{N}_{task_name}
```

Examples:

- `prompt1_speaker_idiosyncratic_style`
- `prompt7_situational_contextual_style_instruction` → stored as caption key after stripping `promptN_` prefix

The instruction-following evaluator automatically strips `prompt\d+_` prefixes and matches against the six canonical task names.

## `metadata` schema

Copy the dataset row into `metadata`, preserving fields your evaluation needs:

| Field | Required for | Description |
| ----- | ------------ | ----------- |
| `source` | All | Upstream corpus id (`ears`, `VCTK`, `eval_giga`, …) |
| `relative_audio_path` | All | Path within the upstream corpus |
| `speakerid` | Speaker metrics | Speaker identifier |
| `transcription` | MetaSet judge | Utterance transcript |
| Style caption fields | MetaSet judge | e.g. `holistic_creative_synthesis`, `speaker_idiosyncratic_style`, … |
| Tag fields | MetaSet judge | e.g. `gender`, `accent`, `pitch`, `intrinsic_tags`, … |
| `ground_truth_caption` | Gigaspeech | SpeechCraft reference caption |

Do **not** include local absolute `audio_path` values from internal manifests when publishing results; `wav_path` on the top-level record is sufficient.

## Building input from StyleInstructCapsDB

Dataset: [StyleInstructCaps/StyleInstructCapsDB](https://huggingface.co/datasets/StyleInstructCaps/StyleInstructCapsDB)

```python
from datasets import load_dataset

REPO = "StyleInstructCaps/StyleInstructCapsDB"

metaset = load_dataset(REPO, "StyleInstructCaps-MetaSet", split="test")
vctk    = load_dataset(REPO, "VCTK_speaker_prompts", split="test")
ood     = load_dataset(REPO, "VCTK_unseen_prompts", split="test")
giga    = load_dataset(REPO, "Gigaspeech_SpeechCraft_captions", split="test")
```

### Workflow

1. **Load** the evaluation config you need (see table below).
2. **Resolve audio**: join `relative_audio_path` with your local copy of the upstream corpus (see dataset card for source mapping).
3. **Run inference**: for each instruction column in the row, call your audio language model and collect `(instruction, caption)` pairs.
4. **Assemble record**:
   - Set `wav_path` to the resolved absolute audio path.
   - Set `generated_captions[<task>] = {"used_instruction": ..., "generated_caption": ...}`.
   - Set `metadata` to the dataset row (as a plain dict).
5. **Save** the list of records as a JSON file. Pass that file to the evaluation scripts.

A minimal template script is in [`examples/manifest_to_input_template.py`](examples/manifest_to_input_template.py).

## Config → evaluation axis mapping

| HF config | Rows | Eval axes | Expected `generated_captions` keys |
| --------- | ---- | --------- | ---------------------------------- |
| `StyleInstructCaps-MetaSet` | 434 | 1, 2, 3 | Six standard task keys |
| `VCTK_speaker_prompts` | 320 | 4, 5, 8 | `prompt1`…`prompt10` × speaker_idiosyncratic_style |
| `VCTK_unseen_prompts` | 32 | 7 | `prompt1`…`prompt36` × all six tasks |
| `Gigaspeech_SpeechCraft_captions` | 600 | 6 (1+2+3) | `holistic_creative_synthesis` (single task) |

## Validation checklist

Before running evaluation, confirm:

- [ ] Top level is a JSON **array** (not `{"annotation": [...]}`).
- [ ] Every record has `wav_path`, `generated_captions`, and `metadata`.
- [ ] Every caption entry has both `used_instruction` and `generated_caption`.
- [ ] MetaSet / Gigaspeech records include rich `metadata` for the Qwen judge.
- [ ] VCTK records include `metadata.speakerid` for speaker verification and consistency.
- [ ] File paths in `wav_path` exist on the machine running evaluation.
