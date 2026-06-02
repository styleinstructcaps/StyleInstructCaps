#!/usr/bin/env python3
"""
Template: convert StyleInstructCapsDB rows + model outputs into evaluation input JSON.

Replace `run_your_audio_lm()` with your model's inference call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from datasets import load_dataset

REPO = "StyleInstructCaps/StyleInstructCapsDB"

# Map upstream source → local audio root on your machine.
AUDIO_ROOTS: Dict[str, Path] = {
    "VCTK": Path("/path/to/VCTK"),
    "ears": Path("/path/to/ears"),
    "eval_giga": Path("/path/to/gigaspeech_eval"),
    # add other sources from the dataset card
}

STANDARD_TASKS = [
    "speaker_idiosyncratic_style",
    "situational_contextual_style",
    "expressive_emotional_style",
    "linguistic_pragmatic_style",
    "perceptual_listener_centric_style",
    "holistic_creative_synthesis",
]


def resolve_wav(row: dict) -> Path:
    root = AUDIO_ROOTS[row["source"]]
    return (root / row["relative_audio_path"]).resolve()


def run_your_audio_lm(wav_path: Path, instruction: str) -> str:
    """Replace with your audio LM inference."""
    raise NotImplementedError("Wire up your model here")


def build_metaset_record(
    row: dict,
    infer_fn: Callable[[Path, str], str] = run_your_audio_lm,
) -> dict:
    wav = resolve_wav(row)
    generated: Dict[str, dict] = {}

    for task in STANDARD_TASKS:
        instr_key = f"{task}_instruction"
        if instr_key not in row:
            continue
        instruction = row[instr_key]
        caption = infer_fn(wav, instruction)
        generated[task] = {
            "used_instruction": instruction,
            "generated_caption": caption,
        }

    metadata = {k: v for k, v in row.items() if not k.endswith("_instruction")}
    return {
        "wav_path": str(wav),
        "status": "success",
        "generated_captions": generated,
        "metadata": metadata,
    }


def build_vctk_prompt_record(
    row: dict,
    infer_fn: Callable[[Path, str], str] = run_your_audio_lm,
    num_prompts: int = 10,
) -> dict:
    wav = resolve_wav(row)
    generated: Dict[str, dict] = {}

    for n in range(1, num_prompts + 1):
        instr_key = f"prompt{n}_speaker_idiosyncratic_style_instruction"
        if instr_key not in row:
            continue
        instruction = row[instr_key]
        caption = infer_fn(wav, instruction)
        cap_key = f"prompt{n}_speaker_idiosyncratic_style"
        generated[cap_key] = {
            "used_instruction": instruction,
            "generated_caption": caption,
        }

    metadata = {
        "source": row["source"],
        "relative_audio_path": row["relative_audio_path"],
        "speakerid": row.get("speakerid"),
    }
    return {
        "wav_path": str(wav),
        "status": "success",
        "generated_captions": generated,
        "metadata": metadata,
    }


def main() -> None:
    ds = load_dataset(REPO, "StyleInstructCaps-MetaSet", split="test")
    records: List[dict] = []
    for row in ds:
        records.append(build_metaset_record(dict(row)))

    out = Path("my_model_metaset_results.json")
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
