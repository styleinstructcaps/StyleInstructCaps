#!/usr/bin/env python3
"""
Build a multi-model embeddings_cache.npz for speaker clustering plots.

For each model JSON it:
  1. Extracts all generated captions + speaker IDs
  2. Computes L2-normalised e5-base-v2 embeddings (prefix "query: ")
  3. Packs everything into a single .npz with the schema expected by
     speaker_cluster_plots.py:

     model_keys   – 1-D object array of canonical slugs
     model_names  – 1-D object array of display names
     seed         – int64 scalar array [seed]
     embeddings__{key}   – float32 (N, 768) per model
     speaker_ids__{key}  – object  (N,)    per model

Usage:
  python build_embeddings_cache.py \
      --models_csv models.csv \
      --output_npz /path/to/embeddings_cache.npz \
      [--batch_size 32] [--device cuda] [--seed 42]

models.csv format (one model per line, no header):
  Model Display Name,/path/to/model_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


def canonical_model_key(name: str) -> str:
    s = name.strip().replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\(\)]", "", s)
    return s


def load_models_csv(path: str) -> List[Tuple[str, str]]:
    """Load (display_name, json_path) pairs from a CSV file."""
    models: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Support comma-separated or tab-separated
            if "\t" in line and "," not in line:
                parts = line.split("\t", 1)
            else:
                reader = csv.reader([line])
                parts = next(reader)
            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{line_no}: expected 'name,json_path' but got: {line!r}"
                )
            name, json_path = parts[0].strip(), parts[1].strip()
            if name and json_path:
                models.append((name, json_path))
    if not models:
        raise ValueError(f"No models found in {path}")
    return models


def extract_captions_and_speakers(json_path: str) -> Tuple[List[str], List[str]]:
    """
    Handles the shared JSON format:
      entry["metadata"]["speakerid"]
      entry["generated_captions"][prompt_key]["generated_caption"]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotation" in data:
        data = data["annotation"]

    captions: List[str] = []
    speaker_ids: List[str] = []

    for entry in data:
        speaker_id = entry.get("metadata", {}).get("speakerid") or entry.get("speakerid")
        if not speaker_id:
            continue

        caption_groups = entry.get("generated_captions", {})
        if not isinstance(caption_groups, dict):
            continue

        for cap_info in caption_groups.values():
            if not isinstance(cap_info, dict):
                continue
            raw = cap_info.get("generated_caption", "")
            caption = raw.replace("<s>", "").replace("</s>", "").strip()
            if caption:
                captions.append(caption)
                speaker_ids.append(speaker_id)

    return captions, speaker_ids


class E5Encoder:
    MODEL_NAME = "intfloat/e5-base-v2"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {self.MODEL_NAME} on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, captions: List[str], batch_size: int = 32) -> np.ndarray:
        texts = [f"query: {c}" for c in captions]
        parts: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"]
            pooled = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            pooled = pooled.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            pooled = F.normalize(pooled, p=2, dim=1)
            parts.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models_csv",
        required=True,
        help="CSV file: one 'display_name,json_path' per line",
    )
    ap.add_argument(
        "--output_npz",
        required=True,
        help="Where to write the cache (e.g. .../embeddings_cache.npz)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_table = load_models_csv(args.models_csv)
    encoder = E5Encoder(device=args.device)

    payload: Dict[str, np.ndarray] = {}
    model_keys_list: List[str] = []
    model_names_list: List[str] = []

    for display_name, json_path in model_table:
        key = canonical_model_key(display_name)
        print(f"\n[{display_name}]  key={key}")
        print(f"  → {json_path}")

        if not os.path.exists(json_path):
            print("  WARNING: file not found, skipping.")
            continue

        captions, speaker_ids = extract_captions_and_speakers(json_path)
        print(f"  captions={len(captions)}  speakers={len(set(speaker_ids))}")

        if not captions:
            print("  WARNING: no captions extracted, skipping.")
            continue

        embeddings = encoder.encode(captions, batch_size=args.batch_size)
        print(f"  embeddings shape: {embeddings.shape}")

        model_keys_list.append(key)
        model_names_list.append(display_name)
        payload[f"embeddings__{key}"] = embeddings
        payload[f"speaker_ids__{key}"] = np.array(speaker_ids, dtype=object)

    if not model_keys_list:
        raise RuntimeError("No models were processed successfully.")

    payload["model_keys"] = np.array(model_keys_list, dtype=object)
    payload["model_names"] = np.array(model_names_list, dtype=object)
    payload["seed"] = np.array([args.seed], dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    print(f"\nSaved → {args.output_npz}  ({len(model_keys_list)} models)")
    print("Done.")


if __name__ == "__main__":
    main()
