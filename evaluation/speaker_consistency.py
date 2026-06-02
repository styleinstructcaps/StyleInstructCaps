#!/usr/bin/env python3
"""
Text Embedding Clustering Evaluation Script (ACL-ready visualization)

This script:
1) Extracts speaking-style captions and speaker labels
2) Computes text embeddings using intfloat/e5-base-v2
3) Clusters embeddings using K-means (K = #speakers)
4) Evaluates clustering quality w.r.t ground-truth speakers
5) Computes embedding geometry metrics
6) Visualizes embeddings using t-SNE (6 speakers + all speakers)

⚠️ Core logic unchanged. Only visualization & I/O improved.
"""

# =============================================================================
# Imports
# =============================================================================

import json
import os
import argparse
import warnings
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# =============================================================================
# Matplotlib styling (ACL / camera-ready)
# =============================================================================

mpl.rcParams.update({
    "figure.figsize": (7.2, 6.4),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,

    "legend.frameon": False,
    "legend.fontsize": 10,

    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Color-blind safe palette (ACL-safe)
TSNE_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#000000"
]

# =============================================================================
# Data Loading
# =============================================================================

def extract_captions_and_speakers(
    json_path: str,
    speaker_filter: str = None
) -> Tuple[List[str], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions, speaker_ids = [], []

    for entry in data:
        speaker_id = entry.get("metadata", {}).get("speakerid")
        if speaker_filter and speaker_id != speaker_filter:
            continue

        caption_groups = entry.get("generated_captions", {})
        if not isinstance(caption_groups, dict):
            continue

        for caption_info in caption_groups.values():
            if not isinstance(caption_info, dict):
                continue
            raw_caption = caption_info.get("generated_caption", "")
            caption = raw_caption.replace("<s>", "").replace("</s>", "").strip()
            if caption and speaker_id:
                captions.append(caption)
                speaker_ids.append(speaker_id)

    print(f"Extracted {len(captions)} captions")
    print(f"Unique speakers: {len(set(speaker_ids))}")
    return captions, speaker_ids

# =============================================================================
# Embedding Utilities
# =============================================================================

def average_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

def compute_embeddings(
    captions: List[str],
    batch_size: int = 32,
    device: str = None
) -> np.ndarray:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "intfloat/e5-base-v2"

    print(f"Loading model {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    captions = [f"query: {c}" for c in captions]
    all_embeddings = []

    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# =============================================================================
# Metrics
# =============================================================================

def compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    total = 0
    for c in np.unique(cluster_labels):
        idx = cluster_labels == c
        counts = np.bincount(true_labels[idx])
        total += np.max(counts)
    return total / len(true_labels)

def compute_clustering_metrics(
    embeddings: np.ndarray,
    speaker_ids: List[str]
) -> Dict[str, float]:

    unique_speakers = sorted(set(speaker_ids))
    label_map = {sp: i for i, sp in enumerate(unique_speakers)}
    true_labels = np.array([label_map[s] for s in speaker_ids])
    n_speakers = len(unique_speakers)

    kmeans = KMeans(n_clusters=n_speakers, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    metrics = {}
    metrics["Adjusted_Rand_Index"] = adjusted_rand_score(true_labels, cluster_labels)
    metrics["Normalized_Mutual_Information"] = normalized_mutual_info_score(
        true_labels, cluster_labels
    )
    metrics["Purity"] = compute_purity(true_labels, cluster_labels)

    metrics["Silhouette_Score_Cosine"] = silhouette_score(
        embeddings, cluster_labels, metric="cosine"
    )
    metrics["Davies_Bouldin_Index"] = davies_bouldin_score(
        embeddings, cluster_labels
    )

    # Geometry
    centroids, intra_vars = [], []
    for i in range(n_speakers):
        emb = embeddings[true_labels == i]
        centroids.append(emb.mean(axis=0))
        intra_vars.append(np.mean(np.var(emb, axis=0)))

    centroids = np.vstack(centroids)
    intra = float(np.mean(intra_vars))

    inter_dists = []
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

    inter = float(np.mean(inter_dists))

    metrics["Intra_class_variance"] = intra
    metrics["Inter_class_separation"] = inter
    metrics["Fisher_Ratio"] = inter / intra if intra > 0 else 0.0

    return metrics

# =============================================================================
# Speaker labeling (unchanged)
# =============================================================================

def get_speaker_label(speaker_id: str) -> str:
    accent_map = {
        "p234": "Scottish – West Dumfries",
        "p238": "Northern Irish – Belfast",
        "p248": "Indian",
        "p253": "Welsh – Cardiff",
        "p283": "Irish – Cork",
        "p294": "American – San Francisco",
        "p314": "South African – Cape Town",
        "p335": "New Zealand English",
        "p237": "Scottish – Fife",
        "p245": "Irish – Dublin",
        "p251": "Indian",
        "p260": "Scottish – Orkney",
        "p292": "Northern Irish – Belfast",
        "p302": "Canadian – Montreal",
        "p326": "Australian English – Sydney",
        "p347": "South African – Johannesburg",
    }

    female_speakers = {"p234","p238","p248","p253","p283","p294","p314","p335"}
    gender = "female" if speaker_id in female_speakers else "male"
    return f"{accent_map.get(speaker_id, 'Unknown')} {gender}"

# =============================================================================
# Visualization
# =============================================================================

def plot_tsne(
    embeddings: np.ndarray,
    speaker_ids: np.ndarray,
    title: str,
    out_png: str,
    out_pdf: str
):
    unique_speakers = sorted(np.unique(speaker_ids))

    tsne = TSNE(
        n_components=2,
        perplexity=min(20, len(embeddings) // 3),
        metric="cosine",
        init="pca",
        random_state=42,
        verbose=1
    )

    emb_2d = tsne.fit_transform(embeddings)

    plt.figure()
    for i, sp in enumerate(unique_speakers):
        idx = speaker_ids == sp
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=38,
            alpha=0.75,
            color=TSNE_COLORS[i % len(TSNE_COLORS)],
            label=get_speaker_label(sp)
        )

    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3
    )

    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved t-SNE → {out_png}, {out_pdf}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--speaker_id",
        default=None,
        help="If set, only use captions for this speaker (e.g., p234)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    captions, speaker_ids = extract_captions_and_speakers(
        args.json_path,
        speaker_filter=args.speaker_id
    )

    if not captions:
        raise ValueError("No captions found for the given criteria.")

    embeddings = compute_embeddings(captions)

    unique_speakers = sorted(set(speaker_ids))
    if len(unique_speakers) < 2:
        print(
            "Only one speaker found; skipping clustering metrics and t-SNE plots."
        )
        return

    metrics = compute_clustering_metrics(embeddings, speaker_ids)

    print("\n========== METRICS ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("=============================\n")

    with open(os.path.join(args.output_dir, "clustering_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    speaker_ids = np.array(speaker_ids)

    # ------------------ 6-speaker plot ------------------
    male = ["p245", "p347", "p251"]
    female = ["p234", "p248", "p294"]
    selected = male + female

    mask = np.isin(speaker_ids, selected)

    plot_tsne(
        embeddings[mask],
        speaker_ids[mask],
        "t-SNE of Caption Embeddings (3 Male + 3 Female)",
        os.path.join(args.output_dir, "tsne_6speakers.png"),
        os.path.join(args.output_dir, "tsne_6speakers.pdf")
    )

    # ------------------ all-speaker plot ------------------
    plot_tsne(
        embeddings,
        speaker_ids,
        "t-SNE of Caption Embeddings (All Speakers)",
        os.path.join(args.output_dir, "tsne_all_speakers.png"),
        os.path.join(args.output_dir, "tsne_all_speakers.pdf")
    )

    print("Done.")

if __name__ == "__main__":
    main()
