#!/usr/bin/env python3
"""
ACL single-column (3.35") t-SNE grid from a precomputed embeddings_cache.npz

Supports any number of models (e.g. 9 -> 3x3, 12 -> 4x3).
Grid layout is always 3 columns wide (configurable with --ncols);
rows are computed automatically. Figure height scales with the number of rows.

Usage:
  python text_embedding_analysis_toegther_plot.py \
    --embeddings_npz /home/.../embeddings_cache_new_12_models.npz \
    --output_dir     /home/.../vctk_spk_plots_12models \
    [--max_models 12] [--ncols 3] [--seed 42] [--dump_metrics]

Outputs (in --output_dir):
  tsne_6speakers_singlecol.pdf/png
  tsne_all_speakers_singlecol.pdf/png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


# ------------------------- ACL single-column style -------------------------

ACL_SINGLECOL_IN = 3.35  # ~8.5 cm

def set_acl_singlecol_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,

        "font.family": "serif",
        "font.size": 7.0,
        "axes.titlesize": 6.8,
        "axes.labelsize": 7.0,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,

        "legend.frameon": False,
        "legend.fontsize": 5.8,

        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
    })


# ------------------------- Speaker labeling -------------------------

def get_speaker_label(speaker_id: str) -> str:
    accent_map = {
        "p234": "Scottish-W. Dumfries",
        "p238": "N. Irish-Belfast",
        "p248": "Indian",
        "p253": "Welsh-Cardiff",
        "p283": "Irish-Cork",
        "p294": "American-SF",
        "p314": "S. African-Cape Town",
        "p335": "NZ English",
        "p237": "Scottish-Fife",
        "p245": "Irish-Dublin",
        "p251": "Indian",
        "p260": "Scottish-Orkney",
        "p292": "N. Irish-Belfast",
        "p302": "Canadian-Montreal",
        "p326": "Australian-Sydney",
        "p347": "S. African-Johannesburg",
    }
    female_speakers = {"p234", "p238", "p248", "p253", "p283", "p294", "p314", "p335"}
    gender = "F" if speaker_id in female_speakers else "M"
    return f"{speaker_id}: {accent_map.get(speaker_id, 'Unknown')} ({gender})"


# ------------------------- Model title handling -------------------------

def canonical_model_key(name: str) -> str:
    s = name.strip().replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\(\)]", "", s)
    return s


def pretty_model_title(name: str) -> str:
    s = name.strip()

    mapping = {
        # ---- 9-model legacy names ----
        "Audio Flamingo 3":                          "Audio\nFlamingo 3",
        "MERaLiON":                                  "MERaLiON",
        "Qwen2-audio":                               "Qwen2-\nAudio",
        "Voxtral-3B":                                "Voxtral\n3B",
        "Voxtral-24B":                               "Voxtral\n24B",
        "SALMONN":                                   "SALMONN",
        "SALMONN(PSC)":                              "SALMONN\n(PSC)",
        "SALMONN(StyleInstructCapsDB) single-task":  "SALMONN\n(SIC-DB) ST",
        "SALMONN(StyleInstructCapsDB) multi-task":   "SALMONN\n(SIC-DB) MT",
        "SALMONN (StyleInstructCapsDB) single-task": "SALMONN\n(SIC-DB) ST",
        "SALMONN (StyleInstructCapsDB) multi-task":  "SALMONN\n(SIC-DB) MT",
        # ---- 12-model names ----
        "Audio-Flamingo-3":        "Audio\nFlamingo 3",
        "Voxtral-Mini-3B":         "Voxtral\nMini-3B",
        "Voxtral-Small-24B":       "Voxtral\nSmall-24B",
        "Qwen2-Audio-7B-Instruct": "Qwen2-Audio\n7B",
        "MOSS-Audio":              "MOSS-\nAudio",
        "Qwen3-Omni":              "Qwen3-\nOmni",
        "af-next":                 "AF-Next",
        "SALMONN (PSC)":           "SALMONN\n(PSC)",
        "SALMONN (SIC-DB) - ST":   "SALMONN\n(SIC-DB) ST",
        "SALMONN (SIC-DB) - MT":   "SALMONN\n(SIC-DB) MT",
    }
    return mapping.get(s, s)


# ------------------------- Cache loading -------------------------

def load_embedding_cache_npz(
    npz_path: str,
) -> Tuple[List[str], List[str], Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    data = np.load(npz_path, allow_pickle=True)
    if "model_keys" not in data or "model_names" not in data:
        raise ValueError(f"Invalid cache: missing model_keys/model_names in {npz_path}")

    model_keys = [str(x) for x in data["model_keys"].tolist()]
    model_names = [str(x) for x in data["model_names"].tolist()]
    seed = int(data["seed"][0]) if "seed" in data else 42

    emb_by_key: Dict[str, np.ndarray] = {}
    spk_by_key: Dict[str, np.ndarray] = {}

    for k in model_keys:
        ek, sk = f"embeddings__{k}", f"speaker_ids__{k}"
        if ek not in data or sk not in data:
            raise ValueError(f"Invalid cache: missing {ek} or {sk}")
        emb_by_key[k] = np.array(data[ek]).astype(np.float32, copy=False)
        spk_by_key[k] = np.array(data[sk], dtype=object)

    print(f"Loaded {npz_path} | models={len(model_keys)}")
    return model_keys, model_names, emb_by_key, spk_by_key, seed


# ------------------------- Metrics (optional JSON dump) -------------------------

def compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    total = 0
    for c in np.unique(cluster_labels):
        idx = cluster_labels == c
        counts = np.bincount(true_labels[idx])
        total += int(np.max(counts)) if len(counts) else 0
    return float(total) / float(len(true_labels)) if len(true_labels) else 0.0


def compute_clustering_metrics(
    embeddings: np.ndarray, speaker_ids: List[str], seed: int
) -> Dict[str, float]:
    uniq = sorted(set(speaker_ids))
    label_map = {sp: i for i, sp in enumerate(uniq)}
    y = np.array([label_map[s] for s in speaker_ids], dtype=int)
    k = len(uniq)

    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    c = km.fit_predict(embeddings)

    return {
        "ARI": float(adjusted_rand_score(y, c)),
        "NMI": float(normalized_mutual_info_score(y, c)),
        "Purity": float(compute_purity(y, c)),
        "Silhouette_cos": float(silhouette_score(embeddings, c, metric="cosine")) if len(set(c)) > 1 else 0.0,
        "DBI": float(davies_bouldin_score(embeddings, c)) if len(set(c)) > 1 else 0.0,
    }


# ------------------------- t-SNE + plotting -------------------------

def make_color_map(speakers: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {sp: cmap(i % 20) for i, sp in enumerate(speakers)}


def tsne_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n = len(embeddings)
    if n < 3:
        return np.zeros((n, 2), dtype=float)
    upper = max(5, min(30, (n - 1) // 3))
    perplexity = min(upper, n - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        random_state=seed,
        learning_rate="auto",
        verbose=0,
    )
    return tsne.fit_transform(embeddings)


def plot_panel(ax, emb2d, sp_ids, speakers_order, color_map, title: str) -> None:
    ax.set_title(pretty_model_title(title), pad=2.0, fontsize=6.8)
    ax.title.set_multialignment("center")
    ax.title.set_linespacing(0.92)

    for sp in speakers_order:
        m = sp_ids == sp
        if not np.any(m):
            continue
        ax.scatter(
            emb2d[m, 0], emb2d[m, 1],
            s=5.0,
            alpha=0.78,
            linewidths=0.0,
            color=color_map[sp],
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def add_legend_inside(fig, speakers, color_map, y: float, ncol: int) -> None:
    handles = [
        mpl.lines.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=3.8,
            markerfacecolor=color_map[sp],
            markeredgewidth=0,
        )
        for sp in speakers
    ]
    labels = [get_speaker_label(sp) for sp in speakers]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        handletextpad=0.35,
        columnspacing=0.85,
        labelspacing=0.35,
        borderaxespad=0.0,
        fontsize=5.8,
    )


def plot_grid_singlecol(
    all_data: List[dict],
    out_pdf: str,
    title: str,
    speakers_for_plot: List[str],
    speakers_for_legend: List[str],
    speaker_to_color: Dict[str, tuple],
    seed: int,
    ncols: int = 3,
    show_global_labels: bool = True,
) -> None:
    n_models = len(all_data)
    nrows = math.ceil(n_models / ncols)

    # Fixed absolute bottom space (inches) for legend + x-label.
    # The all-speaker legend is taller so it gets more room.
    bottom_abs = 0.72 if len(speakers_for_legend) <= 6 else 1.00

    # Per-row panel height in inches.
    row_h = 1.05

    fig_h = bottom_abs + nrows * row_h
    fig = plt.figure(figsize=(ACL_SINGLECOL_IN, fig_h))

    # Normalised bottom boundary of the gridspec.
    bottom = bottom_abs / fig_h

    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.06, right=0.995,
        top=0.94, bottom=bottom,
        wspace=0.22, hspace=0.28,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
    for i, ax in enumerate(axes):
        if i >= n_models:
            ax.axis("off")
            continue

        d = all_data[i]
        sp_ids = d["speaker_ids"]
        embs = d["embeddings"]

        mask = np.isin(sp_ids, speakers_for_plot)
        embs_f = embs[mask]
        sp_f = sp_ids[mask]

        emb2d = tsne_2d(embs_f, seed=seed)
        plot_panel(ax, emb2d, sp_f, speakers_for_plot, speaker_to_color, d["model_name"])

    # Optional figure title (prefer LaTeX caption instead).
    if title:
        fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=7.2)

    # Global axis labels in the bottom margin above the legend.
    if show_global_labels:
        fig.text(0.5, bottom - 0.035, "t-SNE 1", ha="center", va="center", fontsize=7.0)
        fig.text(0.018, 0.55, "t-SNE 2", ha="center", va="center", rotation="vertical", fontsize=7.0)

    # Legend inside the figure box (prevents LaTeX caption overlap).
    add_legend_inside(fig, speakers_for_legend, speaker_to_color, y=0.035, ncol=2)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {out_pdf} (+ .png)")


# ------------------------- Main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_npz", required=True, type=str)
    ap.add_argument("--output_dir",     required=True, type=str)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--max_models", type=int, default=12,
                    help="Cap number of model panels (default: 12).")
    ap.add_argument("--ncols",      type=int, default=3,
                    help="Columns in the panel grid (default: 3).")
    ap.add_argument("--dump_metrics", action="store_true",
                    help="Write metrics__<model>.json for each model.")
    args = ap.parse_args()

    set_acl_singlecol_style()

    model_keys, model_names, emb_by_key, spk_by_key, seed_in_cache = \
        load_embedding_cache_npz(args.embeddings_npz)
    seed = args.seed if args.seed is not None else seed_in_cache

    all_data: List[dict] = []
    all_speakers_set: set = set()

    for k, name in zip(model_keys, model_names):
        embs = emb_by_key[k]
        sp   = spk_by_key[k]
        all_speakers_set.update(sp.tolist())
        all_data.append({
            "model_key":   k,
            "model_name":  name,
            "embeddings":  embs,
            "speaker_ids": np.array(sp, dtype=object),
        })

        if args.dump_metrics:
            m = compute_clustering_metrics(embs, sp.tolist(), seed=seed)
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"metrics__{name}.json"), "w") as fh:
                json.dump(m, fh, indent=2)

    all_data = all_data[:args.max_models]
    all_speakers = sorted(all_speakers_set)
    speaker_to_color = make_color_map(all_speakers)

    # 3M+3F representative set (keep order stable)
    selected_6 = ["p245", "p347", "p251", "p234", "p248", "p294"]
    selected_6 = [s for s in selected_6 if s in all_speakers]

    os.makedirs(args.output_dir, exist_ok=True)

    plot_grid_singlecol(
        all_data=all_data,
        out_pdf=os.path.join(args.output_dir, "tsne_6speakers_singlecol.pdf"),
        title=None,
        speakers_for_plot=selected_6,
        speakers_for_legend=selected_6,
        speaker_to_color=speaker_to_color,
        seed=seed,
        ncols=args.ncols,
        show_global_labels=True,
    )

    plot_grid_singlecol(
        all_data=all_data,
        out_pdf=os.path.join(args.output_dir, "tsne_all_speakers_singlecol.pdf"),
        title=None,
        speakers_for_plot=all_speakers,
        speakers_for_legend=all_speakers,
        speaker_to_color=speaker_to_color,
        seed=seed,
        ncols=args.ncols,
        show_global_labels=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
