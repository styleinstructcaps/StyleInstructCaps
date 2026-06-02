#!/usr/bin/env bash
# Axis 8: Speaker clustering plots (multi-model t-SNE grid)
#
# Usage:
#   bash scripts/run_speaker_cluster_plots.sh MODELS_CSV OUTPUT_DIR [EMBEDDINGS_NPZ] [MAX_MODELS] [NCOLS]
#
# MODELS_CSV: one line per model → "Display Name,/path/to/vctk_results.json"
# If EMBEDDINGS_NPZ is omitted, builds cache at OUTPUT_DIR/embeddings_cache.npz first.
#
# Slurm hint: CPU or GPU, conda env spk_consistency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_CSV="${1:?Usage: $0 MODELS_CSV OUTPUT_DIR [EMBEDDINGS_NPZ] [MAX_MODELS] [NCOLS]}"
OUTPUT_DIR="${2:?Usage: $0 MODELS_CSV OUTPUT_DIR [EMBEDDINGS_NPZ] [MAX_MODELS] [NCOLS]}"
EMBEDDINGS_NPZ="${3:-${OUTPUT_DIR}/embeddings_cache.npz}"
MAX_MODELS="${4:-12}"
NCOLS="${5:-3}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spk_consistency

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${EMBEDDINGS_NPZ}" ]]; then
    echo "=== Building embeddings cache ==="
    python -u "${RELEASE_ROOT}/evaluation/build_embeddings_cache.py" \
        --models_csv "${MODELS_CSV}" \
        --output_npz "${EMBEDDINGS_NPZ}"
fi

echo "=== Rendering speaker cluster plots ==="
python -u "${RELEASE_ROOT}/evaluation/speaker_cluster_plots.py" \
    --embeddings_npz "${EMBEDDINGS_NPZ}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_models "${MAX_MODELS}" \
    --ncols "${NCOLS}" \
    --seed 42 \
    --dump_metrics

echo "Done. Plots → ${OUTPUT_DIR}/tsne_*.pdf"
