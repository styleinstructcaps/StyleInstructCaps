#!/usr/bin/env bash
# Axis 5: Speaker consistency (clustering metrics + t-SNE) on VCTK_speaker_prompts
#
# Usage:
#   bash scripts/run_speaker_consistency.sh INPUT_JSON OUTPUT_DIR
#
# Slurm hint: CPU or GPU, conda env spk_consistency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spk_consistency

mkdir -p "${OUTPUT_DIR}"

python -u "${RELEASE_ROOT}/evaluation/speaker_consistency.py" \
    --json_path "${INPUT_JSON}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done. Results → ${OUTPUT_DIR}/"
