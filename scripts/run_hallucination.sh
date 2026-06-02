#!/usr/bin/env bash
# Axis 2: Hallucination on StyleInstructCaps-MetaSet
#
# Usage:
#   bash scripts/run_hallucination.sh INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE] [NUM_GPUS]
#
# Slurm hint: 1+ GPU (A100/H100), conda env spk_style_eval_env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE] [NUM_GPUS]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE] [NUM_GPUS]}"
MODEL_NAME="${3:-Qwen/Qwen3-32B}"
BATCH_SIZE="${4:-3}"
NUM_GPUS="${5:-}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spk_style_eval_env

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${NUM_GPUS}" ]]; then
    EXTRA_ARGS+=(--num_gpus "${NUM_GPUS}")
fi

python -u "${RELEASE_ROOT}/evaluation/hallucination.py" \
    --input_file "${INPUT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file hallucination.json \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Results → ${OUTPUT_DIR}/hallucination.json"
