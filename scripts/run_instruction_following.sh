#!/usr/bin/env bash
# Axis 3: Instruction following on StyleInstructCaps-MetaSet
# Also used for axes 6 (Gigaspeech) and 7 (VCTK unseen prompts).
#
# Usage:
#   bash scripts/run_instruction_following.sh INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]
#
# Slurm hint: 1+ GPU (A100/H100), conda env spk_style_eval_env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
MODEL_NAME="${3:-Qwen/Qwen3-32B}"
BATCH_SIZE="${4:-4}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spk_style_eval_env

mkdir -p "${OUTPUT_DIR}"

python -u "${RELEASE_ROOT}/evaluation/instruction_following.py" \
    --input_file "${INPUT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file instruction_following.json \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}"

echo "Done. Results → ${OUTPUT_DIR}/instruction_following.json"
