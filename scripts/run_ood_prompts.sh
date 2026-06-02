#!/usr/bin/env bash
# Axis 7: Out-of-domain prompt analysis (VCTK_unseen_prompts)
# Runs instruction following only.
#
# Usage:
#   bash scripts/run_ood_prompts.sh INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]
#
# Slurm hint: 1+ GPU (A100/H100), conda env spk_style_eval_env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
MODEL_NAME="${3:-Qwen/Qwen3-32B}"
BATCH_SIZE="${4:-4}"

OOD_DIR="${OUTPUT_DIR}/vctk_unseen_prompts"
mkdir -p "${OOD_DIR}"

echo "=== Axis 7: Instruction following (VCTK unseen prompts) ==="
bash "${SCRIPT_DIR}/run_instruction_following.sh" \
    "${INPUT_JSON}" "${OOD_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "Done. OOD results → ${OOD_DIR}/instruction_following.json"
