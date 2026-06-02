#!/usr/bin/env bash
# Axis 6: Unseen dataset (Gigaspeech_SpeechCraft_captions)
# Runs metadata groundedness + hallucination + instruction following.
#
# Usage:
#   bash scripts/run_unseen_eval.sh INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]
#
# Slurm hint: 1+ GPU (A100/H100), conda env spk_style_eval_env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR [MODEL_NAME] [BATCH_SIZE]}"
MODEL_NAME="${3:-Qwen/Qwen3-32B}"
BATCH_SIZE="${4:-4}"

UNSEEN_DIR="${OUTPUT_DIR}/gigaspeech"
mkdir -p "${UNSEEN_DIR}"

echo "=== Axis 6a: Metadata groundedness (Gigaspeech) ==="
bash "${SCRIPT_DIR}/run_metadata_groundedness.sh" \
    "${INPUT_JSON}" "${UNSEEN_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "=== Axis 6b: Hallucination (Gigaspeech) ==="
bash "${SCRIPT_DIR}/run_hallucination.sh" \
    "${INPUT_JSON}" "${UNSEEN_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "=== Axis 6c: Instruction following (Gigaspeech) ==="
bash "${SCRIPT_DIR}/run_instruction_following.sh" \
    "${INPUT_JSON}" "${UNSEEN_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "Done. Gigaspeech results → ${UNSEEN_DIR}/"
