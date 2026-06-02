#!/usr/bin/env bash
# Run all 8 evaluation axes.
#
# Usage:
#   bash scripts/eval_all.sh \
#       METASET_JSON \
#       VCTK_JSON \
#       GIGA_JSON \
#       OOD_JSON \
#       MODELS_CSV \
#       OUTPUT_DIR \
#       [MODEL_NAME] [BATCH_SIZE]
#
# Arguments:
#   METASET_JSON  — model results on StyleInstructCaps-MetaSet (434 utt)
#   VCTK_JSON     — model results on VCTK_speaker_prompts (320 utt, 10 prompts)
#   GIGA_JSON     — model results on Gigaspeech_SpeechCraft_captions (600 utt)
#   OOD_JSON      — model results on VCTK_unseen_prompts (32 utt, 36 prompts)
#   MODELS_CSV    — CSV for multi-model cluster plots (axis 8)
#   OUTPUT_DIR    — root directory for all outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METASET_JSON="${1:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
VCTK_JSON="${2:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
GIGA_JSON="${3:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
OOD_JSON="${4:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
MODELS_CSV="${5:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
OUTPUT_DIR="${6:?Usage: $0 METASET_JSON VCTK_JSON GIGA_JSON OOD_JSON MODELS_CSV OUTPUT_DIR}"
MODEL_NAME="${7:-Qwen/Qwen3-32B}"
BATCH_SIZE="${8:-4}"

mkdir -p "${OUTPUT_DIR}"

echo "========== Axis 1: Metadata groundedness (MetaSet) =========="
bash "${SCRIPT_DIR}/run_metadata_groundedness.sh" \
    "${METASET_JSON}" "${OUTPUT_DIR}/metaset" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "========== Axis 2: Hallucination (MetaSet) =========="
bash "${SCRIPT_DIR}/run_hallucination.sh" \
    "${METASET_JSON}" "${OUTPUT_DIR}/metaset" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "========== Axis 3: Instruction following (MetaSet) =========="
bash "${SCRIPT_DIR}/run_instruction_following.sh" \
    "${METASET_JSON}" "${OUTPUT_DIR}/metaset" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "========== Axis 4: Speaker verification (VCTK) =========="
bash "${SCRIPT_DIR}/run_speaker_verification.sh" \
    "${VCTK_JSON}" "${OUTPUT_DIR}/vctk_speaker_verification"

echo "========== Axis 5: Speaker consistency (VCTK) =========="
bash "${SCRIPT_DIR}/run_speaker_consistency.sh" \
    "${VCTK_JSON}" "${OUTPUT_DIR}/vctk_speaker_consistency"

echo "========== Axis 6: Unseen eval (Gigaspeech) =========="
bash "${SCRIPT_DIR}/run_unseen_eval.sh" \
    "${GIGA_JSON}" "${OUTPUT_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "========== Axis 7: OOD prompts (VCTK unseen) =========="
bash "${SCRIPT_DIR}/run_ood_prompts.sh" \
    "${OOD_JSON}" "${OUTPUT_DIR}" "${MODEL_NAME}" "${BATCH_SIZE}"

echo "========== Axis 8: Speaker cluster plots =========="
bash "${SCRIPT_DIR}/run_speaker_cluster_plots.sh" \
    "${MODELS_CSV}" "${OUTPUT_DIR}/speaker_cluster_plots"

echo "All axes complete. Results under ${OUTPUT_DIR}/"
