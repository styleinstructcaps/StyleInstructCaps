#!/usr/bin/env bash
# Axis 4: Speaker caption verification on VCTK_speaker_prompts
#
# Usage:
#   bash scripts/run_speaker_verification.sh INPUT_JSON OUTPUT_DIR [NUM_POSITIVE] [NUM_NEGATIVE]
#
# Slurm hint: CPU or GPU, conda env spk_consistency

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSON="${1:?Usage: $0 INPUT_JSON OUTPUT_DIR [NUM_POSITIVE] [NUM_NEGATIVE]}"
OUTPUT_DIR="${2:?Usage: $0 INPUT_JSON OUTPUT_DIR [NUM_POSITIVE] [NUM_NEGATIVE]}"
NUM_POSITIVE="${3:-4800}"
NUM_NEGATIVE="${4:-4800}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spk_consistency

mkdir -p "${OUTPUT_DIR}"

python -u "${RELEASE_ROOT}/evaluation/speaker_verification.py" \
    --input_json "${INPUT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_positive "${NUM_POSITIVE}" \
    --num_negative "${NUM_NEGATIVE}"

echo "Done. Results → ${OUTPUT_DIR}/"
