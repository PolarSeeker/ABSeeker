#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_load_env.sh"

INPUT_DIR="${INPUT_DIR:-results/example_recover}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/extracted}"
DATASET_PATH="${DATASET_PATH:-dataset/example.json}"
CASE_IDS="${CASE_IDS:-}"
NUM_PROCESSES=60

python -u src/extract_core_clues.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_path "${DATASET_PATH}" \
    --ids "${CASE_IDS}" \
    --num_processes ${NUM_PROCESSES}
