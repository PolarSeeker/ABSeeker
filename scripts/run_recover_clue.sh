#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_load_env.sh"

DATASET="${DATASET:-dataset/example.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/example_recover}"
CASE_IDS="${CASE_IDS:-}"
NUM_PROCESSES=60
MAX_LOOPS=60

python -u src/recover_intermediate_steps_deepseek.py \
    --dataset_path "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --ids "${CASE_IDS}" \
    --num_processes ${NUM_PROCESSES} \
    --max_loops ${MAX_LOOPS}
