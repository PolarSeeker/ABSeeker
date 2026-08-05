#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_load_env.sh"

RECOVER_DIR="${RECOVER_DIR:-results/example_recover/extracted}"
TRAJ_FILE="${TRAJ_FILE:-trajectory/example_sft_format.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/example_recover/scored}"
CASE_IDS="${CASE_IDS:-}"
NUM_PROCESSES=60

python -u src/score_steps.py \
    --ids "${CASE_IDS}" \
    --recover_dir "${RECOVER_DIR}" \
    --traj_file "${TRAJ_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_processes ${NUM_PROCESSES}
