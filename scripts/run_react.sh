#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_load_env.sh"

DATASET_PATH="${DATASET_PATH:-dataset/example.json}"

python -u src/main_react.py \
    --exp_name react \
    --dataset "${DATASET_PATH}" \
    --num_workers 40 \
    --tool_count_max 200 \
    --max_tokens 16384 \
    --temperature 0.7
