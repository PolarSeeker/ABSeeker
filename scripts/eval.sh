#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_load_env.sh"

SKIP_EXISTING="${SKIP_EXISTING:-true}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/dataset/example.json}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/result/react_example_react}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

python -u src/eval.py \
    --result_dir "${RESULT_DIR}" \
    --dataset_path "${DATASET_PATH}" \
    --num_workers 10 \
    ${SKIP_EXISTING:+--skip_existing ${SKIP_EXISTING}}
