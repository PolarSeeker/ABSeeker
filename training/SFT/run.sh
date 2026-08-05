#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

V=4

if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

# Set to 1 to weight each assistant turn by the corresponding score in the dataset.
USE_SCORE_WEIGHT="${USE_SCORE_WEIGHT:-0}"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Multi-node NCCL/Gloo (adjust NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME for your cluster).
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-net0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "USE_SCORE_WEIGHT: $USE_SCORE_WEIGHT"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── Paths (override via env before running) ───────────────────────────────────
SLIME_ROOT="${SLIME_ROOT:-/path/to/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/path/to/Megatron-LM}"
MODEL_HF="${MODEL_HF:-/path/to/Qwen3.5-4B}"
MODEL_TORCH_DIST="${MODEL_TORCH_DIST:-/path/to/Qwen3.5-4B_torch_dist}"
SFT_DATA="${SFT_DATA:-${SCRIPT_DIR}/data/train.jsonl}"
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/id_rsa}"
MPI_HOSTFILE="${MPI_HOSTFILE:-${SCRIPT_DIR}/mpi_hostfile}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

[[ -f "${SSH_KEY}" ]] && chmod 600 "${SSH_KEY}"

export JINJA_PTH="${SCRIPT_DIR}/qwen3_5_template.jinja"

source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

SCORE_WEIGHT_ARGS=()
if [ "$USE_SCORE_WEIGHT" = "1" ]; then
   SCORE_WEIGHT_ARGS+=(--use-score-weight)
fi

if [ "$USE_SCORE_WEIGHT" = "1" ]; then
   CKPT_DIR="${SCRIPT_DIR}/checkpoints_v${V}/lr5e-5_scored"
else
   CKPT_DIR="${SCRIPT_DIR}/checkpoints_v${V}/lr5e-5_scored_baseline"
fi
if [[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
   echo "Resume training from ${CKPT_DIR} (latest iter $(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt"))"
else
   echo "No checkpoint in ${CKPT_DIR}, training from ref-load"
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_HF}"
   --ref-load "${MODEL_TORCH_DIST}"
   --load "${CKPT_DIR}"
   --save "${CKPT_DIR}"
   --save-interval 50
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --loss-mask-type qwen3
   # --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   "${SCORE_WEIGHT_ARGS[@]}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 8
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 30000
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-5
#    --lr-warmup-iters 128
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
#    --use-wandb
#    --wandb-project sft
#    --wandb-group ${WANDB_GROUP}
#    --wandb-mode offline
#    --wandb-dir ${WANDB_DIR:-/path/to/wandb}
#    --wandb-key ${WANDB_KEY}
#    --disable-wandb-random-suffix
#    --wandb-eval-savedir ${WANDB_EVAL_DIR:-/path/to/eval_logs}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   # Safer dist ckpt save under Ray (avoids filesystem_async multiproc BrokenPipe; slower, same parallelism only resume)
   --no-ckpt-fully-parallel-save
)

export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

for WORKER_IP in $(awk '{print $1}' "${MPI_HOSTFILE}"); do
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh -i "${SSH_KEY}" root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"JINJA_PTH\": \"${JINJA_PTH}\",
    \"PYTHONPATH\": \"${MEGATRON_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}\",
    \"TORCH_NCCL_ENABLE_MONITORING\": \"${TORCH_NCCL_ENABLE_MONITORING}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --rollout-num-gpus-per-engine 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
