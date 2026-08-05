#!/bin/bash
# ABSeeker RL training script.
#
#   - No logprob forward pass: USE_INFO_GAIN=false disables prealigned_vectorized.
#   - Advantage estimator grpo_abseeker handles multi-turn token-level reward tensor.
#   - Reward manager: abseeker_batch (reads non_tensor_batch.abseeker_turn_scores).
#   - Per-turn scores from online anchor judge in dr_agent_loop.

set -x

TRAIN_ONLY=false
if [ "$1" = "--train-only" ]; then
    TRAIN_ONLY=true
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "[train_abseeker.sh] WARNING: $SCRIPT_DIR/.env not found; copy .env.example to .env" >&2
fi

export GRPC_PYTHON_BUILD_WITH_CYTHON=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

export project_name="${project_name:-abseeker-rl}"
export RAY_memory_monitor_refresh_ms=0
export PYTHONFAULTHANDLER=1
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_DEBUG=WARN

export experiment_name="${experiment_name:-abseeker-$(date +%Y%m%d-%H%M%S)}"
export PET_NODE_RANK=0

# ── Cluster (aligned with SFT slime config) ───────────────────────────────────
# SFT: --actor-num-nodes 2 --actor-num-gpus-per-node 8
#      --rollout-num-gpus 16 --rollout-num-gpus-per-engine 1
#      --context-parallel-size 8
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"       # rollout-num-gpus-per-engine
MPI_HOSTFILE="${MPI_HOSTFILE:-${SCRIPT_DIR}/mpi_hostfile}"
SSH_KEY="${SSH_KEY:-${SCRIPT_DIR}/id_rsa}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda/envs/verl}"
RAY_BIN="${RAY_BIN:-${CONDA_ENV}/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"

# Multi-node NCCL/Gloo (adjust interface names for your cluster).
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-net0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC TORCH_NCCL_ENABLE_MONITORING
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

NVLINK_COUNT=$(nvidia-smi 2>/dev/null | grep -c "NVLink" || true)
HAS_NVLINK=$([ "${NVLINK_COUNT:-0}" -gt 0 ] && echo 1 || echo 0)

if [ "${NNODES}" -ge 2 ]; then
    START_RAY_CLUSTER="${START_RAY_CLUSTER:-1}"
else
    START_RAY_CLUSTER="${START_RAY_CLUSTER:-0}"
fi

if [ "${START_RAY_CLUSTER}" = "1" ]; then
    MASTER_ADDR="${MASTER_ADDR:-$(awk 'NF{print $1; exit}' "${MPI_HOSTFILE}")}"
    if [ -z "${MASTER_ADDR}" ]; then
        echo "MASTER_ADDR is not set and mpi_hostfile is empty: ${MPI_HOSTFILE}" >&2
        exit 1
    fi
    export MASTER_ADDR
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    [[ -f "${SSH_KEY}" ]] && chmod 600 "${SSH_KEY}"
fi

# ── Method: abseeker (default) | grpo (standard outcome GRPO) ───────────────────────
# TRAIN_METHOD=grpo → adv=grpo, naive_batch, ckpt under output/abseeker_base
TRAIN_METHOD="${TRAIN_METHOD:-abseeker}"
export TRAIN_METHOD
if [ "$TRAIN_METHOD" = "grpo" ]; then
    ADV_ESTIMATOR="grpo"
    REWARD_MANAGER="naive_batch"
    USE_FORMAT_PENALTY=false
    REWARD_USE_FORMAT_PENALTY=false
    FORMAT_PENALTY_SCALE=0
    unset ABSEEKER_ANCHOR_API_BASE   # skip online anchor judge
    _OUTPUT_NAME="abseeker_base"
else
    ADV_ESTIMATOR="grpo_abseeker"
    REWARD_MANAGER="abseeker_batch"
    USE_FORMAT_PENALTY=true
    REWARD_USE_FORMAT_PENALTY=true
    FORMAT_PENALTY_SCALE=2.0   # abseeker: per-turn overwrite magnitude
    _OUTPUT_NAME="abseeker"
fi

# ── Paths (override in .env or env before running) ───────────────────────────
export MODEL_PATH="${MODEL_PATH:-/path/to/sft-checkpoint}"
TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/train.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
export OUTPUT="${OUTPUT:-${SCRIPT_DIR}/output/${_OUTPUT_NAME}}"
export EVAL_LOG_PATH="${SCRIPT_DIR}/output/eval_log"
export TENSORBOARD_DIR="${OUTPUT}/tensorboard_log"
_GPU_NUM="${GPUS_PER_NODE}"
mkdir -p "$OUTPUT"
mkdir -p "$EVAL_LOG_PATH"

TOTAL_TRAINING_STEPS=""

# ── ABSeeker reward ───────────────────────────────────────────────────────
# No logprob info-gain pass needed; per-turn scores come from agent loop LLM judge.
USE_INFO_GAIN=false
TRAIN_REWARD_TYPE="llm"
MASK_TOOL_RESPONSE=true
echo "TRAIN_METHOD=${TRAIN_METHOD} ADV_ESTIMATOR=${ADV_ESTIMATOR} REWARD_MANAGER=${REWARD_MANAGER} OUTPUT=${OUTPUT} FORMAT_PENALTY_SCALE=${FORMAT_PENALTY_SCALE} REWARD_USE_FORMAT_PENALTY=${REWARD_USE_FORMAT_PENALTY}"

# ── Rollout ─────────────────────────────────────────────────────────────────
USE_ASYNC_ROLLOUT=true
ASYNC_NUM_WORKERS="${ASYNC_NUM_WORKERS:-16}"   # agent loop workers (2x rollout GPUs for tool I/O overlap)
ASYNC_MAX_CONCURRENT_SAMPLES=""
ASYNC_COMPLETION_CUTOFF="0.75"

USE_ROBUST_ROLLOUT=false
ROBUST_MAX_RETRIES=3
ROBUST_DEDUP=true
ROBUST_REPETITION=true

# ── Sequence lengths ────────────────────────────────────────────────────────
MAX_MODEL_LEN=261000
MAX_PROMPT_LEN=246000
MAX_RESPONSE_LEN=6000
ULYSSES_SP_SIZE="${ULYSSES_SP_SIZE:-8}"       # context-parallel-size=8
ASYNC_PROMPT_PAD=2048

if [ "$USE_ASYNC_ROLLOUT" = "true" ]; then
    _max_seq=$((ASYNC_PROMPT_PAD + MAX_MODEL_LEN))
else
    _max_seq=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
fi
PPO_MAX_TOKEN_LEN=$(( (_max_seq + ULYSSES_SP_SIZE - 1) / ULYSSES_SP_SIZE + 1000 ))

echo "Mode: $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo ASYNC || echo SYNC)"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}, MAX_PROMPT_LEN=${MAX_PROMPT_LEN}, MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}"
[ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "ASYNC_PROMPT_PAD=${ASYNC_PROMPT_PAD}"
echo "PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN} (effective: $((PPO_MAX_TOKEN_LEN * ULYSSES_SP_SIZE)))"

if [ "${TRAIN_ONLY}" = "false" ] && [ "${START_RAY_CLUSTER}" = "1" ]; then
    pkill -9 sglang 2>/dev/null || true
    "${RAY_BIN}" stop --force || true
    pkill -9 ray 2>/dev/null || true
    sleep 3

    "${RAY_BIN}" start --head \
        --node-ip-address="${MASTER_ADDR}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}"

    _WORKER_PIDS=()
    for WORKER_IP in $(awk '{print $1}' "${MPI_HOSTFILE}"); do
        if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
            continue
        fi
        echo "Starting Ray worker on ${WORKER_IP}"
        ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no root@"${WORKER_IP}" \
            "pkill -9 sglang 2>/dev/null || true; ${RAY_BIN} stop --force || true; pkill -9 python 2>/dev/null || true; ${RAY_BIN} start --address=${MASTER_ADDR}:6379 --num-gpus ${GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
        _WORKER_PIDS+=($!)
    done
    wait "${_WORKER_PIDS[@]}"

    echo "Waiting for Ray dashboard..."
    for i in $(seq 1 30); do
        curl -sf http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version > /dev/null 2>&1 && break
        sleep 2
    done
fi

RUNTIME_ENV_JSON="{
  \"excludes\": [
    \"output/**\",
    \"output_rl_abseeker_archived/**\",
    \"log/**\"
  ],
  \"env_vars\": {
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}\",
    \"TORCH_NCCL_ENABLE_MONITORING\": \"${TORCH_NCCL_ENABLE_MONITORING}\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\",
    \"no_proxy\": \"${no_proxy:-}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR:-}\",
    \"GRPC_PYTHON_BUILD_WITH_CYTHON\": \"1\",
    \"SERPER_KEY_ID\": \"${SERPER_KEY_ID:-}\",
    \"JINA_API_KEYS\": \"${JINA_API_KEYS:-}\",
    \"API_BASE\": \"${API_BASE:-}\",
    \"API_KEY\": \"${API_KEY:-}\",
    \"SUMMARY_MODEL_NAME\": \"${SUMMARY_MODEL_NAME:-}\",
    \"JUDGE_API_BASE\": \"${JUDGE_API_BASE:-}\",
    \"JUDGE_API_KEY\": \"${JUDGE_API_KEY:-}\",
    \"JUDGE_MODEL_NAME\": \"${JUDGE_MODEL_NAME:-}\",
    \"ENABLE_JUDGE_THINKING\": \"${ENABLE_JUDGE_THINKING:-}\",
    \"ABSEEKER_ANCHOR_API_BASE\": \"${ABSEEKER_ANCHOR_API_BASE:-}\",
    \"ABSEEKER_ANCHOR_API_KEY\": \"${ABSEEKER_ANCHOR_API_KEY:-}\",
    \"ABSEEKER_ANCHOR_MODEL_NAME\": \"${ABSEEKER_ANCHOR_MODEL_NAME:-}\",
    \"ABSEEKER_ANCHOR_ENABLE_THINKING\": \"${ABSEEKER_ANCHOR_ENABLE_THINKING:-}\",
    \"ABSEEKER_ANCHOR_TIMEOUT\": \"${ABSEEKER_ANCHOR_TIMEOUT:-}\",
    \"ABSEEKER_ANCHOR_TEMPERATURE\": \"${ABSEEKER_ANCHOR_TEMPERATURE:-}\",
    \"TRAIN_METHOD\": \"${TRAIN_METHOD:-abseeker}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\"
  }
}"

if [ "${TRAIN_ONLY}" = "false" ] && [ "${START_RAY_CLUSTER}" = "1" ]; then
    SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
    "${RAY_BIN}" job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
        --working-dir="${SCRIPT_DIR}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- /bin/bash "${SCRIPT_NAME}" --train-only \
        > >(stdbuf -oL tee "${OUTPUT}/training.log") 2>&1
else
HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 \
ROLLOUT_ROBUST=$([ "$USE_ROBUST_ROLLOUT" = "true" ] && echo 1 || echo 0) \
ROLLOUT_ROBUST_MAX_RETRIES=${ROBUST_MAX_RETRIES} \
ROLLOUT_ROBUST_DEDUP=$([ "$ROBUST_DEDUP" = "true" ] && echo 1 || echo 0) \
ROLLOUT_ROBUST_REPETITION=$([ "$ROBUST_REPETITION" = "true" ] && echo 1 || echo 0) \
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_info_gain=${USE_INFO_GAIN} \
    algorithm.abseeker_norm_mode=scaled_separate \
    algorithm.abseeker_turn_weight=1.0 \
    algorithm.use_format_penalty=${USE_FORMAT_PENALTY} \
    algorithm.format_penalty_scale=${FORMAT_PENALTY_SCALE} \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    +data.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.mask_tool_response=${MASK_TOOL_RESPONSE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP_SIZE} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=1.05 \
    actor_rollout_ref.nccl_timeout=7200 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.gamma=0.25 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    agent_grpo.n=8 \
    max_turns=200 \
    +tool_timeout=90 \
    +trace_save_interval=10 \
    +trace_max_samples=0 \
    search_engine=online_search \
    reward_model.train_reward_type=${TRAIN_REWARD_TYPE} \
    +reward_model.valid_reward_type=llm_em_noformatf1 \
    reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.reward_kwargs.deepthink_disabled=true \
    +reward_model.reward_kwargs.use_format_penalty=${REWARD_USE_FORMAT_PENALTY} \
    +reward_model.reward_kwargs.format_penalty_deduction=${FORMAT_PENALTY_SCALE} \
    data.return_raw_chat=true \
    trainer.total_epochs=1 \
    ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}} \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.mode=async" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.prompt_length=${ASYNC_PROMPT_PAD}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.num_workers=${ASYNC_NUM_WORKERS}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && echo "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/dr_agent_loop.yaml" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_MAX_CONCURRENT_SAMPLES" ] && [ "$ASYNC_MAX_CONCURRENT_SAMPLES" != "null" ] && echo "+actor_rollout_ref.rollout.agent.max_concurrent_samples=${ASYNC_MAX_CONCURRENT_SAMPLES}" || true) \
    $([ "$USE_ASYNC_ROLLOUT" = "true" ] && [ -n "$ASYNC_COMPLETION_CUTOFF" ] && [ "$ASYNC_COMPLETION_CUTOFF" != "null" ] && echo "+actor_rollout_ref.rollout.agent.completion_cutoff=${ASYNC_COMPLETION_CUTOFF}" || true) \
    > >(stdbuf -oL tee "${OUTPUT}/training.log") 2>&1
fi

EXIT_CODE=${PIPESTATUS[0]:-$?}
echo "[$(date)] Training exited with code: $EXIT_CODE" | tee -a ${OUTPUT}/training.log
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== dmesg (last 30 lines) ===" >> ${OUTPUT}/training.log
    dmesg -T 2>/dev/null | tail -30 >> ${OUTPUT}/training.log
    echo "=== nvidia-smi ===" >> ${OUTPUT}/training.log
    nvidia-smi >> ${OUTPUT}/training.log 2>&1
fi
