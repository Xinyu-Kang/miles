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
pkill -9 redis

set -euxo pipefail

# ==================== Platform Detection ====================
if [ -e /dev/kfd ] || python3 -c "import torch; assert torch.version.hip" 2>/dev/null; then
    GPU_VENDOR="amd"
elif command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
else
    echo "ERROR: No supported GPU detected (need NVIDIA or AMD)"
    exit 1
fi
echo "Detected GPU vendor: ${GPU_VENDOR}"

# ==================== Configurable Paths ====================
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
export MODEL_DIR DATA_DIR

# ==================== Platform-Specific Setup ====================
if [ "$GPU_VENDOR" = "amd" ]; then
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    if [ -z "${HIP_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=0
    else
        NUM_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    fi
    HAS_NVLINK=0
else
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
    if [ "$NVLINK_COUNT" -gt 0 ]; then
        HAS_NVLINK=1
    else
        HAS_NVLINK=0
    fi
    echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TRAIN_WORKDIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

# Checkpoint behavior:
# - By default, resume from the previous MILES checkpoint directory.
# - Set RESET_LOAD=1 to ignore stale/incompatible MILES checkpoints and start from torch_dist.
LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_miles/"
if [ "${RESET_LOAD:-0}" = "1" ]; then
   LOAD_DIR_DEFAULT="${MODEL_DIR}/Qwen3-30B-A3B_torch_dist"
fi
LOAD_DIR="${LOAD_DIR:-${LOAD_DIR_DEFAULT}}"
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/Qwen3-30B-A3B_miles/}"

echo "LOAD_DIR=${LOAD_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B
   #--hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-FP8
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B_torch_dist
   --load ${LOAD_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --log-passrate

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
   --skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-expert-parallel-size 8
)

# AMD: disable custom all-reduce
if [ "$GPU_VENDOR" = "amd" ]; then
    SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
   #  SGLANG_ARGS+=(--sglang-attention-backend triton)
fi

export NVTE_USE_CUTLASS_GROUPED_GEMM=1
export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend fused
   # use TE CK grouped GEMM for MoE
   --moe-grouped-gemm
)

# ==================== Multinode Ray Setup ====================
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
RAY_NODE_ROLE="${RAY_NODE_ROLE:-head}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8285}"
RAY_WAIT_FOR_NODES_TIMEOUT_S="${RAY_WAIT_FOR_NODES_TIMEOUT_S:-600}"
RAY_WAIT_INTERVAL_S="${RAY_WAIT_INTERVAL_S:-5}"

DETECTED_LOCAL_NODE_IP=$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i !~ /^127\./) {print $i; exit}}')
LOCAL_NODE_IP="${RAY_NODE_IP:-${DETECTED_LOCAL_NODE_IP}}"
if [ -z "${LOCAL_NODE_IP}" ]; then
    LOCAL_NODE_IP="127.0.0.1"
fi

USER_PROVIDED_MASTER_ADDR=1
if [ -z "${MASTER_ADDR:-}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
    USER_PROVIDED_MASTER_ADDR=0
    export MASTER_ADDR="${LOCAL_NODE_IP}"
else
    export MASTER_ADDR
fi

export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"
echo "MASTER_ADDR=${MASTER_ADDR} LOCAL_NODE_IP=${LOCAL_NODE_IP}"
echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
echo "RAY_NODE_ROLE=${RAY_NODE_ROLE} RAY_HEAD_PORT=${RAY_HEAD_PORT} RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}"

if [ "${RAY_NODE_ROLE}" != "head" ] && [ "${RAY_NODE_ROLE}" != "worker" ]; then
    echo "Invalid RAY_NODE_ROLE=${RAY_NODE_ROLE}. Use head or worker."
    exit 1
fi

if [ "${ACTOR_NUM_NODES}" -gt 1 ] && [ "${RAY_NODE_ROLE}" = "worker" ] && [ "${USER_PROVIDED_MASTER_ADDR}" = "0" ]; then
    echo "In multi-node worker mode, set MASTER_ADDR to the Ray head node IP."
    exit 1
fi

# ── Start Ray ────────────────────────────────────────────────────────────── #
ray stop --force 2>/dev/null || true

if [ "${RAY_NODE_ROLE}" = "head" ]; then
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --port "${RAY_HEAD_PORT}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}"
else
    ray start \
        --address "${MASTER_ADDR}:${RAY_HEAD_PORT}" \
        --node-ip-address "${LOCAL_NODE_IP}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats

    echo "Ray worker started on ${LOCAL_NODE_IP}, joined ${MASTER_ADDR}:${RAY_HEAD_PORT}."
    echo "Waiting for head node to finish training..."
    # Keep the worker process alive so Ray stays up.
    # The launcher (or user) sends SIGTERM/SIGINT to tear down.
    tail -f /dev/null
fi

# ── Head: wait for all workers to join ───────────────────────────────────── #
if [ "${ACTOR_NUM_NODES}" -gt 1 ]; then
    CLUSTER_READY=0
    WAIT_DEADLINE=$((SECONDS + RAY_WAIT_FOR_NODES_TIMEOUT_S))
    while [ "${SECONDS}" -lt "${WAIT_DEADLINE}" ]; do
        ALIVE_NODES=$(python3 - <<'PY' 2>/dev/null || echo 0
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level=40)
print(sum(1 for n in ray.nodes() if n.get("Alive")))
ray.shutdown()
PY
)
        if [ "${ALIVE_NODES}" -ge "${ACTOR_NUM_NODES}" ]; then
            CLUSTER_READY=1
            break
        fi
        echo "Waiting for Ray cluster: ${ALIVE_NODES}/${ACTOR_NUM_NODES} nodes alive"
        sleep "${RAY_WAIT_INTERVAL_S}"
    done
    if [ "${CLUSTER_READY}" != "1" ]; then
        echo "Timed out waiting for Ray workers. Expected ${ACTOR_NUM_NODES} nodes."
        exit 1
    fi
    echo "Ray cluster ready: ${ALIVE_NODES}/${ACTOR_NUM_NODES} nodes."
fi

# ── Head: submit training job ────────────────────────────────────────────── #
MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_SET_CPU_AFFINITY\": \"0\",
    \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"1\",
    \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"1\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --working-dir "${TRAIN_WORKDIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
