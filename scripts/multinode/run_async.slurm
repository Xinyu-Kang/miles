#!/bin/bash
#SBATCH --job-name=miles-multinode-async
##SBATCH --nodelist=useocpm2m-097-[050,067,070,095,110,115,119,122,123]
#SBATCH --nodelist=useocpm2m-097-[076,083,084,086,087,089]
##SBATCH --nodelist=useocpm2m-097-[067,070,080,084]
##SBATCH --nodelist=useocpm2m-097-[086-087]
##SBATCH --nodelist=useocpm2m-097-[014,017,019]
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_6node_%j.log
#SBATCH --error=logs/train_6node_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-miles-multinode}"
TRAIN_SCRIPT="miles/scripts/run-qwen3-30B-A3B-multinode-async.sh"
RESET_LOAD="${RESET_LOAD:-0}"
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

# Async / non-colocate knobs (empty => script default).
TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-2}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"

# NCCL transport. Auto-detect inside the container is unreliable (host IP often
# isn't bound to a container netns iface), so we set the bootstrap iface and IB
# HCAs explicitly here. Override per-run with `sbatch --export=...,NCCL_*=...`.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"

mkdir -p logs
READY_FILE="$(pwd)/logs/ray_head_${SLURM_JOB_ID}.ready"
rm -f "${READY_FILE}"

srun bash -c '
  set -euo pipefail

  port_is_open() {
    python3 -c "import socket, sys; host=sys.argv[1]; port=int(sys.argv[2]); s=socket.socket(); s.settimeout(1);
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
else:
    s.close()
    sys.exit(0)" "$1" "$2"
  }

  run_in_container() {
    docker exec '"${CONTAINER}"' bash -c "\
      MASTER_ADDR=${HEAD_IP} \
      RAY_NODE_ROLE=${ROLE} \
      RAY_HEAD_PORT='"${RAY_HEAD_PORT}"' \
      ACTOR_NUM_NODES=${SLURM_NNODES} \
      TRAIN_NUM_NODES='"${TRAIN_NUM_NODES}"' \
      UPDATE_WEIGHTS_INTERVAL='"${UPDATE_WEIGHTS_INTERVAL}"' \
      ROLLOUT_BATCH_SIZE='"${ROLLOUT_BATCH_SIZE}"' \
      GLOBAL_BATCH_SIZE='"${GLOBAL_BATCH_SIZE}"' \
      RESET_LOAD='"${RESET_LOAD}"' \
      MODEL_DIR='"${MODEL_DIR}"' \
      DATA_DIR='"${DATA_DIR}"' \
      NCCL_SOCKET_IFNAME='"${NCCL_SOCKET_IFNAME}"' \
      NCCL_IB_HCA='"${NCCL_IB_HCA}"' \
      NCCL_DEBUG='"${NCCL_DEBUG}"' \
      bash '"${TRAIN_SCRIPT}"'"
  }

  ensure_clean_container() {
    if ! docker container inspect '"${CONTAINER}"' >/dev/null 2>&1; then
      echo "Container '"${CONTAINER}"' not found on $(hostname). Run setup.slurm first."
      exit 1
    fi

    # Reset the long-lived container before each run so leaked SGLang / Ray
    # processes from prior failures cannot hold onto ports or GPU resources.
    if [ "$(docker inspect -f "{{.State.Running}}" '"${CONTAINER}"')" = "true" ]; then
      docker restart '"${CONTAINER}"' >/dev/null
    else
      docker start '"${CONTAINER}"' >/dev/null
    fi
    sleep 2
  }

  HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_NODELIST | head -1)
  HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk "{print \$1}" | head -1)

  if [ "$SLURM_NODEID" -eq 0 ]; then
    ROLE=head
  else
    ROLE=worker
  fi

  echo "[$(hostname)] role=${ROLE} head=${HEAD_IP} node=${SLURM_NODEID}/${SLURM_NNODES}"

  # Ensure the per-node container exists and start from a clean process state.
  ensure_clean_container

  if [ "${ROLE}" = "head" ]; then
    run_in_container &
    HEAD_PID=$!

    # Wait for any old listener to go away before waiting for the fresh head.
    DEADLINE=$((SECONDS + 60))
    while [ "${SECONDS}" -lt "${DEADLINE}" ] && port_is_open "${HEAD_IP}" '"${RAY_HEAD_PORT}"'; do
      sleep 1
    done

    DEADLINE=$((SECONDS + 180))
    until port_is_open "${HEAD_IP}" '"${RAY_HEAD_PORT}"'; do
      if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "Timed out waiting for Ray head on ${HEAD_IP}:'"${RAY_HEAD_PORT}"'"
        kill "${HEAD_PID}" 2>/dev/null || true
        wait "${HEAD_PID}" || true
        exit 1
      fi
      sleep 2
    done

    touch '"${READY_FILE}"'
    wait "${HEAD_PID}"
  else
    DEADLINE=$((SECONDS + 240))
    while [ ! -f '"${READY_FILE}"' ]; do
      if [ "${SECONDS}" -ge "${DEADLINE}" ]; then
        echo "Timed out waiting for Ray head readiness file: '"${READY_FILE}"'"
        exit 1
      fi
      sleep 2
    done

    run_in_container
  fi
'
