#!/bin/bash
#SBATCH --job-name=miles-multinode
#SBATCH --nodelist=useocpm2m-097-[084-087]
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-miles-multinode}"
TRAIN_SCRIPT="miles/scripts/run-qwen3-30B-A3B-multinode-amd.sh"
RESET_LOAD="${RESET_LOAD:-0}"
MODEL_DIR="${MODEL_DIR:-/root}"
DATA_DIR="${DATA_DIR:-/root}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

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
      RESET_LOAD='"${RESET_LOAD}"' \
      MODEL_DIR='"${MODEL_DIR}"' \
      DATA_DIR='"${DATA_DIR}"' \
      bash '"${TRAIN_SCRIPT}"'"
  }

  HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_NODELIST | head -1)
  HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk "{print \$1}" | head -1)

  if [ "$SLURM_NODEID" -eq 0 ]; then
    ROLE=head
  else
    ROLE=worker
  fi

  echo "[$(hostname)] role=${ROLE} head=${HEAD_IP} node=${SLURM_NODEID}/${SLURM_NNODES}"

  # Ensure container is running
  docker start '"${CONTAINER}"' 2>/dev/null || true
  sleep 1

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
