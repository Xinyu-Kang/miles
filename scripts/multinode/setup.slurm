#!/bin/bash
#SBATCH --job-name=miles-multinode-setup
#SBATCH --nodelist=useocpm2m-097-[084-087]
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.log
#SBATCH --error=logs/setup_%j.err

set -euo pipefail

WORKSPACE=<your-path>/miles-multinode
IMAGE=rlsys/miles:rocm7-mi300-sglang0.5.9-te2.12.0-latest
CONTAINER=miles-multinode

mkdir -p "${WORKSPACE}/logs"

# --------------------------------------------------------------------------- #
#  Per-node setup: launch container and run the init script inside it         #
# --------------------------------------------------------------------------- #
srun bash -c '
  set -euo pipefail
  echo "[$(hostname)] Starting container setup ..."

  docker rm -f '"${CONTAINER}"' 2>/dev/null || true

  # Start container detached so it stays running after setup
  docker run -d \
    --network=host \
    --ipc=host \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME/.ssh:/root/.ssh \
    -v '"${WORKSPACE}"':/workspace \
    --shm-size 128G \
    --name '"${CONTAINER}"' \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -w /workspace \
    '"${IMAGE}"' \
    sleep infinity

  # Run setup inside the running container
  docker exec '"${CONTAINER}"' bash /workspace/miles/scripts/multinode/setup_node.sh

  echo "[$(hostname)] Done."
'
