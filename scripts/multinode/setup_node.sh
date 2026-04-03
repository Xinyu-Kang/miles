#!/bin/bash
# Per-node setup script — runs inside the Docker container.
set -euo pipefail

# Install miles
cd /workspace/miles
pip install -e .

# Download model & datasets
hf download Qwen/Qwen3-30B-A3B                         --local-dir /root/Qwen3-30B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/aime-2024

# Convert checkpoint 
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source /workspace/miles/scripts/models/qwen3-30B-A3B.sh

MEGATRON_LM_PATH=$(python3 -c \
  "import megatron, os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" \
  2>/dev/null || echo "/app/Megatron-LM")

PYTHONPATH="${MEGATRON_LM_PATH}" torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint /root/Qwen3-30B-A3B \
  --save /root/Qwen3-30B-A3B_torch_dist

echo "=== Setup complete on $(hostname) ==="
