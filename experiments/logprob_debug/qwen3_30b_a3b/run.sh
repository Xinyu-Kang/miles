#!/usr/bin/env bash
set -euo pipefail

# Run inside the prepared logprob-blog-tutorial container on the allocated
# MI355X node. CONTAINER_IMAGE_ID and CONTAINER_REPO_DIGEST are obtained with
# docker image inspect on the host and passed through docker exec.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${REPO_ROOT}"

VARIANT="${VARIANT:-baseline}"
RUN_ID="${RUN_ID:-qwen3-30b-logprob-${VARIANT}-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logprob-debug-artifacts}"
DEBUG_EXIT_AFTER_ROLLOUT="${DEBUG_EXIT_AFTER_ROLLOUT:-1}"
if [[ "${VARIANT}" == "b_worker_diagnostic" ]]; then
  NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PROVENANCE_DIR="${RUN_DIR}/provenance"
MODEL_PATH="/root/models/Qwen3-30B-A3B"
TRAIN_CHECKPOINT_PATH="/root/models/Qwen3-30B-A3B_torch_dist"
DATASET_PATH="/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"

if [[ "${CONTAINER_IMAGE_ID:-}" != sha256:* ]]; then
  echo "CONTAINER_IMAGE_ID must be the Docker sha256 image ID" >&2
  exit 2
fi
if [[ "${CONTAINER_REPO_DIGEST:-}" != *@sha256:* ]]; then
  echo "CONTAINER_REPO_DIGEST must include the repository and immutable sha256 digest" >&2
  exit 2
fi

mkdir -p "${PROVENANCE_DIR}"

if [[ "${VARIANT}" == "b_worker_diagnostic" ]]; then
  COMMON_ARGS=(
    --num-rollout "${NUM_ROLLOUT}"
    --debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}"
    --debug-rollout-only
    --rollout-batch-size 1
    --n-samples-per-prompt 1
    --global-batch-size 2
    --rollout-max-response-len 1
    --sglang-max-running-requests 8
    --rollout-health-check-first-wait 180
    --seed 1234
    --rollout-seed 1234
    --custom-generate-function-path tools.diagnose_prefill_workers.generate
    --save-interval 1000000
    --save-retain-interval 1000000
  )
else
  COMMON_ARGS=(
    --num-rollout "${NUM_ROLLOUT}"
    --debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}"
    --rollout-batch-size 4
    --n-samples-per-prompt 2
    --global-batch-size 8
    --rollout-max-response-len 128
    --sglang-max-running-requests 8
    --rollout-health-check-first-wait 180
    --seed 1234
    --rollout-seed 1234
    --debug-compare-decode-prefill-logprobs
    --debug-prefill-logprob-repeats 2
    --debug-trainer-logprob-repeats 2
    --save-interval 1000000
    --save-retain-interval 1000000
  )
fi

# Each ordinary non-baseline variant changes one causal variable. The
# prefill_deterministic variant records SGLang's compound deterministic mode;
# Triton is explicit because this ROCm image lacks the FA3 flash_ops selected
# by SGLang's default deterministic fallback.
VARIANT_ARGS=()
case "${VARIANT}" in
  baseline)
    ;;
  b_worker_diagnostic)
    ;;
  r3)
    VARIANT_ARGS+=(--use-rollout-routing-replay)
    ;;
  graph_off)
    VARIANT_ARGS+=(--sglang-disable-cuda-graph)
    ;;
  concurrency_1)
    VARIANT_ARGS+=(--sglang-max-running-requests 1)
    ;;
  prefill_triton)
    VARIANT_ARGS+=(--sglang-attention-backend triton)
    ;;
  prefill_deterministic)
    VARIANT_ARGS+=(
      --sglang-enable-prefill-only-deterministic-inference
      --sglang-attention-backend triton
    )
    ;;
  overlap_off)
    VARIANT_ARGS+=(--sglang-disable-overlap-schedule)
    ;;
  radix_cache_off)
    VARIANT_ARGS+=(--sglang-disable-radix-cache)
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

EXPERIMENT_ARGS=()
if [[ -n "${MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS:-}" ]]; then
  read -r -a EXPERIMENT_ARGS <<< "${MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS}"
fi
printf -v EXTRA_ARGS '%q ' "${COMMON_ARGS[@]}" "${VARIANT_ARGS[@]}" "${EXPERIMENT_ARGS[@]}"
export RUN_ID OUTPUT_ROOT NUM_ROLLOUT DEBUG_EXIT_AFTER_ROLLOUT EXTRA_ARGS VARIANT PROVENANCE_DIR
export WANDB_API_KEY=""
export SGLANG_RETURN_ORIGINAL_LOGPROB=1
export NCCL_IB_GID_INDEX=1
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export GLOO_SOCKET_IFNAME=ens3
export NCCL_SOCKET_IFNAME=ens3
export TP_SOCKET_IFNAME=ens3
export NCCL_DMABUF_ENABLE=1
RUNTIME_EXTRA_ENV='{"SGLANG_RETURN_ORIGINAL_LOGPROB":"1","NCCL_IB_GID_INDEX":"1","NCCL_IB_HCA":"ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7","GLOO_SOCKET_IFNAME":"ens3","NCCL_SOCKET_IFNAME":"ens3","TP_SOCKET_IFNAME":"ens3","NCCL_DMABUF_ENABLE":"1"}'
if [[ -n "${MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON:-}" ]]; then
  RUNTIME_EXTRA_ENV="$(python experiments/logprob_debug/qwen3_30b_a3b/tutorial/fault_guard.py \
    merge-runtime-env "${RUNTIME_EXTRA_ENV}" "${MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON}")"
fi

if [[ "${VARIANT}" == "b_worker_diagnostic" ]]; then
  DIAGNOSTIC_SAMPLE_PATH="${DIAGNOSTIC_SAMPLE_PATH:-/workspace/logprob-debug-artifacts/qwen3-30b-logprob-fresh-node-107722-20260903/dump_details/rollout_data/0.pt}"
  DIAGNOSTIC_OUTPUT_PATH="${RUN_DIR}/b_worker_diagnostic.json"
  export DIAGNOSTIC_SAMPLE_PATH DIAGNOSTIC_OUTPUT_PATH
  RUNTIME_EXTRA_ENV="${RUNTIME_EXTRA_ENV%?},\"MILES_LOGPROB_WORKER_DIAGNOSTIC_SAMPLE_PATH\":\"${DIAGNOSTIC_SAMPLE_PATH}\",\"MILES_LOGPROB_WORKER_DIAGNOSTIC_OUTPUT_PATH\":\"${DIAGNOSTIC_OUTPUT_PATH}\"}"
fi
export RUNTIME_EXTRA_ENV

python scripts/amd/run_qwen3_30b_a3b.py --help > "${PROVENANCE_DIR}/launcher-help.txt"

git rev-parse HEAD > "${PROVENANCE_DIR}/miles-git-sha.txt"
git status --short --branch > "${PROVENANCE_DIR}/miles-git-status.txt"
git diff --binary HEAD > "${PROVENANCE_DIR}/miles-dirty.diff"
while IFS= read -r untracked; do
  diff_status=0
  git diff --no-index --binary /dev/null "${untracked}" >> "${PROVENANCE_DIR}/miles-dirty.diff" || diff_status=$?
  if [[ "${diff_status}" -ne 0 && "${diff_status}" -ne 1 ]]; then
    exit "${diff_status}"
  fi
done < <(git ls-files --others --exclude-standard)

{
  echo "run_id=${RUN_ID}"
  echo "variant=${VARIANT}"
  echo "utc=$(date -u +%FT%TZ)"
  echo "container_image_id=${CONTAINER_IMAGE_ID}"
  echo "container_repo_digest=${CONTAINER_REPO_DIGEST}"
  echo "container_hostname=$(hostname)"
  echo "repo_root=${REPO_ROOT}"
  echo "model_path=${MODEL_PATH}"
  echo "trainer_checkpoint_path=${TRAIN_CHECKPOINT_PATH}"
  echo "dataset_path=${DATASET_PATH}"
} > "${PROVENANCE_DIR}/run.txt"

{
  cat /etc/os-release
  echo
  cat /proc/self/cgroup
} > "${PROVENANCE_DIR}/container.txt"

{
  command -v hipcc
  hipcc --version
  echo
  command -v rocm-smi
  rocm-smi --showproductname
} > "${PROVENANCE_DIR}/rocm.txt" 2>&1 || true
rocminfo > "${PROVENANCE_DIR}/rocminfo.txt" 2>&1 || true
rocm-smi --showtopo > "${PROVENANCE_DIR}/gpu-topology.txt" 2>&1 || true
rocm-smi --showproductname --showserial --showuniqueid > "${PROVENANCE_DIR}/gpu-identity.txt" 2>&1 || true

python - <<'PY'
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
provenance_dir = Path(os.environ["PROVENANCE_DIR"])

import miles

miles_path = Path(miles.__file__).resolve()
if not miles_path.is_relative_to(repo_root):
    raise RuntimeError(f"miles imports from {miles_path}, expected checkout {repo_root}")

source = {
    "cwd": str(repo_root),
    "miles_file": str(miles_path),
    "sys_path": sys.path,
    "checkout_is_import_source": True,
}
(provenance_dir / "python-source.json").write_text(json.dumps(source, indent=2) + "\n")

packages = {}
for distribution in ("torch", "sglang", "aiter", "megatron-core"):
    try:
        packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        packages[distribution] = None

modules = {}
for name in ("torch", "sglang", "aiter", "megatron"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        modules[name] = {"import_error": repr(exc)}
        continue
    modules[name] = {
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }

torch_module = importlib.import_module("torch")
runtime = {
    "python": sys.version,
    "packages": packages,
    "modules": modules,
    "torch_version": torch_module.__version__,
    "torch_hip": torch_module.version.hip,
    "gpu_count": torch_module.cuda.device_count(),
    "gpus": [
        {
            "index": index,
            "name": torch_module.cuda.get_device_name(index),
            "properties": str(torch_module.cuda.get_device_properties(index)),
        }
        for index in range(torch_module.cuda.device_count())
    ],
}

source_repositories = {}
for name, source_path in {
    "sglang": Path("/sgl-workspace/sglang"),
    "aiter": Path("/sgl-workspace/aiter"),
    "megatron": Path("/root/Megatron-LM"),
}.items():
    if not (source_path / ".git").exists():
        continue
    source_repositories[name] = {
        "path": str(source_path),
        "git_sha": subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "-C", str(source_path), "status", "--short"],
            text=True,
        ).splitlines(),
    }
runtime["source_repositories"] = source_repositories

(provenance_dir / "software.json").write_text(json.dumps(runtime, indent=2) + "\n")
PY

{
  echo "model_path=${MODEL_PATH}"
  find "${MODEL_PATH}" -maxdepth 2 -type f -printf '%P\n' | sort
  echo
  echo "trainer_checkpoint_path=${TRAIN_CHECKPOINT_PATH}"
  find "${TRAIN_CHECKPOINT_PATH}" -maxdepth 2 -type f -printf '%P\n' | sort
  echo
  echo "dataset_path=${DATASET_PATH}"
  find "$(dirname "${DATASET_PATH}")" -maxdepth 2 -type f -printf '%P\n' | sort
} > "${PROVENANCE_DIR}/model-dataset-manifest.txt"

{
  find "${MODEL_PATH}/.cache/huggingface/download" -type f -name '*.metadata' -print -exec sed -n '1,8p' {} \;
  find "$(dirname "${DATASET_PATH}")/.cache/huggingface/download" -type f -name '*.metadata' -print -exec sed -n '1,8p' {} \;
} > "${PROVENANCE_DIR}/huggingface-revisions.txt" 2>&1 || true

sha256sum "${MODEL_PATH}/config.json" "${MODEL_PATH}/model.safetensors.index.json" "${TRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" "${DATASET_PATH}" > "${PROVENANCE_DIR}/model-dataset-sha256.txt"

python - <<'PY'
import argparse
import importlib
import json
import os
import shlex
from pathlib import Path

from miles.utils.arguments import get_miles_extra_args_provider

launcher = importlib.import_module("scripts.amd.run_qwen3_30b_a3b")
captured = {}


def capture_execute_train(**kwargs):
    captured.update(kwargs)


launcher.U.execute_train = capture_execute_train
config = launcher.ScriptArgs(
    mode="debug_minimal",
    hardware="MI355X",
    num_gpus_per_node=8,
    run_id=os.environ["RUN_ID"],
    output_dir=os.environ["OUTPUT_ROOT"],
    extra_env_vars=os.environ["RUNTIME_EXTRA_ENV"],
    extra_args=os.environ["EXTRA_ARGS"],
)
launcher.execute(config)

train_args = captured["train_args"]
model_args = launcher.U.shell_safe_model_args(captured["megatron_model_type"])
train_script = str(Path(launcher.U.repo_base_dir) / "train.py")
command = shlex.join(["python3", train_script, *shlex.split(model_args), *shlex.split(train_args)])

parser = argparse.ArgumentParser(add_help=False)
get_miles_extra_args_provider()(parser)
resolved, _unknown = parser.parse_known_args(shlex.split(train_args))
last_value_checks = {
    "num_rollout": resolved.num_rollout,
    "debug_exit_after_rollout": resolved.debug_exit_after_rollout,
    "rollout_batch_size": resolved.rollout_batch_size,
    "n_samples_per_prompt": resolved.n_samples_per_prompt,
    "global_batch_size": resolved.global_batch_size,
    "rollout_max_response_len": resolved.rollout_max_response_len,
    "sglang_max_running_requests": resolved.sglang_max_running_requests,
    "sglang_enable_prefill_only_deterministic_inference": resolved.sglang_enable_prefill_only_deterministic_inference,
    "sglang_attention_backend": resolved.sglang_attention_backend,
    "rollout_health_check_first_wait": resolved.rollout_health_check_first_wait,
    "debug_prefill_logprob_repeats": resolved.debug_prefill_logprob_repeats,
    "debug_trainer_logprob_repeats": resolved.debug_trainer_logprob_repeats,
    "debug_compare_decode_prefill_logprobs": resolved.debug_compare_decode_prefill_logprobs,
    "debug_rollout_only": resolved.debug_rollout_only,
    "custom_generate_function_path": resolved.custom_generate_function_path,
}
diagnostic = os.environ["VARIANT"] == "b_worker_diagnostic"
triton_attention = os.environ["VARIANT"] in {"prefill_triton", "prefill_deterministic"}
expected = {
    "num_rollout": int(os.environ.get("NUM_ROLLOUT", "3")),
    "debug_exit_after_rollout": int(os.environ["DEBUG_EXIT_AFTER_ROLLOUT"]),
    "rollout_batch_size": 1 if diagnostic else 4,
    "n_samples_per_prompt": 1 if diagnostic else 2,
    "global_batch_size": 2 if diagnostic else 8,
    "rollout_max_response_len": 1 if diagnostic else 128,
    "sglang_max_running_requests": 1 if os.environ["VARIANT"] == "concurrency_1" else 8,
    "sglang_enable_prefill_only_deterministic_inference": os.environ["VARIANT"] == "prefill_deterministic",
    "sglang_attention_backend": "triton" if triton_attention else None,
    "rollout_health_check_first_wait": 180.0,
    "debug_prefill_logprob_repeats": 1 if diagnostic else 2,
    "debug_trainer_logprob_repeats": 1 if diagnostic else 2,
    "debug_compare_decode_prefill_logprobs": not diagnostic,
    "debug_rollout_only": diagnostic,
    "custom_generate_function_path": "tools.diagnose_prefill_workers.generate" if diagnostic else None,
}
if last_value_checks != expected:
    raise RuntimeError(f"CLI override resolution mismatch: observed={last_value_checks}, expected={expected}")

provenance_dir = Path(os.environ["PROVENANCE_DIR"])
(provenance_dir / "resolved-command.txt").write_text(command + "\n")
(provenance_dir / "resolved-train-args.txt").write_text(train_args.strip() + "\n")
(provenance_dir / "resolved-overrides.json").write_text(
    json.dumps(last_value_checks, indent=2, sort_keys=True) + "\n"
)
(provenance_dir / "runtime-extra-env.json").write_text(
    json.dumps(json.loads(os.environ["RUNTIME_EXTRA_ENV"]), indent=2, sort_keys=True) + "\n"
)
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight_dir=${PROVENANCE_DIR}"
  exit 0
fi

capture_sglang_server_info() {
  local attempt captured port target
  local server_info_host="${SERVER_INFO_HOST:-$(python -c 'from ray._private.services import get_node_ip_address; print(get_node_ip_address())')}"

  for port in 15000 15003 15006 15009; do
    target="${PROVENANCE_DIR}/resolved-sglang-server-info-${port}.json"
    captured=0
    for ((attempt = 1; attempt <= 450; attempt++)); do
      if curl --fail --silent --max-time 5 \
        "http://${server_info_host}:${port}/server_info" --output "${target}"; then
        captured=1
        break
      fi
      sleep 2
    done
    if [[ "${captured}" -ne 1 ]]; then
      echo "Failed to capture /server_info from ${server_info_host}:${port}" >&2
      return 1
    fi
  done
  sha256sum "${PROVENANCE_DIR}"/resolved-sglang-server-info-*.json \
    > "${PROVENANCE_DIR}/resolved-sglang-server-info-sha256.txt"
}

capture_sglang_server_info > "${PROVENANCE_DIR}/server-info-capture.log" 2>&1 &
server_info_capture_pid=$!

set +e
python scripts/amd/run_qwen3_30b_a3b.py --mode debug_minimal --hardware MI355X --num-gpus-per-node 8 --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}" --extra-env-vars "${RUNTIME_EXTRA_ENV}" --extra-args "${EXTRA_ARGS}" 2>&1 | tee "${RUN_DIR}/launcher.log"
launch_status=${PIPESTATUS[0]}
set -e

server_info_status=0
if kill -0 "${server_info_capture_pid}" 2>/dev/null; then
  kill "${server_info_capture_pid}" 2>/dev/null || true
  wait "${server_info_capture_pid}" 2>/dev/null || true
  server_info_status=1
else
  wait "${server_info_capture_pid}" || server_info_status=$?
fi
if [[ "${launch_status}" -eq 0 && "${server_info_status}" -ne 0 ]]; then
  echo "Training succeeded but complete SGLang /server_info capture failed" >&2
  launch_status="${server_info_status}"
fi

if [[ -f "${RUN_DIR}/dump_details/rollout_data/0.pt" ]] && compgen -G "${RUN_DIR}/dump_details/train_data/0_*.pt" > /dev/null; then
  python tools/analyze_logprob_abc.py --dump-details "${RUN_DIR}/dump_details" --rollout-id 0 --output-dir "${RUN_DIR}/analysis/rollout_0" 2>&1 | tee "${RUN_DIR}/analysis-rollout-0.log"
fi

echo "run_dir=${RUN_DIR}"
echo "launch_status=${launch_status}"
exit "${launch_status}"
