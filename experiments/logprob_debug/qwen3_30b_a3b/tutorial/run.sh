#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PARENT_RUNNER="${SCRIPT_DIR}/../run.sh"
FAULT_GUARD="${SCRIPT_DIR}/fault_guard.py"
cd "${REPO_ROOT}"

TUTORIAL_CASE="${TUTORIAL_CASE:-case0}"
TUTORIAL_ARM="${TUTORIAL_ARM:-clean}"
# Use the previously validated node-072 deterministic-prefill configuration as
# the tutorial base so both replay floors stay below the 0.01 localization
# gate. The parent runner records its complete resolved SGLang ServerArgs.
TUTORIAL_BASE_VARIANT="${TUTORIAL_BASE_VARIANT:-prefill_deterministic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/logprob-debug-artifacts}"
RUN_ID="${RUN_ID:-qwen3-30b-logprob-tutorial-${TUTORIAL_CASE}-${TUTORIAL_ARM}-$(date -u +%Y%m%d-%H%M%S)}"
ENABLE_FAULT="${MILES_ENABLE_LOGPROB_FAULT_INJECTION:-0}"
ACTIVE_FAULT="${MILES_LOGPROB_ACTIVE_FAULT:-}"
TEMPERATURE=1.0
UPDATE_INTERVAL=1
LEARNING_RATE=0.000001
NUM_ROLLOUT=3
DEBUG_EXIT_AFTER_ROLLOUT=1
SKIP_UPDATE_AT=""
FAULT_NAME=""
FAULT_STRENGTH=""
CASE2_FAULT_MODE="${TUTORIAL_CASE2_FAULT_MODE:-temperature}"
CASE3_MOE_RUNNER_BACKEND="${TUTORIAL_CASE3_MOE_RUNNER_BACKEND:-}"
EXTRA_ARGS=()

case "${TUTORIAL_CASE}" in
  case0)
    if [[ "${TUTORIAL_ARM}" != "clean" ]]; then
      echo "case0 only supports TUTORIAL_ARM=clean" >&2
      exit 2
    fi
    ;;
  case1)
    case "${TUTORIAL_ARM}" in
      clean|fixed)
        ;;
      fault)
        FAULT_NAME=decode_logprob_offset
        FAULT_STRENGTH="${TUTORIAL_FAULT_DELTA_NAT:-0.10}"
        EXTRA_ARGS+=(
          --rollout-all-samples-process-path
          experiments.logprob_debug.qwen3_30b_a3b.tutorial.decode_logprob_offset.process
        )
        ;;
      *)
        echo "case1 arm must be clean, fault, or fixed" >&2
        exit 2
        ;;
    esac
    ;;
  case2)
    case "${TUTORIAL_ARM}" in
      clean|fixed)
        ;;
      fault)
        case "${CASE2_FAULT_MODE}" in
          temperature)
            FAULT_NAME=temperature_definition_mismatch
            TEMPERATURE="${TUTORIAL_ROLLOUT_TEMPERATURE:-0.9}"
            FAULT_STRENGTH="${TEMPERATURE}"
            ;;
          trainer_offset)
            # The real 0.9/0.8 mismatch is retained as a measured weak-signal
            # pilot. This explicitly synthetic fallback moves only frozen C.
            FAULT_NAME=trainer_normalizer_offset
            FAULT_STRENGTH="${TUTORIAL_TRAINER_NORMALIZER_OFFSET_NAT:-0.10}"
            EXTRA_ARGS+=(
              --custom-megatron-before-log-prob-hook-path
              experiments.logprob_debug.qwen3_30b_a3b.tutorial.trainer_normalizer_offset.install
            )
            ;;
          *)
            echo "Unknown TUTORIAL_CASE2_FAULT_MODE=${CASE2_FAULT_MODE}" >&2
            exit 2
            ;;
        esac
        ;;
      *)
        echo "case2 arm must be clean, fault, or fixed" >&2
        exit 2
        ;;
    esac
    ;;
  case3)
    # Keep rollout 2 non-final: Miles always checkpoints the final configured
    # rollout even when --save-interval is very large.  The debug exit still
    # stops after exactly three rollouts, before rollout 3 is generated.
    NUM_ROLLOUT=4
    DEBUG_EXIT_AFTER_ROLLOUT=3
    LEARNING_RATE="${TUTORIAL_CASE3_LEARNING_RATE:-0.00005}"
    EXTRA_ARGS+=(
      --rollout-all-samples-process-path
      experiments.logprob_debug.qwen3_30b_a3b.tutorial.case3_training_signal.process
    )
    if [[ -n "${CASE3_MOE_RUNNER_BACKEND}" ]]; then
      if [[ "${CASE3_MOE_RUNNER_BACKEND}" != "triton" ]]; then
        echo "case3 diagnostic only supports TUTORIAL_CASE3_MOE_RUNNER_BACKEND=triton" >&2
        exit 2
      fi
      EXTRA_ARGS+=(--sglang-moe-runner-backend "${CASE3_MOE_RUNNER_BACKEND}")
    fi
    case "${TUTORIAL_ARM}" in
      clean|fixed)
        ;;
      fault)
        FAULT_NAME=stale_rollout_weights
        SKIP_UPDATE_AT="${TUTORIAL_SKIP_UPDATE_AT:-0}"
        FAULT_STRENGTH=1
        EXTRA_ARGS+=(--debug-skip-rollout-weight-update-at "${SKIP_UPDATE_AT}")
        ;;
      *)
        echo "case3 arm must be clean, fault, or fixed" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown TUTORIAL_CASE=${TUTORIAL_CASE}" >&2
    exit 2
    ;;
esac

if [[ "${TUTORIAL_ARM}" == "fault" ]]; then
  if [[ "${ENABLE_FAULT}" != "1" ]]; then
    echo "Fault arms require MILES_ENABLE_LOGPROB_FAULT_INJECTION=1" >&2
    exit 2
  fi
  if [[ -n "${ACTIVE_FAULT}" && "${ACTIVE_FAULT}" != "${FAULT_NAME}" ]]; then
    echo "Requested fault ${FAULT_NAME}, but MILES_LOGPROB_ACTIVE_FAULT=${ACTIVE_FAULT}" >&2
    exit 2
  fi
  case "${CI:-}" in
    ""|0|false|False|FALSE|no|No|NO|off|Off|OFF)
      ;;
    *)
      echo "Tutorial fault injection is forbidden in CI" >&2
      exit 2
      ;;
  esac
  python "${FAULT_GUARD}" banner --fault "${FAULT_NAME}" --strength "${FAULT_STRENGTH}"
else
  if [[ "${ENABLE_FAULT}" == "1" || -n "${ACTIVE_FAULT}" ]]; then
    echo "Clean/fixed arms require fault injection environment variables to be unset" >&2
    exit 2
  fi
fi

EXTRA_ARGS+=(--rollout-temperature "${TEMPERATURE}")
EXTRA_ARGS+=(--update-weights-interval "${UPDATE_INTERVAL}")
EXTRA_ARGS+=(--lr "${LEARNING_RATE}")
printf -v MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS '%s ' "${EXTRA_ARGS[@]}"

MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON=""
if [[ "${TUTORIAL_CASE}" == "case3" ]]; then
  MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON='{"MILES_ENABLE_LOGPROB_TRAINING_STIMULUS":"1"}'
fi
if [[ "${TUTORIAL_ARM}" == "fault" ]]; then
  BUILD_RUNTIME_ARGS=(
    build-runtime-env
    --fault "${FAULT_NAME}"
    --strength "${FAULT_STRENGTH}"
  )
  if [[ "${TUTORIAL_CASE}" == "case1" ]]; then
    BUILD_RUNTIME_ARGS+=(--set "MILES_LOGPROB_DECODE_OFFSET_NAT=${FAULT_STRENGTH}")
  elif [[ "${TUTORIAL_CASE}" == "case2" && "${CASE2_FAULT_MODE}" == "trainer_offset" ]]; then
    BUILD_RUNTIME_ARGS+=(--set "MILES_LOGPROB_TRAINER_NORMALIZER_OFFSET_NAT=${FAULT_STRENGTH}")
  fi
  FAULT_RUNTIME_ENV_JSON="$(
    python "${FAULT_GUARD}" "${BUILD_RUNTIME_ARGS[@]}"
  )"
  if [[ -n "${MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON}" ]]; then
    MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON="$(
      python "${FAULT_GUARD}" merge-runtime-env \
        "${MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON}" "${FAULT_RUNTIME_ENV_JSON}"
    )"
  else
    MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON="${FAULT_RUNTIME_ENV_JSON}"
  fi
fi

export RUN_ID OUTPUT_ROOT NUM_ROLLOUT DEBUG_EXIT_AFTER_ROLLOUT
export VARIANT="${TUTORIAL_BASE_VARIANT}"
export MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS
export MILES_LOGPROB_EXPERIMENT_RUNTIME_ENV_JSON

PREFLIGHT_ONLY=1 "${PARENT_RUNNER}"

PROVENANCE_DIR="${OUTPUT_ROOT}/${RUN_ID}/provenance"
VALIDATOR_ARGS=(
  --train-args "${PROVENANCE_DIR}/resolved-train-args.txt"
  --runtime-env "${PROVENANCE_DIR}/runtime-extra-env.json"
  --case "${TUTORIAL_CASE}"
  --arm "${TUTORIAL_ARM}"
  --temperature "${TEMPERATURE}"
  --update-interval "${UPDATE_INTERVAL}"
  --learning-rate "${LEARNING_RATE}"
  --exit-after "${DEBUG_EXIT_AFTER_ROLLOUT}"
  --output "${PROVENANCE_DIR}/tutorial-config.json"
)
if [[ "${TUTORIAL_ARM}" == "fault" ]]; then
  VALIDATOR_ARGS+=(--fault "${FAULT_NAME}" --strength "${FAULT_STRENGTH}")
fi
if [[ -n "${SKIP_UPDATE_AT}" ]]; then
  VALIDATOR_ARGS+=(--skip-update-at "${SKIP_UPDATE_AT}")
fi
python -m experiments.logprob_debug.qwen3_30b_a3b.tutorial.validate_run_config "${VALIDATOR_ARGS[@]}"

{
  echo "TUTORIAL_CASE=${TUTORIAL_CASE}"
  echo "TUTORIAL_ARM=${TUTORIAL_ARM}"
  echo "TUTORIAL_BASE_VARIANT=${TUTORIAL_BASE_VARIANT}"
  echo "RUN_ID=${RUN_ID}"
  echo "NUM_ROLLOUT=${NUM_ROLLOUT}"
  echo "DEBUG_EXIT_AFTER_ROLLOUT=${DEBUG_EXIT_AFTER_ROLLOUT}"
  echo "MILES_ENABLE_LOGPROB_FAULT_INJECTION=${ENABLE_FAULT}"
  echo "MILES_LOGPROB_ACTIVE_FAULT=${FAULT_NAME}"
  echo "FAULT_STRENGTH=${FAULT_STRENGTH}"
  echo "CASE2_FAULT_MODE=${CASE2_FAULT_MODE}"
  echo "CASE3_MOE_RUNNER_BACKEND=${CASE3_MOE_RUNNER_BACKEND}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "SKIP_UPDATE_AT=${SKIP_UPDATE_AT}"
  echo "MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS=${MILES_LOGPROB_EXPERIMENT_EXTRA_ARGS}"
} > "${PROVENANCE_DIR}/tutorial-invocation.txt"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "tutorial_preflight_dir=${PROVENANCE_DIR}"
  exit 0
fi

PREFLIGHT_ONLY=0 "${PARENT_RUNNER}"

RAY_LOG_DIR="/tmp/ray/session_latest/logs"
SGLANG_ARGS_PATH="${PROVENANCE_DIR}/sglang-server-args.txt"
shopt -s nullglob
worker_error_logs=("${RAY_LOG_DIR}"/worker-*.err)
job_driver_logs=("${RAY_LOG_DIR}"/job-driver-*.log)
shopt -u nullglob
if (( ${#worker_error_logs[@]} == 0 )); then
  echo "No Ray worker error logs found under ${RAY_LOG_DIR}" >&2
  exit 1
fi
grep -h "server_args=ServerArgs" "${worker_error_logs[@]}" > "${SGLANG_ARGS_PATH}"
if [[ "$(wc -l < "${SGLANG_ARGS_PATH}")" -ne 4 ]]; then
  echo "Expected four complete SGLang ServerArgs records in ${SGLANG_ARGS_PATH}" >&2
  exit 1
fi
if (( ${#job_driver_logs[@]} != 1 )); then
  echo "Expected exactly one Ray job-driver log under ${RAY_LOG_DIR}" >&2
  exit 1
fi
cp "${job_driver_logs[0]}" "${PROVENANCE_DIR}/ray-job.log"

if [[ "${TUTORIAL_CASE}" == "case2" && "${TUTORIAL_ARM}" == "fault" ]]; then
  python -m experiments.logprob_debug.qwen3_30b_a3b.tutorial.capture_case2_semantics \
    --resolved-args "${PROVENANCE_DIR}/resolved-train-args.txt" \
    --runtime-env "${PROVENANCE_DIR}/runtime-extra-env.json" \
    --server-args "${SGLANG_ARGS_PATH}" \
    --output "${PROVENANCE_DIR}/case2-logprob-semantics.json"
fi

for ((rollout_id = 0; rollout_id < DEBUG_EXIT_AFTER_ROLLOUT; rollout_id++)); do
  rollout_dump="${OUTPUT_ROOT}/${RUN_ID}/dump_details/rollout_data/${rollout_id}.pt"
  if [[ ! -f "${rollout_dump}" ]]; then
    echo "Missing required rollout dump ${rollout_dump}" >&2
    exit 1
  fi
  python tools/analyze_logprob_abc.py \
    --dump-details "${OUTPUT_ROOT}/${RUN_ID}/dump_details" \
    --rollout-id "${rollout_id}" \
    --output-dir "${OUTPUT_ROOT}/${RUN_ID}/analysis/rollout_${rollout_id}"
done

echo "tutorial_run_dir=${OUTPUT_ROOT}/${RUN_ID}"
