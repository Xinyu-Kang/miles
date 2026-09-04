#!/usr/bin/env python3
"""Validate resolved tutorial args and write a machine-readable arm manifest."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

from .fault_guard import ACTIVE_FAULT_ENV, ENABLE_ENV, STRENGTH_ENV

_CASE1_HOOK = (
    "experiments.logprob_debug.qwen3_30b_a3b.tutorial.decode_logprob_offset.process"
)
_CASE2_HOOK = (
    "experiments.logprob_debug.qwen3_30b_a3b.tutorial.trainer_normalizer_offset.install"
)
_CASE3_STIMULUS_HOOK = (
    "experiments.logprob_debug.qwen3_30b_a3b.tutorial.case3_training_signal.process"
)


def _last_value(tokens: list[str], option: str, default: str | None = None) -> str | None:
    values = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens):
                raise ValueError(f"{option} has no value")
            values.append(tokens[index + 1])
    return values[-1] if values else default


def _count(tokens: list[str], option: str) -> int:
    return sum(token == option for token in tokens)


def validate(
    *,
    train_args_path: Path,
    runtime_env_path: Path,
    case: str,
    arm: str,
    expected_temperature: float,
    expected_update_interval: int,
    expected_exit_after: int,
    expected_fault: str | None,
    expected_strength: float | None,
    expected_learning_rate: float = 1e-6,
    expected_skip_update_at: int | None = None,
) -> dict[str, Any]:
    train_args = train_args_path.read_text(encoding="utf-8")
    tokens = shlex.split(train_args)
    runtime_env = json.loads(runtime_env_path.read_text(encoding="utf-8"))

    actual = {
        "rollout_batch_size": int(_last_value(tokens, "--rollout-batch-size")),
        "n_samples_per_prompt": int(_last_value(tokens, "--n-samples-per-prompt")),
        "global_batch_size": int(_last_value(tokens, "--global-batch-size")),
        "rollout_max_response_len": int(_last_value(tokens, "--rollout-max-response-len")),
        "debug_prefill_logprob_repeats": int(
            _last_value(tokens, "--debug-prefill-logprob-repeats")
        ),
        "debug_trainer_logprob_repeats": int(
            _last_value(tokens, "--debug-trainer-logprob-repeats")
        ),
        "debug_exit_after_rollout": int(_last_value(tokens, "--debug-exit-after-rollout")),
        "rollout_temperature": float(_last_value(tokens, "--rollout-temperature")),
        "update_weights_interval": int(
            _last_value(tokens, "--update-weights-interval", "1")
        ),
        "learning_rate": float(_last_value(tokens, "--lr", "0.000001")),
        "debug_skip_rollout_weight_update_at": (
            None
            if _last_value(tokens, "--debug-skip-rollout-weight-update-at") is None
            else int(_last_value(tokens, "--debug-skip-rollout-weight-update-at"))
        ),
        "rollout_all_samples_process_path": _last_value(
            tokens, "--rollout-all-samples-process-path"
        ),
        "custom_megatron_before_log_prob_hook_path": _last_value(
            tokens, "--custom-megatron-before-log-prob-hook-path"
        ),
        "dump_details": _last_value(tokens, "--dump-details"),
        "seed": int(_last_value(tokens, "--seed")),
        "rollout_seed": int(_last_value(tokens, "--rollout-seed")),
    }
    expected = {
        "rollout_batch_size": 4,
        "n_samples_per_prompt": 2,
        "global_batch_size": 8,
        "rollout_max_response_len": 128,
        "debug_prefill_logprob_repeats": 2,
        "debug_trainer_logprob_repeats": 2,
        "debug_exit_after_rollout": expected_exit_after,
        "rollout_temperature": expected_temperature,
        "update_weights_interval": expected_update_interval,
        "learning_rate": expected_learning_rate,
        "debug_skip_rollout_weight_update_at": expected_skip_update_at,
        "rollout_all_samples_process_path": (
            _CASE1_HOOK
            if case == "case1" and arm == "fault"
            else _CASE3_STIMULUS_HOOK if case == "case3" else None
        ),
        "custom_megatron_before_log_prob_hook_path": (
            _CASE2_HOOK
            if case == "case2" and expected_fault == "trainer_normalizer_offset"
            else None
        ),
        "seed": 1234,
        "rollout_seed": 1234,
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(f"Resolved tutorial arguments do not match the arm: {mismatches}")
    if not actual["dump_details"]:
        raise ValueError("Tutorial runs require --dump-details")
    for flag in (
        "--debug-compare-decode-prefill-logprobs",
        "--dump-details",
    ):
        if _count(tokens, flag) == 0:
            raise ValueError(f"Resolved tutorial args are missing {flag}")
    for forbidden in ("--fully-async", "--true-on-policy-mode", "--use-tis"):
        if _count(tokens, forbidden):
            raise ValueError(f"Tutorial runs forbid {forbidden}")

    indicators = {
        "decode_logprob_offset": actual["rollout_all_samples_process_path"] == _CASE1_HOOK,
        "temperature_definition_mismatch": not math.isclose(
            actual["rollout_temperature"], 1.0
        ),
        "trainer_normalizer_offset": (
            actual["custom_megatron_before_log_prob_hook_path"] == _CASE2_HOOK
        ),
        "stale_rollout_weights": actual["debug_skip_rollout_weight_update_at"] is not None,
    }
    active_indicators = [name for name, enabled in indicators.items() if enabled]
    if arm == "fault":
        if active_indicators != [expected_fault]:
            raise ValueError(
                f"Exactly fault {expected_fault!r} must be active, got {active_indicators}"
            )
        if runtime_env.get(ENABLE_ENV) != "1":
            raise ValueError(f"Fault arm is missing {ENABLE_ENV}=1 in worker runtime")
        if runtime_env.get(ACTIVE_FAULT_ENV) != expected_fault:
            raise ValueError("Fault arm runtime name does not match the resolved fault")
        observed_strength = float(runtime_env.get(STRENGTH_ENV, "nan"))
        if expected_strength is None or observed_strength != expected_strength:
            raise ValueError(
                f"Fault strength mismatch: expected {expected_strength}, got {observed_strength}"
            )
        if expected_fault == "trainer_normalizer_offset":
            offset = float(runtime_env.get("MILES_LOGPROB_TRAINER_NORMALIZER_OFFSET_NAT", "nan"))
            if offset != expected_strength:
                raise ValueError(
                    f"Trainer-normalizer offset mismatch: expected {expected_strength}, got {offset}"
                )
    else:
        if active_indicators:
            raise ValueError(f"{arm} arm unexpectedly activates faults: {active_indicators}")
        leaked = [key for key in (ENABLE_ENV, ACTIVE_FAULT_ENV, STRENGTH_ENV) if key in runtime_env]
        if leaked:
            raise ValueError(f"{arm} arm leaks fault runtime keys: {leaked}")
    if case == "case3" and runtime_env.get("MILES_ENABLE_LOGPROB_TRAINING_STIMULUS") != "1":
        raise ValueError("Case 3 requires the common nonzero training stimulus")

    return {
        "schema_version": 1,
        "case": case,
        "arm": arm,
        "deliberately_injected": arm == "fault",
        "active_fault": expected_fault,
        "fault_strength": expected_strength,
        "resolved": actual,
        "fault_indicators": indicators,
        "runtime_fault_environment": {
            key: runtime_env.get(key)
            for key in (ENABLE_ENV, ACTIVE_FAULT_ENV, STRENGTH_ENV)
            if key in runtime_env
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-args", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", choices=("clean", "fault", "fixed"), required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--update-interval", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--skip-update-at", type=int)
    parser.add_argument("--exit-after", type=int, required=True)
    parser.add_argument("--fault")
    parser.add_argument("--strength", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = validate(
        train_args_path=args.train_args,
        runtime_env_path=args.runtime_env,
        case=args.case,
        arm=args.arm,
        expected_temperature=args.temperature,
        expected_update_interval=args.update_interval,
        expected_exit_after=args.exit_after,
        expected_fault=args.fault,
        expected_strength=args.strength,
        expected_learning_rate=args.learning_rate,
        expected_skip_update_at=args.skip_update_at,
    )
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validated tutorial configuration: {args.case}/{args.arm}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
