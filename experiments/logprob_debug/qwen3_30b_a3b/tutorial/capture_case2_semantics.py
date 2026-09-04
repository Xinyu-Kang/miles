#!/usr/bin/env python3
"""Capture source/runtime evidence for the Case 2 temperature mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any


EVIDENCE = (
    (
        "generation_temperature",
        Path("miles/rollout/sglang_rollout.py"),
        "temperature=args.rollout_temperature",
    ),
    (
        "prefill_greedy_raw_score",
        Path("miles/rollout/generate_utils/prefill_logprobs.py"),
        '"temperature": 0',
    ),
    (
        "trainer_temperature_argument",
        Path("miles/backends/training_utils/loss_hub/logit_processors.py"),
        "temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature",
    ),
    (
        "trainer_fp32_temperature_scaling",
        Path("miles/backends/training_utils/loss_hub/math_utils.py"),
        "chunk.div_(temperature)",
    ),
    (
        "sglang_raw_logprob_before_temperature",
        Path("/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py"),
        "original_logprobs = torch.log_softmax(logits, dim=-1)",
    ),
    (
        "sglang_selects_raw_logprob",
        Path("/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py"),
        "logprobs = original_logprobs",
    ),
)
FALLBACK_HOOK = (
    "experiments.logprob_debug.qwen3_30b_a3b.tutorial.trainer_normalizer_offset.install"
)
FALLBACK_EVIDENCE = (
    "trainer_normalizer_offset",
    Path("experiments/logprob_debug/qwen3_30b_a3b/tutorial/trainer_normalizer_offset.py"),
    "return _offset_result(result, float(state[\"offset\"]))",
)


def _last_option(tokens: list[str], option: str) -> str:
    values = [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == option]
    if not values:
        raise ValueError(f"Missing {option} in resolved command")
    return values[-1]


def _optional_last_option(tokens: list[str], option: str) -> str | None:
    values = [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == option]
    return values[-1] if values else None


def _source_record(name: str, path: Path, anchor: str, context: int = 5) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if anchor in line]
    if len(matches) != 1:
        raise ValueError(f"Expected one {anchor!r} in {path}, found {len(matches)}")
    index = matches[0]
    first = max(0, index - context)
    last = min(len(lines), index + context + 1)
    return {
        "name": name,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "anchor": anchor,
        "anchor_line": index + 1,
        "excerpt_first_line": first + 1,
        "excerpt_last_line": last,
        "excerpt": "\n".join(f"{line_number + 1}: {lines[line_number]}" for line_number in range(first, last)),
    }


def capture(
    *,
    resolved_args_path: Path,
    runtime_env_path: Path,
    server_args_path: Path,
) -> dict[str, Any]:
    tokens = shlex.split(resolved_args_path.read_text(encoding="utf-8"))
    temperature = float(_last_option(tokens, "--rollout-temperature"))
    custom_hook = _optional_last_option(tokens, "--custom-megatron-before-log-prob-hook-path")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Case 2 requires a positive finite rollout temperature, got {temperature}")
    if custom_hook is None and temperature == 1.0:
        raise ValueError("Native Case 2 requires a non-unit rollout temperature")
    if custom_hook == FALLBACK_HOOK and temperature != 1.0:
        raise ValueError("Fallback Case 2 requires rollout_temperature=1.0")
    if custom_hook not in (None, FALLBACK_HOOK):
        raise ValueError(f"Unexpected Case 2 Megatron logprob hook: {custom_hook}")
    if "--true-on-policy-mode" in tokens:
        raise ValueError("Case 2 raw-versus-processed trace assumes true-on-policy mode is disabled")
    runtime_env = json.loads(runtime_env_path.read_text(encoding="utf-8"))
    if runtime_env.get("SGLANG_RETURN_ORIGINAL_LOGPROB") != "1":
        raise ValueError("Case 2 requires SGLANG_RETURN_ORIGINAL_LOGPROB=1")
    offset = None
    if custom_hook == FALLBACK_HOOK:
        offset = float(runtime_env.get("MILES_LOGPROB_TRAINER_NORMALIZER_OFFSET_NAT", "nan"))
        if not math.isfinite(offset) or offset <= 0:
            raise ValueError("Fallback Case 2 requires a finite positive trainer offset")
        implementation = "guarded_trainer_normalizer_offset_fallback"
        stages = {
            "A": "unit-temperature generation returns SGLang raw logprobs",
            "B": "exact-token max_new_tokens=0 replay returns raw input-token logprobs",
            "C": "the existing Megatron pre-logprob hook subtracts the configured offset only from frozen C/repeat scores",
        }
        conclusion = "The explicit fallback leaves A/B raw and deliberately offsets frozen trainer C"
        evidence = (*EVIDENCE, FALLBACK_EVIDENCE)
    else:
        implementation = "native_non_unit_temperature"
        stages = {
            "A": "generation samples at the configured temperature but SGLang returns the cached pre-temperature log_softmax",
            "B": "exact-token max_new_tokens=0 replay forces temperature=0 and returns raw input-token logprobs",
            "C": "Megatron divides fp32 logits by rollout_temperature before computing target-token logprobs",
        }
        conclusion = "A and B are raw-logit scores while C is temperature-scaled in this non-true-on-policy configuration"
        evidence = EVIDENCE
    server_lines = [line for line in server_args_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(server_lines) != 4:
        raise ValueError(f"Expected four complete SGLang ServerArgs lines, got {len(server_lines)}")
    sources = [_source_record(*item) for item in evidence]
    return {
        "schema_version": 1,
        "fault_implementation": implementation,
        "trainer_normalizer_offset_nat": offset,
        "rollout_temperature": temperature,
        "SGLANG_RETURN_ORIGINAL_LOGPROB": "1",
        "true_on_policy_mode": False,
        "sglang_worker_server_args_count": len(server_lines),
        "stages": stages,
        "conclusion": conclusion,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-args", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--server-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(
        resolved_args_path=args.resolved_args,
        runtime_env_path=args.runtime_env,
        server_args_path=args.server_args,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote Case 2 source/runtime semantics to {args.output}")


if __name__ == "__main__":
    main()
