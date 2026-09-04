#!/usr/bin/env python3
"""Analyze A/B/C logprobs from one Miles --dump-details rollout.

A is the generation-time SGLang rollout logprob, B is a clean-cache
exact-token SGLang prefill replay, and C is Megatron's old-policy forward
score captured before training. The analyzer fails closed on any identity,
alignment, length, policy-version, or finite-value violation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from miles.dashboard.dump_reader import DumpReader, TrainRow
from miles.utils.types import Sample

_COMPARISON_NAMES = ("A_minus_B", "B_minus_C", "A_minus_C", "B0_minus_B1", "C0_minus_C1")


@dataclass(frozen=True)
class SampleScores:
    sample_index: int
    sample_occurrence: int
    group_index: int | None
    status: str
    policy_version: str
    response_token_ids: tuple[int, ...]
    active: torch.Tensor
    a: torch.Tensor
    b_repeats: tuple[torch.Tensor, ...]
    c: torch.Tensor
    c_repeat_1: torch.Tensor

    @property
    def key(self) -> tuple[int, int]:
        return self.sample_index, self.sample_occurrence

    def delta(self, name: str) -> torch.Tensor:
        if name == "A_minus_B":
            return self.a - self.b_repeats[0]
        if name == "B_minus_C":
            return self.b_repeats[0] - self.c
        if name == "A_minus_C":
            return self.a - self.c
        if name == "B0_minus_B1":
            return self.b_repeats[0] - self.b_repeats[1]
        if name == "C0_minus_C1":
            return self.c - self.c_repeat_1
        raise KeyError(name)


def _as_f64(values: Iterable[float] | torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        result = values.detach().to(device="cpu", dtype=torch.float64)
    else:
        result = torch.as_tensor(list(values), dtype=torch.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={tuple(result.shape)}")
    if not torch.isfinite(result).all():
        bad = int((~torch.isfinite(result)).sum())
        raise ValueError(f"{name} contains {bad} NaN or Inf value(s)")
    return result


def _as_token_ids(values: Iterable[int] | torch.Tensor, *, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        result = values.detach().cpu()
    else:
        result = torch.as_tensor(list(values))
    if result.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={tuple(result.shape)}")
    return [int(value) for value in result]


def _as_mask(values: Iterable[int] | torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        result = values.detach().cpu()
    else:
        result = torch.as_tensor(list(values))
    if result.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={tuple(result.shape)}")
    if not torch.logical_or(result == 0, result == 1).all():
        raise ValueError(f"{name} contains values other than 0 and 1")
    return result.bool()


def _stats(delta: torch.Tensor) -> dict[str, float | int]:
    values = delta.detach().double().flatten()
    if values.numel() == 0:
        raise ValueError("No active tokens for statistics")
    if not torch.isfinite(values).all():
        bad = int((~torch.isfinite(values)).sum())
        raise ValueError(f"Found {bad} non-finite differences")

    absolute = values.abs()
    return {
        "count": int(values.numel()),
        "signed_mean": float(values.mean()),
        "mean_abs": float(absolute.mean()),
        "rmse": float(torch.sqrt(torch.mean(values.square()))),
        "p50_abs": float(torch.quantile(absolute, 0.50)),
        "p95_abs": float(torch.quantile(absolute, 0.95)),
        "p99_abs": float(torch.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        "frac_abs_gt_1e-3": float((absolute > 1e-3).double().mean()),
        "frac_abs_gt_1e-2": float((absolute > 1e-2).double().mean()),
        "frac_abs_gt_5e-2": float((absolute > 5e-2).double().mean()),
        "frac_abs_gt_1e-1": float((absolute > 1e-1).double().mean()),
    }


def _optional_stats(parts: Sequence[torch.Tensor]) -> dict[str, float | int] | None:
    nonempty = [part.flatten() for part in parts if part.numel()]
    return _stats(torch.cat(nonempty)) if nonempty else None


def _policy_version(sample: Sample, train: TrainRow, *, key: tuple[int, int]) -> str:
    rollout_versions = [str(value) for value in sample.weight_versions]
    if not rollout_versions:
        raise ValueError(f"Missing rollout policy version for sample key {key}")
    if len(set(rollout_versions)) != 1:
        raise ValueError(f"Mixed rollout policy versions for sample key {key}: {rollout_versions}")

    train_versions = None if train.weight_versions is None else [str(value) for value in train.weight_versions]
    if train_versions != rollout_versions:
        raise ValueError(
            f"Rollout/train policy-version metadata mismatch for sample key {key}: "
            f"rollout={rollout_versions}, train={train_versions}"
        )
    return rollout_versions[0]


def _debug_record(sample: Sample, *, key: tuple[int, int]) -> dict[str, Any]:
    record = (sample.metadata or {}).get("logprob_debug")
    if not isinstance(record, dict):
        raise ValueError(f"Missing logprob_debug metadata for sample key {key}")
    required = {"response_token_ids", "decode_logprobs", "prefill_logprobs_repeats"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Incomplete logprob_debug metadata for sample key {key}: missing={sorted(missing)}")
    return record


def _validate_identity(sample: Sample, train: TrainRow, record: dict[str, Any], *, key: tuple[int, int]) -> list[int]:
    response_length = int(sample.response_length)
    if response_length <= 0:
        raise ValueError(f"Sample key {key} has no response tokens")

    rollout_tokens = [int(token) for token in sample.tokens]
    response_token_ids = rollout_tokens[-response_length:]
    debug_token_ids = [int(token) for token in record["response_token_ids"]]
    if debug_token_ids != response_token_ids:
        raise ValueError(
            f"Rollout/debug response token mismatch for sample key {key}: "
            f"rollout={response_token_ids[:8]}, debug={debug_token_ids[:8]}"
        )

    train_tokens = _as_token_ids(train.tokens, name=f"trainer tokens[{key}]")
    if train_tokens != rollout_tokens:
        raise ValueError(
            f"Rollout/trainer token mismatch for sample key {key}: "
            f"rollout_len={len(rollout_tokens)}, train_len={len(train_tokens)}"
        )
    if int(train.total_length) != len(rollout_tokens) or int(train.response_length) != response_length:
        raise ValueError(
            f"Rollout/trainer length metadata mismatch for sample key {key}: "
            f"rollout_total={len(rollout_tokens)}, train_total={train.total_length}, "
            f"rollout_response={response_length}, train_response={train.response_length}"
        )
    return response_token_ids


def _load_scores(sample: Sample, train: TrainRow, *, key: tuple[int, int]) -> SampleScores:
    if train.alignment_failed:
        raise ValueError(f"CP/PP/TP alignment failed for sample key {key}")
    if train.log_probs is None:
        raise ValueError(
            f"Trainer log_probs are absent for sample key {key}; "
            "C must be the pre-update old-policy forward score"
        )

    if train.debug_repeat_1_log_probs is None:
        raise ValueError(
            f"Trainer repeat log_probs are absent for sample key {key}; "
            "C0 and C1 must both be captured before the optimizer update"
        )

    record = _debug_record(sample, key=key)
    response_token_ids = _validate_identity(sample, train, record, key=key)
    expected = len(response_token_ids)
    a = _as_f64(record["decode_logprobs"], name=f"A[{key}]")
    rollout_a = _as_f64(sample.rollout_log_probs or [], name=f"sample.rollout_log_probs[{key}]")
    if not torch.equal(a, rollout_a):
        raise ValueError(f"Debug mode replaced or changed generation-time A for sample key {key}")

    raw_repeats = record["prefill_logprobs_repeats"]
    if not isinstance(raw_repeats, list) or len(raw_repeats) < 2:
        raise ValueError(f"Expected at least two B repeats for sample key {key}, got {len(raw_repeats)}")
    b_repeats = tuple(
        _as_f64(values, name=f"B{repeat}[{key}]") for repeat, values in enumerate(raw_repeats)
    )
    c = _as_f64(train.log_probs, name=f"C[{key}]")
    c_repeat_1 = _as_f64(train.debug_repeat_1_log_probs, name=f"C1[{key}]")
    active = _as_mask(train.loss_mask, name=f"loss_mask[{key}]")

    lengths = {
        "tokens": expected,
        "A": int(a.numel()),
        "C": int(c.numel()),
        "C1": int(c_repeat_1.numel()),
        "mask": int(active.numel()),
        **{f"B{repeat}": int(values.numel()) for repeat, values in enumerate(b_repeats)},
    }
    if any(length != expected for length in lengths.values()):
        raise ValueError(f"Length mismatch for sample key {key}: expected={expected}, observed={lengths}")

    if sample.loss_mask is not None:
        rollout_mask = _as_mask(sample.loss_mask, name=f"rollout loss_mask[{key}]")
        if not torch.equal(active, rollout_mask):
            raise ValueError(f"Rollout/trainer loss-mask mismatch for sample key {key}")

    return SampleScores(
        sample_index=key[0],
        sample_occurrence=key[1],
        group_index=sample.group_index,
        status=sample.status.value,
        policy_version=_policy_version(sample, train, key=key),
        response_token_ids=tuple(response_token_ids),
        active=active,
        a=a,
        b_repeats=b_repeats,
        c=c,
        c_repeat_1=c_repeat_1,
    )


def _comparison_summary(scores: Sequence[SampleScores], name: str) -> dict[str, Any]:
    overall_parts = []
    position_zero_parts = []
    position_later_parts = []
    per_position_parts: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
    per_sample = []

    for score in scores:
        delta = score.delta(name)
        active_delta = delta[score.active]
        overall_parts.append(active_delta)
        if score.active[0]:
            position_zero_parts.append(delta[0:1])
        position_later_parts.append(delta[1:][score.active[1:]])
        for position in torch.nonzero(score.active, as_tuple=False).flatten().tolist():
            per_position_parts[int(position)].append(delta[position : position + 1])
        per_sample.append(
            {
                "sample_index": score.sample_index,
                "sample_occurrence": score.sample_occurrence,
                "group_index": score.group_index,
                "response_length": len(score.response_token_ids),
                "active_tokens": int(score.active.sum()),
                "stats": _optional_stats([active_delta]),
            }
        )

    return {
        "overall": _optional_stats(overall_parts),
        "position_0": _optional_stats(position_zero_parts),
        "position_1_plus": _optional_stats(position_later_parts),
        "per_sample": per_sample,
        "per_position": [
            {"token_position": position, "stats": _optional_stats(parts)}
            for position, parts in sorted(per_position_parts.items())
        ],
    }


def _token_rows(scores: Sequence[SampleScores], rollout_id: int) -> list[dict[str, Any]]:
    rows = []
    for score in scores:
        deltas = {name: score.delta(name) for name in _COMPARISON_NAMES}
        for position, token_id in enumerate(score.response_token_ids):
            row = {
                "rollout_id": rollout_id,
                "sample_index": score.sample_index,
                "sample_occurrence": score.sample_occurrence,
                "group_index": score.group_index,
                "status": score.status,
                "policy_version": score.policy_version,
                "response_length": len(score.response_token_ids),
                "token_position": position,
                "token_id": token_id,
                "active": int(score.active[position]),
                "A_decode": float(score.a[position]),
                "C_trainer": float(score.c[position]),
                "C1_trainer_repeat": float(score.c_repeat_1[position]),
                **{name: float(values[position]) for name, values in deltas.items()},
            }
            for repeat, values in enumerate(score.b_repeats):
                row[f"B{repeat}_prefill"] = float(values[position])
            rows.append(row)
    return rows


def analyze(dump_details: Path, rollout_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    joined = DumpReader(dump_details).load_joined(rollout_id)
    occurrences: defaultdict[int, int] = defaultdict(int)
    scores = []

    for sample in joined.samples:
        if sample.index is None:
            raise ValueError("A rollout sample has no Sample.index")
        sample_index = int(sample.index)
        key = sample_index, occurrences[sample_index]
        occurrences[sample_index] += 1
        train = joined.train_rows.get(key)
        if train is None:
            raise ValueError(f"No joined train row for sample key {key}")
        scores.append(_load_scores(sample, train, key=key))

    if not scores:
        raise ValueError(f"Rollout {rollout_id} contains no samples")
    policy_versions = sorted({score.policy_version for score in scores})
    if len(policy_versions) != 1:
        raise ValueError(f"Samples span multiple policy versions: {policy_versions}")
    repeat_counts = sorted({len(score.b_repeats) for score in scores})
    if len(repeat_counts) != 1:
        raise ValueError(f"Samples have inconsistent B repeat counts: {repeat_counts}")

    token_rows = _token_rows(scores, rollout_id)
    summary = {
        "schema_version": 1,
        "rollout_id": rollout_id,
        "sample_count": len(scores),
        "response_token_count": len(token_rows),
        "active_token_count": sum(row["active"] for row in token_rows),
        "policy_version": policy_versions[0],
        "prefill_repeat_count": repeat_counts[0],
        "trainer_repeat_count": 2,
        "comparison_definitions": {
            "A_minus_B": "SGLang generation-time decode minus SGLang clean-cache prefill repeat 0",
            "B_minus_C": "SGLang clean-cache prefill repeat 0 minus Megatron pre-update forward",
            "A_minus_C": "SGLang generation-time decode minus Megatron pre-update forward",
            "B0_minus_B1": "SGLang clean-cache prefill repeat 0 minus repeat 1",
            "C0_minus_C1": "Megatron pre-update frozen-batch forward 0 minus forward 1",
        },
        "comparisons": {name: _comparison_summary(scores, name) for name in _COMPARISON_NAMES},
    }
    if summary["active_token_count"] == 0:
        raise ValueError("No active response tokens were found")
    return token_rows, summary


def _markdown_table(summary: dict[str, Any], section: str) -> list[str]:
    lines = [
        f"## {section.replace('_', ' ')}",
        "",
        "| comparison | count | signed mean | mean abs | RMSE | p50 abs | p95 abs | p99 abs | max abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _COMPARISON_NAMES:
        stats = summary["comparisons"][name][section]
        if stats is None:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {stats['count']} | {stats['signed_mean']:.6g} | {stats['mean_abs']:.6g} | "
            f"{stats['rmse']:.6g} | {stats['p50_abs']:.6g} | {stats['p95_abs']:.6g} | "
            f"{stats['p99_abs']:.6g} | {stats['max_abs']:.6g} |"
        )
    return [*lines, ""]


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# A/B/C logprob summary",
        "",
        f"- rollout: {summary['rollout_id']}",
        f"- policy version: {summary['policy_version']}",
        f"- samples: {summary['sample_count']}",
        f"- active response tokens: {summary['active_token_count']}",
        f"- clean-cache prefill repeats: {summary['prefill_repeat_count']}",
        f"- frozen-batch trainer repeats: {summary['trainer_repeat_count']}",
        "",
    ]
    for section in ("overall", "position_0", "position_1_plus"):
        lines.extend(_markdown_table(summary, section))

    lines.extend(
        [
            "## Threshold fractions",
            "",
            "| comparison | >1e-3 | >1e-2 | >5e-2 | >1e-1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in _COMPARISON_NAMES:
        stats = summary["comparisons"][name]["overall"]
        lines.append(
            f"| {name} | {stats['frac_abs_gt_1e-3']:.6g} | {stats['frac_abs_gt_1e-2']:.6g} | "
            f"{stats['frac_abs_gt_5e-2']:.6g} | {stats['frac_abs_gt_1e-1']:.6g} |"
        )

    lines.extend(
        [
            "",
            "Full per-sample and per-position statistics are in summary.json; "
            "every response-token value is in tokens.csv.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, token_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    fieldnames = list(dict.fromkeys(key for row in token_rows for key in row))
    with (output_dir / "tokens.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(token_rows)

    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-details", type=Path, required=True)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    token_rows, summary = analyze(args.dump_details, args.rollout_id)
    write_outputs(args.output_dir, token_rows, summary)
    headline = {
        name: summary["comparisons"][name]["overall"]
        for name in _COMPARISON_NAMES
    }
    print(json.dumps(headline, indent=2, sort_keys=True))
    print(f"Wrote {args.output_dir / 'tokens.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
