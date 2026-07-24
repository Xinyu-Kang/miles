"""Compare two offline AMD qualification summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from amd.agentic.io import write_json

COMPARABLE_METRICS = (
    "valid_count",
    "invalid_count",
    "average_turn_count",
    "average_tool_call_count",
    "average_prompt_tokens",
    "average_response_tokens",
    "average_trainable_tokens",
    "average_duration_seconds",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = baseline.get("validation", {})
    candidate_metrics = candidate.get("validation", {})
    metrics: dict[str, Any] = {}
    for name in COMPARABLE_METRICS:
        before = baseline_metrics.get(name)
        after = candidate_metrics.get(name)
        delta = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        metrics[name] = {"baseline": before, "candidate": after, "delta": delta}
    return {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_commit": baseline.get("reproducibility", {}).get("miles_commit"),
        "candidate_commit": candidate.get("reproducibility", {}).get("miles_commit"),
        "metrics": metrics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison = compare_summaries(_load(args.baseline), _load(args.candidate))
    if args.output:
        write_json(args.output, comparison)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
