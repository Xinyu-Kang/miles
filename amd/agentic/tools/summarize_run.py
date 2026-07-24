"""Create an offline run summary from a preflight manifest and validation report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from amd.agentic import SCHEMA_VERSION
from amd.agentic.io import write_json


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_summary(manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    results = validation.get("results", [])
    valid_results = [result for result in results if result.get("valid")]
    metrics = [result.get("metrics", {}) for result in valid_results]

    status_counts = Counter(str(metric.get("status")) for metric in metrics)
    issue_counts = Counter(
        issue.get("code", "unknown")
        for result in results
        for issue in result.get("issues", [])
        if isinstance(issue, dict)
    )

    def average(key: str) -> float:
        values = [float(metric[key]) for metric in metrics if isinstance(metric.get(key), (int, float))]
        return mean(values) if values else 0.0

    source = manifest.get("source", {})
    container = manifest.get("container", {})
    gpus = manifest.get("gpus", {})
    qualification = manifest.get("qualification", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
        "workload": manifest.get("workload"),
        "reproducibility": {
            "preflight_passed": qualification.get("passed", False),
            "preflight_error_count": len(qualification.get("errors", [])),
            "preflight_warning_count": len(qualification.get("warnings", [])),
            "miles_commit": source.get("commit"),
            "miles_dirty": source.get("dirty"),
            "image_ref": container.get("image_ref"),
            "image_digest": container.get("image_digest"),
            "gpu_count": gpus.get("count"),
            "gpu_architectures": gpus.get("architectures", []),
        },
        "validation": {
            "passed": validation.get("passed", False),
            "trace_count": validation.get("trace_count", 0),
            "repeat": validation.get("repeat", 1),
            "validation_count": validation.get("validation_count", len(results)),
            "valid_count": len(valid_results),
            "invalid_count": len(results) - len(valid_results),
            "status_counts": dict(sorted(status_counts.items())),
            "issue_counts": dict(sorted(issue_counts.items())),
            "average_turn_count": average("turn_count"),
            "average_tool_call_count": average("tool_call_count"),
            "average_prompt_tokens": average("prompt_tokens"),
            "average_response_tokens": average("response_tokens"),
            "average_trainable_tokens": average("trainable_tokens"),
            "average_duration_seconds": average("duration_seconds"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(_load_json(args.manifest), _load_json(args.validation_report))
    write_json(args.output, summary)
    print(f"wrote {args.output}")
    return 0 if summary["reproducibility"]["preflight_passed"] and summary["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
