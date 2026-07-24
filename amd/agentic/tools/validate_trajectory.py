"""Validate AMD agentic trajectory traces without a model or WandB.

The trace format records exact token IDs at each generation boundary. It is an
AMD qualification sidecar and does not replace Miles' native rollout dumps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from amd.agentic import SCHEMA_VERSION
from amd.agentic.io import write_json

ALLOWED_STATUSES = {"completed", "truncated", "aborted", "failed"}
STATUS_FINISH_REASON = {"completed": "stop", "truncated": "length", "aborted": "abort"}
ALLOWED_FINISH_REASONS = {"stop", "length", "abort", "tool_call"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    event_index: int | None = None


@dataclass
class ValidationResult:
    trajectory_id: str
    valid: bool
    issues: list[ValidationIssue]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "valid": self.valid,
            "issues": [asdict(issue) for issue in self.issues],
            "metrics": self.metrics,
        }


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value)


def _is_finite_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in value
    )


def validate_trajectory(trace: dict[str, Any]) -> ValidationResult:
    issues: list[ValidationIssue] = []
    trajectory_id = str(trace.get("trajectory_id", ""))

    def add(code: str, message: str, event_index: int | None = None) -> None:
        issues.append(ValidationIssue(code, message, event_index))

    initial_input_ids = trace.get("initial_input_ids")
    if not _is_int_list(initial_input_ids):
        add("invalid_initial_input_ids", "initial_input_ids must be a list of integer token IDs")
        initial_input_ids = []

    events = trace.get("events")
    if not isinstance(events, list):
        add("invalid_events", "events must be a list")
        events = []

    accumulated = list(initial_input_ids)
    response_tokens: list[int] = []
    response_loss_mask: list[int] = []
    response_logprobs: list[float] = []
    generation_count = 0
    observation_count = 0
    tool_call_count = 0
    last_generation_turn = -1
    last_generation_finish: str | None = None
    previous_timestamp: float | None = None
    finalized = False
    deleted = False

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            add("invalid_event", "event must be an object", event_index)
            continue

        kind = event.get("kind")
        if deleted:
            add("event_after_delete", "no event may follow delete", event_index)
        elif finalized and kind != "delete":
            add("event_after_finalize", "only delete may follow finalize", event_index)

        sequence = event.get("sequence")
        if sequence != event_index:
            add("non_monotonic_sequence", f"expected sequence {event_index}, observed {sequence}", event_index)

        timestamp = event.get("timestamp")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
        ):
            add("invalid_timestamp", "timestamp must be a finite number", event_index)
        else:
            current_timestamp = float(timestamp)
            if previous_timestamp is not None and current_timestamp < previous_timestamp:
                add("non_monotonic_timestamp", "timestamps must be monotonic", event_index)
            previous_timestamp = current_timestamp

        if kind == "generation":
            turn = event.get("turn")
            if turn != generation_count:
                add("non_monotonic_turn", f"expected generation turn {generation_count}, observed {turn}", event_index)
            last_generation_turn = turn if isinstance(turn, int) else last_generation_turn

            input_ids = event.get("input_ids")
            if not _is_int_list(input_ids):
                add("invalid_generation_input_ids", "generation input_ids must be integer token IDs", event_index)
            elif input_ids != accumulated:
                add(
                    "prefix_mismatch",
                    f"generation input prefix differs at turn {turn}: expected {len(accumulated)} tokens, "
                    f"observed {len(input_ids)}",
                    event_index,
                )

            output_ids = event.get("output_ids")
            if not _is_int_list(output_ids) or not output_ids:
                add(
                    "invalid_generation_output_ids",
                    "generation output_ids must be a non-empty integer list",
                    event_index,
                )
                output_ids = []

            loss_mask = event.get("loss_mask")
            if not _is_int_list(loss_mask) or len(loss_mask) != len(output_ids):
                add("generation_loss_mask_length", "generation loss_mask must align with output_ids", event_index)
                loss_mask = []
            elif any(value != 1 for value in loss_mask):
                add("generation_loss_mask_value", "all generated assistant tokens must have loss mask 1", event_index)

            logprobs = event.get("logprobs")
            if not _is_finite_number_list(logprobs) or len(logprobs) != len(output_ids):
                add(
                    "generation_logprob_alignment",
                    "finite generation logprobs must align with output_ids",
                    event_index,
                )
                logprobs = []

            declared_count = event.get("generated_token_count")
            if declared_count is not None and declared_count != len(output_ids):
                add(
                    "generated_token_count_mismatch",
                    f"declared {declared_count} generated tokens, observed {len(output_ids)}",
                    event_index,
                )

            finish_reason = event.get("finish_reason")
            if finish_reason not in ALLOWED_FINISH_REASONS:
                add("invalid_finish_reason", f"unsupported finish_reason {finish_reason!r}", event_index)
            if finish_reason == "tool_call":
                tool_call_count += 1
            last_generation_finish = finish_reason if isinstance(finish_reason, str) else None

            accumulated.extend(output_ids)
            response_tokens.extend(output_ids)
            response_loss_mask.extend(loss_mask)
            response_logprobs.extend(float(value) for value in logprobs)
            generation_count += 1

        elif kind == "observation":
            turn = event.get("turn")
            if turn != last_generation_turn:
                add(
                    "observation_turn_mismatch",
                    f"observation turn {turn} does not match generation turn {last_generation_turn}",
                    event_index,
                )
            if last_generation_finish != "tool_call":
                add("unexpected_observation", "observation must follow a tool_call generation", event_index)

            token_ids = event.get("token_ids")
            if not _is_int_list(token_ids) or not token_ids:
                add(
                    "invalid_observation_token_ids",
                    "observation token_ids must be a non-empty integer list",
                    event_index,
                )
                token_ids = []

            loss_mask = event.get("loss_mask")
            if not _is_int_list(loss_mask) or len(loss_mask) != len(token_ids):
                add("observation_loss_mask_length", "observation loss_mask must align with token_ids", event_index)
                loss_mask = []
            elif any(value != 0 for value in loss_mask):
                add("observation_loss_mask_value", "all observation tokens must have loss mask 0", event_index)

            logprobs = event.get("logprobs")
            if not _is_finite_number_list(logprobs) or len(logprobs) != len(token_ids):
                add(
                    "observation_logprob_alignment",
                    "finite observation logprobs must align with token_ids",
                    event_index,
                )
                logprobs = []
            elif any(float(value) != 0.0 for value in logprobs):
                add("observation_logprob_value", "observation logprobs must be zero placeholders", event_index)

            accumulated.extend(token_ids)
            response_tokens.extend(token_ids)
            response_loss_mask.extend(loss_mask)
            response_logprobs.extend(float(value) for value in logprobs)
            observation_count += 1

        elif kind == "finalize":
            if finalized:
                add("duplicate_finalize", "trajectory may be finalized only once", event_index)
            finalized = True

        elif kind == "delete":
            if not finalized:
                add("delete_before_finalize", "delete must follow finalize", event_index)
            if deleted:
                add("duplicate_delete", "trajectory may be deleted only once", event_index)
            deleted = True

        else:
            add("invalid_event_kind", f"unsupported event kind {kind!r}", event_index)

    declared_tool_calls = trace.get("tool_call_count")
    if declared_tool_calls != tool_call_count:
        add("tool_call_count_mismatch", f"declared {declared_tool_calls}, observed {tool_call_count}")
    declared_turns = trace.get("turn_count")
    if declared_turns != generation_count:
        add("turn_count_mismatch", f"declared {declared_turns}, observed {generation_count}")

    status = trace.get("status")
    if status not in ALLOWED_STATUSES:
        add("invalid_status", f"unsupported status {status!r}")
    expected_finish = STATUS_FINISH_REASON.get(status)
    if expected_finish and last_generation_finish != expected_finish:
        add(
            "finish_status_mismatch",
            f"status {status!r} requires final generation finish_reason {expected_finish!r}, "
            f"observed {last_generation_finish!r}",
        )

    lifecycle = trace.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        add("invalid_lifecycle", "lifecycle must be an object")
        lifecycle = {}
    if lifecycle.get("requires_finalize") and not finalized:
        add("missing_finalize", "trajectory requires a finalize event")
    if lifecycle.get("requires_delete") and not deleted:
        add("missing_delete", "trajectory requires a delete event")
    if lifecycle.get("finalized") is not None and bool(lifecycle["finalized"]) != finalized:
        add("finalized_state_mismatch", "declared lifecycle.finalized does not match events")
    if lifecycle.get("deleted") is not None and bool(lifecycle["deleted"]) != deleted:
        add("deleted_state_mismatch", "declared lifecycle.deleted does not match events")

    assembled = trace.get("assembled")
    if not isinstance(assembled, dict):
        add("missing_assembled_sample", "assembled sample is required")
        assembled = {}
    if assembled.get("tokens") != accumulated:
        add("assembled_tokens_mismatch", "assembled tokens differ from event-derived tokens")
    if assembled.get("response_length") != len(response_tokens):
        add(
            "assembled_response_length_mismatch",
            f"expected response_length {len(response_tokens)}, observed {assembled.get('response_length')}",
        )
    if assembled.get("loss_mask") != response_loss_mask:
        add("assembled_loss_mask_mismatch", "assembled loss_mask differs from event-derived loss mask")
    assembled_logprobs = assembled.get("rollout_log_probs")
    if not _is_finite_number_list(assembled_logprobs) or len(assembled_logprobs) != len(response_tokens):
        add("assembled_logprob_alignment", "assembled rollout_log_probs must be finite and response-aligned")
    elif [float(value) for value in assembled_logprobs] != response_logprobs:
        add("assembled_logprob_mismatch", "assembled rollout_log_probs differ from event-derived values")

    duration = 0.0
    timestamps = [event.get("timestamp") for event in events if isinstance(event, dict)]
    numeric_timestamps = [
        float(value)
        for value in timestamps
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    if len(numeric_timestamps) >= 2:
        duration = numeric_timestamps[-1] - numeric_timestamps[0]

    metrics = {
        "status": status,
        "turn_count": generation_count,
        "tool_call_count": tool_call_count,
        "observation_count": observation_count,
        "prompt_tokens": len(initial_input_ids),
        "response_tokens": len(response_tokens),
        "trainable_tokens": sum(response_loss_mask),
        "duration_seconds": duration,
        "finalized": finalized,
        "deleted": deleted,
    }
    return ValidationResult(trajectory_id=trajectory_id, valid=not issues, issues=issues, metrics=metrics)


def load_traces(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text()
    if source.suffix == ".jsonl":
        traces = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        traces = value if isinstance(value, list) else [value]
    if not all(isinstance(trace, dict) for trace in traces):
        raise ValueError(f"{source} must contain trajectory objects")
    return traces


def build_validation_report(traces: list[dict[str, Any]], repeat: int = 1) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    results = [validate_trajectory(trace) for _ in range(repeat) for trace in traces]
    issue_counts = Counter(issue.code for result in results for issue in result.issues)
    status_counts = Counter(str(result.metrics["status"]) for result in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_count": len(traces),
        "repeat": repeat,
        "validation_count": len(results),
        "valid_count": sum(result.valid for result in results),
        "invalid_count": sum(not result.valid for result in results),
        "passed": all(result.valid for result in results),
        "issue_counts": dict(sorted(issue_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "results": [result.to_dict() for result in results],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_validation_report(load_traces(args.trace), repeat=args.repeat)
    if args.output:
        write_json(args.output, report)
        print(f"wrote {args.output}")
    print(
        f"validated={report['validation_count']} valid={report['valid_count']} "
        f"invalid={report['invalid_count']}"
    )
    if report["issue_counts"]:
        print(json.dumps(report["issue_counts"], sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
