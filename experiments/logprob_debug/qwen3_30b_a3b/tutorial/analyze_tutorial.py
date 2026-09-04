#!/usr/bin/env python3
"""Validate and compare clean/fault/fixed logprob tutorial artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


COMPARISONS = ("A_minus_B", "B_minus_C", "A_minus_C", "B0_minus_B1", "C0_minus_C1")
BOUNDARIES = ("A_minus_B", "B_minus_C", "A_minus_C")
REPEAT_LIMIT = 0.01
PARITY_LIMIT = 0.01
FAULT_MINIMUM = 0.05


@dataclass(frozen=True)
class RolloutArtifact:
    rollout_id: int
    summary: dict[str, Any]
    tokens: list[dict[str, str]]


@dataclass(frozen=True)
class ArmArtifact:
    name: str
    run_dir: Path
    manifest: dict[str, Any]
    rollouts: dict[int, RolloutArtifact]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_tokens(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No token rows in {path}")
    numeric = {
        "A_decode", "B0_prefill", "B1_prefill", "C_trainer",
        "C1_trainer_repeat", *COMPARISONS,
    }
    for row_index, row in enumerate(rows):
        for field in numeric:
            if field not in row:
                raise ValueError(f"Missing {field} in {path}")
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} at row {row_index} in {path}")
    return rows


def load_arm(name: str, run_dir: Path) -> ArmArtifact:
    manifest = _read_json(run_dir / "provenance" / "tutorial-config.json")
    analysis_dir = run_dir / "analysis"
    rollout_dirs = sorted(analysis_dir.glob("rollout_*"))
    if not rollout_dirs:
        raise ValueError(f"No analyzed rollouts under {analysis_dir}")
    rollouts = {}
    for directory in rollout_dirs:
        rollout_id = int(directory.name.removeprefix("rollout_"))
        summary = _read_json(directory / "summary.json")
        tokens = _read_tokens(directory / "tokens.csv")
        if int(summary["rollout_id"]) != rollout_id:
            raise ValueError(f"Rollout ID mismatch in {directory}")
        if int(summary["response_token_count"]) != len(tokens):
            raise ValueError(f"Token count mismatch in {directory}")
        rollouts[rollout_id] = RolloutArtifact(rollout_id, summary, tokens)
    return ArmArtifact(name, run_dir.resolve(), manifest, rollouts)


def _token_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(row["rollout_id"]),
        int(row["sample_index"]),
        int(row["sample_occurrence"]),
        int(row["token_position"]),
    )


def _identity_map(artifact: RolloutArtifact) -> dict[tuple[int, int, int, int], tuple[int, int, int, int]]:
    result = {}
    for row in artifact.tokens:
        key = _token_key(row)
        if key in result:
            raise ValueError(f"Duplicate token identity {key}")
        result[key] = (
            int(row["token_id"]),
            int(row["active"]),
            int(row["response_length"]),
            int(row["group_index"]),
        )
    return result


def validate_cross_arm_tokens(reference: ArmArtifact, others: list[ArmArtifact]) -> dict[str, Any]:
    """Case 1 is bookkeeping-only, so response tokens must match across arms."""

    reference_ids = set(reference.rollouts)
    for other in others:
        if set(other.rollouts) != reference_ids:
            raise ValueError(
                f"Rollout sets differ: {reference.name}={sorted(reference_ids)}, "
                f"{other.name}={sorted(other.rollouts)}"
            )
        for rollout_id in sorted(reference_ids):
            expected = _identity_map(reference.rollouts[rollout_id])
            observed = _identity_map(other.rollouts[rollout_id])
            if observed != expected:
                first = next(
                    (key for key in sorted(expected.keys() | observed.keys()) if expected.get(key) != observed.get(key)),
                    None,
                )
                raise ValueError(
                    f"Cross-arm token identity changed in rollout {rollout_id} at {first}: "
                    f"{reference.name}={expected.get(first)}, {other.name}={observed.get(first)}"
                )
    payload = json.dumps(
        {str(rid): sorted(_identity_map(reference.rollouts[rid]).items()) for rid in reference_ids},
        separators=(",", ":"),
    ).encode()
    return {
        "matched_arms": [reference.name, *[other.name for other in others]],
        "rollout_ids": sorted(reference_ids),
        "token_rows": sum(len(value.tokens) for value in reference.rollouts.values()),
        "identity_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sample_sequences(arm: ArmArtifact) -> dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...]]]:
    result = {}
    for rollout_id in sorted(arm.rollouts):
        dump_path = arm.run_dir / "dump_details" / "rollout_data" / f"{rollout_id}.pt"
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload.samples
        occurrences: dict[int, int] = {}
        for sample in samples:
            sample_index = int(_get(sample, "index"))
            occurrence = occurrences.get(sample_index, 0)
            occurrences[sample_index] = occurrence + 1
            key = rollout_id, sample_index, occurrence
            response_length = int(_get(sample, "response_length"))
            tokens = tuple(int(value) for value in _get(sample, "tokens"))
            result[key] = (tokens[:-response_length], tokens[-response_length:])
    return result


def validate_cross_arm_prompts(reference: ArmArtifact, others: list[ArmArtifact]) -> dict[str, Any]:
    """Reruns must use identical prompts; sampled responses are reported, not assumed paired."""

    expected = _sample_sequences(reference)
    response_mismatches = {}
    for other in others:
        observed = _sample_sequences(other)
        if observed.keys() != expected.keys():
            raise ValueError(
                f"Cross-arm sample identities differ: {reference.name}={sorted(expected)}, "
                f"{other.name}={sorted(observed)}"
            )
        prompt_mismatch = next(
            (key for key in sorted(expected) if expected[key][0] != observed[key][0]),
            None,
        )
        if prompt_mismatch is not None:
            raise ValueError(f"Cross-arm prompt tokens changed at {prompt_mismatch}")
        mismatched = [key for key in sorted(expected) if expected[key][1] != observed[key][1]]
        response_mismatches[other.name] = {
            "mismatched_sample_count": len(mismatched),
            "first_mismatch": list(mismatched[0]) if mismatched else None,
            "paired_response_claim_allowed": not mismatched,
        }
    prompt_payload = json.dumps(
        {str(key): list(value[0]) for key, value in sorted(expected.items())},
        separators=(",", ":"),
    ).encode()
    return {
        "sample_count": len(expected),
        "prompt_token_ids_match": True,
        "prompt_identity_sha256": hashlib.sha256(prompt_payload).hexdigest(),
        "response_comparison": response_mismatches,
        "note": "Rerun responses are aggregate comparisons unless paired_response_claim_allowed is true",
    }


def _get(sample: Any, key: str) -> Any:
    return sample[key] if isinstance(sample, dict) else getattr(sample, key)


def validate_case1_fault_metadata(fault: ArmArtifact, expected_delta: float) -> dict[str, Any]:
    checked_samples = 0
    checked_positions = 0
    for rollout_id in sorted(fault.rollouts):
        dump_path = fault.run_dir / "dump_details" / "rollout_data" / f"{rollout_id}.pt"
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload.samples
        for sample in samples:
            response_length = int(_get(sample, "response_length"))
            tokens = [int(value) for value in _get(sample, "tokens")]
            rollout_a = [float(value) for value in _get(sample, "rollout_log_probs")]
            metadata = _get(sample, "metadata") or {}
            fault_record = metadata.get("fault_injection")
            debug_record = metadata.get("logprob_debug")
            if not isinstance(fault_record, dict) or not isinstance(debug_record, dict):
                raise ValueError("Case 1 fault sample is missing fault/debug metadata")
            if fault_record.get("name") != "decode_logprob_offset":
                raise ValueError(f"Unexpected Case 1 fault name: {fault_record.get('name')!r}")
            delta = float(fault_record.get("delta", "nan"))
            if delta != expected_delta:
                raise ValueError(f"Case 1 delta mismatch: expected {expected_delta}, got {delta}")
            response_ids = tokens[-response_length:]
            fields = {
                "fault response tokens": [int(value) for value in fault_record["response_token_ids"]],
                "debug response tokens": [int(value) for value in debug_record["response_token_ids"]],
            }
            if any(value != response_ids for value in fields.values()):
                raise ValueError(f"Case 1 response-token metadata mismatch: {fields}")
            clean = [float(value) for value in fault_record["clean_decode_logprobs"]]
            faulted = [float(value) for value in fault_record["faulted_decode_logprobs"]]
            debug_a = [float(value) for value in debug_record["decode_logprobs"]]
            if any(len(value) != response_length for value in (clean, faulted, debug_a, rollout_a)):
                raise ValueError("Case 1 fault logprob length mismatch")
            if not all(math.isfinite(value) for values in (clean, faulted, debug_a, rollout_a) for value in values):
                raise ValueError("Case 1 fault metadata contains NaN or Inf")
            if faulted != rollout_a or debug_a != rollout_a:
                raise ValueError("Case 1 stored/debug A is not the faulted A")
            for position, (before, after) in enumerate(zip(clean, faulted, strict=True)):
                expected = before if position == 0 else before - expected_delta
                if not math.isclose(after, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"Case 1 injection mismatch at sample {_get(sample, 'index')} "
                        f"position {position}: expected {expected}, got {after}"
                    )
                checked_positions += 1
            checked_samples += 1
    if not checked_samples:
        raise ValueError("Case 1 fault run has no samples")
    return {
        "fault": "decode_logprob_offset",
        "delta_nat": expected_delta,
        "samples": checked_samples,
        "positions": checked_positions,
        "position_0_unchanged": True,
        "positions_1_plus_shifted": True,
        "tokens_unchanged_within_hook": True,
    }


def _mean_abs(rollout: RolloutArtifact, comparison: str, section: str = "overall") -> float:
    value = rollout.summary["comparisons"][comparison][section]
    if value is None:
        raise ValueError(f"Missing {comparison}/{section} statistics")
    result = float(value["mean_abs"])
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {comparison}/{section} mean absolute difference")
    return result


def first_failed_gate(rollout: RolloutArtifact) -> str:
    for comparison, label in (
        ("B0_minus_B1", "B_repeatability"),
        ("C0_minus_C1", "C_repeatability"),
        ("A_minus_B", "A_vs_B"),
        ("B_minus_C", "B_vs_C"),
        ("A_minus_C", "A_vs_C"),
    ):
        if _mean_abs(rollout, comparison) >= (REPEAT_LIMIT if "repeat" in label else PARITY_LIMIT):
            return label
    return "none"


def _validate_manifests(case: str, arms: list[ArmArtifact]) -> dict[str, Any]:
    by_name = {arm.name: arm for arm in arms}
    for expected in ("clean", "fault", "fixed"):
        if expected not in by_name:
            raise ValueError(f"Missing {expected} arm")
    if by_name["fault"].manifest.get("case") != case or by_name["fault"].manifest.get("arm") != "fault":
        raise ValueError("Fault manifest does not match requested tutorial case")
    if by_name["fixed"].manifest.get("case") != case or by_name["fixed"].manifest.get("arm") != "fixed":
        raise ValueError("Fixed manifest does not match requested tutorial case")
    clean_manifest = by_name["clean"].manifest
    if clean_manifest.get("arm") != "clean" or clean_manifest.get("case") not in ("case0", case):
        raise ValueError("Clean manifest is not a compatible clean control")
    if clean_manifest.get("deliberately_injected") or by_name["fixed"].manifest.get("deliberately_injected"):
        raise ValueError("Clean/fixed manifest claims a deliberate fault")
    if not by_name["fault"].manifest.get("deliberately_injected"):
        raise ValueError("Fault manifest is not labeled deliberately injected")

    ignored = {"dump_details"}
    allowed = {
        "case1": {"rollout_all_samples_process_path"},
        "case2": (
            {"custom_megatron_before_log_prob_hook_path"}
            if by_name["fault"].manifest.get("active_fault") == "trainer_normalizer_offset"
            else {"rollout_temperature"}
        ),
        "case3": {"debug_skip_rollout_weight_update_at"},
    }[case]
    def normalized(arm: ArmArtifact) -> dict[str, Any]:
        result = dict(arm.manifest["resolved"])
        result.setdefault("learning_rate", 1e-6)
        result.setdefault("debug_skip_rollout_weight_update_at", None)
        result.setdefault("custom_megatron_before_log_prob_hook_path", None)
        return result

    clean_resolved = normalized(by_name["clean"])
    fixed_resolved = normalized(by_name["fixed"])
    clean_fixed_drift = {
        key for key in clean_resolved.keys() | fixed_resolved.keys()
        if key not in ignored and clean_resolved.get(key) != fixed_resolved.get(key)
    }
    if clean_fixed_drift:
        raise ValueError(f"Clean/fixed resolved configuration drift: {sorted(clean_fixed_drift)}")
    fault_resolved = normalized(by_name["fault"])
    fault_drift = {
        key for key in clean_resolved.keys() | fault_resolved.keys()
        if key not in ignored and clean_resolved.get(key) != fault_resolved.get(key)
    }
    if fault_drift != allowed:
        raise ValueError(
            f"Expected only {sorted(allowed)} to differ in {case}, got {sorted(fault_drift)}"
        )
    return {"clean_fixed_drift": [], "fault_only_drift": sorted(fault_drift)}


def _validate_common_provenance(arms: list[ArmArtifact]) -> dict[str, Any]:
    required = (
        "container_image_id", "container_repo_digest", "container_hostname",
        "model_path", "trainer_checkpoint_path", "dataset_path",
    )
    observed: dict[str, dict[str, str]] = {}
    for arm in arms:
        values = {}
        for line in (arm.run_dir / "provenance" / "run.txt").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        observed[arm.name] = values
    for key in required:
        values = {arm: fields.get(key) for arm, fields in observed.items()}
        if None in values.values() or len(set(values.values())) != 1:
            raise ValueError(f"Cross-arm provenance mismatch for {key}: {values}")
    return {key: observed[arms[0].name][key] for key in required}


def _validate_signature(
    case: str,
    arms: list[ArmArtifact],
    *,
    allow_failures: bool = False,
) -> dict[str, Any]:
    by_name = {arm.name: arm for arm in arms}
    for arm in arms:
        for rollout in arm.rollouts.values():
            if _mean_abs(rollout, "B0_minus_B1") >= REPEAT_LIMIT:
                raise ValueError(f"{arm.name} rollout {rollout.rollout_id} B repeatability is not <0.01")
            if _mean_abs(rollout, "C0_minus_C1") >= REPEAT_LIMIT:
                raise ValueError(f"{arm.name} rollout {rollout.rollout_id} C repeatability is not <0.01")
    baseline_checks = {}
    for arm_name in ("clean", "fixed"):
        for rollout in by_name[arm_name].rollouts.values():
            for boundary in BOUNDARIES:
                key = f"{arm_name}_rollout_{rollout.rollout_id}_{boundary}_below_0.01"
                baseline_checks[key] = _mean_abs(rollout, boundary) < PARITY_LIMIT
    fault = by_name["fault"].rollouts[0]
    if case == "case1":
        checks = baseline_checks | {
            "position_0_A_minus_B_below_0.01": _mean_abs(fault, "A_minus_B", "position_0") < PARITY_LIMIT,
            "position_1_plus_A_minus_B_above_0.05": _mean_abs(fault, "A_minus_B", "position_1_plus") >= FAULT_MINIMUM,
            "B_minus_C_below_0.01": _mean_abs(fault, "B_minus_C") < PARITY_LIMIT,
            "A_minus_C_above_0.05": _mean_abs(fault, "A_minus_C", "position_1_plus") >= FAULT_MINIMUM,
        }
    elif case == "case2":
        checks = baseline_checks | {
            "A_minus_B_below_0.01": _mean_abs(fault, "A_minus_B") < PARITY_LIMIT,
            "B_minus_C_between_0.07_and_0.20": 0.07 <= _mean_abs(fault, "B_minus_C") <= 0.20,
            "A_minus_C_above_0.05": _mean_abs(fault, "A_minus_C") >= FAULT_MINIMUM,
        }
    else:
        expected_rollouts = {0, 1, 2}
        if set(by_name["fault"].rollouts) != expected_rollouts:
            raise ValueError(
                f"Case 3 fault must contain rollouts {sorted(expected_rollouts)}, "
                f"got {sorted(by_name['fault'].rollouts)}"
            )
        before = by_name["fault"].rollouts[0]
        stale = by_name["fault"].rollouts[1]
        recovered = by_name["fault"].rollouts[2]
        checks = baseline_checks | {
            "rollout_0_all_boundaries_below_0.01": all(
                _mean_abs(before, boundary) < PARITY_LIMIT for boundary in BOUNDARIES
            ),
            "rollout_1_A_minus_B_below_0.01": _mean_abs(stale, "A_minus_B") < PARITY_LIMIT,
            "rollout_1_B_minus_C_above_0.05": _mean_abs(stale, "B_minus_C") >= FAULT_MINIMUM,
            "rollout_1_A_minus_C_above_0.05": _mean_abs(stale, "A_minus_C") >= FAULT_MINIMUM,
            "rollout_2_all_boundaries_below_0.01": all(
                _mean_abs(recovered, boundary) < PARITY_LIMIT for boundary in BOUNDARIES
            ),
        }
    failed = [name for name, passed in checks.items() if not passed]
    if failed and not allow_failures:
        raise ValueError(f"Observed {case} signature failed checks: {failed}")
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "failures_allowed_for_artifact_generation": bool(allow_failures),
    }


def _table_rows(arms: list[ArmArtifact]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        for rollout_id, artifact in sorted(arm.rollouts.items()):
            row = {
                "case": arm.manifest["case"],
                "arm": arm.name,
                "rollout": rollout_id,
                "policy_version": artifact.summary["policy_version"],
                "A-B MAE": _mean_abs(artifact, "A_minus_B"),
                "B-C MAE": _mean_abs(artifact, "B_minus_C"),
                "A-C MAE": _mean_abs(artifact, "A_minus_C"),
                "B-repeat": _mean_abs(artifact, "B0_minus_B1"),
                "C-repeat": _mean_abs(artifact, "C0_minus_C1"),
                "first failed gate": first_failed_gate(artifact),
            }
            rows.append(row)
    return rows


def _write_plots(case: str, arms: list[ArmArtifact], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths = []
    focus_rollout = 1 if case == "case3" else 0
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, comparison in zip(axes, BOUNDARIES, strict=True):
        for arm in arms:
            stats = arm.rollouts[focus_rollout].summary["comparisons"][comparison]["per_position"]
            axis.plot(
                [int(item["token_position"]) for item in stats],
                [float(item["stats"]["mean_abs"]) for item in stats],
                label=arm.name,
            )
        axis.axhline(PARITY_LIMIT, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(f"{comparison}\nmean |delta|")
        axis.legend()
    axes[-1].set_xlabel("response token position")
    fig.tight_layout()
    path = output_dir / "per_position_abs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, comparison in zip(axes, BOUNDARIES, strict=True):
        for arm in arms:
            values = [
                float(row[comparison])
                for row in arm.rollouts[focus_rollout].tokens
                if int(row["active"])
            ]
            axis.hist(values, bins=50, alpha=0.45, label=arm.name)
        axis.set_title(comparison)
        axis.set_xlabel("signed delta (nat)")
        axis.legend()
    fig.tight_layout()
    path = output_dir / "signed_delta_histograms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path.name)

    labels = [arm.name for arm in arms]
    width = 0.25
    positions = list(range(len(labels)))
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, comparison in enumerate(BOUNDARIES):
        values = [_mean_abs(arm.rollouts[focus_rollout], comparison) for arm in arms]
        axis.bar([position + (offset - 1) * width for position in positions], values, width, label=comparison)
    axis.axhline(PARITY_LIMIT, color="black", linestyle="--", linewidth=0.8, label="0.01 gate")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("mean absolute difference (nat)")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "clean_fault_fixed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path.name)
    return plot_paths


def _digest_rows(rows: list[dict[str, str]], field: str) -> str:
    """Hash a score vector with token identities as a lightweight functional witness."""

    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(
            (
                f"{row['sample_index']}:{row['sample_occurrence']}:"
                f"{row['token_position']}:{row['token_id']}:{float(row[field]).hex()}\n"
            ).encode()
        )
    return hasher.hexdigest()


def _logical_digest(state_index: int) -> str:
    """Hash a controller-level state label; this is not a tensor checksum."""

    return hashlib.sha256(f"trainer_state_T{state_index}".encode()).hexdigest()


_GRAD_NORM_PATTERN = re.compile(
    r"log_utils\.py:544 - step (?P<step>\d+): .*?'train/grad_norm': (?P<value>[^,}\s]+)"
)


def _case3_grad_norms(arms: list[ArmArtifact]) -> dict[str, dict[int, float]]:
    result = {}
    for arm in arms:
        job_log = (arm.run_dir / "provenance" / "ray-job.log").read_text(encoding="utf-8")
        values: dict[int, float] = {}
        for match in _GRAD_NORM_PATTERN.finditer(job_log):
            step = int(match.group("step"))
            value = float(match.group("value"))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Case 3 {arm.name} step {step} has invalid grad norm {value}")
            if step in values and values[step] != value:
                raise ValueError(
                    f"Case 3 {arm.name} step {step} has conflicting grad norms: "
                    f"{values[step]} and {value}"
                )
            values[step] = value
        expected = set(arm.rollouts)
        if set(values) != expected:
            raise ValueError(
                f"Case 3 {arm.name} grad-norm steps differ: "
                f"expected {sorted(expected)}, got {sorted(values)}"
            )
        result[arm.name] = values
    return result


def _case3_timeline(
    arms: list[ArmArtifact],
    output_dir: Path,
    grad_norms: dict[str, dict[int, float]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        expected_rollouts = {0, 1, 2}
        if set(arm.rollouts) != expected_rollouts:
            raise ValueError(
                f"Case 3 {arm.name} must contain rollouts {sorted(expected_rollouts)}, "
                f"got {sorted(arm.rollouts)}"
            )
        skip_at = arm.manifest["resolved"].get("debug_skip_rollout_weight_update_at")
        skip_at = None if skip_at is None else int(skip_at)
        job_log = (arm.run_dir / "provenance" / "ray-job.log").read_text(encoding="utf-8")
        skip_marker = "DELIBERATE tutorial fault stale_rollout_weights"
        if (skip_marker in job_log) != (arm.name == "fault"):
            raise ValueError(f"Case 3 {arm.name} skip-update log marker does not match its arm")

        rollout_state = 0
        for rollout_id in sorted(arm.rollouts):
            artifact = arm.rollouts[rollout_id]
            trainer_before = rollout_id
            trainer_after = rollout_id + 1
            rollout_before = rollout_state
            sync_event = "skipped" if rollout_id == skip_at else "completed"
            if sync_event == "completed":
                rollout_state = trainer_after
            rows.append(
                {
                    "arm": arm.name,
                    "rollout": rollout_id,
                    "observed_rollout_policy_version": str(artifact.summary["policy_version"]),
                    "trainer_grad_norm": grad_norms[arm.name][rollout_id],
                    "trainer_before_score_state": f"T{trainer_before}",
                    "rollout_before_score_state": f"T{rollout_before}",
                    "same_logical_state_for_A_B_C": rollout_before == trainer_before,
                    "trainer_after_optimizer_state": f"T{trainer_after}",
                    "sync_after_rollout": sync_event,
                    "rollout_after_sync_state": f"T{rollout_state}",
                    "trainer_before_logical_digest": _logical_digest(trainer_before),
                    "rollout_before_logical_digest": _logical_digest(rollout_before),
                    "C_functional_score_digest": _digest_rows(artifact.tokens, "C_trainer"),
                    "B_functional_score_digest": _digest_rows(artifact.tokens, "B0_prefill"),
                }
            )

    path = output_dir / "policy_update_timeline.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, arm in zip(axes, arms, strict=True):
        arm_rows = [row for row in rows if row["arm"] == arm.name]
        x = [row["rollout"] for row in arm_rows]
        trainer = [int(str(row["trainer_before_score_state"])[1:]) for row in arm_rows]
        rollout = [int(str(row["rollout_before_score_state"])[1:]) for row in arm_rows]
        axis.plot(x, trainer, marker="o", label="trainer C state")
        axis.plot(x, rollout, marker="s", label="rollout A/B state")
        axis.set_title(arm.name)
        axis.set_xlabel("rollout")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("logical policy state index")
    axes[-1].legend()
    fig.tight_layout()
    timeline_plot = output_dir / "policy_update_timeline.png"
    fig.savefig(timeline_plot, dpi=160)
    plt.close(fig)
    return {
        "csv": path.name,
        "plot": timeline_plot.name,
        "digest_kind": (
            "logical digests hash controller state labels and are not parameter tensor checksums; "
            "functional digests hash exact token/score rows"
        ),
        "rows": rows,
    }


def _write_outliers(fault: ArmArtifact, output_dir: Path, limit: int = 12) -> str:
    rows = []
    for rollout in fault.rollouts.values():
        for row in rollout.tokens:
            if not int(row["active"]):
                continue
            copied: dict[str, Any] = dict(row)
            copied["largest_abs_boundary_delta"] = max(abs(float(row[name])) for name in BOUNDARIES)
            rows.append(copied)
    rows.sort(key=lambda row: float(row["largest_abs_boundary_delta"]), reverse=True)
    path = output_dir / "fault_outliers.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:limit])
    return path.name


def _render_markdown(case: str, rows: list[dict[str, Any]], validations: dict[str, Any]) -> str:
    lines = [
        f"# {case} clean → fault → fixed",
        "",
        "The fault arm below is deliberately injected for this tutorial.",
        "",
        "| arm | rollout | policy | A−B MAE | B−C MAE | A−C MAE | B repeat | C repeat | first failed gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['rollout']} | {row['policy_version']} | "
            f"{row['A-B MAE']:.6f} | {row['B-C MAE']:.6f} | {row['A-C MAE']:.6f} | "
            f"{row['B-repeat']:.6f} | {row['C-repeat']:.6f} | {row['first failed gate']} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Repeatability gate: both B and C MAE must be strictly below {REPEAT_LIMIT}.",
            f"- Parity gate: each A/B/C boundary MAE must be strictly below {PARITY_LIMIT}.",
            f"- Checks: `{json.dumps(validations, sort_keys=True)}`",
            "",
            "Observation, inference, known injected cause, and fix verification are recorded in TUTORIAL_RESULTS.md.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_case(
    case: str,
    clean_dir: Path,
    fault_dir: Path,
    fixed_dir: Path,
    output_dir: Path,
    *,
    allow_signature_failures: bool = False,
) -> dict[str, Any]:
    arms = [load_arm("clean", clean_dir), load_arm("fault", fault_dir), load_arm("fixed", fixed_dir)]
    validations: dict[str, Any] = {
        "manifests": _validate_manifests(case, arms),
        "common_provenance": _validate_common_provenance(arms),
    }
    validations["cross_arm_prompts"] = validate_cross_arm_prompts(arms[0], arms[1:])
    if case == "case1":
        expected_delta = float(arms[1].manifest["fault_strength"])
        validations["fault_metadata"] = validate_case1_fault_metadata(arms[1], expected_delta)
    validations["signature"] = _validate_signature(
        case,
        arms,
        allow_failures=allow_signature_failures,
    )

    rows = _table_rows(arms)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = _write_plots(case, arms, output_dir)
    outliers = _write_outliers(arms[1], output_dir)
    result = {
        "schema_version": 1,
        "case": case,
        "thresholds": {"repeatability_mae_lt": REPEAT_LIMIT, "parity_mae_lt": PARITY_LIMIT},
        "arms": {arm.name: str(arm.run_dir) for arm in arms},
        "validations": validations,
        "rows": rows,
        "plots": plots,
        "outliers": outliers,
    }
    if case == "case3":
        grad_norms = _case3_grad_norms(arms)
        result["validations"]["finite_nonzero_grad_norms"] = grad_norms
        result["policy_update_timeline"] = _case3_timeline(arms, output_dir, grad_norms)
        result["plots"].append(result["policy_update_timeline"]["plot"])
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(case, rows, validations), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("case1", "case2", "case3"), required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--fault-run", type=Path, required=True)
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-signature-failures",
        action="store_true",
        help="Emit artifacts while retaining failed signature checks in the report",
    )
    args = parser.parse_args()
    result = analyze_case(
        args.case,
        args.clean_run,
        args.fault_run,
        args.fixed_run,
        args.output_dir,
        allow_signature_failures=args.allow_signature_failures,
    )
    print(json.dumps(result["rows"], indent=2))
    print(f"Wrote tutorial analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
