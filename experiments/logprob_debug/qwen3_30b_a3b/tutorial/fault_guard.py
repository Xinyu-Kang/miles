#!/usr/bin/env python3
"""Fail-closed activation guard shared by tutorial-only logprob faults."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

ENABLE_ENV = "MILES_ENABLE_LOGPROB_FAULT_INJECTION"
ACTIVE_FAULT_ENV = "MILES_LOGPROB_ACTIVE_FAULT"
STRENGTH_ENV = "MILES_LOGPROB_FAULT_STRENGTH"
_DISABLED_VALUES = {"", "0", "false", "no", "off"}
_BANNER_EDGE = "=" * 88

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaultActivation:
    name: str
    strength: float
    banner: str


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in _DISABLED_VALUES


def _validated_strength(value: str | float) -> float:
    strength = float(value)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError(f"Fault strength must be positive and finite, got {value!r}")
    return strength


def fault_banner(name: str, strength: str | float) -> str:
    return (
        f"{_BANNER_EDGE}\n"
        f"DELIBERATE LOGPROB FAULT INJECTION ACTIVE: name={name} strength={strength}\n"
        "THIS IS A CONTROLLED TUTORIAL FAULT, NOT A NATURALLY DISCOVERED FRAMEWORK BUG.\n"
        f"{_BANNER_EDGE}"
    )


def require_fault(args: Any, *, expected_name: str, strength: str | float) -> FaultActivation:
    """Validate every safety gate at the point where a fault mutates data."""

    if os.environ.get(ENABLE_ENV) != "1":
        raise RuntimeError(f"{expected_name} requires {ENABLE_ENV}=1")
    active_name = os.environ.get(ACTIVE_FAULT_ENV)
    if active_name != expected_name:
        raise RuntimeError(
            f"Expected exactly one active fault {expected_name!r}, got {active_name!r}"
        )
    if getattr(args, "dump_details", None) is None:
        raise RuntimeError(f"{expected_name} requires --dump-details")
    if bool(getattr(args, "ci_test", False)) or _truthy(os.environ.get("CI")):
        raise RuntimeError(f"{expected_name} is forbidden in CI")

    value = _validated_strength(strength)
    banner = fault_banner(expected_name, value)
    logger.critical("\n%s", banner)
    print(banner, flush=True)
    return FaultActivation(name=expected_name, strength=value, banner=banner)


def require_no_fault_environment() -> None:
    """Reject leaked fault activation from a clean or fixed tutorial arm."""

    if os.environ.get(ENABLE_ENV) == "1" or os.environ.get(ACTIVE_FAULT_ENV):
        raise RuntimeError(
            "Clean/fixed tutorial arms require fault injection environment variables to be unset"
        )


def merge_runtime_env(base_json: str, extra_json: str) -> str:
    """Merge two JSON string maps without allowing protected-key replacement."""

    base = json.loads(base_json)
    extra = json.loads(extra_json)
    if not isinstance(base, dict) or not isinstance(extra, dict):
        raise TypeError("Runtime environments must be JSON objects")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in base.items()):
        raise TypeError("Base runtime environment must map strings to strings")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in extra.items()):
        raise TypeError("Experiment runtime environment must map strings to strings")
    overlap = sorted(base.keys() & extra.keys())
    if overlap:
        raise ValueError(f"Experiment runtime environment overwrites protected keys: {overlap}")
    base.update(extra)
    return json.dumps(base, separators=(",", ":"), sort_keys=True)


def build_fault_runtime_env(
    fault: str,
    strength: str | float,
    assignments: list[str],
) -> str:
    value = _validated_strength(strength)
    result = {
        ENABLE_ENV: "1",
        ACTIVE_FAULT_ENV: fault,
        STRENGTH_ENV: str(value),
    }
    for assignment in assignments:
        key, separator, assigned = assignment.partition("=")
        if not separator or not key.startswith("MILES_") or key in result:
            raise ValueError(f"Invalid or duplicate tutorial runtime assignment: {assignment!r}")
        result[key] = assigned
    return json.dumps(result, separators=(",", ":"), sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge-runtime-env")
    merge_parser.add_argument("base_json")
    merge_parser.add_argument("extra_json")

    build_parser = subparsers.add_parser("build-runtime-env")
    build_parser.add_argument("--fault", required=True)
    build_parser.add_argument("--strength", required=True)
    build_parser.add_argument("--set", action="append", default=[])

    banner_parser = subparsers.add_parser("banner")
    banner_parser.add_argument("--fault", required=True)
    banner_parser.add_argument("--strength", required=True)

    args = parser.parse_args()
    if args.command == "merge-runtime-env":
        print(merge_runtime_env(args.base_json, args.extra_json))
    elif args.command == "build-runtime-env":
        print(build_fault_runtime_env(args.fault, args.strength, args.set))
    else:
        print(fault_banner(args.fault, _validated_strength(args.strength)))


if __name__ == "__main__":
    main()
