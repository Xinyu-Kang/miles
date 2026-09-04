"""Case 1: deliberately offset stored decode logprobs without changing tokens."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from typing import Any

from miles.utils.types import Sample

from .fault_guard import require_fault

FAULT_NAME = "decode_logprob_offset"
DELTA_ENV = "MILES_LOGPROB_DECODE_OFFSET_NAT"

logger = logging.getLogger(__name__)


def _iter_samples(groups: list[list[Sample | list[Sample]]]) -> Iterator[Sample]:
    for group in groups:
        for item in group:
            if isinstance(item, list):
                yield from item
            else:
                yield item


def _validate_logprobs(sample: Sample, values: list[float], label: str) -> None:
    if len(values) != sample.response_length:
        raise ValueError(
            f"{label} length mismatch for sample {sample.index}: "
            f"expected {sample.response_length}, got {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains NaN or Inf for sample {sample.index}")


def process(args: Any, all_samples: list[list[Sample | list[Sample]]], data_source: Any) -> None:
    """Inject a -delta bookkeeping error at response positions 1+."""

    del data_source
    raw_delta = os.environ.get(DELTA_ENV, "0.10")
    activation = require_fault(args, expected_name=FAULT_NAME, strength=raw_delta)
    affected = 0

    for sample in _iter_samples(all_samples):
        if sample.response_length <= 0 or sample.rollout_log_probs is None:
            continue
        if "fault_injection" in (sample.metadata or {}):
            raise RuntimeError(f"Sample {sample.index} already carries fault_injection metadata")

        original_tokens = tuple(int(token) for token in sample.tokens)
        clean = [float(value) for value in sample.rollout_log_probs]
        _validate_logprobs(sample, clean, "clean generation logprobs")

        faulted = list(clean)
        for position in range(1, len(faulted)):
            faulted[position] -= activation.strength
        _validate_logprobs(sample, faulted, "faulted generation logprobs")

        sample.metadata = dict(sample.metadata or {})
        sample.metadata["fault_injection"] = {
            "name": FAULT_NAME,
            "delta": activation.strength,
            "clean_decode_logprobs": clean,
            "faulted_decode_logprobs": list(faulted),
            "response_token_ids": [
                int(token) for token in sample.tokens[-sample.response_length :]
            ],
        }
        sample.rollout_log_probs = faulted

        if tuple(int(token) for token in sample.tokens) != original_tokens:
            raise AssertionError(f"Fault injection changed token IDs for sample {sample.index}")
        if clean and faulted[0] != clean[0]:
            raise AssertionError(f"Fault injection changed position 0 for sample {sample.index}")
        affected += 1

    if affected == 0:
        raise RuntimeError("decode_logprob_offset did not find any completed response logprobs")
    logger.critical(
        "DELIBERATE tutorial fault %s applied to %d samples at response positions 1+",
        FAULT_NAME,
        affected,
    )
