"""Common Case 3 stimulus that guarantees a nonzero policy update."""

from __future__ import annotations

import logging
import os
from typing import Any

from miles.utils.types import Sample

ENABLE_ENV = "MILES_ENABLE_LOGPROB_TRAINING_STIMULUS"

logger = logging.getLogger(__name__)


def _flatten(group: list[Sample | list[Sample]]) -> list[Sample]:
    result = []
    for item in group:
        result.extend(item if isinstance(item, list) else [item])
    return result


def process(args: Any, all_samples: list[list[Sample | list[Sample]]], data_source: Any) -> None:
    """Assign balanced +/-1 rewards without touching tokens or logprobs."""

    del data_source
    if os.environ.get(ENABLE_ENV) != "1":
        raise RuntimeError(f"Case 3 training stimulus requires {ENABLE_ENV}=1")
    if getattr(args, "dump_details", None) is None:
        raise RuntimeError("Case 3 training stimulus requires --dump-details")
    if bool(getattr(args, "ci_test", False)) or os.environ.get("CI", "").lower() not in ("", "0", "false", "no", "off"):
        raise RuntimeError("Case 3 training stimulus is forbidden in CI")

    changed = 0
    for group_index, group in enumerate(all_samples):
        samples = _flatten(group)
        if len(samples) < 2:
            raise ValueError(f"Case 3 stimulus requires at least two samples in group {group_index}")
        for ordinal, sample in enumerate(samples):
            tokens_before = tuple(int(token) for token in sample.tokens)
            logprobs_before = None if sample.rollout_log_probs is None else tuple(float(value) for value in sample.rollout_log_probs)
            original_reward = sample.reward
            sample.reward = -1.0 if ordinal % 2 == 0 else 1.0
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["tutorial_training_signal"] = {
                "name": "balanced_nonzero_rewards",
                "group_index": group_index,
                "ordinal": ordinal,
                "original_reward": original_reward,
                "assigned_reward": sample.reward,
            }
            if tuple(int(token) for token in sample.tokens) != tokens_before:
                raise AssertionError("Case 3 training stimulus changed token IDs")
            after = None if sample.rollout_log_probs is None else tuple(float(value) for value in sample.rollout_log_probs)
            if after != logprobs_before:
                raise AssertionError("Case 3 training stimulus changed rollout logprobs")
            changed += 1
    if changed == 0:
        raise RuntimeError("Case 3 training stimulus found no samples")
    logger.warning(
        "CONTROLLED TUTORIAL TRAINING STIMULUS ACTIVE: assigned balanced +/-1 rewards to %d samples",
        changed,
    )
