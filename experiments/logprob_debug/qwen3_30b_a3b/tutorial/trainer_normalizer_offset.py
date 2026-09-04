"""Fallback Case 2 fault: offset only frozen trainer-forward logprobs."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

from .fault_guard import require_fault

FAULT_NAME = "trainer_normalizer_offset"
OFFSET_ENV = "MILES_LOGPROB_TRAINER_NORMALIZER_OFFSET_NAT"

logger = logging.getLogger(__name__)


def _offset_result(
    result: tuple[torch.Tensor, torch.Tensor | None], offset: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return an out-of-place logprob offset without changing entropy."""

    log_probs, entropy = result
    return log_probs - offset, entropy


def install(args: Any, model: Any, store_prefix: str) -> None:
    """Install/update the guarded forward-only C-score offset.

    Miles calls this existing plugin hook immediately before each frozen
    trainer scoring pass.  The wrapper is installed once per trainer process;
    later hook calls only select whether the imminent pass is C/ref/repeat.
    Training forwards remain untouched because they execute with gradients
    enabled.
    """

    del model
    raw_offset = os.environ.get(OFFSET_ENV, "")
    try:
        offset = float(raw_offset)
    except ValueError as exc:
        raise ValueError(f"{OFFSET_ENV} must be a finite positive float") from exc
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError(f"{OFFSET_ENV} must be a finite positive float, got {raw_offset!r}")
    require_fault(args, expected_name=FAULT_NAME, strength=offset)

    from miles.backends.training_utils.loss_hub import logit_processors

    current = logit_processors.calculate_log_probs_and_entropy
    state = getattr(current, "_tutorial_trainer_offset_state", None)
    if state is None:
        state = {"store_prefix": store_prefix, "offset": offset}
        original = current

        def wrapped(*call_args: Any, **call_kwargs: Any):
            result = original(*call_args, **call_kwargs)
            prefix = str(state["store_prefix"])
            if torch.is_grad_enabled() or prefix.startswith("ref_"):
                return result
            return _offset_result(result, float(state["offset"]))

        wrapped._tutorial_trainer_offset_state = state  # type: ignore[attr-defined]
        wrapped._tutorial_trainer_offset_original = original  # type: ignore[attr-defined]
        logit_processors.calculate_log_probs_and_entropy = wrapped
    else:
        state["store_prefix"] = store_prefix
        state["offset"] = offset

    logger.critical(
        "DELIBERATE tutorial fault %s: frozen trainer score prefix=%r offset=-%.6f nat",
        FAULT_NAME,
        store_prefix,
        offset,
    )
