"""Case 3 controller guard for one deliberately skipped rollout-weight sync."""

from __future__ import annotations

import logging
from typing import Any

from .fault_guard import require_fault

FAULT_NAME = "stale_rollout_weights"

logger = logging.getLogger(__name__)


def should_skip_weight_update(args: Any, rollout_id: int) -> bool:
    """Authorize the configured one-shot skipped sync, failing closed."""

    target = getattr(args, "debug_skip_rollout_weight_update_at", None)
    if target is None or int(target) != rollout_id:
        return False
    if getattr(args, "fully_async", False):
        raise RuntimeError(f"{FAULT_NAME} tutorial hook supports only the synchronous driver")
    require_fault(args, expected_name=FAULT_NAME, strength=1)
    logger.critical(
        "DELIBERATE tutorial fault %s: skipping actor-to-rollout weight update after rollout %d",
        FAULT_NAME,
        rollout_id,
    )
    return True
