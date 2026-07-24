"""Guarded entry point for the future MI355X ReTool workload."""

from __future__ import annotations

from pathlib import Path

from amd.agentic.launch.workload import launch

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "retool_qwen3_4b_mi355x.yaml"


def main() -> int:
    return launch("retool", DEFAULT_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
