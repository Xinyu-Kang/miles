"""Guarded entry point for the future MI355X Qwen3-Coder SWE workload."""

from __future__ import annotations

from pathlib import Path

from amd.agentic.launch.workload import launch

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "swe_qwen3_coder_mi355x.yaml"


def main() -> int:
    return launch("swe", DEFAULT_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
