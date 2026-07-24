"""Shared guarded workload launcher used by the AMD overlay entry points."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from amd.agentic.io import load_structured_file
from amd.agentic.launch.preflight import collect_source


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a manifest object")
    return value


def launch(workload_kind: str, default_config: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Guarded AMD {workload_kind} launcher")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the pinned command; without this flag only print it",
    )
    args = parser.parse_args(argv)

    config = load_structured_file(args.config)
    manifest = _load_manifest(args.manifest)
    if config.get("kind") != workload_kind:
        raise ValueError(f"{args.config} is kind {config.get('kind')!r}, expected {workload_kind!r}")
    if not manifest.get("qualification", {}).get("passed", False):
        raise RuntimeError(f"{args.manifest} did not pass preflight")
    if manifest.get("source", {}).get("dirty"):
        raise RuntimeError(f"{args.manifest} records a dirty source tree")
    if manifest.get("workload", {}).get("kind") != workload_kind:
        raise RuntimeError(f"{args.manifest} was collected for a different workload")
    if not config.get("enabled", False):
        raise RuntimeError(f"{workload_kind} is disabled until its artifacts and command are pinned")

    current_source = collect_source(args.repo_root.resolve())
    recorded_source = manifest.get("source", {})
    if current_source["dirty"]:
        raise RuntimeError("current source tree is dirty; collect a new manifest only after reviewing the changes")
    if current_source["commit"] != recorded_source.get("commit"):
        raise RuntimeError(
            f"current source revision {current_source['commit']} differs from manifest "
            f"{recorded_source.get('commit')}"
        )

    command = config.get("launch", {}).get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("launch.command must be a non-empty string list")

    print(shlex.join(command))
    if not args.execute:
        return 0
    return subprocess.run(command, check=False).returncode
