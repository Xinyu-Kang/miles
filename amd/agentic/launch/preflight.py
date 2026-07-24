"""Collect a reproducible AMD agentic-RL run manifest.

The preflight is deliberately independent of Miles imports so it can diagnose a
broken Python environment. Training launchers must use strict mode and refuse to
run when the resulting manifest does not pass.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amd.agentic import SCHEMA_VERSION
from amd.agentic.io import directory_layout, load_structured_file, sha256_bytes, sha256_file, write_json

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "retool_qwen3_4b_mi355x.yaml"
DEFAULT_PACKAGE_NAMES = (
    "miles",
    "torch",
    "ray",
    "sglang",
    "transformers",
    "flash_attn",
    "amd-aiter",
    "transformer_engine",
)
SECRET_NAME_PATTERN = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|CREDENTIAL)", re.IGNORECASE)
MAX_PATCH_BYTES = 1024 * 1024


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # Bind-mounted repositories can be owned by the host user. The one-command
    # override avoids mutating either global or repository git configuration.
    return _run(
        ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), *arguments],
        timeout=60,
    )


def collect_source(repo_root: Path) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(repo_root, "diff", "--binary", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"{repo_root} is not a readable git repository: {head.stderr.strip()}")

    patch = diff.stdout
    patch_bytes = patch.encode()
    patch_truncated = len(patch_bytes) > MAX_PATCH_BYTES
    if patch_truncated:
        patch = patch_bytes[:MAX_PATCH_BYTES].decode(errors="replace")
    status_lines = [line for line in status.stdout.splitlines() if line]
    return {
        "repo_root": str(repo_root.resolve()),
        "commit": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty": bool(status_lines),
        "status_lines": status_lines,
        "tracked_patch": patch,
        "tracked_patch_sha256": sha256_bytes(diff.stdout.encode()),
        "tracked_patch_truncated": patch_truncated,
        "remotes": _collect_remotes(repo_root),
    }


def _collect_remotes(repo_root: Path) -> dict[str, str]:
    result = _git(repo_root, "remote", "-v")
    remotes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] == "(fetch)":
            remotes[fields[0]] = fields[1]
    return remotes


def collect_runtime(package_names: tuple[str, ...] = DEFAULT_PACKAGE_NAMES) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "rocm_version": _read_rocm_version(),
    }


def _read_rocm_version() -> str | None:
    candidates = (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
        Path("/opt/rocm/.info/version-hip-libraries"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text().strip()
    result = _run(["hipcc", "--version"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "HIP version" in line:
                return line.partition(":")[2].strip()
    return None


def collect_gpus() -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    result = _run(["rocm-smi", "--showproductname", "--showuniqueid", "--showmeminfo", "vram", "--json"], timeout=60)
    raw: dict[str, Any] | None = None
    if result.returncode == 0:
        try:
            candidate = json.loads(result.stdout)
            raw = candidate if isinstance(candidate, dict) else None
        except json.JSONDecodeError:
            raw = None
    if raw:
        for device_id, metadata in sorted(raw.items()):
            if not isinstance(metadata, dict):
                continue
            devices.append(
                {
                    "id": device_id,
                    "name": _first_value(
                        metadata,
                        "Card Series",
                        "Card series",
                        "Product Name",
                        "Card Model",
                        "Card model",
                        "Card SKU",
                    ),
                    "unique_id": _first_value(metadata, "Unique ID", "Unique ID (Hex)"),
                    "vram_total_bytes": _first_integer(
                        metadata,
                        "VRAM Total Memory (B)",
                        "VRAM Total Used Memory (B)",
                    ),
                    "arch": _first_value(metadata, "GFX Version"),
                }
            )

    archs = sorted({device["arch"] for device in devices if device.get("arch")}) or _collect_gpu_archs()
    if devices and len(archs) == 1:
        for device in devices:
            device["arch"] = archs[0]
    return {
        "count": len(devices),
        "architectures": archs,
        "devices": devices,
        "collector_error": None if result.returncode == 0 else result.stderr.strip(),
    }


def _first_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _first_integer(metadata: dict[str, Any], *keys: str) -> int | None:
    value = _first_value(metadata, *keys)
    if value is None:
        return None
    match = re.search(r"\d+", str(value).replace(",", ""))
    return int(match.group()) if match else None


def _collect_gpu_archs() -> list[str]:
    result = _run(["rocminfo"], timeout=60)
    if result.returncode != 0:
        return []
    return sorted(
        {
            match.group(1)
            for line in result.stdout.splitlines()
            if (match := re.match(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", line, flags=re.IGNORECASE))
        }
    )


def collect_assets(asset_specs: list[dict[str, Any]], repo_root: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for spec in asset_specs:
        raw_path = str(spec.get("path", ""))
        path = Path(raw_path)
        if raw_path and not path.is_absolute():
            path = repo_root / path
        exists = bool(raw_path) and path.exists()
        observed: dict[str, Any] = {
            "name": str(spec.get("name", "")),
            "kind": str(spec.get("kind", "")),
            "path": str(path) if raw_path else "",
            "required": bool(spec.get("required", True)),
            "declared_revision": spec.get("revision"),
            "declared_sha256": spec.get("sha256"),
            "exists": exists,
        }
        if exists and path.is_file():
            observed.update(
                {
                    "type": "file",
                    "size_bytes": path.stat().st_size,
                    "observed_sha256": sha256_file(path) if spec.get("verify_content_sha256") else None,
                }
            )
        elif exists and path.is_dir():
            observed.update({"type": "directory", **directory_layout(path)})
        else:
            observed["type"] = None
        assets.append(observed)
    return assets


def collect_environment(names: list[str], environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in sorted(set(names)):
        if name not in environ:
            values[name] = None
        elif SECRET_NAME_PATTERN.search(name):
            values[name] = {"present": True, "value": "<redacted>"}
        else:
            values[name] = environ[name]
    return values


def collect_services(service_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    for spec in service_specs:
        url = str(spec.get("url", ""))
        started = time.monotonic()
        status: int | None = None
        error: str | None = None
        if url:
            try:
                request = urllib.request.Request(url, method=str(spec.get("method", "GET")).upper())
                with urllib.request.urlopen(request, timeout=float(spec.get("timeout_seconds", 3))) as response:
                    status = response.status
            except (urllib.error.URLError, ValueError, TimeoutError) as caught:
                error = str(caught)
        else:
            error = "missing URL"
        services.append(
            {
                "name": str(spec.get("name", "")),
                "url": url,
                "required": bool(spec.get("required", True)),
                "status": status,
                "healthy": status is not None and 200 <= status < 400,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "error": error,
            }
        )
    return services


def qualify(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def error(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    def warning(code: str, message: str) -> None:
        warnings.append({"code": code, "message": message})

    if config.get("schema_version") != SCHEMA_VERSION:
        error(
            "config_schema_version",
            f"expected config schema {SCHEMA_VERSION}, observed {config.get('schema_version')!r}",
        )
    if not config.get("name") or not config.get("kind"):
        error("invalid_workload_identity", "config must declare non-empty name and kind")

    expected = config.get("expected", {})
    source = manifest["source"]
    expected_commit = expected.get("miles_revision")
    if expected_commit and source["commit"] != expected_commit:
        error("source_revision_mismatch", f"expected Miles {expected_commit}, observed {source['commit']}")
    if source["dirty"] and not bool(expected.get("allow_dirty", False)):
        error("dirty_worktree", "Miles worktree has uncommitted changes")

    container = manifest["container"]
    if not container.get("image_ref"):
        error("missing_image_ref", "container image reference is not declared")
    if expected.get("require_image_digest", True) and not container.get("image_digest"):
        error("missing_image_digest", "container image digest is required")
    elif container.get("image_digest") and not re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_digest"]):
        error("invalid_image_digest", "container image digest must be sha256:<64 lowercase hex characters>")

    gpus = manifest["gpus"]
    expected_gpu_count = expected.get("gpu_count")
    if expected_gpu_count is not None and gpus["count"] != int(expected_gpu_count):
        error("gpu_count_mismatch", f"expected {expected_gpu_count} GPUs, observed {gpus['count']}")
    expected_arch = expected.get("gpu_arch")
    if expected_arch and expected_arch not in gpus["architectures"]:
        error("gpu_arch_mismatch", f"expected {expected_arch}, observed {gpus['architectures']}")

    for asset in manifest["assets"]:
        if asset["required"] and not asset["exists"]:
            error("missing_asset", f"{asset['name'] or asset['path']} is missing")
        if asset["required"] and not (asset.get("declared_revision") or asset.get("declared_sha256")):
            error("unpinned_asset", f"{asset['name'] or asset['path']} has no revision or SHA-256")
        declared = asset.get("declared_sha256")
        observed = asset.get("observed_sha256")
        if declared and observed and declared != observed:
            error("asset_checksum_mismatch", f"{asset['name']}: expected {declared}, observed {observed}")

    for service in manifest["services"]:
        if service["required"] and not service["healthy"]:
            error("service_unhealthy", f"{service['name'] or service['url']} is unhealthy: {service['error']}")

    if not config.get("launch", {}).get("arguments"):
        warning("launch_arguments_empty", "effective launch arguments are not configured yet")
    if not config.get("enabled", False):
        warning("workload_disabled", "workload is intentionally disabled until its assets and command are pinned")

    return {"passed": not errors, "errors": errors, "warnings": warnings}


def build_manifest(
    config: dict[str, Any],
    *,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    source = collect_source(repo_root)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "name": config.get("name"),
            "kind": config.get("kind"),
            "enabled": bool(config.get("enabled", False)),
        },
        "source": source,
        "container": {
            "image_ref": config.get("container", {}).get("image_ref"),
            "image_digest": config.get("container", {}).get("image_digest"),
        },
        "runtime": collect_runtime(),
        "gpus": collect_gpus(),
        "assets": collect_assets(config.get("assets", []), repo_root),
        "launch": {
            "config_path": str(config_path.resolve()),
            "arguments": config.get("launch", {}).get("arguments", []),
            "command": config.get("launch", {}).get("command", []),
        },
        "environment": collect_environment(config.get("environment_allowlist", []), environ),
        "services": collect_services(config.get("services", [])),
    }
    manifest["qualification"] = qualify(manifest, config)
    return manifest


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="write the manifest but exit zero even when qualification fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_structured_file(args.config)
    output = args.output or Path(__file__).parents[1] / "results" / args.run_id / "manifest.json"
    manifest = build_manifest(
        config,
        config_path=args.config,
        repo_root=args.repo_root.resolve(),
        run_id=args.run_id,
    )
    write_json(output, manifest)
    qualification = manifest["qualification"]
    print(f"wrote {output}")
    print(
        f"qualification passed={qualification['passed']} "
        f"errors={len(qualification['errors'])} warnings={len(qualification['warnings'])}"
    )
    for issue in qualification["errors"] + qualification["warnings"]:
        print(f"{issue['code']}: {issue['message']}")
    if qualification["passed"] or args.collect_only:
        return 0
    print(
        "preflight blocked launch; inspect the manifest or rerun collection with --collect-only",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
