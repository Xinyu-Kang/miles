"""Small, dependency-free I/O helpers for the AMD qualification harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_structured_file(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML.

    The checked-in ``.yaml`` configs intentionally use JSON syntax, which is a
    valid YAML subset and keeps the preflight path dependency-free. Conventional
    YAML is accepted when PyYAML is installed.
    """

    source = Path(path)
    text = os.path.expandvars(source.read_text())
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml
        except ImportError as import_error:
            raise ValueError(
                f"{source} is not JSON-compatible YAML and PyYAML is not installed"
            ) from import_error
        value = yaml.safe_load(text)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"{source} must contain a top-level object") from json_error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a top-level object")
    return value


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)
    return destination


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_layout(path: str | Path) -> dict[str, Any]:
    """Return a cheap, deterministic directory inventory.

    This is intentionally a layout fingerprint, not a content checksum. A
    publishable run must still declare an immutable asset revision or checksum.
    """

    root = Path(path)
    rows: list[str] = []
    total_bytes = 0
    file_count = 0
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        size = item.stat().st_size
        rows.append(f"{item.relative_to(root).as_posix()}\0{size}")
        file_count += 1
        total_bytes += size
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "layout_sha256": sha256_bytes("\n".join(rows).encode()),
    }
