import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).parents[3]
MANIFEST_SCHEMA = ROOT / "amd" / "agentic" / "manifests" / "schema.json"
TRAJECTORY_SCHEMA = ROOT / "amd" / "agentic" / "manifests" / "trajectory_schema.json"
TRAJECTORY = ROOT / "amd" / "agentic" / "tests" / "fixtures" / "two_tool_trajectory.json"


def _load(path):
    return json.loads(path.read_text())


def test_checked_in_schemas_are_valid_and_accept_trajectory_fixture():
    manifest_schema = _load(MANIFEST_SCHEMA)
    trajectory_schema = _load(TRAJECTORY_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    jsonschema.Draft202012Validator.check_schema(trajectory_schema)
    jsonschema.validate(_load(TRAJECTORY), trajectory_schema)


def test_manifest_schema_accepts_minimal_complete_manifest():
    manifest = {
        "schema_version": "1.0",
        "run_id": "schema-test",
        "created_at": "2026-07-24T00:00:00+00:00",
        "workload": {"name": "test", "kind": "qualification", "enabled": True},
        "source": {
            "repo_root": "/workspace/miles",
            "commit": "a" * 40,
            "branch": "amd/agentic-multiturn",
            "dirty": False,
            "status_lines": [],
            "tracked_patch": "",
            "tracked_patch_sha256": "b" * 64,
            "tracked_patch_truncated": False,
            "remotes": {},
        },
        "container": {
            "image_ref": "image:test",
            "image_digest": "sha256:" + "c" * 64,
        },
        "runtime": {
            "python": "3.10",
            "python_implementation": "CPython",
            "platform": "Linux",
            "executable": "/usr/bin/python",
            "packages": {},
            "rocm_version": "7.2",
        },
        "gpus": {
            "count": 8,
            "architectures": ["gfx950"],
            "devices": [],
            "collector_error": None,
        },
        "assets": [],
        "launch": {"config_path": "config.yaml", "arguments": [], "command": []},
        "environment": {},
        "services": [],
        "qualification": {"passed": True, "errors": [], "warnings": []},
    }

    jsonschema.validate(manifest, _load(MANIFEST_SCHEMA))
