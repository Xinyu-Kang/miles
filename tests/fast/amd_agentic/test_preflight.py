import hashlib
import subprocess
from pathlib import Path

from amd.agentic.launch import preflight


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _clean_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "tracked.txt").write_text("tracked\n")
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=AMD Test",
        "-c",
        "user.email=amd-test@example.com",
        "commit",
        "-m",
        "initial",
    )
    return repo, _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_collect_source_records_clean_and_dirty_states(tmp_path):
    repo, commit = _clean_repo(tmp_path)

    clean = preflight.collect_source(repo)
    assert clean["commit"] == commit
    assert clean["dirty"] is False
    assert clean["tracked_patch"] == ""

    (repo / "tracked.txt").write_text("changed\n")
    (repo / "untracked.txt").write_text("untracked\n")
    dirty = preflight.collect_source(repo)
    assert dirty["dirty"] is True
    assert any("tracked.txt" in line for line in dirty["status_lines"])
    assert any("untracked.txt" in line for line in dirty["status_lines"])
    assert dirty["tracked_patch"]
    assert len(dirty["tracked_patch_sha256"]) == 64


def test_collect_environment_is_allowlist_only_and_redacts_secret_names():
    observed = preflight.collect_environment(
        ["SAFE_VALUE", "SERVICE_API_KEY", "MISSING"],
        {"SAFE_VALUE": "visible", "SERVICE_API_KEY": "do-not-log", "OTHER": "ignored"},
    )
    assert observed == {
        "MISSING": None,
        "SAFE_VALUE": "visible",
        "SERVICE_API_KEY": {"present": True, "value": "<redacted>"},
    }


def test_build_manifest_passes_with_pinned_inputs(tmp_path, monkeypatch):
    repo, commit = _clean_repo(tmp_path)
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"model")
    asset_sha = hashlib.sha256(b"model").hexdigest()
    config = {
        "schema_version": "1.0",
        "name": "phase0-test",
        "kind": "retool",
        "enabled": True,
        "container": {
            "image_ref": "example/image:test",
            "image_digest": "sha256:" + "a" * 64,
        },
        "expected": {
            "miles_revision": commit,
            "allow_dirty": False,
            "require_image_digest": True,
            "gpu_count": 8,
            "gpu_arch": "gfx950",
        },
        "assets": [
            {
                "name": "fixture",
                "kind": "model",
                "path": str(asset),
                "sha256": asset_sha,
                "verify_content_sha256": True,
                "required": True,
            }
        ],
        "environment_allowlist": ["SAFE_VALUE"],
        "services": [],
        "launch": {"arguments": ["--one-step"], "command": ["python", "train.py"]},
    }
    monkeypatch.setattr(preflight, "collect_runtime", lambda: {"python": "3.10"})
    monkeypatch.setattr(
        preflight,
        "collect_gpus",
        lambda: {"count": 8, "architectures": ["gfx950"], "devices": [], "collector_error": None},
    )

    manifest = preflight.build_manifest(
        config,
        config_path=tmp_path / "config.yaml",
        repo_root=repo,
        run_id="phase0-test",
        environ={"SAFE_VALUE": "yes"},
    )

    assert manifest["qualification"] == {"passed": True, "errors": [], "warnings": []}
    assert manifest["source"]["commit"] == commit
    assert manifest["assets"][0]["observed_sha256"] == asset_sha
    assert manifest["environment"] == {"SAFE_VALUE": "yes"}


def test_qualification_blocks_missing_unpinned_assets():
    manifest = {
        "source": {"commit": "a" * 40, "dirty": False},
        "container": {"image_ref": "image", "image_digest": "sha256:" + "b" * 64},
        "gpus": {"count": 8, "architectures": ["gfx950"]},
        "assets": [
            {
                "name": "missing-model",
                "path": "/missing",
                "required": True,
                "exists": False,
                "declared_revision": None,
                "declared_sha256": None,
            }
        ],
        "services": [],
    }
    config = {
        "schema_version": "1.0",
        "name": "blocked-test",
        "kind": "retool",
        "enabled": False,
        "expected": {
            "allow_dirty": False,
            "require_image_digest": True,
            "gpu_count": 8,
            "gpu_arch": "gfx950",
        },
        "launch": {"arguments": []},
    }

    result = preflight.qualify(manifest, config)

    assert result["passed"] is False
    assert {issue["code"] for issue in result["errors"]} == {"missing_asset", "unpinned_asset"}
    assert {issue["code"] for issue in result["warnings"]} == {
        "launch_arguments_empty",
        "workload_disabled",
    }
