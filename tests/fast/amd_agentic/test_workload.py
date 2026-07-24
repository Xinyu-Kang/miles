import json
import subprocess

import pytest

from amd.agentic.launch.workload import launch


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def test_disabled_workload_cannot_launch(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "kind": "retool",
                "enabled": False,
                "launch": {"command": ["python", "train.py"]},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "qualification": {"passed": True},
                "source": {"dirty": False},
                "workload": {"kind": "retool"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="disabled"):
        launch("retool", config, ["--config", str(config), "--manifest", str(manifest)])


def test_failed_preflight_cannot_launch(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"kind": "retool", "enabled": True, "launch": {"command": ["true"]}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "qualification": {"passed": False},
                "source": {"dirty": False},
                "workload": {"kind": "retool"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        launch("retool", config, ["--config", str(config), "--manifest", str(manifest)])


def test_manifest_revision_must_match_current_source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "tracked").write_text("one\n")
    _git(repo, "add", "tracked")
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
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"kind": "retool", "enabled": True, "launch": {"command": ["true"]}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "qualification": {"passed": True},
                "source": {"dirty": False, "commit": "a" * 40},
                "workload": {"kind": "retool"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="differs from manifest"):
        launch(
            "retool",
            config,
            [
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--repo-root",
                str(repo),
            ],
        )
