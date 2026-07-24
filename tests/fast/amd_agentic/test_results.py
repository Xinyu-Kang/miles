from amd.agentic.tools.compare_runs import compare_summaries
from amd.agentic.tools.summarize_run import build_summary


def _manifest():
    return {
        "run_id": "run-a",
        "created_at": "2026-07-24T00:00:00+00:00",
        "workload": {"name": "fixture", "kind": "retool", "enabled": True},
        "source": {"commit": "a" * 40, "dirty": False},
        "container": {"image_ref": "image:test", "image_digest": "sha256:" + "b" * 64},
        "gpus": {"count": 8, "architectures": ["gfx950"]},
        "qualification": {"passed": True, "errors": [], "warnings": []},
    }


def _validation(duration=6.0):
    return {
        "passed": True,
        "trace_count": 1,
        "repeat": 1,
        "validation_count": 1,
        "results": [
            {
                "valid": True,
                "issues": [],
                "metrics": {
                    "status": "completed",
                    "turn_count": 3,
                    "tool_call_count": 2,
                    "prompt_tokens": 3,
                    "response_tokens": 10,
                    "trainable_tokens": 5,
                    "duration_seconds": duration,
                },
            }
        ],
    }


def test_summary_is_regenerated_without_wandb():
    summary = build_summary(_manifest(), _validation())

    assert summary["reproducibility"]["preflight_passed"] is True
    assert summary["validation"]["passed"] is True
    assert summary["validation"]["average_tool_call_count"] == 2
    assert summary["validation"]["average_trainable_tokens"] == 5


def test_compare_reports_numeric_delta():
    baseline = build_summary(_manifest(), _validation(duration=6.0))
    candidate_manifest = _manifest()
    candidate_manifest["run_id"] = "run-b"
    candidate = build_summary(candidate_manifest, _validation(duration=4.5))

    comparison = compare_summaries(baseline, candidate)

    assert comparison["baseline_run_id"] == "run-a"
    assert comparison["candidate_run_id"] == "run-b"
    assert comparison["metrics"]["average_duration_seconds"]["delta"] == -1.5
