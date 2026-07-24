import copy
from pathlib import Path

from amd.agentic.tools.validate_trajectory import build_validation_report, load_traces, validate_trajectory

FIXTURE = Path(__file__).parents[3] / "amd" / "agentic" / "tests" / "fixtures" / "two_tool_trajectory.json"


def _fixture():
    return load_traces(FIXTURE)[0]


def _codes(result):
    return {issue.code for issue in result.issues}


def test_deterministic_two_tool_trajectory_passes_100_times():
    report = build_validation_report([_fixture()], repeat=100)

    assert report["passed"] is True
    assert report["validation_count"] == 100
    assert report["valid_count"] == 100
    assert report["invalid_count"] == 0
    assert report["issue_counts"] == {}
    assert {result["metrics"]["tool_call_count"] for result in report["results"]} == {2}


def test_corrupted_generation_prefix_is_rejected():
    trace = copy.deepcopy(_fixture())
    trace["events"][2]["input_ids"][-1] = 999

    result = validate_trajectory(trace)

    assert result.valid is False
    assert "prefix_mismatch" in _codes(result)


def test_corrupted_observation_mask_is_rejected():
    trace = copy.deepcopy(_fixture())
    trace["events"][1]["loss_mask"][0] = 1
    trace["assembled"]["loss_mask"][2] = 1

    result = validate_trajectory(trace)

    assert result.valid is False
    assert "observation_loss_mask_value" in _codes(result)


def test_non_finite_logprob_is_rejected():
    trace = copy.deepcopy(_fixture())
    trace["events"][0]["logprobs"][0] = float("inf")
    trace["assembled"]["rollout_log_probs"][0] = float("inf")

    result = validate_trajectory(trace)

    assert result.valid is False
    assert "generation_logprob_alignment" in _codes(result)
    assert "assembled_logprob_alignment" in _codes(result)


def test_event_after_finalize_is_rejected():
    trace = copy.deepcopy(_fixture())
    trace["events"].insert(
        -1,
        {
            "kind": "observation",
            "sequence": 6,
            "timestamp": 6.5,
            "turn": 2,
            "token_ids": [999],
            "loss_mask": [0],
            "logprobs": [0.0],
        },
    )
    trace["events"][-1]["sequence"] = 7

    result = validate_trajectory(trace)

    assert result.valid is False
    assert "event_after_finalize" in _codes(result)


def test_assembled_token_mismatch_is_rejected():
    trace = copy.deepcopy(_fixture())
    trace["assembled"]["tokens"][-1] = 999

    result = validate_trajectory(trace)

    assert result.valid is False
    assert "assembled_tokens_mismatch" in _codes(result)
