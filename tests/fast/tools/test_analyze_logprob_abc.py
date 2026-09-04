import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.dashboard.dump_reader import TrainRow
from miles.utils.types import Sample

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "analyze_logprob_abc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_logprob_abc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_and_train():
    sample = Sample(
        group_index=3,
        index=9,
        tokens=[10, 20, 21],
        response_length=2,
        loss_mask=[1, 1],
        weight_versions=["7"],
        rollout_log_probs=[-1.0, -2.0],
        status=Sample.Status.COMPLETED,
        metadata={
            "logprob_debug": {
                "response_token_ids": [20, 21],
                "decode_logprobs": [-1.0, -2.0],
                "prefill_logprobs_repeats": [
                    [-1.1, -2.2],
                    [-1.1, -2.1],
                ],
            }
        },
    )
    train = TrainRow(
        sample_index=9,
        rank=0,
        tokens=torch.tensor([10, 20, 21]),
        response_length=2,
        total_length=3,
        reward=1.0,
        loss_mask=torch.tensor([1, 1]),
        log_probs=torch.tensor([-0.9, -2.3]),
        debug_repeat_1_log_probs=torch.tensor([-1.0, -2.2]),
        rollout_log_probs=torch.tensor([-1.0, -2.0]),
        ref_log_probs=None,
        entropy=None,
        ref_entropy=None,
        advantages=None,
        returns=None,
        raw_reward=1.0,
        truncated=0,
        weight_versions=["7"],
    )
    return sample, train


def test_analyze_emits_all_breakdowns(monkeypatch, tmp_path):
    module = _load_module()
    sample, train = _sample_and_train()
    joined = SimpleNamespace(samples=[sample], train_rows={(9, 0): train})

    class FakeReader:
        def __init__(self, dump_details):
            assert dump_details == tmp_path / "dump"

        def load_joined(self, rollout_id):
            assert rollout_id == 0
            return joined

    monkeypatch.setattr(module, "DumpReader", FakeReader)

    token_rows, summary = module.analyze(tmp_path / "dump", 0)
    module.write_outputs(tmp_path / "analysis", token_rows, summary)

    assert summary["policy_version"] == "7"
    assert summary["sample_count"] == 1
    assert summary["active_token_count"] == 2
    assert summary["prefill_repeat_count"] == 2
    assert set(summary["comparisons"]) == {
        "A_minus_B",
        "B_minus_C",
        "A_minus_C",
        "B0_minus_B1",
        "C0_minus_C1",
    }
    for comparison in summary["comparisons"].values():
        assert comparison["overall"]["count"] == 2
        assert comparison["position_0"]["count"] == 1
        assert comparison["position_1_plus"]["count"] == 1
        assert comparison["per_sample"][0]["stats"]["count"] == 2
        assert [row["token_position"] for row in comparison["per_position"]] == [0, 1]

    assert token_rows[0]["token_id"] == 20
    assert token_rows[0]["A_minus_B"] == pytest.approx(0.1)
    assert token_rows[1]["B0_minus_B1"] == pytest.approx(-0.1)
    assert token_rows[0]["C0_minus_C1"] == pytest.approx(0.1)
    assert (tmp_path / "analysis" / "summary.md").is_file()
    with (tmp_path / "analysis" / "summary.json").open() as stream:
        assert json.load(stream)["comparisons"]["A_minus_C"]["overall"]["count"] == 2
    with (tmp_path / "analysis" / "tokens.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [int(row["token_id"]) for row in csv_rows] == [20, 21]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda sample, train: setattr(train, "tokens", torch.tensor([10, 20, 999])),
            "Rollout/trainer token mismatch",
        ),
        (
            lambda sample, train: setattr(train, "log_probs", torch.tensor([float("nan"), -2.3])),
            "NaN or Inf",
        ),
        (
            lambda sample, train: sample.weight_versions.append("8"),
            "Mixed rollout policy versions",
        ),
    ],
)
def test_analyze_fails_closed(monkeypatch, tmp_path, mutate, message):
    module = _load_module()
    sample, train = _sample_and_train()
    mutate(sample, train)
    joined = SimpleNamespace(samples=[sample], train_rows={(9, 0): train})

    class FakeReader:
        def __init__(self, dump_details):
            pass

        def load_joined(self, rollout_id):
            return joined

    monkeypatch.setattr(module, "DumpReader", FakeReader)

    with pytest.raises(ValueError, match=message):
        module.analyze(tmp_path / "dump", 0)
