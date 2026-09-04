import json
from copy import deepcopy
from types import SimpleNamespace

import torch
import pytest

from experiments.logprob_debug.qwen3_30b_a3b.tutorial import (
    case3_training_signal,
    decode_logprob_offset,
    stale_weight_fault,
    trainer_normalizer_offset,
)
from experiments.logprob_debug.qwen3_30b_a3b.tutorial.analyze_tutorial import (
    ArmArtifact,
    RolloutArtifact,
    _validate_signature,
    first_failed_gate,
    validate_cross_arm_tokens,
)
from experiments.logprob_debug.qwen3_30b_a3b.tutorial.fault_guard import (
    ACTIVE_FAULT_ENV,
    ENABLE_ENV,
    STRENGTH_ENV,
    build_fault_runtime_env,
    merge_runtime_env,
    require_fault,
)
from experiments.logprob_debug.qwen3_30b_a3b.tutorial.validate_run_config import validate
from miles.utils.types import Sample


def _args(**overrides):
    values = {"dump_details": "/tmp/details", "ci_test": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(index=4):
    return Sample(
        index=index,
        tokens=[10, 20, 21, 22],
        response_length=3,
        rollout_log_probs=[-1.0, -2.0, -3.0],
        status=Sample.Status.COMPLETED,
    )


def _enable_case1(monkeypatch, delta="0.10"):
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(ACTIVE_FAULT_ENV, decode_logprob_offset.FAULT_NAME)
    monkeypatch.setenv(decode_logprob_offset.DELTA_ENV, delta)
    monkeypatch.delenv("CI", raising=False)


def test_decode_offset_changes_only_stored_positions_one_plus(monkeypatch):
    _enable_case1(monkeypatch)
    first = _sample()
    second = _sample(index=5)
    original_tokens = [list(first.tokens), list(second.tokens)]

    decode_logprob_offset.process(_args(), [[first, [second]]], data_source=object())

    for sample, tokens in zip((first, second), original_tokens, strict=True):
        assert sample.tokens == tokens
        assert sample.rollout_log_probs == pytest.approx([-1.0, -2.1, -3.1])
        assert sample.metadata["fault_injection"] == {
            "name": "decode_logprob_offset",
            "delta": 0.1,
            "clean_decode_logprobs": [-1.0, -2.0, -3.0],
            "faulted_decode_logprobs": pytest.approx([-1.0, -2.1, -3.1]),
            "response_token_ids": [20, 21, 22],
        }


def test_disabled_fault_fails_before_mutation(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(ACTIVE_FAULT_ENV, raising=False)
    sample = _sample()
    before = deepcopy((sample.tokens, sample.rollout_log_probs, sample.metadata))

    with pytest.raises(RuntimeError, match="requires MILES_ENABLE"):
        decode_logprob_offset.process(_args(), [[sample]], data_source=None)

    assert (sample.tokens, sample.rollout_log_probs, sample.metadata) == before


@pytest.mark.parametrize(
    ("args", "environment", "message"),
    [
        (_args(dump_details=None), {}, "requires --dump-details"),
        (_args(ci_test=True), {}, "forbidden in CI"),
        (_args(), {"CI": "1"}, "forbidden in CI"),
    ],
)
def test_fault_guard_rejects_unsafe_activation(monkeypatch, args, environment, message):
    _enable_case1(monkeypatch)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match=message):
        require_fault(args, expected_name=decode_logprob_offset.FAULT_NAME, strength=0.1)


def test_runtime_env_merge_rejects_protected_overwrite():
    with pytest.raises(ValueError, match="protected keys"):
        merge_runtime_env('{"SGLANG_RETURN_ORIGINAL_LOGPROB":"1"}', '{"SGLANG_RETURN_ORIGINAL_LOGPROB":"0"}')


def test_fault_runtime_environment_is_explicit_and_single():
    result = json.loads(
        build_fault_runtime_env(
            "decode_logprob_offset",
            0.1,
            ["MILES_LOGPROB_DECODE_OFFSET_NAT=0.1"],
        )
    )

    assert result == {
        ENABLE_ENV: "1",
        ACTIVE_FAULT_ENV: "decode_logprob_offset",
        STRENGTH_ENV: "0.1",
        "MILES_LOGPROB_DECODE_OFFSET_NAT": "0.1",
    }


def _write_resolved(
    tmp_path,
    *,
    temperature,
    interval,
    hook=None,
    megatron_hook=None,
    runtime=None,
    exit_after=1,
    learning_rate=1e-6,
    skip_update_at=None,
):
    tokens = [
        "--rollout-batch-size", "4",
        "--n-samples-per-prompt", "2",
        "--global-batch-size", "8",
        "--rollout-max-response-len", "128",
        "--debug-prefill-logprob-repeats", "2",
        "--debug-trainer-logprob-repeats", "2",
        "--debug-exit-after-rollout", str(exit_after),
        "--debug-compare-decode-prefill-logprobs",
        "--dump-details", "/tmp/details",
        "--rollout-temperature", str(temperature),
        "--update-weights-interval", str(interval),
        "--lr", str(learning_rate),
        "--seed", "1234",
        "--rollout-seed", "1234",
    ]
    if skip_update_at is not None:
        tokens.extend(["--debug-skip-rollout-weight-update-at", str(skip_update_at)])
    if hook is not None:
        tokens.extend(["--rollout-all-samples-process-path", hook])
    if megatron_hook is not None:
        tokens.extend(["--custom-megatron-before-log-prob-hook-path", megatron_hook])
    train_args = tmp_path / "args.txt"
    train_args.write_text(" ".join(tokens))
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime or {}))
    return train_args, runtime_path


def test_resolved_case1_fault_has_exactly_one_indicator(tmp_path):
    hook = "experiments.logprob_debug.qwen3_30b_a3b.tutorial.decode_logprob_offset.process"
    runtime = json.loads(build_fault_runtime_env("decode_logprob_offset", 0.1, []))
    train_args, runtime_path = _write_resolved(
        tmp_path,
        temperature=1.0,
        interval=1,
        hook=hook,
        runtime=runtime,
    )

    result = validate(
        train_args_path=train_args,
        runtime_env_path=runtime_path,
        case="case1",
        arm="fault",
        expected_temperature=1.0,
        expected_update_interval=1,
        expected_exit_after=1,
        expected_fault="decode_logprob_offset",
        expected_strength=0.1,
    )

    assert result["fault_indicators"] == {
        "decode_logprob_offset": True,
        "temperature_definition_mismatch": False,
        "trainer_normalizer_offset": False,
        "stale_rollout_weights": False,
    }


def test_resolved_case1_rejects_second_temperature_fault(tmp_path):
    hook = "experiments.logprob_debug.qwen3_30b_a3b.tutorial.decode_logprob_offset.process"
    runtime = json.loads(build_fault_runtime_env("decode_logprob_offset", 0.1, []))
    train_args, runtime_path = _write_resolved(
        tmp_path,
        temperature=0.9,
        interval=1,
        hook=hook,
        runtime=runtime,
    )

    with pytest.raises(ValueError, match="do not match"):
        validate(
            train_args_path=train_args,
            runtime_env_path=runtime_path,
            case="case1",
            arm="fault",
            expected_temperature=1.0,
            expected_update_interval=1,
            expected_exit_after=1,
            expected_fault="decode_logprob_offset",
            expected_strength=0.1,
        )


def test_resolved_clean_arm_rejects_fault_environment(tmp_path):
    train_args, runtime_path = _write_resolved(
        tmp_path,
        temperature=1.0,
        interval=1,
        runtime={ENABLE_ENV: "1"},
    )

    with pytest.raises(ValueError, match="leaks fault runtime"):
        validate(
            train_args_path=train_args,
            runtime_env_path=runtime_path,
            case="case0",
            arm="clean",
            expected_temperature=1.0,
            expected_update_interval=1,
            expected_exit_after=1,
            expected_fault=None,
            expected_strength=None,
        )


def _rollout_with_maes(**overrides):
    maes = {
        "A_minus_B": 0.002,
        "B_minus_C": 0.003,
        "A_minus_C": 0.004,
        "B0_minus_B1": 0.005,
        "C0_minus_C1": 0.0,
    }
    maes.update(overrides)
    return RolloutArtifact(
        rollout_id=0,
        summary={
            "policy_version": "1",
            "comparisons": {
                name: {"overall": {"mean_abs": value}}
                for name, value in maes.items()
            },
        },
        tokens=[],
    )


def test_first_failed_gate_requires_repeatability_before_boundary_localization():
    assert first_failed_gate(_rollout_with_maes(B0_minus_B1=0.01, A_minus_B=0.1)) == "B_repeatability"
    assert first_failed_gate(_rollout_with_maes(C0_minus_C1=0.01, A_minus_B=0.1)) == "C_repeatability"
    assert first_failed_gate(_rollout_with_maes(A_minus_B=0.1)) == "A_vs_B"
    assert first_failed_gate(_rollout_with_maes()) == "none"

def _case2_arms(clean, fault, fixed):
    return [
        ArmArtifact("clean", None, {}, {0: clean}),
        ArmArtifact("fault", None, {}, {0: fault}),
        ArmArtifact("fixed", None, {}, {0: fixed}),
    ]


def test_signature_failure_can_be_retained_for_artifact_generation():
    clean = _rollout_with_maes(B_minus_C=0.012)
    fault = _rollout_with_maes(B_minus_C=0.1, A_minus_C=0.1)
    fixed = _rollout_with_maes(B_minus_C=0.012)
    arms = _case2_arms(clean, fault, fixed)

    with pytest.raises(ValueError, match="signature failed checks"):
        _validate_signature("case2", arms)

    result = _validate_signature("case2", arms, allow_failures=True)
    assert result["passed"] is False
    assert result["failures_allowed_for_artifact_generation"] is True
    assert "clean_rollout_0_B_minus_C_below_0.01" in result["failed_checks"]
    assert "fixed_rollout_0_B_minus_C_below_0.01" in result["failed_checks"]


def test_signature_override_never_allows_repeatability_failure():
    clean = _rollout_with_maes(B0_minus_B1=0.01)
    fault = _rollout_with_maes(B_minus_C=0.1, A_minus_C=0.1)
    fixed = _rollout_with_maes()

    with pytest.raises(ValueError, match="B repeatability is not <0.01"):
        _validate_signature(
            "case2", _case2_arms(clean, fault, fixed), allow_failures=True
        )



def test_cross_arm_token_validation_fails_on_first_changed_token(tmp_path):
    row = {
        "rollout_id": "0",
        "sample_index": "3",
        "sample_occurrence": "0",
        "token_position": "0",
        "token_id": "42",
        "active": "1",
        "response_length": "1",
        "group_index": "1",
    }
    expected = ArmArtifact(
        "clean", tmp_path, {}, {0: RolloutArtifact(0, {}, [row])}
    )
    changed = deepcopy(row)
    changed["token_id"] = "43"
    observed = ArmArtifact(
        "fault", tmp_path, {}, {0: RolloutArtifact(0, {}, [changed])}
    )

    with pytest.raises(ValueError, match="token identity changed"):
        validate_cross_arm_tokens(expected, [observed])


def test_stale_weight_hook_default_path_is_inactive(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(ACTIVE_FAULT_ENV, raising=False)
    args = _args(debug_skip_rollout_weight_update_at=None, fully_async=False)

    assert stale_weight_fault.should_skip_weight_update(args, rollout_id=0) is False


def test_stale_weight_hook_requires_fault_guard(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(ACTIVE_FAULT_ENV, raising=False)
    args = _args(debug_skip_rollout_weight_update_at=0, fully_async=False)

    with pytest.raises(RuntimeError, match="requires MILES_ENABLE"):
        stale_weight_fault.should_skip_weight_update(args, rollout_id=0)


def test_case3_training_signal_changes_only_rewards(monkeypatch):
    monkeypatch.setenv(case3_training_signal.ENABLE_ENV, "1")
    monkeypatch.delenv("CI", raising=False)
    first, second = _sample(index=0), _sample(index=1)
    before = [(list(sample.tokens), list(sample.rollout_log_probs)) for sample in (first, second)]

    case3_training_signal.process(_args(), [[first, second]], data_source=None)

    assert [first.reward, second.reward] == [-1.0, 1.0]
    assert [(sample.tokens, sample.rollout_log_probs) for sample in (first, second)] == before
    assert first.metadata["tutorial_training_signal"]["assigned_reward"] == -1.0
    assert second.metadata["tutorial_training_signal"]["assigned_reward"] == 1.0


def test_trainer_normalizer_offset_is_out_of_place():
    log_probs = torch.tensor([[-1.0], [-2.0]])
    entropy = torch.tensor([0.3, 0.4])

    shifted, observed_entropy = trainer_normalizer_offset._offset_result(
        (log_probs, entropy), 0.1
    )

    assert torch.allclose(shifted, torch.tensor([[-1.1], [-2.1]]))
    assert observed_entropy is entropy
    assert log_probs.tolist() == [[-1.0], [-2.0]]


def test_trainer_normalizer_hook_fails_before_install_without_guard(monkeypatch):
    from miles.backends.training_utils.loss_hub import logit_processors

    monkeypatch.setenv(trainer_normalizer_offset.OFFSET_ENV, "0.1")
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(ACTIVE_FAULT_ENV, raising=False)
    before = logit_processors.calculate_log_probs_and_entropy

    with pytest.raises(RuntimeError, match="requires MILES_ENABLE"):
        trainer_normalizer_offset.install(_args(), model=None, store_prefix="")

    assert logit_processors.calculate_log_probs_and_entropy is before


def test_trainer_normalizer_hook_is_idempotent(monkeypatch):
    from miles.backends.training_utils.loss_hub import logit_processors

    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv(ACTIVE_FAULT_ENV, trainer_normalizer_offset.FAULT_NAME)
    monkeypatch.setenv(STRENGTH_ENV, "0.1")
    monkeypatch.setenv(trainer_normalizer_offset.OFFSET_ENV, "0.1")
    monkeypatch.delenv("CI", raising=False)
    clean_log_probs = torch.tensor([[-1.0], [-2.0]])
    entropy = torch.tensor([0.3, 0.4])

    def fake(*args, **kwargs):
        return clean_log_probs, entropy

    monkeypatch.setattr(logit_processors, "calculate_log_probs_and_entropy", fake)
    trainer_normalizer_offset.install(_args(), model=None, store_prefix="")
    installed = logit_processors.calculate_log_probs_and_entropy
    trainer_normalizer_offset.install(_args(), model=None, store_prefix="debug_repeat_1_")

    assert logit_processors.calculate_log_probs_and_entropy is installed
    with torch.no_grad():
        shifted, observed_entropy = installed()
    assert torch.allclose(shifted, torch.tensor([[-1.1], [-2.1]]))
    assert observed_entropy is entropy
    assert clean_log_probs.tolist() == [[-1.0], [-2.0]]


def test_resolved_case2_fallback_has_exactly_one_indicator(tmp_path):
    hook = (
        "experiments.logprob_debug.qwen3_30b_a3b.tutorial."
        "trainer_normalizer_offset.install"
    )
    runtime = json.loads(
        build_fault_runtime_env(
            "trainer_normalizer_offset",
            0.1,
            ["MILES_LOGPROB_TRAINER_NORMALIZER_OFFSET_NAT=0.1"],
        )
    )
    train_args, runtime_path = _write_resolved(
        tmp_path,
        temperature=1.0,
        interval=1,
        megatron_hook=hook,
        runtime=runtime,
    )

    result = validate(
        train_args_path=train_args,
        runtime_env_path=runtime_path,
        case="case2",
        arm="fault",
        expected_temperature=1.0,
        expected_update_interval=1,
        expected_exit_after=1,
        expected_fault="trainer_normalizer_offset",
        expected_strength=0.1,
    )

    assert result["fault_indicators"] == {
        "decode_logprob_offset": False,
        "temperature_definition_mismatch": False,
        "trainer_normalizer_offset": True,
        "stale_rollout_weights": False,
    }


def test_resolved_case3_fault_has_exactly_one_indicator(tmp_path):
    hook = (
        "experiments.logprob_debug.qwen3_30b_a3b.tutorial."
        "case3_training_signal.process"
    )
    runtime = json.loads(build_fault_runtime_env("stale_rollout_weights", 1.0, []))
    runtime[case3_training_signal.ENABLE_ENV] = "1"
    train_args, runtime_path = _write_resolved(
        tmp_path,
        temperature=1.0,
        interval=1,
        hook=hook,
        runtime=runtime,
        exit_after=3,
        learning_rate=1e-4,
        skip_update_at=0,
    )

    result = validate(
        train_args_path=train_args,
        runtime_env_path=runtime_path,
        case="case3",
        arm="fault",
        expected_temperature=1.0,
        expected_update_interval=1,
        expected_exit_after=3,
        expected_fault="stale_rollout_weights",
        expected_strength=1.0,
        expected_learning_rate=1e-4,
        expected_skip_update_at=0,
    )

    assert result["fault_indicators"] == {
        "decode_logprob_offset": False,
        "temperature_definition_mismatch": False,
        "trainer_normalizer_offset": False,
        "stale_rollout_weights": True,
    }
