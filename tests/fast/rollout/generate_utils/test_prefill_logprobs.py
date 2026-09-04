from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import prefill_logprobs
from miles.utils.types import AdapterRef, Sample


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_uses_response_tail(monkeypatch):
    sample = Sample(
        tokens=[10, 11, 12, 20, 21, 22],
        response_length=3,
        rollout_log_probs=[-9.0, -9.0, -9.0],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)
    seen = {}

    async def fake_post(url, payload, headers=None):
        seen["url"] = url
        seen["payload"] = payload
        seen["headers"] = headers
        return {
            "meta_info": {
                "input_token_logprobs": [
                    (None, 12),
                    (-0.1, 20),
                    (-0.2, 21),
                    (-0.3, 22),
                ]
            }
        }

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
        args,
        sample,
        url="http://localhost/generate",
        sampling_params={"temperature": 1, "max_new_tokens": 128},
        headers={"X-Test": "1"},
    )

    assert sample.rollout_log_probs == [-0.1, -0.2, -0.3]
    assert sample.metadata["rollout_log_probs_source"] == "sglang_prefill_recompute"
    assert seen["url"] == "http://localhost/generate"
    assert seen["headers"] == {"X-Test": "1"}
    assert seen["payload"]["input_ids"] == sample.tokens
    assert seen["payload"]["return_logprob"] is True
    assert seen["payload"]["logprob_start_len"] == 2
    assert seen["payload"]["sampling_params"]["max_new_tokens"] == 0
    assert seen["payload"]["sampling_params"]["temperature"] == 0


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_checks_token_alignment(monkeypatch):
    sample = Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED)
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)

    async def fake_post(url, payload, headers=None):
        return {"meta_info": {"input_token_logprobs": [(None, 11), (-0.1, 999)]}}

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    with pytest.raises(ValueError, match="token alignment mismatch"):
        await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
            args,
            sample,
            url="http://localhost/generate",
            sampling_params={},
        )


@pytest.mark.asyncio
async def test_recompute_samples_flushes_each_batch_and_batches_prefill_score(monkeypatch):
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 21], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {"meta_info": {"input_token_logprobs": [(None, 11), (-float(tokens[-1]), tokens[-1])]}}
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [[-20.0], [-21.0]]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert [call[2] for call in calls] == ["post", "post"]
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 21]]
    assert calls[1][1]["logprob_start_len"] == 1


@pytest.mark.asyncio
async def test_recompute_uses_per_sample_adapter_lora_path(monkeypatch):
    """Multi-LoRA: the scoring request must go to the sample's own slot adapter,
    not the single-adapter name (which is never registered on those engines)."""
    sample = Sample(
        tokens=[10, 11, 20],
        response_length=1,
        status=Sample.Status.COMPLETED,
        adapter=AdapterRef(name="run-a", slot=3),
    )
    # Multi-LoRA forces lora_rank > 0, so is_lora_enabled(args) is always true.
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, lora_rank=8)
    seen = {}

    async def fake_post(url, payload, headers=None):
        seen["payload"] = payload
        return {"meta_info": {"input_token_logprobs": [(None, 11), (-0.5, 20)]}}

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
        args,
        sample,
        url="http://localhost/generate",
        sampling_params={},
    )

    assert seen["payload"]["lora_path"] == "__miles_slot_3"
    assert sample.rollout_log_probs == [-0.5]


@pytest.mark.asyncio
async def test_recompute_samples_batches_group_by_adapter(monkeypatch):
    """Samples from different adapters must not share a batch payload: each
    batch carries a single lora_path."""
    samples = [
        Sample(
            tokens=[10, 11, 20],
            response_length=1,
            status=Sample.Status.COMPLETED,
            adapter=AdapterRef(name="run-a", slot=0),
        ),
        Sample(
            tokens=[10, 11, 21],
            response_length=1,
            status=Sample.Status.COMPLETED,
            adapter=AdapterRef(name="run-b", slot=1),
        ),
        Sample(
            tokens=[10, 11, 22],
            response_length=1,
            status=Sample.Status.COMPLETED,
            adapter=AdapterRef(name="run-a", slot=0),
        ),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        lora_rank=8,
        sglang_router_policy="round_robin",
    )
    generate_payloads = []

    async def fake_post(url, payload, action="post", headers=None):
        if url.endswith("/flush_cache"):
            return {}
        generate_payloads.append(payload)
        return [
            {"meta_info": {"input_token_logprobs": [(None, 11), (-float(tokens[-1]), tokens[-1])]}}
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [[-20.0], [-21.0], [-22.0]]
    by_lora_path = {payload["lora_path"]: payload["input_ids"] for payload in generate_payloads}
    assert by_lora_path == {
        "__miles_slot_0": [[10, 11, 20], [10, 11, 22]],
        "__miles_slot_1": [[10, 11, 21]],
    }


def test_batch_payload_rejects_mixed_lora_paths():
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, adapter=AdapterRef(name="run-a", slot=0)),
        Sample(tokens=[10, 11, 21], response_length=1, adapter=AdapterRef(name="run-b", slot=1)),
    ]
    args = SimpleNamespace(lora_rank=8)

    with pytest.raises(ValueError, match="shared lora_path"):
        prefill_logprobs._build_batch_prefill_scoring_payload(args, samples, {})


@pytest.mark.asyncio
async def test_recompute_samples_batches_by_logprob_start_len(monkeypatch):
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 12, 21], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 22], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, tokens[-2]),
                        (-float(tokens[-1]), tokens[-1]),
                    ]
                }
            }
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [
        [-20.0],
        [-21.0],
        [-22.0],
    ]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert calls[1][1]["logprob_start_len"] == 1
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 22]]
    assert calls[3][1]["logprob_start_len"] == 2
    assert calls[3][1]["input_ids"] == [[10, 11, 12, 21]]


@pytest.mark.asyncio
async def test_debug_compare_records_repeats_without_replacing_decode_logprobs(monkeypatch):
    samples = [
        Sample(
            tokens=[10, 11, 20, 21],
            response_length=2,
            rollout_log_probs=[-1.0, -2.0],
            status=Sample.Status.COMPLETED,
        )
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=False,
        debug_compare_decode_prefill_logprobs=True,
        debug_prefill_logprob_repeats=2,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []
    generate_calls = 0

    async def fake_post(url, payload, headers=None):
        nonlocal generate_calls
        calls.append(url)
        if url.endswith("/flush_cache"):
            return {}
        logprobs = [[-1.1, -2.1], [-1.2, -2.2]][generate_calls]
        generate_calls += 1
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, 11),
                        (logprobs[0], 20),
                        (logprobs[1], 21),
                    ]
                }
            }
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={},
    )

    sample = samples[0]
    assert sample.rollout_log_probs == [-1.0, -2.0]
    assert sample.metadata["logprob_debug"] == {
        "response_token_ids": [20, 21],
        "decode_logprobs": [-1.0, -2.0],
        "prefill_logprobs_repeats": [[-1.1, -2.1], [-1.2, -2.2]],
    }
    assert calls == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]


@pytest.mark.asyncio
async def test_debug_compare_checks_exact_prefill_tokens(monkeypatch):
    sample = Sample(
        tokens=[10, 11, 20],
        response_length=1,
        rollout_log_probs=[-1.0],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=False,
        debug_compare_decode_prefill_logprobs=True,
        debug_prefill_logprob_repeats=1,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )

    async def fake_post(url, payload, headers=None):
        if url.endswith("/flush_cache"):
            return {}
        return [{"meta_info": {"input_token_logprobs": [(None, 11), (-1.1, 999)]}}]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    with pytest.raises(ValueError, match="token alignment mismatch"):
        await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
            args,
            [sample],
            url="http://localhost/generate",
            sampling_params={},
        )


@pytest.mark.asyncio
async def test_debug_compare_rejects_nonfinite_decode_logprobs():
    sample = Sample(
        tokens=[10, 11, 20],
        response_length=1,
        rollout_log_probs=[float("inf")],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=False,
        debug_compare_decode_prefill_logprobs=True,
        debug_prefill_logprob_repeats=1,
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
            args,
            [sample],
            url="http://localhost/generate",
            sampling_params={},
        )


def test_debug_compare_rejects_nonfinite_prefill_logprobs():
    sample = Sample(
        tokens=[10, 11, 20],
        response_length=1,
        rollout_log_probs=[-1.0],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=False,
        debug_compare_decode_prefill_logprobs=True,
    )
    prefill_logprobs._initialize_logprob_debug(sample)

    with pytest.raises(ValueError, match="NaN or Inf"):
        prefill_logprobs._record_prefill_logprobs(args, sample, [float("nan")])
