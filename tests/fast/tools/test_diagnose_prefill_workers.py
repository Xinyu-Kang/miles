import pytest

from tools.diagnose_prefill_workers import _generate_count, _pairwise_stats, _request_worker


def test_generate_count_sums_matching_label_sets():
    metrics = """
# HELP sglang:http_requests_total Total requests
sglang:http_requests_total{endpoint="/generate",method="POST"} 3.0
sglang:http_requests_total{endpoint="/health",method="GET"} 11.0
sglang:http_requests_total{model_name="qwen",endpoint="/generate",method="POST"} 2.0
"""

    assert _generate_count(metrics) == 5


def test_generate_count_allows_zero_generate_requests():
    assert _generate_count(
        'sglang:http_requests_total{endpoint="/health",method="GET"} 1.0'
    ) == 0


def test_generate_count_requires_request_counter_family():
    with pytest.raises(ValueError, match="no HTTP request counter family"):
        _generate_count('sglang:num_requests_total 1.0')


def test_pairwise_stats_aggregates_every_token_and_pair():
    records = [
        {"label": "a", "logprobs": [0.0, 1.0]},
        {"label": "b", "logprobs": [1.0, 1.0]},
        {"label": "c", "logprobs": [2.0, 3.0]},
    ]

    result = _pairwise_stats(records)

    assert result["overall"]["count"] == 6
    assert result["overall"]["mean_abs"] == pytest.approx(8 / 6)
    assert len(result["pairs"]) == 3


def test_request_worker_requires_one_exact_success_increment():
    worker, deltas = _request_worker({"w0": 2, "w1": 4}, {"w0": 2, "w1": 5})

    assert worker == "w1"
    assert deltas == {"w0": 0, "w1": 1}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"w0": 1}, {"w0": 1}),
        ({"w0": 1, "w1": 1}, {"w0": 2, "w1": 2}),
        ({"w0": 1}, {"w0": 3}),
    ],
)
def test_request_worker_rejects_ambiguous_deltas(before, after):
    with pytest.raises(ValueError, match="exactly one router worker"):
        _request_worker(before, after)
