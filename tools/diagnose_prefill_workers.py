#!/usr/bin/env python3
"""Controlled SGLang exact-token prefill repeatability diagnostic.

This module is used as a Miles custom generate function in
``--debug-rollout-only`` mode.  It loads one previously dumped rollout sample,
scores its exact token IDs directly and through the SGLang router, and writes a
strict JSON report.  No trainer is created and no response is resampled.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.prefill_logprobs import (
    _build_prefill_scoring_payload,
    _extract_response_logprobs,
)
from miles.utils.types import Sample


_SAMPLE_PATH_ENV = "MILES_LOGPROB_WORKER_DIAGNOSTIC_SAMPLE_PATH"
_OUTPUT_PATH_ENV = "MILES_LOGPROB_WORKER_DIAGNOSTIC_OUTPUT_PATH"
_HTTP_REQUEST_RE = re.compile(r"^sglang:http_requests_total\{([^}]*)\}\s+([0-9.eE+-]+)\s*$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _load_source_sample(path: Path) -> Sample:
    dumped = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dumped, dict) or not isinstance(dumped.get("samples"), list):
        raise ValueError(f"{path} is not a Miles rollout dump")
    if not dumped["samples"]:
        raise ValueError(f"{path} contains no samples")
    raw = dumped["samples"][0]
    sample = raw if isinstance(raw, Sample) else Sample.from_dict(raw)
    sample.validate()
    if sample.response_length <= 0:
        raise ValueError("diagnostic source sample has no response tokens")
    return sample


def _validate_scores(sample: Sample, scores: list[float], label: str) -> None:
    if len(scores) != sample.response_length:
        raise ValueError(f"{label} length mismatch: expected {sample.response_length}, got {len(scores)}")
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError(f"{label} contains NaN or Inf")


async def _json_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    response = await client.request(method, url, json=payload)
    response.raise_for_status()
    if not response.content:
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text


async def _score(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    sample: Sample,
    label: str,
) -> list[float]:
    output = await _json_request(client, "POST", f"{url}/generate", payload=payload)
    if not isinstance(output, dict) or not isinstance(output.get("meta_info"), dict):
        raise ValueError(f"{label} returned malformed output: {type(output).__name__}")
    scores = [float(value) for value in _extract_response_logprobs(sample, output["meta_info"])]
    _validate_scores(sample, scores, label)
    return scores


async def _flush(client: httpx.AsyncClient, url: str) -> Any:
    return await _json_request(client, "POST", f"{url}/flush_cache", payload={})


async def _worker_urls(client: httpx.AsyncClient, router_url: str) -> list[str]:
    data = await _json_request(client, "GET", f"{router_url}/workers")
    if not isinstance(data, dict) or not isinstance(data.get("workers"), list):
        raise ValueError(f"router /workers returned malformed output: {data!r}")
    urls = sorted({worker["url"].rstrip("/") for worker in data["workers"]})
    if len(urls) != 4:
        raise ValueError(f"expected four SGLang workers, found {len(urls)}: {urls}")
    return urls


def _generate_count(metrics_text: str) -> int:
    count = 0
    family_present = "sglang:http_requests_total" in metrics_text
    for line in metrics_text.splitlines():
        match = _HTTP_REQUEST_RE.match(line)
        if not match:
            continue
        labels = dict(_LABEL_RE.findall(match.group(1)))
        if labels.get("endpoint") != "/generate" or labels.get("method") != "POST":
            continue
        count += int(float(match.group(2)))
    if not family_present:
        raise ValueError("worker metrics contain no HTTP request counter family")
    return count


async def _worker_generate_counts(
    client: httpx.AsyncClient, worker_urls: list[str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for worker_url in worker_urls:
        response = await client.get(f"{worker_url}/metrics")
        response.raise_for_status()
        counts[worker_url] = _generate_count(response.text)
    return counts


def _request_worker(before: dict[str, int], after: dict[str, int]) -> tuple[str, dict[str, int]]:
    deltas = {url: after.get(url, 0) - before.get(url, 0) for url in set(before) | set(after)}
    winners = [url for url, delta in deltas.items() if delta == 1]
    unexpected = {url: delta for url, delta in deltas.items() if delta != 0}
    if len(winners) != 1 or any(delta != 1 for delta in unexpected.values()):
        raise ValueError(f"could not identify exactly one router worker from metrics: {unexpected}")
    return winners[0], dict(sorted(deltas.items()))


def _diff_stats(differences: list[float]) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("difference vector is empty or non-finite")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "signed_mean": float(values.mean()),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "p50_abs": float(np.quantile(absolute, 0.50)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
    }


def _pairwise_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    combined: list[float] = []
    for left, right in itertools.combinations(records, 2):
        differences = [a - b for a, b in zip(left["logprobs"], right["logprobs"], strict=True)]
        stats = _diff_stats(differences)
        combined.extend(differences)
        pairs.append({"left": left["label"], "right": right["label"], "stats": stats})
    return {"overall": _diff_stats(combined), "pairs": pairs}


async def _collect(input: GenerateFnInput, source: Sample, source_path: Path) -> dict[str, Any]:
    args = input.args
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}".rstrip("/")
    payload = _build_prefill_scoring_payload(args, source, input.sampling_params)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        workers = await _worker_urls(client, router_url)
        server_info = {
            url: await _json_request(client, "GET", f"{url}/get_server_info") for url in workers
        }

        # Validate worker-level attribution before issuing any diagnostic score.
        # Router Prometheus metrics are served on a separate, dynamically chosen
        # port, while each engine exposes this counter on its traffic URL.
        await _worker_generate_counts(client, workers)

        fixed_worker = workers[0]
        within_records = []
        for repeat in range(5):
            await _flush(client, fixed_worker)
            scores = await _score(client, fixed_worker, payload, source, f"fixed-worker repeat {repeat}")
            within_records.append(
                {"label": f"fixed-{repeat}", "worker_url": fixed_worker, "logprobs": scores}
            )

        cross_records = []
        for worker_index, worker_url in enumerate(workers):
            await _flush(client, worker_url)
            scores = await _score(client, worker_url, payload, source, f"worker {worker_index}")
            cross_records.append(
                {"label": f"worker-{worker_index}", "worker_url": worker_url, "logprobs": scores}
            )

        router_records = []
        for repeat in range(5):
            flush_result = await _flush(client, router_url)
            before = await _worker_generate_counts(client, workers)
            scores = await _score(client, router_url, payload, source, f"router repeat {repeat}")
            after = await _worker_generate_counts(client, workers)
            worker_url, deltas = _request_worker(before, after)
            router_records.append(
                {
                    "label": f"router-{repeat}",
                    "worker_url": worker_url,
                    "worker_request_deltas": deltas,
                    "router_flush_result": flush_result,
                    "logprobs": scores,
                }
            )

    token_bytes = json.dumps(source.tokens, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "source": {
            "path": str(source_path),
            "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "sample_index": source.index,
            "prompt_length": len(source.tokens) - source.response_length,
            "response_length": source.response_length,
            "token_count": len(source.tokens),
            "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "response_token_ids": list(source.tokens[-source.response_length :]),
        },
        "router_url": router_url,
        "workers": workers,
        "fixed_worker": fixed_worker,
        "resolved_sglang_server_info": server_info,
        "within_worker": {"records": within_records, "pairwise": _pairwise_stats(within_records)},
        "cross_worker": {"records": cross_records, "pairwise": _pairwise_stats(cross_records)},
        "router_mediated": {"records": router_records, "pairwise": _pairwise_stats(router_records)},
    }


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Miles custom generate hook that runs the B-only diagnostic once."""

    source_path = Path(_require_env(_SAMPLE_PATH_ENV)).resolve()
    output_path = Path(_require_env(_OUTPUT_PATH_ENV)).resolve()
    source = _load_source_sample(source_path)
    report = await _collect(input, source, source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    # Return the preserved sample itself so rollout-only mode exits without
    # generating new response tokens or invoking a reward model.
    source.reward = 0.0
    source.loss_mask = [1] * source.response_length
    source.status = Sample.Status.TRUNCATED
    source.generate_function_path = None
    source.validate()
    return GenerateFnOutput(samples=[source, deepcopy(source)])
