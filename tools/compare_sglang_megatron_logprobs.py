#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _load_pt(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _to_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _float_list(value: Any, *, name: str) -> list[float]:
    values = _to_list(value)
    if values is None:
        raise ValueError(f"{name} is missing")
    return [float(x) for x in values]


def _int_list(value: Any, *, name: str) -> list[int]:
    values = _to_list(value)
    if values is None:
        raise ValueError(f"{name} is missing")
    return [int(x) for x in values]


def _scalar(value: Any) -> Any:
    values = _to_list(value)
    if values is None:
        return None
    if len(values) != 1:
        return values
    return values[0]


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _abs_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return _finite(abs(a - b))


def _signed_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return _finite(a - b)


def _stats(values: list[float | None]) -> dict[str, float | int | None]:
    finite = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}

    def percentile(q: float) -> float:
        if len(finite) == 1:
            return finite[0]
        pos = q * (len(finite) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return finite[lo]
        return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)

    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": finite[-1],
    }


def _parse_rollout_ids(spec: str | None, dump_details: Path) -> list[int]:
    if not spec or spec == "all":
        ids = []
        for path in sorted((dump_details / "rollout_data").glob("*.pt")):
            if path.stem.isdigit():
                ids.append(int(path.stem))
        return ids

    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid rollout id range: {part}")
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def _normalize_partition(value: Any, size: int) -> list[int]:
    if value is None:
        return list(range(size))
    if isinstance(value, range):
        return list(value)
    values = _to_list(value)
    if values is None:
        return list(range(size))
    return [int(x) for x in values]


def _load_tokenizer(path: str | None) -> Any | None:
    if not path:
        return None
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        tokenizer_file = Path(path) / "tokenizer.json"
        if tokenizer_file.exists():
            return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        raise


def _decode_token(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception as exc:
        return f"<decode-error:{type(exc).__name__}>"


def _route_summary(routed_experts: Any, route_index: int) -> tuple[str | None, Any | None]:
    routes = _to_list(routed_experts)
    if routes is None or route_index < 0 or route_index >= len(routes):
        return None, None
    route = routes[route_index]
    route_list = _to_list(route)
    if route_list is None:
        return None, None
    route_json = json.dumps(route_list, sort_keys=True, separators=(",", ":"))
    route_hash = hashlib.sha1(route_json.encode()).hexdigest()[:12]
    return route_hash, route_list[:2]


def _normalize_sglang_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.path or parsed.path == "/":
        return urllib.parse.urlunparse(parsed._replace(path="/generate"))
    return url


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _flush_sglang_cache(generate_url: str, timeout: float) -> None:
    base = generate_url.rsplit("/", 1)[0]
    _post_json(f"{base}/flush_cache", {}, timeout)


def _score_sglang_prefill(
    *,
    generate_url: str,
    tokens: list[int],
    response_length: int,
    timeout: float,
    flush_cache: bool,
) -> list[float]:
    if response_length == 0:
        return []
    prompt_len = len(tokens) - response_length
    if prompt_len <= 0:
        raise ValueError(f"Cannot prefill-score sample with prompt_len={prompt_len}")
    if flush_cache:
        _flush_sglang_cache(generate_url, timeout)
    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_len - 1,
    }
    output = _post_json(generate_url, payload, timeout)
    meta_info = output.get("meta_info", {})
    input_token_logprobs = meta_info.get("input_token_logprobs")
    if not input_token_logprobs:
        raise ValueError("SGLang prefill response did not include input_token_logprobs")

    response_items = input_token_logprobs[-response_length:]
    response_tokens = tokens[-response_length:]
    scored_tokens = [int(item[1]) for item in response_items]
    if scored_tokens != response_tokens:
        raise ValueError(
            "SGLang prefill token alignment mismatch: "
            f"expected tail={response_tokens[:8]} len={len(response_tokens)}, "
            f"got={scored_tokens[:8]} len={len(scored_tokens)}"
        )
    logprobs = [item[0] for item in response_items]
    if any(value is None for value in logprobs):
        raise ValueError("SGLang prefill returned None for a response-token logprob")
    return [float(value) for value in logprobs]


def _load_rollout_samples(dump_details: Path, rollout_id: int) -> list[Any]:
    path = dump_details / "rollout_data" / f"{rollout_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = _load_pt(path)
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples list")
    return samples


def _load_train_rollout_data(dump_details: Path, rollout_id: int, rank: int) -> dict[str, Any]:
    path = dump_details / "train_data" / f"{rollout_id}_{rank}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = _load_pt(path)
    rollout_data = data.get("rollout_data")
    if not isinstance(rollout_data, dict):
        raise ValueError(f"{path} does not contain rollout_data")
    return rollout_data


def _validate_response_length(
    *,
    rollout_id: int,
    sample_pos: int,
    name: str,
    values: list[Any],
    response_length: int,
    allow_local_slices: bool,
) -> None:
    if len(values) == response_length:
        return
    if allow_local_slices:
        return
    raise ValueError(
        f"rollout={rollout_id} sample={sample_pos} {name} length={len(values)} "
        f"does not match response_length={response_length}. This looks like a CP-local slice; "
        "rerun with CP=1 or pass --allow-local-slices for aggregate-only local comparison."
    )


def _compare_rollout(
    *,
    dump_details: Path,
    rollout_id: int,
    rank: int,
    tokenizer: Any | None,
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict[str, Any]]:
    samples = _load_rollout_samples(dump_details, rollout_id)
    train_data = _load_train_rollout_data(dump_details, rollout_id, rank)
    if "log_probs" not in train_data:
        raise ValueError(
            f"train_data/{rollout_id}_{rank}.pt does not contain log_probs. "
            "Run with mismatch metrics enabled so Megatron scoring is dumped."
        )
    if "rollout_log_probs" not in train_data:
        raise ValueError(f"train_data/{rollout_id}_{rank}.pt does not contain rollout_log_probs")

    train_tokens = train_data["tokens"]
    partition = _normalize_partition(train_data.get("partition"), len(train_tokens))
    sample_indices = _to_list(train_data.get("sample_indices")) or [None] * len(train_tokens)
    total_lengths = _to_list(train_data.get("total_lengths")) or [None] * len(train_tokens)
    response_lengths = _to_list(train_data.get("response_lengths")) or [None] * len(train_tokens)
    loss_masks = _to_list(train_data.get("loss_masks")) or [None] * len(train_tokens)

    if len(partition) != len(train_tokens):
        raise ValueError(f"partition length {len(partition)} does not match train sample count {len(train_tokens)}")

    rows: list[dict[str, Any]] = []
    generate_url = _normalize_sglang_url(args.sglang_url) if args.sglang_url else None

    for train_pos, raw_pos in enumerate(partition):
        if raw_pos >= len(samples):
            raise ValueError(f"partition index {raw_pos} out of range for rollout {rollout_id}")
        sample = samples[raw_pos]
        tokens = _int_list(train_tokens[train_pos], name="train tokens")
        raw_tokens = _int_list(_sample_get(sample, "tokens"), name="raw sample tokens")
        if tokens != raw_tokens:
            raise ValueError(
                f"rollout={rollout_id} train_pos={train_pos} raw_pos={raw_pos} token mismatch: "
                f"train first={tokens[:8]} len={len(tokens)}, raw first={raw_tokens[:8]} len={len(raw_tokens)}"
            )

        response_length = int(_scalar(response_lengths[train_pos]))
        total_length = int(_scalar(total_lengths[train_pos]) or len(tokens))
        prompt_length = total_length - response_length
        if response_length < 0 or prompt_length < 0:
            raise ValueError(
                f"rollout={rollout_id} train_pos={train_pos} invalid total/response lengths: "
                f"{total_length}/{response_length}"
            )

        raw_decode = _float_list(_sample_get(sample, "rollout_log_probs"), name="raw rollout_log_probs")
        train_rollout = _float_list(train_data["rollout_log_probs"][train_pos], name="train rollout_log_probs")
        megatron_train = _float_list(train_data["log_probs"][train_pos], name="train log_probs")
        mask = _to_list(loss_masks[train_pos]) or _sample_get(sample, "loss_mask") or [1] * response_length
        mask = [int(x) for x in mask]

        _validate_response_length(
            rollout_id=rollout_id,
            sample_pos=train_pos,
            name="raw rollout_log_probs",
            values=raw_decode,
            response_length=response_length,
            allow_local_slices=args.allow_local_slices,
        )
        _validate_response_length(
            rollout_id=rollout_id,
            sample_pos=train_pos,
            name="train rollout_log_probs",
            values=train_rollout,
            response_length=response_length,
            allow_local_slices=args.allow_local_slices,
        )
        _validate_response_length(
            rollout_id=rollout_id,
            sample_pos=train_pos,
            name="megatron log_probs",
            values=megatron_train,
            response_length=response_length,
            allow_local_slices=args.allow_local_slices,
        )

        prefill = None
        if args.compare_sglang_prefill:
            if generate_url is None:
                raise ValueError("--compare-sglang-prefill requires --sglang-url")
            prefill = _score_sglang_prefill(
                generate_url=generate_url,
                tokens=tokens,
                response_length=response_length,
                timeout=args.timeout,
                flush_cache=not args.no_prefill_flush_cache,
            )

        compare_len = min(len(raw_decode), len(train_rollout), len(megatron_train), response_length)
        if prefill is not None:
            compare_len = min(compare_len, len(prefill))
        if compare_len != response_length and not args.allow_local_slices:
            raise ValueError(
                f"rollout={rollout_id} train_pos={train_pos} compare_len={compare_len} "
                f"response_length={response_length}"
            )
        if compare_len != response_length:
            warnings.append(
                f"rollout={rollout_id} train_pos={train_pos} using local compare_len={compare_len} "
                f"for response_length={response_length}"
            )

        routed_experts = _sample_get(sample, "rollout_routed_experts")
        status = _sample_get(sample, "status")
        sample_index = _scalar(sample_indices[train_pos])

        for response_pos in range(compare_len):
            token_id = tokens[prompt_length + response_pos] if prompt_length + response_pos < len(tokens) else None
            route_hash, route_preview = _route_summary(routed_experts, prompt_length + response_pos - 1)
            raw_lp = _finite(raw_decode[response_pos])
            train_rollout_lp = _finite(train_rollout[response_pos])
            megatron_lp = _finite(megatron_train[response_pos])
            prefill_lp = _finite(prefill[response_pos]) if prefill is not None else None
            row = {
                "rollout_id": rollout_id,
                "rank": rank,
                "train_sample_pos": train_pos,
                "raw_sample_pos": raw_pos,
                "sample_index": sample_index,
                "status": status.value if hasattr(status, "value") else status,
                "response_pos": response_pos,
                "absolute_token_pos": prompt_length + response_pos,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "total_length": total_length,
                "token_id": token_id,
                "token_text": _decode_token(tokenizer, int(token_id)) if token_id is not None else "",
                "loss_mask": mask[response_pos] if response_pos < len(mask) else None,
                "sglang_decode_logprob": raw_lp,
                "train_dump_rollout_logprob": train_rollout_lp,
                "sglang_prefill_logprob": prefill_lp,
                "megatron_train_logprob": megatron_lp,
                "abs_diff_decode_megatron": _abs_diff(raw_lp, megatron_lp),
                "signed_diff_decode_megatron": _signed_diff(raw_lp, megatron_lp),
                "abs_diff_train_dump_rollout_megatron": _abs_diff(train_rollout_lp, megatron_lp),
                "abs_diff_raw_vs_train_dump_rollout": _abs_diff(raw_lp, train_rollout_lp),
                "abs_diff_decode_prefill": _abs_diff(raw_lp, prefill_lp),
                "abs_diff_prefill_megatron": _abs_diff(prefill_lp, megatron_lp),
                "routed_experts_hash": route_hash,
                "routed_experts_preview": route_preview,
            }
            rows.append(row)

    return rows


def _diff_values(rows: list[dict[str, Any]], key: str, *, active_only: bool) -> list[float | None]:
    values = []
    for row in rows:
        if active_only and not row.get("loss_mask", 1):
            continue
        values.append(row.get(key))
    return values


def _sample_mean_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if not row.get("loss_mask", 1):
            continue
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        group_key = (row["rollout_id"], row["rank"], row["train_sample_pos"])
        groups.setdefault(group_key, []).append(float(value))
    return _stats([sum(values) / len(values) for values in groups.values() if values])


def _pair_stats(rows: list[dict[str, Any]], *, active_only: bool) -> dict[str, dict[str, float | int | None]]:
    return {
        "decode_vs_megatron": _stats(_diff_values(rows, "abs_diff_decode_megatron", active_only=active_only)),
        "train_dump_rollout_vs_megatron": _stats(
            _diff_values(rows, "abs_diff_train_dump_rollout_megatron", active_only=active_only)
        ),
        "raw_vs_train_dump_rollout": _stats(
            _diff_values(rows, "abs_diff_raw_vs_train_dump_rollout", active_only=active_only)
        ),
        "decode_vs_prefill": _stats(_diff_values(rows, "abs_diff_decode_prefill", active_only=active_only)),
        "prefill_vs_megatron": _stats(_diff_values(rows, "abs_diff_prefill_megatron", active_only=active_only)),
    }


def _sample_mean_pair_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "decode_vs_megatron": _sample_mean_stats(rows, "abs_diff_decode_megatron"),
        "train_dump_rollout_vs_megatron": _sample_mean_stats(rows, "abs_diff_train_dump_rollout_megatron"),
        "raw_vs_train_dump_rollout": _sample_mean_stats(rows, "abs_diff_raw_vs_train_dump_rollout"),
        "decode_vs_prefill": _sample_mean_stats(rows, "abs_diff_decode_prefill"),
        "prefill_vs_megatron": _sample_mean_stats(rows, "abs_diff_prefill_megatron"),
    }


def _classify(summary: dict[str, Any], threshold: float) -> str:
    pairs = summary["sample_mean_pairs"]
    raw_train = pairs["raw_vs_train_dump_rollout"]["mean"]
    decode_megatron = pairs["decode_vs_megatron"]["mean"]
    decode_prefill = pairs["decode_vs_prefill"]["mean"]
    prefill_megatron = pairs["prefill_vs_megatron"]["mean"]

    if raw_train is not None and raw_train > threshold:
        return "raw SGLang rollout logprobs differ from train-dump rollout logprobs; inspect Miles slicing/partitioning first"
    if decode_prefill is None or prefill_megatron is None:
        return "offline decode-vs-Megatron comparison only; add --compare-sglang-prefill to separate SGLang decode from prefill scoring"

    close_decode_prefill = decode_prefill <= threshold
    close_prefill_megatron = prefill_megatron <= threshold
    close_decode_megatron = decode_megatron is not None and decode_megatron <= threshold

    if close_decode_prefill and not close_prefill_megatron:
        return "SGLang decode and prefill agree, but Megatron diverges; suspect Megatron scoring, weight mapping/update, or replay parity"
    if close_prefill_megatron and not close_decode_prefill:
        return "SGLang prefill and Megatron agree, but decode diverges; suspect SGLang decode/logprob runtime"
    if close_decode_megatron and not close_prefill_megatron:
        return "SGLang decode and Megatron agree, but prefill diverges; suspect SGLang prefill/scoring path"
    if not close_decode_prefill and not close_prefill_megatron and not close_decode_megatron:
        return "all compared paths diverge; suspect alignment, stale weights, routing/indexer replay, or mixed runtime state"
    return "all compared paths are within threshold"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_top_csv(path: Path, rows: list[dict[str, Any]], top_k: int) -> None:
    fields = [
        "rollout_id",
        "rank",
        "train_sample_pos",
        "raw_sample_pos",
        "sample_index",
        "response_pos",
        "absolute_token_pos",
        "token_id",
        "token_text",
        "loss_mask",
        "sglang_decode_logprob",
        "sglang_prefill_logprob",
        "megatron_train_logprob",
        "abs_diff_decode_megatron",
        "signed_diff_decode_megatron",
        "abs_diff_decode_prefill",
        "abs_diff_prefill_megatron",
        "abs_diff_raw_vs_train_dump_rollout",
        "routed_experts_hash",
        "routed_experts_preview",
    ]
    sorted_rows = sorted(rows, key=lambda r: r.get("abs_diff_decode_megatron") or -1, reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows[:top_k]:
            writer.writerow(row)


def _logprob_pairs(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    logprob_min: float,
    logprob_max: float,
) -> tuple[list[float], list[float], int]:
    x_values = []
    y_values = []
    skipped_out_of_range = 0
    for row in rows:
        if not row.get("loss_mask", 1):
            continue
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if not (logprob_min <= x <= logprob_max and logprob_min <= y <= logprob_max):
            skipped_out_of_range += 1
            continue
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values, skipped_out_of_range


def _write_logprob_pair_chart(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    x_legend: str,
    y_legend: str,
    filename: str,
    plt: Any,
    np: Any,
    args: argparse.Namespace,
    warnings: list[str],
) -> None:
    x_values, y_values, skipped = _logprob_pairs(
        rows,
        x_key,
        y_key,
        logprob_min=args.plot_logprob_min,
        logprob_max=args.plot_logprob_max,
    )
    if skipped:
        warnings.append(f"{filename}: skipped {skipped} active tokens outside plot range")
    if not x_values:
        return

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    eps = 1e-10
    hist2d, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=args.plot_bins,
        range=[[args.plot_logprob_min, args.plot_logprob_max], [args.plot_logprob_min, args.plot_logprob_max]],
        density=True,
    )
    hist2d = np.nan_to_num(hist2d, nan=0.0, posinf=0.0, neginf=0.0)

    fig = plt.figure(figsize=(9.5, 9.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.35], hspace=0.16)
    heat_ax = fig.add_subplot(grid[0])
    density_ax = fig.add_subplot(grid[1], sharex=heat_ax)

    heatmap = heat_ax.pcolormesh(
        x_edges,
        y_edges,
        np.log(hist2d.T + eps),
        shading="auto",
        cmap="viridis",
    )
    heat_ax.plot(
        [args.plot_logprob_min, args.plot_logprob_max],
        [args.plot_logprob_min, args.plot_logprob_max],
        color="white",
        linewidth=0.8,
        alpha=0.85,
    )
    heat_ax.set_xlim(args.plot_logprob_min, args.plot_logprob_max)
    heat_ax.set_ylim(args.plot_logprob_min, args.plot_logprob_max)
    heat_ax.set_aspect("equal", adjustable="box")
    heat_ax.set_xlabel(x_label)
    heat_ax.set_ylabel(y_label)
    colorbar = fig.colorbar(heatmap, ax=heat_ax)
    colorbar.set_label("Log Frequency")

    x_density, edges = np.histogram(
        x,
        bins=args.plot_bins,
        range=(args.plot_logprob_min, args.plot_logprob_max),
        density=True,
    )
    y_density, _ = np.histogram(
        y,
        bins=args.plot_bins,
        range=(args.plot_logprob_min, args.plot_logprob_max),
        density=True,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    x_density = np.nan_to_num(x_density, nan=0.0, posinf=0.0, neginf=0.0)
    y_density = np.nan_to_num(y_density, nan=0.0, posinf=0.0, neginf=0.0)
    density_ax.plot(centers, np.log(x_density + eps), label=x_legend)
    density_ax.plot(centers, np.log(y_density + eps), label=y_legend)
    density_ax.set_xlim(args.plot_logprob_min, args.plot_logprob_max)
    density_ax.set_xlabel("log-prob")
    density_ax.set_ylabel("Log Density")
    density_ax.legend()

    fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _maybe_write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    warnings: list[str],
    args: argparse.Namespace,
) -> None:
    try:
        import numpy as np
        from matplotlib import pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; skipped plots ({type(exc).__name__})")
        return

    diffs = [row["abs_diff_decode_megatron"] for row in rows if row["abs_diff_decode_megatron"] is not None]
    if not diffs:
        return

    plt.figure(figsize=(10, 4))
    plt.hist(diffs, bins=80)
    plt.xlabel("|SGLang decode - Megatron train|")
    plt.ylabel("token count")
    plt.tight_layout()
    plt.savefig(output_dir / "diff_hist.png")
    plt.close()

    by_pos: dict[int, list[float]] = {}
    for row in rows:
        diff = row["abs_diff_decode_megatron"]
        if diff is None:
            continue
        by_pos.setdefault(int(row["response_pos"]), []).append(float(diff))
    positions = sorted(by_pos)
    means = [sum(by_pos[pos]) / len(by_pos[pos]) for pos in positions]
    plt.figure(figsize=(10, 4))
    plt.plot(positions, means)
    plt.xlabel("response token position")
    plt.ylabel("mean abs diff")
    plt.tight_layout()
    plt.savefig(output_dir / "diff_by_position.png")
    plt.close()

    _write_logprob_pair_chart(
        output_dir=output_dir,
        rows=rows,
        x_key="sglang_decode_logprob",
        y_key="megatron_train_logprob",
        x_label="SGLang decode log-prob",
        y_label="Megatron train log-prob",
        x_legend="SGLang decode",
        y_legend="Megatron train",
        filename="logprob_heatmap_sglang_decode_vs_megatron_train.png",
        plt=plt,
        np=np,
        args=args,
        warnings=warnings,
    )
    _write_logprob_pair_chart(
        output_dir=output_dir,
        rows=rows,
        x_key="sglang_decode_logprob",
        y_key="sglang_prefill_logprob",
        x_label="SGLang decode log-prob",
        y_label="SGLang prefill log-prob",
        x_legend="SGLang decode",
        y_legend="SGLang prefill",
        filename="logprob_heatmap_sglang_decode_vs_sglang_prefill.png",
        plt=plt,
        np=np,
        args=args,
        warnings=warnings,
    )
    _write_logprob_pair_chart(
        output_dir=output_dir,
        rows=rows,
        x_key="sglang_prefill_logprob",
        y_key="megatron_train_logprob",
        x_label="SGLang prefill log-prob",
        y_label="Megatron train log-prob",
        x_legend="SGLang prefill",
        y_legend="Megatron train",
        filename="logprob_heatmap_sglang_prefill_vs_megatron_train.png",
        plt=plt,
        np=np,
        args=args,
        warnings=warnings,
    )


def _build_summary(rows: list[dict[str, Any]], warnings: list[str], args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "num_rows": len(rows),
        "num_active_rows": sum(1 for row in rows if row.get("loss_mask", 1)),
        "rollout_ids": sorted({row["rollout_id"] for row in rows}),
        "rank": args.rank,
        "threshold": args.close_threshold,
        "pairs": _pair_stats(rows, active_only=True),
        "pairs_all_tokens": _pair_stats(rows, active_only=False),
        "sample_mean_pairs": _sample_mean_pair_stats(rows),
        "by_rollout": {},
        "warnings": warnings,
    }
    for rollout_id in summary["rollout_ids"]:
        rollout_rows = [row for row in rows if row["rollout_id"] == rollout_id]
        summary["by_rollout"][str(rollout_id)] = {
            "num_rows": len(rollout_rows),
            "num_active_rows": sum(1 for row in rollout_rows if row.get("loss_mask", 1)),
            "pairs": _pair_stats(rollout_rows, active_only=True),
            "sample_mean_pairs": _sample_mean_pair_stats(rollout_rows),
        }
    summary["classification"] = _classify(summary, args.close_threshold)
    return summary


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print(f"Wrote logprob comparison to {output_dir}")
    print(
        f"Rows: {summary['num_rows']} active={summary['num_active_rows']} "
        f"rollouts: {summary['rollout_ids']} rank: {summary['rank']}"
    )
    print("Active-token stats:")
    for name, stats in summary["pairs"].items():
        mean = stats["mean"]
        p99 = stats["p99"]
        max_value = stats["max"]
        if mean is None:
            print(f"{name}: n=0")
        else:
            print(f"{name}: n={stats['count']} mean={mean:.6g} p99={p99:.6g} max={max_value:.6g}")
    sample_stats = summary["sample_mean_pairs"]["decode_vs_megatron"]
    if sample_stats["mean"] is not None:
        print(f"sample_mean_decode_vs_megatron: mean={sample_stats['mean']:.6g} n={sample_stats['count']}")
    print(f"classification: {summary['classification']}")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SGLang rollout logprobs with Megatron train logprobs.")
    parser.add_argument("--dump-details", type=Path, required=True)
    parser.add_argument("--rollout-ids", default="all", help="Comma/range spec, e.g. 0-2,5. Default: all.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sglang-url", default=None, help="SGLang /generate URL, or host URL where /generate is appended.")
    parser.add_argument("--compare-sglang-prefill", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-prefill-flush-cache", action="store_true")
    parser.add_argument("--close-threshold", type=float, default=0.03)
    parser.add_argument("--allow-local-slices", action="store_true")
    parser.add_argument("--plot-bins", type=int, default=120)
    parser.add_argument("--plot-logprob-min", type=float, default=-40.0)
    parser.add_argument("--plot-logprob-max", type=float, default=0.0)
    args = parser.parse_args()

    dump_details = args.dump_details
    if not dump_details.exists():
        raise FileNotFoundError(dump_details)
    rollout_ids = _parse_rollout_ids(args.rollout_ids, dump_details)
    if not rollout_ids:
        raise ValueError(f"No rollout ids found under {dump_details / 'rollout_data'}")

    tokenizer = _load_tokenizer(args.tokenizer)
    output_dir = args.output_dir or dump_details / "logprob_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for rollout_id in rollout_ids:
        rows.extend(
            _compare_rollout(
                dump_details=dump_details,
                rollout_id=rollout_id,
                rank=args.rank,
                tokenizer=tokenizer,
                args=args,
                warnings=warnings,
            )
        )

    summary = _build_summary(rows, warnings, args)
    _write_jsonl(output_dir / "tokens.jsonl", rows)
    _write_top_csv(output_dir / "top_diffs.csv", rows, args.top_k)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    _maybe_write_plots(output_dir, rows, warnings, args)
    if warnings:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    _print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
