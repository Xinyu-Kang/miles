#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Json = dict[str, Any]


def _read_jsonl(path: Path) -> list[Json]:
    rows: list[Json] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _normalize_url(url: str, endpoint: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.path and parsed.path != "/":
        return url.rstrip("/")
    return urllib.parse.urlunparse(parsed._replace(path=endpoint)).rstrip("/")


def _post_json(url: str, payload: Json, timeout: float) -> Any:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _flush_sglang_cache(base_url: str, timeout: float) -> None:
    url = _normalize_url(base_url, "/flush_cache")
    _post_json(url, {}, timeout)


def _tokenize_sglang(base_url: str, text: str, timeout: float, add_special_tokens: bool) -> list[int]:
    url = _normalize_url(base_url, "/tokenize")
    output = _post_json(
        url,
        {"prompt": text, "add_special_tokens": add_special_tokens},
        timeout,
    )
    tokens = output.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"SGLang /tokenize returned malformed response: {output}")
    return [int(x) for x in tokens]


def _detokenize_sglang(base_url: str, tokens: list[int], timeout: float) -> str:
    url = _normalize_url(base_url, "/detokenize")
    output = _post_json(url, {"tokens": tokens}, timeout)
    text = output.get("text", output.get("prompt"))
    if not isinstance(text, str):
        return ""
    return text


def _sample_id(row: Json, fallback: int) -> str:
    for key in ("id", "sample_id", "idx", "index"):
        if key in row:
            return str(row[key])
    return str(fallback)


def _int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"{name} is missing")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    return [int(x) for x in value]


def _float_list(value: Any, *, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{name} is missing")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    return [float(x) for x in value]


def _record_prompt_ids(
    row: Json,
    *,
    sample_pos: int,
    sglang_base_url: str,
    timeout: float,
    add_special_tokens: bool,
) -> tuple[str, list[int], str | None]:
    sample_id = _sample_id(row, sample_pos)
    if "prompt_ids" in row:
        return sample_id, _int_list(row["prompt_ids"], name="prompt_ids"), row.get("prompt")
    if "input_ids" in row:
        return sample_id, _int_list(row["input_ids"], name="input_ids"), row.get("prompt")
    if "prompt" not in row:
        raise ValueError(f"sample {sample_id} must contain prompt_ids, input_ids, or prompt")
    prompt = str(row["prompt"])
    return (
        sample_id,
        _tokenize_sglang(sglang_base_url, prompt, timeout, add_special_tokens),
        prompt,
    )


def _full_ids_from_record(row: Json) -> list[int]:
    if row.get("full_ids") is not None:
        return _int_list(row["full_ids"], name="full_ids")
    if row.get("tokens") is not None:
        return _int_list(row["tokens"], name="tokens")
    if row.get("input_ids") is not None:
        return _int_list(row["input_ids"], name="input_ids")
    if row.get("prompt_ids") is not None and row.get("completion_ids") is not None:
        return _int_list(row["prompt_ids"], name="prompt_ids") + _int_list(
            row["completion_ids"], name="completion_ids"
        )
    raise ValueError("record must contain full_ids, tokens, input_ids, or prompt_ids+completion_ids")


def _extract_output_token_logprobs(output: Json) -> tuple[list[int], list[float]]:
    meta = output.get("meta_info")
    if not isinstance(meta, dict):
        raise ValueError(f"SGLang response lacks meta_info: {output}")
    items = meta.get("output_token_logprobs")
    if not isinstance(items, list):
        raise ValueError(f"SGLang response lacks output_token_logprobs: {output}")
    token_ids: list[int] = []
    logprobs: list[float] = []
    for item in items:
        if not isinstance(item, list) and not isinstance(item, tuple):
            raise ValueError(f"Malformed output_token_logprobs item: {item}")
        if len(item) < 2:
            raise ValueError(f"Malformed output_token_logprobs item: {item}")
        logprob, token_id = item[0], item[1]
        if logprob is None:
            raise ValueError(f"SGLang returned None logprob for token {token_id}")
        token_ids.append(int(token_id))
        logprobs.append(float(logprob))
    return token_ids, logprobs


def _trim_stop_suffixes(
    *,
    base_url: str,
    token_ids: list[int],
    logprobs: list[float],
    stop_strings: list[str],
    timeout: float,
) -> tuple[list[int], list[float]]:
    for stop in stop_strings:
        if not stop:
            continue
        stop_ids = _tokenize_sglang(base_url, stop, timeout, add_special_tokens=False)
        if stop_ids and token_ids[-len(stop_ids) :] == stop_ids:
            return token_ids[: -len(stop_ids)], logprobs[: -len(stop_ids)]
    return token_ids, logprobs


def _extract_input_token_logprobs(output: Json, expected_tokens: list[int]) -> list[float]:
    meta = output.get("meta_info")
    if not isinstance(meta, dict):
        raise ValueError(f"SGLang response lacks meta_info: {output}")
    items = meta.get("input_token_logprobs")
    if not isinstance(items, list):
        raise ValueError(f"SGLang response lacks input_token_logprobs: {output}")
    if len(items) < len(expected_tokens):
        raise ValueError(
            f"SGLang returned {len(items)} input logprobs, expected at least {len(expected_tokens)}"
        )
    tail = items[-len(expected_tokens) :] if expected_tokens else []
    seen_tokens = [int(item[1]) for item in tail]
    if seen_tokens != expected_tokens:
        raise ValueError(
            "SGLang prefill token alignment mismatch: "
            f"expected={expected_tokens[:8]} len={len(expected_tokens)} "
            f"got={seen_tokens[:8]} len={len(seen_tokens)}"
        )
    logprobs = [item[0] for item in tail]
    if any(x is None for x in logprobs):
        raise ValueError("SGLang prefill returned None for at least one selected-token logprob")
    return [float(x) for x in logprobs]


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def command_sglang_generate(args: argparse.Namespace) -> None:
    base_url = args.sglang_url.rstrip("/")
    generate_url = _normalize_url(base_url, "/generate")
    input_rows = _read_jsonl(args.input)
    out_rows: list[Json] = []

    for batch in _chunks(list(enumerate(input_rows)), args.batch_size):
        payloads: list[Json] = []
        metadata: list[tuple[str, Json, list[int], str | None]] = []
        for sample_pos, row in batch:
            sample_id, prompt_ids, prompt_text = _record_prompt_ids(
                row,
                sample_pos=sample_pos,
                sglang_base_url=base_url,
                timeout=args.timeout,
                add_special_tokens=args.add_special_tokens,
            )
            sampling_params: Json = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "skip_special_tokens": args.skip_special_tokens,
            }
            if args.stop:
                sampling_params["stop"] = args.stop
            payloads.append(
                {
                    "input_ids": prompt_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                }
            )
            metadata.append((sample_id, row, prompt_ids, prompt_text))

        for payload, (sample_id, source_row, prompt_ids, prompt_text) in zip(payloads, metadata, strict=True):
            if args.flush_cache:
                _flush_sglang_cache(base_url, args.timeout)
            output = _post_json(generate_url, payload, args.timeout)
            completion_ids, completion_logprobs = _extract_output_token_logprobs(output)
            if args.stop and not args.keep_stop_tokens:
                completion_ids, completion_logprobs = _trim_stop_suffixes(
                    base_url=base_url,
                    token_ids=completion_ids,
                    logprobs=completion_logprobs,
                    stop_strings=args.stop,
                    timeout=args.timeout,
                )
            full_ids = prompt_ids + completion_ids
            text = output.get("text")
            if text is None and completion_ids:
                text = _detokenize_sglang(base_url, completion_ids, args.timeout)

            out_rows.append(
                {
                    "id": sample_id,
                    "backend": "sglang_decode",
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "completion_logprobs": completion_logprobs,
                    "full_ids": full_ids,
                    "score_start": max(len(prompt_ids) - 1, 0),
                    "score_length": len(completion_ids),
                    "text": text,
                    "source": source_row,
                }
            )
            if args.progress:
                print(f"sglang-generate id={sample_id} completion_tokens={len(completion_ids)}")

    _write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} SGLang generation records to {args.output}")


def command_sglang_score(args: argparse.Namespace) -> None:
    base_url = args.sglang_url.rstrip("/")
    generate_url = _normalize_url(base_url, "/generate")
    input_rows = _read_jsonl(args.input)
    out_rows: list[Json] = []

    for pos, row in enumerate(input_rows):
        sample_id = _sample_id(row, pos)
        full_ids = _full_ids_from_record(row)
        score_length = int(row.get("score_length", len(row.get("completion_ids", []))))
        if score_length < 0 or score_length > len(full_ids):
            raise ValueError(f"id={sample_id} invalid score_length={score_length}")
        score_start = int(row.get("score_start", len(full_ids) - score_length - 1))
        if score_start < 0:
            score_start = 0
        expected_tokens = full_ids[-score_length:] if score_length else []
        if args.flush_cache:
            _flush_sglang_cache(base_url, args.timeout)
        output = _post_json(
            generate_url,
            {
                "input_ids": full_ids,
                "sampling_params": {
                    "max_new_tokens": 0,
                    "temperature": 0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": score_start,
            },
            args.timeout,
        )
        score_logprobs = _extract_input_token_logprobs(output, expected_tokens)
        out = dict(row)
        out.update(
            {
                "id": sample_id,
                "backend": "sglang_prefill",
                "score_token_ids": expected_tokens,
                "score_logprobs": score_logprobs,
            }
        )
        out_rows.append(out)
        if args.progress:
            print(f"sglang-score id={sample_id} scored_tokens={len(score_logprobs)}")

    _write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} SGLang score records to {args.output}")


def command_make_score_input(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input)
    out_rows = []
    for pos, row in enumerate(rows):
        sample_id = _sample_id(row, pos)
        full_ids = _full_ids_from_record(row)
        score_length = int(row.get("score_length", len(row.get("completion_ids", []))))
        expected = full_ids[-score_length:] if score_length else []
        out_rows.append(
            {
                "id": sample_id,
                "full_ids": full_ids,
                "score_start": int(row.get("score_start", len(full_ids) - score_length - 1)),
                "score_length": score_length,
                "score_token_ids": expected,
                "prompt_ids": row.get("prompt_ids"),
                "completion_ids": row.get("completion_ids"),
            }
        )
    _write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} score-input records to {args.output}")


def command_lm_eval_samples_to_prompts(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input)
    out_rows: list[Json] = []
    for pos, row in enumerate(rows):
        if args.limit is not None and len(out_rows) >= args.limit:
            break
        arguments = row.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{args.input}:{pos + 1} does not have an arguments object")
        gen_args = arguments.get(args.argument_key)
        if gen_args is None and args.argument_key == "gen_args_0" and arguments:
            gen_args = next(iter(arguments.values()))
        if not isinstance(gen_args, dict) or "arg_0" not in gen_args:
            raise ValueError(
                f"{args.input}:{pos + 1} does not have arguments.{args.argument_key}.arg_0"
            )
        prompt = gen_args["arg_0"]
        if not isinstance(prompt, str):
            raise ValueError(f"{args.input}:{pos + 1} prompt is not a string")
        doc_id = row.get("doc_id", pos)
        sample_id = f"{args.id_prefix}_{doc_id}" if args.id_prefix else str(doc_id)
        out_rows.append({"id": sample_id, "prompt": prompt})

    _write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} prompt records to {args.output}")


def _torch_dtype(name: str) -> Any:
    import torch

    normalized = name.lower()
    if normalized in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "f16", "torch.float16"):
        return torch.float16
    if normalized in ("fp32", "float32", "f32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _dist_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _init_megatron_parallel(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    rank, world_size, local_rank = _dist_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = args.dist_backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=args.context_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )
    if torch.cuda.is_available():
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(1234, force_reset_rng=True)


def _patch_dsv4_router_tid2eid_device() -> None:
    import torch
    from megatron.core.transformer.moe.router import TopKRouter

    if getattr(TopKRouter, "_miles_dsv4_tid2eid_device_patch", False):
        return

    original = TopKRouter._init_routing_mode

    def _init_routing_mode(self, layer_number):
        if getattr(self.config, "experimental_attention_variant", None) != "dsv4":
            return original(self, layer_number)

        assert not self._routing_mode_initialized
        self._routing_mode_initialized = True

        mode_hash = layer_number <= self.config.dsv4_n_hash_layers
        self.enable_expert_bias = self.config.moe_router_enable_expert_bias and not mode_hash
        if self.enable_expert_bias:
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                "expert_bias",
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        if self.config.freeze_e_score_correction_bias and self.enable_expert_bias:
            self._frozen_expert_bias_snapshot = None

        if mode_hash:
            full_kwargs = {"dtype": torch.int32}
            if torch.cuda.is_available():
                full_kwargs["device"] = torch.cuda.current_device()
            self.tid2eid = torch.nn.Parameter(
                torch.full(
                    (self.config.vocab_size, self.topk),
                    fill_value=-1,
                    **full_kwargs,
                ),
                requires_grad=False,
            )

    TopKRouter._init_routing_mode = _init_routing_mode
    TopKRouter._miles_dsv4_tid2eid_device_patch = True


def _build_megatron_model(args: argparse.Namespace) -> Any:
    import torch

    _init_megatron_parallel(args)
    _patch_dsv4_router_tid2eid_device()

    from miles.utils.transformers_patch import apply_transformers_patch

    apply_transformers_patch()
    import miles_plugins.mbridge  # noqa: F401
    from mbridge import AutoBridge

    dtype = _torch_dtype(args.dtype)
    bridge = AutoBridge.from_pretrained(args.model_path, trust_remote_code=True)
    if not args.keep_mtp and getattr(bridge.config, "mtp_num_layers", None):
        if _dist_rank_info()[0] == 0:
            print(
                "Disabling MTP block for selected-token logprob scoring "
                f"(checkpoint has mtp_num_layers={bridge.config.mtp_num_layers})."
            )
        bridge.config.mtp_num_layers = None
        if hasattr(bridge.config, "mtp_loss_scaling_factor"):
            bridge.config.mtp_loss_scaling_factor = None
    extra_provider_args = json.loads(args.extra_provider_args) if args.extra_provider_args else {}
    weight_path = args.weight_path or args.model_path
    model = bridge.get_model(
        weight_path=None,
        wrap_with_ddp=False,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        extra_provider_args=extra_provider_args,
    )
    bridge.load_weights(
        model,
        bridge._get_actual_hf_path(weight_path),
        memory_efficient=True,
    )
    if isinstance(model, list):
        if len(model) != 1:
            raise ValueError(
                "This standalone scorer currently supports PP=1 only; "
                f"bridge returned {len(model)} pipeline chunks"
            )
        model = model[0]
    model.eval()
    return model


def _set_miles_parallel_state_from_megatron() -> None:
    from megatron.core import mpu
    from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

    def _optional(call):
        try:
            return call()
        except Exception:
            return None

    def _group_info(rank, size, group, gloo_group=None):
        return GroupInfo(rank=rank, size=size, group=group, gloo_group=gloo_group)

    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None or vpp_size <= 1:
        vpp_size = 1

    state = ParallelState(
        intra_dp=_group_info(
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
            mpu.get_data_parallel_group(with_context_parallel=False),
            _optional(lambda: mpu.get_data_parallel_group_gloo(with_context_parallel=False)),
        ),
        intra_dp_cp=_group_info(
            mpu.get_data_parallel_rank(with_context_parallel=True),
            mpu.get_data_parallel_world_size(with_context_parallel=True),
            mpu.get_data_parallel_group(with_context_parallel=True),
            _optional(lambda: mpu.get_data_parallel_group_gloo(with_context_parallel=True)),
        ),
        cp=_group_info(
            mpu.get_context_parallel_rank(),
            mpu.get_context_parallel_world_size(),
            mpu.get_context_parallel_group(),
        ),
        tp=_group_info(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_tensor_model_parallel_group(),
        ),
        pp=_group_info(
            mpu.get_pipeline_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_world_size(),
            mpu.get_pipeline_model_parallel_group(),
        ),
        ep=_group_info(
            mpu.get_expert_model_parallel_rank(),
            mpu.get_expert_model_parallel_world_size(),
            mpu.get_expert_model_parallel_group(),
        ),
        etp=_group_info(
            mpu.get_expert_tensor_parallel_rank(),
            mpu.get_expert_tensor_parallel_world_size(),
            mpu.get_expert_tensor_parallel_group(),
        ),
        cp_comm_type=None,
        is_pp_last_stage=mpu.is_pipeline_last_stage(),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=None,
    )
    set_parallel_state(state)


def _pad_bshd_max_len(lengths: list[int], tp_size: int, multiplier: int) -> int:
    pad_size = max(1, tp_size * multiplier)
    max_len = max(lengths) if lengths else 0
    return (max_len + pad_size - 1) // pad_size * pad_size


def command_megatron_score_mbridge(args: argparse.Namespace) -> None:
    """Experimental direct Megatron scorer using mbridge and Miles logprob helpers.

    This is intentionally a separate subcommand: run it with torchrun if TP > 1.
    It reads the canonical JSONL produced by make-score-input and writes the same
    JSONL plus score_logprobs.
    """

    import torch
    import torch.distributed as dist
    from argparse import Namespace

    model = _build_megatron_model(args)

    from megatron.core import parallel_state
    from miles.backends.megatron_utils.parallel import get_packed_seq_params
    from miles.backends.training_utils.data import DataIterator, get_batch
    from miles.backends.training_utils.loss import get_log_probs_and_entropy

    _set_miles_parallel_state_from_megatron()

    rank, _, _ = _dist_rank_info()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    rows = _read_jsonl(args.input)
    out_rows: list[Json] = []

    runtime_args = Namespace(
        qkv_format=args.qkv_format,
        data_pad_size_multiplier=args.data_pad_size_multiplier,
        allgather_cp=False,
        log_probs_chunk_size=args.log_probs_chunk_size,
        true_on_policy_mode=args.true_on_policy_mode,
        rollout_temperature=args.temperature,
        use_rollout_entropy=False,
        bf16=_torch_dtype(args.dtype) is torch.bfloat16,
        fp16=_torch_dtype(args.dtype) is torch.float16,
        vocab_size=args.vocab_size,
    )

    for batch_rows in _chunks(rows, args.batch_size):
        token_tensors = []
        loss_masks = []
        total_lengths = []
        response_lengths = []
        for row in batch_rows:
            full_ids = _full_ids_from_record(row)
            score_length = int(row.get("score_length", len(row.get("score_token_ids", []))))
            token_tensors.append(torch.tensor(full_ids, dtype=torch.long, device=device))
            loss_masks.append(torch.ones(score_length, dtype=torch.int, device=device))
            total_lengths.append(len(full_ids))
            response_lengths.append(score_length)

        rollout_data: dict[str, Any] = {
            "tokens": token_tensors,
            "loss_masks": loss_masks,
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }
        if args.qkv_format == "bshd":
            max_seq_len = _pad_bshd_max_len(total_lengths, tp_size, args.data_pad_size_multiplier)
            rollout_data["max_seq_lens"] = [max_seq_len] * len(batch_rows)

        iterator = DataIterator(rollout_data, micro_batch_size=len(batch_rows))
        batch = get_batch(
            iterator,
            ["tokens", "loss_masks", "total_lengths", "response_lengths", "max_seq_lens"],
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=False,
        )
        packed_seq_params = get_packed_seq_params(batch, runtime_args)
        with torch.no_grad():
            logits = model(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=packed_seq_params,
                loss_mask=batch["full_loss_masks"],
            )
            result = get_log_probs_and_entropy(
                logits,
                args=runtime_args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                with_entropy=False,
                max_seq_lens=batch.get("max_seq_lens", None),
            )

        score_rows = result["log_probs"]
        if rank == 0:
            for row, score_tensor in zip(batch_rows, score_rows, strict=True):
                score_logprobs = [float(x) for x in score_tensor.detach().cpu().tolist()]
                out = dict(row)
                out.update(
                    {
                        "backend": "megatron_mbridge",
                        "score_logprobs": score_logprobs,
                    }
                )
                if "score_token_ids" not in out:
                    full_ids = _full_ids_from_record(out)
                    out["score_token_ids"] = full_ids[-len(score_logprobs) :] if score_logprobs else []
                out_rows.append(out)
                if args.progress:
                    print(f"megatron-score id={_sample_id(out, len(out_rows) - 1)} tokens={len(score_logprobs)}")

        del logits, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rank == 0:
        _write_jsonl(args.output, out_rows)
        print(f"Wrote {len(out_rows)} Megatron score records to {args.output}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _field_with_fallback(row: Json, field: str | None, fallbacks: tuple[str, ...], *, name: str) -> list[float]:
    if field:
        return _float_list(row.get(field), name=f"{name}.{field}")
    for key in fallbacks:
        if key in row:
            return _float_list(row[key], name=f"{name}.{key}")
    raise ValueError(f"{name} does not contain any of {fallbacks}; pass an explicit field")


def _tokens_for_compare(row: Json, score_len: int) -> list[int] | None:
    for key in ("score_token_ids", "completion_ids"):
        if key in row:
            return _int_list(row[key], name=key)
    if "full_ids" in row:
        full_ids = _int_list(row["full_ids"], name="full_ids")
        return full_ids[-score_len:] if score_len else []
    return None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(x for x in values if math.isfinite(x))
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


@dataclass
class PairRows:
    token_rows: list[Json]
    warnings: list[str]


def build_compare_rows(args: argparse.Namespace) -> PairRows:
    left_rows = _read_jsonl(args.left)
    right_rows = _read_jsonl(args.right)
    right_by_id = {_sample_id(row, pos): row for pos, row in enumerate(right_rows)}

    token_rows: list[Json] = []
    warnings: list[str] = []
    for left_pos, left in enumerate(left_rows):
        sample_id = _sample_id(left, left_pos)
        right = right_by_id.get(sample_id)
        if right is None:
            warnings.append(f"id={sample_id} missing from right file")
            continue

        left_lp = _field_with_fallback(
            left,
            args.left_field,
            ("score_logprobs", "completion_logprobs", "logprobs", "megatron_logprobs"),
            name=f"left id={sample_id}",
        )
        right_lp = _field_with_fallback(
            right,
            args.right_field,
            ("score_logprobs", "completion_logprobs", "logprobs", "megatron_logprobs"),
            name=f"right id={sample_id}",
        )
        compare_len = min(len(left_lp), len(right_lp))
        if len(left_lp) != len(right_lp):
            message = f"id={sample_id} length mismatch left={len(left_lp)} right={len(right_lp)}"
            if args.strict_lengths:
                raise ValueError(message)
            warnings.append(message)

        left_tokens = _tokens_for_compare(left, compare_len)
        right_tokens = _tokens_for_compare(right, compare_len)
        if left_tokens is not None and right_tokens is not None:
            if left_tokens[:compare_len] != right_tokens[:compare_len]:
                message = f"id={sample_id} token mismatch between left and right"
                if args.strict_tokens:
                    raise ValueError(message)
                warnings.append(message)

        tokens = left_tokens or right_tokens or [None] * compare_len
        for token_pos in range(compare_len):
            a = float(left_lp[token_pos])
            b = float(right_lp[token_pos])
            token_id = tokens[token_pos] if token_pos < len(tokens) else None
            token_rows.append(
                {
                    "id": sample_id,
                    "token_pos": token_pos,
                    "token_id": token_id,
                    "left_logprob": a,
                    "right_logprob": b,
                    "signed_diff_left_minus_right": a - b,
                    "abs_diff": abs(a - b),
                    "left_prob": math.exp(max(min(a, 0.0), -80.0)),
                    "right_prob": math.exp(max(min(b, 0.0), -80.0)),
                }
            )

    return PairRows(token_rows=token_rows, warnings=warnings)


def _write_top_csv(path: Path, rows: list[Json], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "token_pos",
        "token_id",
        "left_logprob",
        "right_logprob",
        "signed_diff_left_minus_right",
        "abs_diff",
        "left_prob",
        "right_prob",
    ]
    sorted_rows = sorted(rows, key=lambda row: float(row["abs_diff"]), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows[:top_k]:
            writer.writerow(row)


def _maybe_write_plots(output_dir: Path, rows: list[Json], warnings: list[str], args: argparse.Namespace) -> None:
    if args.no_plots:
        return
    try:
        import numpy as np
        from matplotlib import pyplot as plt
    except Exception as exc:
        warnings.append(f"plotting skipped: {type(exc).__name__}: {exc}")
        return
    if not rows:
        return

    left = np.asarray([float(row["left_logprob"]) for row in rows], dtype=np.float64)
    right = np.asarray([float(row["right_logprob"]) for row in rows], dtype=np.float64)
    diff = np.abs(left - right)
    left_label = getattr(args, "left_label", "left")
    right_label = getattr(args, "right_label", "right")

    plt.figure(figsize=(10, 4))
    plt.hist(diff, bins=args.plot_bins)
    plt.xlabel("|left - right| logprob")
    plt.ylabel("token count")
    plt.tight_layout()
    plt.savefig(output_dir / "diff_hist.png", dpi=160)
    plt.close()

    eps = 1e-10
    hist2d, x_edges, y_edges = np.histogram2d(
        left,
        right,
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
    )
    heat_ax.set_xlim(args.plot_logprob_min, args.plot_logprob_max)
    heat_ax.set_ylim(args.plot_logprob_min, args.plot_logprob_max)
    heat_ax.set_aspect("equal", adjustable="box")
    heat_ax.set_xlabel(f"{left_label} log-prob")
    heat_ax.set_ylabel(f"{right_label} log-prob")
    colorbar = fig.colorbar(heatmap, ax=heat_ax)
    colorbar.set_label("Log Frequency")

    left_density, edges = np.histogram(
        left,
        bins=args.plot_bins,
        range=(args.plot_logprob_min, args.plot_logprob_max),
        density=True,
    )
    right_density, _ = np.histogram(
        right,
        bins=args.plot_bins,
        range=(args.plot_logprob_min, args.plot_logprob_max),
        density=True,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    left_density = np.nan_to_num(left_density, nan=0.0, posinf=0.0, neginf=0.0)
    right_density = np.nan_to_num(right_density, nan=0.0, posinf=0.0, neginf=0.0)
    density_ax.plot(centers, np.log(left_density + eps), label=left_label)
    density_ax.plot(centers, np.log(right_density + eps), label=right_label)
    density_ax.set_xlim(args.plot_logprob_min, args.plot_logprob_max)
    density_ax.set_xlabel("log-prob")
    density_ax.set_ylabel("Log Density")
    density_ax.legend()

    fig.savefig(output_dir / "logprob_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def command_compare(args: argparse.Namespace) -> dict[str, Any]:
    pair_rows = build_compare_rows(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    token_rows = pair_rows.token_rows
    diffs = [float(row["abs_diff"]) for row in token_rows]
    signed = [float(row["signed_diff_left_minus_right"]) for row in token_rows]
    probs_diff = [float(row["left_prob"]) - float(row["right_prob"]) for row in token_rows]
    summary = {
        "left": str(args.left),
        "right": str(args.right),
        "left_field": args.left_field,
        "right_field": args.right_field,
        "num_tokens": len(token_rows),
        "abs_diff": _stats(diffs),
        "signed_diff_left_minus_right": _stats(signed),
        "prob_diff_left_minus_right": _stats(probs_diff),
        "threshold": args.close_threshold,
        "pass": bool(diffs) and _stats(diffs)["mean"] is not None and float(_stats(diffs)["mean"]) <= args.close_threshold,
        "warnings": pair_rows.warnings,
    }
    _write_jsonl(output_dir / "tokens.jsonl", token_rows)
    _write_top_csv(output_dir / "top_diffs.csv", token_rows, args.top_k)
    _write_json(output_dir / "summary.json", summary)
    _maybe_write_plots(output_dir, token_rows, pair_rows.warnings, args)
    if pair_rows.warnings:
        summary["warnings"] = pair_rows.warnings
        _write_json(output_dir / "summary.json", summary)

    stats = summary["abs_diff"]
    print(f"Wrote comparison to {output_dir}")
    print(
        "abs_diff: "
        f"n={stats['count']} mean={stats['mean']} p99={stats['p99']} max={stats['max']}"
    )
    print(f"pass_mean_threshold={summary['pass']} threshold={args.close_threshold}")
    for warning in pair_rows.warnings:
        print(f"warning: {warning}")
    return summary


def command_run_pair(args: argparse.Namespace) -> None:
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    sglang_decode = work_dir / "sglang_decode.jsonl"
    megatron_input = work_dir / "score_input.jsonl"
    megatron_output = work_dir / "megatron_score.jsonl"
    compare_dir = work_dir / "compare_sglang_decode_vs_megatron"

    gen_args = argparse.Namespace(**vars(args))
    gen_args.output = sglang_decode
    command_sglang_generate(gen_args)

    score_args = argparse.Namespace(input=sglang_decode, output=megatron_input)
    command_make_score_input(score_args)

    command = args.megatron_command.format(
        input=shlex.quote(str(megatron_input)),
        output=shlex.quote(str(megatron_output)),
        work_dir=shlex.quote(str(work_dir)),
    )
    print(f"Running Megatron scoring command: {command}")
    subprocess.run(command, shell=True, check=True)
    if not megatron_output.exists():
        raise FileNotFoundError(
            f"Megatron command completed but did not create expected output {megatron_output}"
        )

    cmp_args = argparse.Namespace(
        left=sglang_decode,
        right=megatron_output,
        left_field=args.left_field or "completion_logprobs",
        right_field=args.right_field or "score_logprobs",
        left_label=args.left_label,
        right_label=args.right_label,
        output_dir=compare_dir,
        top_k=args.top_k,
        close_threshold=args.close_threshold,
        strict_lengths=args.strict_lengths,
        strict_tokens=args.strict_tokens,
        no_plots=args.no_plots,
        plot_bins=args.plot_bins,
        plot_logprob_min=args.plot_logprob_min,
        plot_logprob_max=args.plot_logprob_max,
    )
    command_compare(cmp_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone logprob comparator for independently run engines. "
            "Use SGLang to generate canonical token sequences, score those same "
            "sequences with Megatron, then compare chosen-token logprobs."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("sglang-generate", help="Generate with an existing SGLang HTTP server.")
    gen.add_argument("--input", type=Path, required=True, help="JSONL with prompt_ids, input_ids, or prompt.")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    gen.add_argument("--batch-size", type=int, default=1)
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=-1)
    gen.add_argument("--stop", action="append", default=[])
    gen.add_argument("--keep-stop-tokens", action="store_true")
    gen.add_argument("--skip-special-tokens", action="store_true")
    gen.add_argument("--add-special-tokens", action="store_true")
    gen.add_argument("--flush-cache", action="store_true")
    gen.add_argument("--timeout", type=float, default=600.0)
    gen.add_argument("--progress", action="store_true")
    gen.set_defaults(func=command_sglang_generate)

    score = sub.add_parser("sglang-score", help="Prefill-score existing full_ids with SGLang.")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    score.add_argument("--flush-cache", action="store_true")
    score.add_argument("--timeout", type=float, default=600.0)
    score.add_argument("--progress", action="store_true")
    score.set_defaults(func=command_sglang_score)

    mk = sub.add_parser("make-score-input", help="Write canonical score-input JSONL for Megatron.")
    mk.add_argument("--input", type=Path, required=True)
    mk.add_argument("--output", type=Path, required=True)
    mk.set_defaults(func=command_make_score_input)

    lm_samples = sub.add_parser(
        "lm-eval-samples-to-prompts",
        help="Extract prompt JSONL from lm-eval sample logs for larger comparisons.",
    )
    lm_samples.add_argument("--input", type=Path, required=True)
    lm_samples.add_argument("--output", type=Path, required=True)
    lm_samples.add_argument("--limit", type=int, default=None)
    lm_samples.add_argument("--argument-key", default="gen_args_0")
    lm_samples.add_argument("--id-prefix", default="")
    lm_samples.set_defaults(func=command_lm_eval_samples_to_prompts)

    meg = sub.add_parser(
        "megatron-score-mbridge",
        help="Experimental Megatron scorer for canonical score-input JSONL. Run with torchrun for TP > 1.",
    )
    meg.add_argument("--input", type=Path, required=True)
    meg.add_argument("--output", type=Path, required=True)
    meg.add_argument("--model-path", required=True, help="HF checkpoint/model path used by mbridge.")
    meg.add_argument("--weight-path", default=None, help="Optional weight path; defaults to --model-path.")
    meg.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    meg.add_argument("--batch-size", type=int, default=1)
    meg.add_argument("--qkv-format", choices=["bshd", "thd"], default="bshd")
    meg.add_argument("--temperature", type=float, default=1.0)
    meg.add_argument("--tensor-model-parallel-size", type=int, default=1)
    meg.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    meg.add_argument("--context-parallel-size", type=int, default=1)
    meg.add_argument("--expert-model-parallel-size", type=int, default=1)
    meg.add_argument("--expert-tensor-parallel-size", type=int, default=1)
    meg.add_argument("--data-pad-size-multiplier", type=int, default=128)
    meg.add_argument("--log-probs-chunk-size", type=int, default=-1)
    meg.add_argument("--true-on-policy-mode", action="store_true")
    meg.add_argument(
        "--keep-mtp",
        action="store_true",
        help="Build the checkpoint's MTP block. By default it is disabled because normal logprob scoring does not use it.",
    )
    meg.add_argument("--vocab-size", type=int, default=None)
    meg.add_argument("--dist-backend", default="auto", choices=["auto", "nccl", "gloo", "mpi"])
    meg.add_argument("--extra-provider-args", default=None, help="JSON dict passed to mbridge get_model().")
    meg.add_argument("--progress", action="store_true")
    meg.set_defaults(func=command_megatron_score_mbridge)

    cmp_parser = sub.add_parser("compare", help="Compare two JSONL score files.")
    cmp_parser.add_argument("--left", type=Path, required=True)
    cmp_parser.add_argument("--right", type=Path, required=True)
    cmp_parser.add_argument("--left-field", default=None)
    cmp_parser.add_argument("--right-field", default=None)
    cmp_parser.add_argument("--left-label", default="left")
    cmp_parser.add_argument("--right-label", default="right")
    cmp_parser.add_argument("--output-dir", type=Path, required=True)
    cmp_parser.add_argument("--top-k", type=int, default=50)
    cmp_parser.add_argument("--close-threshold", type=float, default=0.03)
    cmp_parser.add_argument("--strict-lengths", action="store_true")
    cmp_parser.add_argument("--strict-tokens", action="store_true")
    cmp_parser.add_argument("--no-plots", action="store_true")
    cmp_parser.add_argument("--plot-bins", type=int, default=120)
    cmp_parser.add_argument("--plot-logprob-min", type=float, default=-40.0)
    cmp_parser.add_argument("--plot-logprob-max", type=float, default=0.0)
    cmp_parser.set_defaults(func=command_compare)

    run_pair = sub.add_parser(
        "run-pair",
        help=(
            "Run SGLang generation, call a user-provided Megatron scoring command, "
            "then compare. The command may use {input}, {output}, and {work_dir}."
        ),
    )
    run_pair.add_argument("--input", type=Path, required=True)
    run_pair.add_argument("--work-dir", type=Path, required=True)
    run_pair.add_argument("--megatron-command", required=True)
    run_pair.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    run_pair.add_argument("--batch-size", type=int, default=1)
    run_pair.add_argument("--max-new-tokens", type=int, default=128)
    run_pair.add_argument("--temperature", type=float, default=0.0)
    run_pair.add_argument("--top-p", type=float, default=1.0)
    run_pair.add_argument("--top-k", type=int, default=-1)
    run_pair.add_argument("--stop", action="append", default=[])
    run_pair.add_argument("--skip-special-tokens", action="store_true")
    run_pair.add_argument("--add-special-tokens", action="store_true")
    run_pair.add_argument("--flush-cache", action="store_true")
    run_pair.add_argument("--timeout", type=float, default=600.0)
    run_pair.add_argument("--progress", action="store_true")
    run_pair.add_argument("--left-field", default=None)
    run_pair.add_argument("--right-field", default=None)
    run_pair.add_argument("--left-label", default="SGLang decode")
    run_pair.add_argument("--right-label", default="Megatron")
    run_pair.add_argument("--top-k-diffs", dest="top_k", type=int, default=50)
    run_pair.add_argument("--close-threshold", type=float, default=0.03)
    run_pair.add_argument("--strict-lengths", action="store_true")
    run_pair.add_argument("--strict-tokens", action="store_true")
    run_pair.add_argument("--no-plots", action="store_true")
    run_pair.add_argument("--plot-bins", type=int, default=120)
    run_pair.add_argument("--plot-logprob-min", type=float, default=-40.0)
    run_pair.add_argument("--plot-logprob-max", type=float, default=0.0)
    run_pair.set_defaults(func=command_run_pair)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.perf_counter()
    args.func(args)
    print(f"done_seconds={time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
