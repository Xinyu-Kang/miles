import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


_DEFAULT_AITER_CONFIG_FMOE = Path(__file__).resolve().parent / "amd" / "dsv4_flash_fp4_tp8_fmoe.csv"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()

    hf_checkpoint: str = "deepseek-ai/DeepSeek-V4-Flash"
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V4-Flash"
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    num_rollout: int | None = None
    rollout_batch_size: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_max_response_len: int | None = None
    num_steps_per_rollout: int | None = None
    micro_batch_size: int = 1
    max_tokens_per_gpu: int | None = None
    log_probs_max_tokens_per_gpu: int | None = None
    context_length: int = 8192

    sglang_mem_fraction_static: float | None = None
    sglang_max_running_requests: int | None = None
    sglang_max_total_tokens: int | None = None
    sglang_disable_cuda_graph: bool = True
    accumulate_allreduce_grads_in_fp32: bool = False
    train_memory_margin_bytes: int = 2 * 1024 * 1024 * 1024

    aiter_config_fmoe: str = str(_DEFAULT_AITER_CONFIG_FMOE)
    download_model: bool = False
    download_data: bool = True
    skip_saving: bool = True
    enable_eval: bool = False
    extra_args: str = ""


def _pick(args: ScriptArgs, name: str, debug_value, normal_value):
    value = getattr(args, name)
    if value is not None:
        return value
    return debug_value if args.mode == "debug_minimal" else normal_value


def _is_local_path(value: str) -> bool:
    return value.startswith("/") or value.startswith(".")


def _prepare(args: ScriptArgs) -> None:
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")

    if args.download_model:
        local_checkpoint = Path(args.model_dir) / args.model_name
        U.exec_command(f"hf download {args.model_org}/{args.model_name} --local-dir {local_checkpoint}")
        args.hf_checkpoint = str(local_checkpoint)

    if not _is_local_path(args.hf_checkpoint):
        from huggingface_hub import snapshot_download

        args.hf_checkpoint = snapshot_download(args.hf_checkpoint)

    if _is_local_path(args.hf_checkpoint) and not Path(args.hf_checkpoint).exists():
        raise FileNotFoundError(f"{args.hf_checkpoint} does not exist. Download it or pass a Hugging Face repo id.")

    if args.download_data:
        U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        if args.enable_eval:
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    _resolve_aiter_config(args)


def _resolve_aiter_config(args: ScriptArgs) -> str:
    configured = Path(args.aiter_config_fmoe)
    if configured.exists():
        return str(configured)

    tmp_default = Path("/tmp/dsv4_flash_fp4_tp8_fmoe.csv")
    if tmp_default.exists():
        args.aiter_config_fmoe = str(tmp_default)
        return str(tmp_default)

    raise FileNotFoundError(
        f"AITER FMOE config not found at {configured}. "
        "This MI355 DeepSeek-V4-Flash FP4 rollout path needs AITER_CONFIG_FMOE."
    )


def _execute(args: ScriptArgs) -> None:
    if args.num_nodes != 1:
        raise ValueError("DeepSeek-V4-Flash MI355 launcher is configured for one 8-GPU node.")
    if args.num_gpus_per_node != 8:
        raise ValueError("DeepSeek-V4-Flash MI355 launcher expects exactly 8 GPUs on the node.")

    checkpoint = args.hf_checkpoint
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {checkpoint} "
        # Miles raw checkpoint validation uses ref_load as the initial HF load path
        # when --load is omitted. KL/ref computation is still disabled below.
        f"--ref-load {checkpoint} "
    )
    if not args.skip_saving:
        ckpt_args += f"--save {load_save_path} --save-interval 20 --save-retain-interval 20 "

    num_rollout = _pick(args, "num_rollout", 1, 300)
    rollout_batch_size = _pick(args, "rollout_batch_size", 1, 4)
    n_samples = _pick(args, "n_samples_per_prompt", 1, 4)
    response_len = _pick(args, "rollout_max_response_len", 64, 4096)
    num_steps = _pick(args, "num_steps_per_rollout", 1, 1)
    max_tokens_per_gpu = _pick(args, "max_tokens_per_gpu", 2048, 2048)
    log_probs_max_tokens_per_gpu = args.log_probs_max_tokens_per_gpu or max_tokens_per_gpu

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"thinking\":true}' "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples} "
        f"--rollout-max-response-len {response_len} "
        "--rollout-temperature 0.8 "
        f"--num-steps-per-rollout {num_steps} "
        "--balance-data "
    )

    eval_args = ""
    if args.mode != "debug_minimal" and args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 8 "
            "--eval-max-response-len 4096 "
            "--eval-top-p 0.7 "
        )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        f"--seq-length {args.context_length} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        f"--micro-batch-size {args.micro_batch_size} "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        f"--log-probs-max-tokens-per-gpu {log_probs_max_tokens_per_gpu} "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    sglang_mem_fraction = _pick(args, "sglang_mem_fraction_static", 0.30, 0.32)
    sglang_max_running = _pick(args, "sglang_max_running_requests", 2, 4)
    sglang_max_total_tokens = _pick(args, "sglang_max_total_tokens", 2048, 8192)
    sglang_cuda_graph_args = (
        "--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph "
        if args.sglang_disable_cuda_graph
        else ""
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-tp-size 8 "
        "--sglang-dp-size 1 "
        # Keep rollout on the validated DeepSeek-V4-Flash SGLang path: TP=8 only.
        # Megatron still uses EP=8 for training above.
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        f"--sglang-max-running-requests {sglang_max_running} "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 1024 "
        f"--sglang-context-length {args.context_length} "
        f"--sglang-max-total-tokens {sglang_max_total_tokens} "
        "--sglang-disable-radix-cache "
        "--sglang-disable-custom-all-reduce "
        "--sglang-disable-shared-experts-fusion "
        "--sglang-moe-runner-backend aiter "
        "--sglang-schedule-conservativeness 1.0 "
        f"--sglang-mem-fraction-static {sglang_mem_fraction} "
        "--sglang-tool-call-parser deepseekv4 "
        "--sglang-reasoning-parser deepseek-v4 "
        f"{sglang_cuda_graph_args}"
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--load-hf-with-mbridge "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--grad-reduce-in-bf16 "
        "--update-weight-buffer-size 268435456 "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--model-name deepseekv4 "
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--colocate "
        "--offload-train "
        "--offload-rollout "
        "--offload-rollout-level kv_cache weight "
        "--use-fault-tolerance "
        "--use-miles-router "
        "--disable-weights-backuper "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    if args.accumulate_allreduce_grads_in_fp32:
        misc_args += "--accumulate-allreduce-grads-in-fp32 "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="deepseek-v4-flash",
        extra_env_vars=_extra_env(args),
        megatron_path=args.megatron_path,
    )


def _extra_env(args: ScriptArgs) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        path for path in (str(U.repo_base_dir), args.megatron_path, os.environ.get("PYTHONPATH", "")) if path
    )
    visible_devices = ",".join(str(i) for i in range(args.num_gpus_per_node))
    return {
        "PYTHONPATH": pythonpath,
        "HIP_VISIBLE_DEVICES": visible_devices,
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "HIP_FORCE_DEV_KERNARG": "1",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_MIN_NCHANNELS": "112",
        "NCCL_ALGO": "Ring",
        "NCCL_NVLS_ENABLE": "0",
        "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "USE_ROCM": "1",
        "USE_CUDA": "0",
        "ROCM_HOME": "/opt/rocm",
        "ROCM_PATH": "/opt/rocm",
        "USE_ROCM_AITER_ROPE_BACKEND": "0",
        "MILES_DSV4_CKPT_VERSION": "2604",
        "MILES_DSV4_2604_SUBMODE": "2604B",
        "MEGATRON_USE_KV_QAT": "1",
        "MILES_HACK_TRAIN_TORCH_DETERMINISTIC": "1",
        "MILES_MBRIDGE_MEMORY_EFFICIENT_LOAD": "1",
        "AITER_CONFIG_FMOE": _resolve_aiter_config(args),
        "AITER_BF16_FP8_MOE_BOUND": "0",
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
        "SGLANG_DSV4_MODE": "2604",
        "SGLANG_DSV4_2604_SUBMODE": "2604B",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_ENABLE_THINKING": "1",
        "SGLANG_USE_AITER": "1",
        "SGLANG_USE_ROCM700A": "1",
        "SGLANG_MOE_PADDING": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
        "SGLANG_OPT_USE_FUSED_COMPRESS": "false",
        "SGLANG_OPT_USE_OLD_COMPRESSOR": "true",
        "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
        "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
        "SGLANG_OPT_USE_FUSED_HASH_TOPK": "false",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
        "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
        "SGLANG_OPT_DPSK_V4_RADIX": "0",
        "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "false",
        "SGLANG_OPT_USE_FUSED_STORE_CACHE": "false",
        "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
        "SGLANG_HACK_FLASHMLA_BACKEND": "tilelang",
        "SGLANG_REASONING_EFFORT": "max",
        "SGLANG_DSV4_FP4_EXPERTS": "true",
        "SGLANG_FORCE_TRITON_MOE_FP8": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "RAY_DEDUP_LOGS": "0",
    }


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
