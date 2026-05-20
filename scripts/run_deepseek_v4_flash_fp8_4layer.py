import os
from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    hf_checkpoint: str = "/root/models/DeepSeek-V4-Flash-FP8-4layer"
    model_org: str = "Pinaster"
    model_name: str = "DeepSeek-V4-Flash-FP8-4layer"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    num_rollout: int = 5
    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 1
    rollout_max_response_len: int = 128
    download_model: bool = False
    download_data: bool = True
    skip_saving: bool = True
    enable_eval: bool = False
    extra_args: str = ""


def _prepare(args: ScriptArgs) -> None:
    checkpoint = Path(args.hf_checkpoint)
    U.exec_command(f"mkdir -p {checkpoint.parent} {args.data_dir}")
    if args.download_model:
        U.exec_command(f"hf download {args.model_org}/{args.model_name} --local-dir {checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} does not exist. Download it or pass --download-model.")

    if args.download_data:
        U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        if args.enable_eval:
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _execute(args: ScriptArgs) -> None:
    if args.num_nodes != 1:
        raise ValueError("This 4-layer smoke script is only configured for one 8-GPU node.")

    checkpoint = args.hf_checkpoint
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {checkpoint} --ref-load {checkpoint} "
    if not args.skip_saving:
        ckpt_args += f"--load {checkpoint} --save {load_save_path} --save-interval 20 --save-retain-interval 20 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"thinking\":true}' "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
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
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-tp-size 8 "
        "--sglang-dp-size 1 "
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        "--sglang-max-running-requests 8 "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 1024 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
        "--sglang-max-total-tokens 4096 "
        "--sglang-disable-cuda-graph "
        "--sglang-disable-custom-all-reduce "
        "--sglang-schedule-conservativeness 1.0 "
        "--sglang-ep-size 8 "
        "--sglang-disable-radix-cache "
        "--sglang-mem-fraction-static 0.24 "
    )

    misc_args = (
        "--load-hf-with-mbridge "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--update-weight-buffer-size 1073741824 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--model-name deepseekv4 "
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--colocate "
        "--no-offload-train "
        "--no-offload-rollout "
        "--use-fault-tolerance "
        "--use-miles-router "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )

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
        megatron_model_type="deepseek-v4-flash-4layer",
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
        "NCCL_MIN_NCHANNELS": "112",
        "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
        "USE_ROCM": "1",
        "USE_CUDA": "0",
        "ROCM_HOME": "/opt/rocm",
        "ROCM_PATH": "/opt/rocm",
        "USE_ROCM_AITER_ROPE_BACKEND": "0",
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
        "SGLANG_ENABLE_THINKING": "1",
        "SGLANG_USE_AITER": "1",
        "SGLANG_USE_ROCM700A": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_MOE_PADDING": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
        "SGLANG_OPT_USE_FUSED_COMPRESS": "false",
        "SGLANG_OPT_USE_OLD_COMPRESSOR": "true",
        "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": "false",
        "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
        "SGLANG_OPT_USE_FUSED_HASH_TOPK": "false",
        "SGLANG_HACK_FLASHMLA_BACKEND": "torch",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
        "SGLANG_TOPK_TRANSFORM_512_TORCH": "1",
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_OPT_DPSK_V4_RADIX": "0",
        "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "false",
        "SGLANG_OPT_USE_FUSED_STORE_CACHE": "false",
        "SGLANG_FORCE_TRITON_MOE_FP8": "1",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "MILES_DSV4_CKPT_VERSION": "0415",
        "MEGATRON_USE_KV_QAT": "1",
        "MILES_HACK_TRAIN_TORCH_DETERMINISTIC": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
