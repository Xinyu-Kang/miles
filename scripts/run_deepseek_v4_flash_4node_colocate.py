import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


_DEFAULT_AITER_CONFIG_FMOE = Path(__file__).resolve().parent / "amd" / "dsv4_flash_fp4_tp8_fmoe.csv"
_FULLY_ASYNC_DIR = Path(__file__).resolve().parents[1] / "examples" / "fully_async"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()

    # This launcher is for a colocated 4-node bringup:
    #   four nodes worth of GPUs are shared by Megatron actor and SGLang rollout.
    # Miles colocate is synchronous (`train.py`); `train_async.py` explicitly
    # rejects --colocate.
    num_nodes: int = 4
    actor_num_nodes: int = 4
    rollout_num_nodes: int = 4

    hf_checkpoint: str = "deepseek-ai/DeepSeek-V4-Flash"
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V4-Flash"
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    num_rollout: int | None = 1
    rollout_batch_size: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_max_response_len: int | None = None
    num_steps_per_rollout: int | None = None
    micro_batch_size: int = 1
    max_tokens_per_gpu: int | None = None
    log_probs_max_tokens_per_gpu: int | None = None
    context_length: int | None = None

    sglang_mem_fraction_static: float | None = None
    sglang_max_running_requests: int | None = None
    sglang_max_total_tokens: int | None = None
    sglang_disable_cuda_graph: bool = True
    pause_generation_mode: Literal["in_place", "retract", "abort"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast"] = "broadcast"
    max_weight_staleness: int | None = None
    use_tis: bool = False

    accumulate_allreduce_grads_in_fp32: bool = False
    # Keep enough slack for ROCm allocator spikes without artificially
    # recreating the 2-node optimizer-margin OOM.
    train_memory_margin_bytes: int = 16 * 1024 * 1024 * 1024
    offload_train: bool = True
    offload_rollout: bool = True
    offload_rollout_level: str = "kv_cache weight"
    optimizer: Literal["adam", "sgd"] = "adam"
    # ROCm7 precision-aware Adam currently trips HSA memory faults in this DSV4
    # path. The 4-node script instead relies on DP=4 plus optimizer-state
    # offload to keep non-precision-aware Adam within memory.
    precision_aware_optimizer: bool = False
    main_params_dtype: Literal["fp16", "fp32"] = "fp32"
    optimizer_state_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    num_layers: int = 43
    decoder_first_pipeline_num_layers: int | None = None
    decoder_last_pipeline_num_layers: int | None = None

    aiter_config_fmoe: str = str(_DEFAULT_AITER_CONFIG_FMOE)
    download_model: bool = False
    download_data: bool = True
    skip_saving: bool = True
    enable_eval: bool = False
    wait_for_ray_gpus: bool = True
    ray_wait_timeout_secs: int = 900
    extra_args: str = "--offload-optimizer-states"


def _pick(args: ScriptArgs, name: str, debug_value, normal_value):
    value = getattr(args, name)
    if value is not None:
        return value
    return debug_value if args.mode == "debug_minimal" else normal_value


def _is_local_path(value: str) -> bool:
    return value.startswith("/") or value.startswith(".")


@contextmanager
def _without_ray_address():
    # RAY_ADDRESS=http://... is useful for `ray job submit`, but Ray's Python API
    # also reads it and then treats "http" as a Ray Client scheme.
    old_value = os.environ.pop("RAY_ADDRESS", None)
    try:
        yield
    finally:
        if old_value is not None:
            os.environ["RAY_ADDRESS"] = old_value


def _run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _detect_socket_ifname() -> str:
    if os.environ.get("NCCL_SOCKET_IFNAME"):
        return os.environ["NCCL_SOCKET_IFNAME"]

    target_ip = None
    for env_name in ("RAY_NODE_IP", "MASTER_ADDR"):
        value = os.environ.get(env_name)
        if value and value not in {"localhost", "127.0.0.1"}:
            target_ip = value
            break

    if target_ip:
        for line in _run_text(["ip", "-o", "-4", "addr", "show", "scope", "global"]).splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[3].split("/", 1)[0] == target_ip:
                return parts[1].split("@", 1)[0]

    route = _run_text(["ip", "-o", "-4", "route", "show", "to", "default"]).split()
    if "dev" in route:
        return route[route.index("dev") + 1]
    return "eth0"


def _detect_nccl_ib_hca() -> str:
    if os.environ.get("NCCL_IB_HCA"):
        return os.environ["NCCL_IB_HCA"]

    base = Path("/sys/class/infiniband")
    if not base.exists():
        return ""

    hcas = []
    for dev in sorted(path for path in base.iterdir() if path.is_dir()):
        ndev_path = dev / "ports" / "1" / "gid_attrs" / "ndevs" / "0"
        ndev = _read_optional(ndev_path) if ndev_path.exists() else ""
        if dev.name.startswith("rdma") or ndev.startswith(("rdma", "tw-eth")):
            hcas.append(dev.name)
    return ",".join(hcas)


def _read_optional(path: Path) -> str:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _gid_is_nonzero(gid: str) -> bool:
    return bool(gid) and gid != "0000:0000:0000:0000:0000:0000:0000:0000"


def _detect_roce_gid_index() -> str:
    if os.environ.get("NCCL_IB_GID_INDEX"):
        return os.environ["NCCL_IB_GID_INDEX"]

    base = Path("/sys/class/infiniband")
    if not base.exists():
        return "1"

    first_nonzero = None
    for dev in sorted(path for path in base.iterdir() if path.is_dir()):
        gids_dir = dev / "ports" / "1" / "gids"
        if not gids_dir.exists():
            continue
        gid_files = sorted(
            (path for path in gids_dir.iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for gid_file in gid_files:
            gid = _read_optional(gid_file)
            if not _gid_is_nonzero(gid):
                continue
            gid_type = _read_optional(dev / "ports" / "1" / "gid_attrs" / "types" / gid_file.name)
            if first_nonzero is None:
                first_nonzero = gid_file.name
            if gid_type == "RoCE v2" and ":ffff:" in gid:
                return gid_file.name
    return first_nonzero or "1"


def _nccl_multinode_env() -> dict[str, str]:
    socket_ifname = _detect_socket_ifname()
    return {
        "NCCL_SOCKET_IFNAME": socket_ifname,
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", socket_ifname),
        "TP_SOCKET_IFNAME": os.environ.get("TP_SOCKET_IFNAME", socket_ifname),
        "NCCL_IB_HCA": _detect_nccl_ib_hca(),
        "NCCL_IB_GID_INDEX": _detect_roce_gid_index(),
        "NCCL_IB_TC": os.environ.get("NCCL_IB_TC", "160"),
        "NCCL_IB_TIMEOUT": os.environ.get("NCCL_IB_TIMEOUT", "22"),
        "NCCL_IB_RETRY_CNT": os.environ.get("NCCL_IB_RETRY_CNT", "7"),
        "NCCL_IB_QPS_PER_CONNECTION": os.environ.get("NCCL_IB_QPS_PER_CONNECTION", "8"),
        "NCCL_PXN_DISABLE": os.environ.get("NCCL_PXN_DISABLE", "0"),
        "NCCL_NET_GDR_LEVEL": os.environ.get("NCCL_NET_GDR_LEVEL", "0"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "VERSION"),
    }
    print(
        "NCCL/RDMA env: "
        f"NCCL_SOCKET_IFNAME={env['NCCL_SOCKET_IFNAME']} "
        f"NCCL_IB_HCA={env['NCCL_IB_HCA'] or '<none>'} "
        f"NCCL_IB_GID_INDEX={env['NCCL_IB_GID_INDEX']}",
        flush=True,
    )


def _download_model_all_nodes(model_id: str, local_dir: str, args: ScriptArgs) -> None:
    with _without_ray_address():
        U.exec_command_all_ray_node(
            f"mkdir -p {Path(local_dir).parent} && "
            f"if [ ! -d {local_dir} ]; then "
            f"hf download {model_id} --local-dir {local_dir}; "
            f"else echo 'Model already exists at {local_dir}'; "
            f"fi",
            num_nodes=args.num_nodes,
        )


def _download_dataset_all_nodes(full_name: str, args: ScriptArgs) -> None:
    _, partial_name = full_name.split("/")
    with _without_ray_address():
        U.exec_command_all_ray_node(
            f"mkdir -p {args.data_dir} && "
            f"hf download --repo-type dataset {full_name} --local-dir {args.data_dir}/{partial_name}",
            num_nodes=args.num_nodes,
        )


def _verify_container_runtime_patches(args: ScriptArgs) -> None:
    check_code = r'''
from pathlib import Path

checks = [
    (
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py",
        "MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_METHOD_PATCH",
        "SGLang online post_process_weights process-only hook",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py",
        "MILES_FP4_NCCL_TRANSPORT_PATCH",
        "SGLang FP4 NCCL transport view",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py",
        "MILES_BUCKET_FP4_NCCL_TRANSPORT_PATCH",
        "SGLang bucketed FP4 NCCL transport view",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py",
        "MILES_TENSOR_BUCKET_FFN_FP4_NCCL_TRANSPORT_PATCH",
        "SGLang FFN FP4 tensor-bucket transport view",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py",
        "MILES_DSV4_LOADER_FP4_EXPERT_VIEW_PATCH",
        "DeepSeek-V4 FP4 expert loader view",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
        "MILES_FUSED_MOE_FP4_EXPERT_COPY_PATCH",
        "SGLang FusedMoE FP4 payload copy",
    ),
    (
        "/sgl-workspace/sglang/python/sglang/srt/utils/weight_checker.py",
        "MILES_WEIGHT_CHECKER_FP4_FINAL_SKIP_PATCH",
        "SGLang FP4-aware weight checker",
    ),
]

missing = []
for filename, sentinel, label in checks:
    path = Path(filename)
    try:
        text = path.read_text()
    except OSError as exc:
        missing.append(f"{label}: cannot read {filename}: {exc}")
        continue
    if sentinel not in text:
        missing.append(f"{label}: missing {sentinel} in {filename}")

if missing:
    details = "\n".join(f"  - {item}" for item in missing)
    raise RuntimeError(
        "This container is missing DeepSeek-V4-Flash build-time patches. "
        "Rebuild from miles/docker/Dockerfile.rocm_MI350-5_DSV4 on every node.\n"
        + details
    )

print("DeepSeek-V4-Flash build-time patches verified")
'''
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{check_code}\nPY", num_nodes=args.num_nodes)


def _patch_sglang_tokenizer_compat(args: ScriptArgs) -> None:
    # Transformers 5.6 tries AutoConfig before honoring tokenizer_config.json.
    # DeepSeek-V4 is served by SGLang's plain config shim, so AutoTokenizer can
    # fall back to PreTrainedConfig and trip on RoPE standardization. Load the
    # declared fast tokenizer directly for that exact compatibility failure.
    patch_code = r"""
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py")
text = path.read_text()

auto_marker = "MILES_DSV4_TOKENIZER_AUTO_ATTR_FALLBACK_PATCH"
resolve_marker = "MILES_DSV4_TOKENIZER_RESOLVE_ATTR_FALLBACK_PATCH"
changed = False

old_auto = '''        return tokenizer
    except TypeError as e:
'''
new_auto = '''        return tokenizer
    except AttributeError as e:
        if "max_position_embeddings" in str(e):
            tokenizer = _load_tokenizer_by_declared_class(
                tokenizer_name, *args, **common_kwargs
            )
            if tokenizer is not None:
                logging.getLogger(tokenizer.__class__.__module__).addFilter(
                    TokenizerWarningsFilter()
                )
                return tokenizer
        raise  # MILES_DSV4_TOKENIZER_AUTO_ATTR_FALLBACK_PATCH
    except TypeError as e:
'''
if auto_marker not in text:
    if old_auto in text:
        text = text.replace(old_auto, new_auto, 1)
        changed = True
    elif "max_position_embeddings" not in text:
        raise RuntimeError("Could not patch SGLang AutoTokenizer AttributeError fallback")

old_resolve = '''    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
'''
new_resolve = '''    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
    except AttributeError as e:
        if "max_position_embeddings" in str(e):
            tokenizer = _load_tokenizer_by_declared_class(
                tokenizer_name, *args, **common_kwargs
            )
            if tokenizer is not None:
                return tokenizer
        raise  # MILES_DSV4_TOKENIZER_RESOLVE_ATTR_FALLBACK_PATCH
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
'''
if resolve_marker not in text:
    if old_resolve in text:
        text = text.replace(old_resolve, new_resolve, 1)
        changed = True
    elif "max_position_embeddings" not in text:
        raise RuntimeError("Could not patch SGLang TokenizersBackend retry AttributeError fallback")

if changed:
    path.write_text(text)
    print("SGLang DeepSeek-V4 tokenizer compatibility patch applied")
else:
    print("SGLang DeepSeek-V4 tokenizer compatibility patch already present")
"""
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{patch_code}\nPY", num_nodes=args.num_nodes)


def _patch_tensor_backuper_noop(args: ScriptArgs) -> None:
    # When --disable-weights-backuper is used, TensorBackuperNoop should not
    # hash the live model during backup/restore. DSV4 ranks are large enough
    # that the hash path can allocate hundreds of MiB after wake-up and trip the
    # train memory margin before the optimizer step.
    patch_code = r"""
from pathlib import Path

path = Path("/workspace/miles/miles/utils/tensor_backper.py")
text = path.read_text()

marker = "MILES_TENSOR_BACKUPER_NOOP_SKIP_HASH_PATCH"
old = '''class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        torch.cuda.synchronize()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        torch.cuda.synchronize()
'''
new = '''class _TensorBackuperNoop(TensorBackuper):
    # MILES_TENSOR_BACKUPER_NOOP_SKIP_HASH_PATCH
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        assert tag == self._single_tag
        return {k: v.detach() for k, v in self._source_getter()}

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
'''

if marker in text:
    print("TensorBackuperNoop hash-skip patch already present")
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print("TensorBackuperNoop hash-skip patch applied")
elif "return {k: v.detach() for k, v in self._source_getter()}" in text:
    path.write_text(text.replace("class _TensorBackuperNoop(TensorBackuper):", new.splitlines()[0] + "\n    # " + marker, 1))
    print("TensorBackuperNoop hash-skip patch marker added")
else:
    raise RuntimeError("Could not patch TensorBackuperNoop hash path")
"""
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{patch_code}\nPY", num_nodes=args.num_nodes)


def _prepare(args: ScriptArgs) -> None:
    _validate_layout(args)

    with _without_ray_address():
        U.exec_command_all_ray_node(f"mkdir -p {args.model_dir} {args.data_dir}", num_nodes=args.num_nodes)

    _verify_container_runtime_patches(args)
    _patch_sglang_tokenizer_compat(args)
    _patch_tensor_backuper_noop(args)

    if args.download_model:
        local_checkpoint = Path(args.model_dir) / args.model_name
        _download_model_all_nodes(f"{args.model_org}/{args.model_name}", str(local_checkpoint), args)
        args.hf_checkpoint = str(local_checkpoint)
    elif not _is_local_path(args.hf_checkpoint):
        local_checkpoint = Path(args.model_dir) / args.model_name
        _download_model_all_nodes(args.hf_checkpoint, str(local_checkpoint), args)
        args.hf_checkpoint = str(local_checkpoint)
    else:
        _download_model_all_nodes(f"{args.model_org}/{args.model_name}", args.hf_checkpoint, args)

    if args.download_data:
        _download_dataset_all_nodes("zhuzilin/dapo-math-17k", args)
        if args.enable_eval:
            _download_dataset_all_nodes("zhuzilin/aime-2024", args)

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


def _validate_layout(args: ScriptArgs) -> None:
    if args.num_nodes != 4:
        raise ValueError("This launcher is intentionally scoped to exactly 4 Ray nodes.")
    if args.actor_num_nodes != 4 or args.rollout_num_nodes != 4:
        raise ValueError("This colocate launcher expects actor and rollout to share all four nodes.")
    if args.num_gpus_per_node != 8:
        raise ValueError("DeepSeek-V4-Flash MI355 4-node colocate bringup expects exactly 8 GPUs per node.")
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError(
            "Four-node runs must use an already-running Ray cluster. Start Ray on all nodes, "
            "then run this launcher on the head with MILES_SCRIPT_EXTERNAL_RAY=1."
        )
    if args.update_weight_transfer_mode != "broadcast":
        raise ValueError("AMD MI355 4-node colocate bringup uses NCCL broadcast weight sync only.")
    bad_levels = set(args.offload_rollout_level.split()) - {"kv_cache", "weight"}
    if bad_levels:
        raise ValueError(f"Unsupported offload_rollout_level entries: {sorted(bad_levels)}")
    # SGLangEngine enables weights CPU backup whenever rollout weight offload is active.
    if args.pipeline_model_parallel_size != 1:
        raise ValueError(
            "ROCm7 DeepSeek-V4-Flash Megatron PP>1 currently gives incorrect train/rollout logprobs. "
            "Use pipeline_model_parallel_size=1 for the 4-node correctness path."
        )
    train_world_size = args.actor_num_nodes * args.num_gpus_per_node
    model_parallel_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    if train_world_size % model_parallel_size != 0:
        raise ValueError(
            "actor world size must be divisible by tensor*pipeline*context parallel size: "
            f"world={train_world_size}, tp={args.tensor_model_parallel_size}, "
            f"pp={args.pipeline_model_parallel_size}, cp={args.context_parallel_size}."
        )
    if 256 % args.expert_model_parallel_size != 0:
        raise ValueError(
            "DeepSeek-V4-Flash has 256 routed experts; expert_model_parallel_size must divide 256."
        )
    expert_model_pipeline_size = (
        args.expert_tensor_parallel_size * args.expert_model_parallel_size * args.pipeline_model_parallel_size
    )
    if train_world_size % expert_model_pipeline_size != 0:
        raise ValueError(
            "actor world size must be divisible by expert_tensor*expert*pipeline parallel size: "
            f"world={train_world_size}, etp={args.expert_tensor_parallel_size}, "
            f"ep={args.expert_model_parallel_size}, pp={args.pipeline_model_parallel_size}."
        )
    _resolve_uneven_pipeline_split(args)


def _resolve_uneven_pipeline_split(args: ScriptArgs) -> None:
    if args.pipeline_model_parallel_size <= 1:
        return
    if args.decoder_first_pipeline_num_layers is not None or args.decoder_last_pipeline_num_layers is not None:
        return
    if args.num_layers % args.pipeline_model_parallel_size == 0:
        return

    layers_per_early_stage = (args.num_layers + args.pipeline_model_parallel_size - 1) // args.pipeline_model_parallel_size
    args.decoder_last_pipeline_num_layers = args.num_layers - layers_per_early_stage * (
        args.pipeline_model_parallel_size - 1
    )
    if args.decoder_last_pipeline_num_layers <= 0:
        raise ValueError(
            f"Cannot compute uneven pipeline split for num_layers={args.num_layers}, "
            f"pipeline_model_parallel_size={args.pipeline_model_parallel_size}."
        )


def _wait_for_ray_gpus(args: ScriptArgs) -> None:
    if not args.wait_for_ray_gpus:
        return

    import ray

    expected = args.num_nodes * args.num_gpus_per_node
    deadline = time.time() + args.ray_wait_timeout_secs
    with _without_ray_address():
        ray.init(address="auto", ignore_reinit_error=True)
        try:
            while True:
                available = int(ray.cluster_resources().get("GPU", 0))
                print(f"Waiting for Ray GPUs: {available}/{expected}", flush=True)
                if available >= expected:
                    return
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for {expected} Ray GPUs. Only saw {available}. "
                        "Check that the worker joined the head Ray cluster."
                    )
                time.sleep(5)
        finally:
            ray.shutdown()


def _execute(args: ScriptArgs) -> None:
    _validate_layout(args)

    checkpoint = args.hf_checkpoint
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    rollout_num_gpus = args.rollout_num_nodes * args.num_gpus_per_node

    ckpt_args = (
        f"--hf-checkpoint {checkpoint} "
        # Miles raw checkpoint validation uses ref_load as the initial HF load path
        # when --load is omitted. KL/ref computation is still disabled below.
        f"--ref-load {checkpoint} "
    )
    if not args.skip_saving:
        ckpt_args += f"--save {load_save_path} --save-interval 20 --save-retain-interval 20 "

    num_rollout = _pick(args, "num_rollout", 1, 300)
    rollout_batch_size = _pick(args, "rollout_batch_size", 4, 4)
    n_samples = _pick(args, "n_samples_per_prompt", 1, 4)
    response_len = _pick(args, "rollout_max_response_len", 64, 4096)
    num_steps = _pick(args, "num_steps_per_rollout", 1, 1)
    context_length = args.context_length or (512 if args.mode == "debug_minimal" else 8192)
    max_tokens_per_gpu = _pick(args, "max_tokens_per_gpu", 512, 2048)
    log_probs_max_tokens_per_gpu = args.log_probs_max_tokens_per_gpu or max_tokens_per_gpu
    actor_dp_size = (
        args.actor_num_nodes
        * args.num_gpus_per_node
        // (
            args.tensor_model_parallel_size
            * args.pipeline_model_parallel_size
            * args.context_parallel_size
        )
    )
    if rollout_batch_size % (args.micro_batch_size * actor_dp_size) != 0:
        raise ValueError(
            "Megatron global batch must be divisible by micro_batch_size * data_parallel_size: "
            f"rollout_batch_size={rollout_batch_size}, micro_batch_size={args.micro_batch_size}, "
            f"data_parallel_size={actor_dp_size}."
        )

    staleness_args = ""
    if args.max_weight_staleness is not None:
        staleness_args = f"--max-weight-staleness {args.max_weight_staleness} "

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
        f"{staleness_args}"
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
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        f"--expert-tensor-parallel-size {args.expert_tensor_parallel_size} "
        f"--seq-length {context_length} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        f"--micro-batch-size {args.micro_batch_size} "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        f"--log-probs-max-tokens-per-gpu {log_probs_max_tokens_per_gpu} "
    )
    if args.decoder_first_pipeline_num_layers is not None:
        perf_args += f"--decoder-first-pipeline-num-layers {args.decoder_first_pipeline_num_layers} "
    if args.decoder_last_pipeline_num_layers is not None:
        perf_args += f"--decoder-last-pipeline-num-layers {args.decoder_last_pipeline_num_layers} "

    if args.optimizer == "adam":
        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
            "--clip-grad 0.0 "
        )
        if args.precision_aware_optimizer:
            optimizer_args += (
                "--use-precision-aware-optimizer "
                "--main-grads-dtype bf16 "
                f"--main-params-dtype {args.main_params_dtype} "
                f"--exp-avg-dtype {args.optimizer_state_dtype} "
                f"--exp-avg-sq-dtype {args.optimizer_state_dtype} "
            )
    else:
        optimizer_args = (
            "--optimizer sgd "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--clip-grad 0.0 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if args.use_tis:
        grpo_args += "--use-tis "

    sglang_mem_fraction = _pick(args, "sglang_mem_fraction_static", 0.50, 0.70)
    sglang_max_running = _pick(args, "sglang_max_running_requests", 2, 8)
    sglang_max_total_tokens = _pick(args, "sglang_max_total_tokens", 2048, 32768)
    sglang_cuda_graph_args = (
        "--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph "
        if args.sglang_disable_cuda_graph
        else ""
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        f"--rollout-num-gpus {rollout_num_gpus} "
        "--sglang-tp-size 8 "
        "--sglang-dp-size 1 "
        # Keep rollout on the validated DeepSeek-V4-Flash SGLang path: TP=8 only.
        # Megatron still uses EP=8 for training above.
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        f"--sglang-max-running-requests {sglang_max_running} "
        f"--sglang-chunked-prefill-size {context_length} "
        "--sglang-server-concurrency 1024 "
        f"--sglang-context-length {context_length} "
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
        f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--model-name deepseekv4 "
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
        "--colocate "
        "--use-fault-tolerance "
        "--use-miles-router "
        "--disable-weights-backuper "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    if args.offload_train:
        misc_args += "--offload-train "
    else:
        misc_args += "--no-offload-train "
    if args.offload_rollout:
        misc_args += "--offload-rollout "
        misc_args += f"--offload-rollout-level {args.offload_rollout_level} "
    else:
        misc_args += "--no-offload-rollout "
    if args.accumulate_allreduce_grads_in_fp32:
        misc_args += "--accumulate-allreduce-grads-in-fp32 "

    extra_args = args.extra_args
    if (
        args.optimizer == "adam"
        and not args.precision_aware_optimizer
        and "--offload-optimizer-states" not in extra_args
    ):
        extra_args = f"{extra_args} --offload-optimizer-states".strip()

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
        f"{extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="deepseek-v4-flash",
        train_script="train.py",
        before_ray_job_submit=lambda: _wait_for_ray_gpus(args),
        extra_env_vars=_extra_env(args),
        megatron_path=args.megatron_path,
    )


def _extra_env(args: ScriptArgs) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        path
        for path in (
            str(U.repo_base_dir),
            str(_FULLY_ASYNC_DIR),
            args.megatron_path,
            os.environ.get("PYTHONPATH", ""),
        )
        if path
    )
    visible_devices = ",".join(str(i) for i in range(args.num_gpus_per_node))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or f"localhost,127.0.0.1,0.0.0.0,{master_addr}"
    env = {
        "PYTHONPATH": pythonpath,
        "MASTER_ADDR": master_addr,
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
        "HIP_VISIBLE_DEVICES": visible_devices,
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "HIP_FORCE_DEV_KERNARG": "1",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "0",
        **_nccl_multinode_env(),
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
        "MILES_TE_ADAM_CHUNK_ELEMS": os.environ.get("MILES_TE_ADAM_CHUNK_ELEMS", "8000000"),
        "AITER_CONFIG_FMOE": _resolve_aiter_config(args),
        "AITER_BF16_FP8_MOE_BOUND": "0",
        "MC_TRANSFER_TIMEOUT": "300",
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
        "SGLANG_DSV4_MODE": "2604",
        "SGLANG_DSV4_2604_SUBMODE": "2604B",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_ENABLE_THINKING": "1",
        "SGLANG_USE_AITER": "1",
        "SGLANG_USE_ROCM700A": "1",
        "SGLANG_MOE_PADDING": "1",
        "SGLANG_SET_CPU_AFFINITY": "0",
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
    return env


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
