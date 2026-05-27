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

    # This launcher is for a colocated 2-node bringup:
    #   two nodes worth of GPUs are shared by Megatron actor and SGLang rollout.
    # Miles colocate is synchronous (`train.py`); `train_async.py` explicitly
    # rejects --colocate.
    num_nodes: int = 2
    actor_num_nodes: int = 2
    rollout_num_nodes: int = 2

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
    train_memory_margin_bytes: int = 2 * 1024 * 1024 * 1024
    offload_train: bool = True
    offload_rollout: bool = True
    offload_rollout_level: str = "kv_cache weight"
    optimizer: Literal["adam", "sgd"] = "adam"
    main_params_dtype: Literal["fp16", "fp32"] = "fp32"
    optimizer_state_dtype: Literal["fp8", "bf16", "fp16", "fp32"] = "fp8"
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 2
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
    extra_args: str = ""


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
    env = {
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
    return env


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


def _patch_sglang_e8m0_nccl_transport_all_nodes(args: ScriptArgs) -> None:
    # RCCL/NCCL in this ROCm stack cannot broadcast torch.float8_e8m0fnu
    # tensors directly. DeepSeek-V4-Flash FP4 experts also arrive as packed
    # int8 payloads but must be viewed as torch.float4_e2m1fn_x2 before SGLang
    # copies them into its FP4 expert parameters.
    patch_code = r'''
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py")
e8m0_sentinel = "MILES_E8M0_NCCL_TRANSPORT_PATCH"
fp4_sentinel = "MILES_FP4_NCCL_TRANSPORT_PATCH"
bucket_fp4_sentinel = "MILES_BUCKET_FP4_NCCL_TRANSPORT_PATCH"
ffn_fp4_sentinel = "MILES_FFN_FP4_NCCL_TRANSPORT_PATCH"
direct_ffn_fp4_sentinel = "MILES_DIRECT_FFN_FP4_NCCL_TRANSPORT_PATCH"
tensor_bucket_ffn_fp4_sentinel = "MILES_TENSOR_BUCKET_FFN_FP4_NCCL_TRANSPORT_PATCH"
text = path.read_text()

if e8m0_sentinel not in text:
    old = """            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()
"""
    new = """            weights = []
            handles = []
            e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)  # MILES_E8M0_NCCL_TRANSPORT_PATCH
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)  # MILES_FP4_NCCL_TRANSPORT_PATCH
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                transport_dtype = torch.uint8 if target_dtype == e8m0_dtype else target_dtype
                weight = torch.empty(shape, dtype=transport_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                if transport_dtype != target_dtype:
                    weight = weight.view(target_dtype)
                if (
                    fp4_dtype is not None
                    and target_dtype in (torch.int8, torch.uint8)
                    and (
                        name.endswith(".mlp.experts.w13_weight")
                        or name.endswith(".mlp.experts.w2_weight")
                    )
                ):
                    weight = weight.view(fp4_dtype)
                weights.append((name, weight))
            for handle in handles:
                handle.wait()
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang update_weights_from_distributed block in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {e8m0_sentinel} and {fp4_sentinel}")

elif fp4_sentinel not in text:
    old = """            e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)  # MILES_E8M0_NCCL_TRANSPORT_PATCH
            for name, dtype, shape in zip(names, dtypes, shapes):
"""
    new = """            e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)  # MILES_E8M0_NCCL_TRANSPORT_PATCH
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)  # MILES_FP4_NCCL_TRANSPORT_PATCH
            for name, dtype, shape in zip(names, dtypes, shapes):
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang e8m0 dtype block in {path}")
    text = text.replace(old, new)

    old = """                if transport_dtype != target_dtype:
                    weight = weight.view(target_dtype)
                weights.append((name, weight))
"""
    new = """                if transport_dtype != target_dtype:
                    weight = weight.view(target_dtype)
                if (
                    fp4_dtype is not None
                    and target_dtype in (torch.int8, torch.uint8)
                    and (
                        name.endswith(".mlp.experts.w13_weight")
                        or name.endswith(".mlp.experts.w2_weight")
                    )
                ):
                    weight = weight.view(fp4_dtype)
                weights.append((name, weight))
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang weights append block in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {fp4_sentinel}")
else:
    print(f"{path}: {e8m0_sentinel} and {fp4_sentinel} already applied")

if bucket_fp4_sentinel not in text:
    old = """            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
"""
    new = """            reconstructed_tensors = bucket.reconstruct_tensors()
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)  # MILES_BUCKET_FP4_NCCL_TRANSPORT_PATCH
            if fp4_dtype is not None:
                reconstructed_tensors = [
                    (
                        name,
                        tensor.view(fp4_dtype)
                        if (
                            tensor.dtype in (torch.int8, torch.uint8)
                            and (
                                name.endswith(".mlp.experts.w13_weight")
                                or name.endswith(".mlp.experts.w2_weight")
                            )
                        )
                        else tensor,
                    )
                    for name, tensor in reconstructed_tensors
                ]
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang bucketed NCCL load block in {path}")
    text = text.replace(old, new)

    old = """        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"
"""
    new = """        # Load the reconstructed tensors using the standard method
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)  # MILES_BUCKET_FP4_NCCL_TRANSPORT_PATCH
        if fp4_dtype is not None:
            reconstructed_tensors = [
                (
                    name,
                    tensor.view(fp4_dtype)
                    if (
                        tensor.dtype in (torch.int8, torch.uint8)
                        and (
                            name.endswith(".mlp.experts.w13_weight")
                            or name.endswith(".mlp.experts.w2_weight")
                        )
                    )
                    else tensor,
                )
                for name, tensor in reconstructed_tensors
            ]
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang tensor bucket load block in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {bucket_fp4_sentinel}")
else:
    print(f"{path}: {bucket_fp4_sentinel} already applied")

if ffn_fp4_sentinel not in text:
    old = """                                name.endswith(".mlp.experts.w13_weight")
                                or name.endswith(".mlp.experts.w2_weight")
"""
    new = """                                name.endswith(".mlp.experts.w13_weight")
                                or name.endswith(".mlp.experts.w2_weight")
                                or name.endswith(".ffn.experts.w13_weight")
                                or name.endswith(".ffn.experts.w2_weight")  # MILES_FFN_FP4_NCCL_TRANSPORT_PATCH
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang FP4 expert suffix checks in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {ffn_fp4_sentinel}")
else:
    print(f"{path}: {ffn_fp4_sentinel} already applied")

if direct_ffn_fp4_sentinel not in text:
    old = """                        name.endswith(".mlp.experts.w13_weight")
                        or name.endswith(".mlp.experts.w2_weight")
                    )
                ):
"""
    new = """                        name.endswith(".mlp.experts.w13_weight")
                        or name.endswith(".mlp.experts.w2_weight")
                        or name.endswith(".ffn.experts.w13_weight")
                        or name.endswith(".ffn.experts.w2_weight")  # MILES_DIRECT_FFN_FP4_NCCL_TRANSPORT_PATCH
                    )
                ):
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang direct FP4 expert suffix checks in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {direct_ffn_fp4_sentinel}")
else:
    print(f"{path}: {direct_ffn_fp4_sentinel} already applied")

if tensor_bucket_ffn_fp4_sentinel not in text:
    old = """                            name.endswith(".mlp.experts.w13_weight")
                            or name.endswith(".mlp.experts.w2_weight")
                        )
                    )
                    else tensor,
"""
    new = """                            name.endswith(".mlp.experts.w13_weight")
                            or name.endswith(".mlp.experts.w2_weight")
                            or name.endswith(".ffn.experts.w13_weight")
                            or name.endswith(".ffn.experts.w2_weight")  # MILES_TENSOR_BUCKET_FFN_FP4_NCCL_TRANSPORT_PATCH
                        )
                    )
                    else tensor,
"""
    if old not in text:
        raise RuntimeError(f"Could not find SGLang tensor-bucket FP4 expert suffix checks in {path}")
    text = text.replace(old, new)
    path.write_text(text)
    print(f"{path}: applied {tensor_bucket_ffn_fp4_sentinel}")
else:
    print(f"{path}: {tensor_bucket_ffn_fp4_sentinel} already applied")

model_path = Path("/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py")
loader_fp4_sentinel = "MILES_DSV4_LOADER_FP4_EXPERT_VIEW_PATCH"
model_text = model_path.read_text()
if loader_fp4_sentinel not in model_text:
    old = """                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
"""
    new = """                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
                    if (
                        fp4_dtype is not None
                        and loaded_weight.dtype in (torch.int8, torch.uint8)
                        and (
                            name.endswith(".mlp.experts.w13_weight")
                            or name.endswith(".mlp.experts.w2_weight")
                            or name.endswith(".ffn.experts.w13_weight")
                            or name.endswith(".ffn.experts.w2_weight")
                        )
                    ):
                        loaded_weight = loaded_weight.view(fp4_dtype)  # MILES_DSV4_LOADER_FP4_EXPERT_VIEW_PATCH

                    layer_id = get_layer_id(name)
"""
    if old not in model_text:
        raise RuntimeError(f"Could not find DeepSeek-V4 loader remap block in {model_path}")
    model_text = model_text.replace(old, new)
    model_path.write_text(model_text)
    print(f"{model_path}: applied {loader_fp4_sentinel}")
else:
    print(f"{model_path}: {loader_fp4_sentinel} already applied")

fused_moe_path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py")
fused_moe_fp4_sentinel = "MILES_FUSED_MOE_FP4_EXPERT_COPY_PATCH"
fused_text = fused_moe_path.read_text()
if fused_moe_fp4_sentinel not in fused_text:
    old = """_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
"""
    new = """_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


def _miles_view_fp4_payload_if_needed(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if (
        fp4_dtype is not None
        and dst.dtype == fp4_dtype
        and src.dtype in (torch.int8, torch.uint8)
    ):
        return src.view(fp4_dtype)
    return src

# MILES_FUSED_MOE_FP4_EXPERT_COPY_PATCH
"""
    if old not in fused_text:
        raise RuntimeError(f"Could not find SGLang fused MoE globals in {fused_moe_path}")
    fused_text = fused_text.replace(old, new, 1)

    old = "expert_data.copy_(loaded_weight)"
    new = "expert_data.copy_(_miles_view_fp4_payload_if_needed(expert_data, loaded_weight))"
    if old not in fused_text:
        raise RuntimeError(f"Could not find SGLang fused MoE expert copy sites in {fused_moe_path}")
    fused_text = fused_text.replace(old, new)

    old = "param.data[:, :dim1, :dim2].copy_(loaded_weight)"
    new = (
        "param.data[:, :dim1, :dim2].copy_("
        "_miles_view_fp4_payload_if_needed(param.data[:, :dim1, :dim2], loaded_weight)"
        ")"
    )
    if old not in fused_text:
        raise RuntimeError(f"Could not find SGLang fused MoE static FP4 copy sites in {fused_moe_path}")
    fused_text = fused_text.replace(old, new)

    fused_moe_path.write_text(fused_text)
    print(f"{fused_moe_path}: applied {fused_moe_fp4_sentinel}")
else:
    print(f"{fused_moe_path}: {fused_moe_fp4_sentinel} already applied")
'''
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{patch_code}\nPY", num_nodes=args.num_nodes)


def _patch_megatron_skip_grad_norm_when_unclipped_all_nodes(args: ScriptArgs) -> None:
    # TE's fused multi-tensor l2norm has produced HIP memory faults on MI355
    # during DSV4 Flash bringup. With --clip-grad 0.0, ChainedOptimizer should
    # not need grad norm, so skip that fused path completely.
    patch_code = r'''
from pathlib import Path

path = Path("/root/Megatron-LM/megatron/core/optimizer/optimizer.py")
sentinel = "MILES_SKIP_CHAINED_GRAD_NORM_WHEN_UNCLIPPED"
text = path.read_text()
if sentinel in text:
    print(f"{path}: {sentinel} already applied")
else:
    old = """        grad_norm = self.get_grad_norm()

        # Clip gradients.
"""
    new = """        should_compute_grad_norm = any(
            not (hasattr(optimizer, 'is_stub_optimizer') and optimizer.is_stub_optimizer)
            and optimizer.config.clip_grad > 0.0
            for optimizer in self.chained_optimizers
        )
        grad_norm = self.get_grad_norm() if should_compute_grad_norm else 0.0

        # Clip gradients.  # MILES_SKIP_CHAINED_GRAD_NORM_WHEN_UNCLIPPED
"""
    if old not in text:
        raise RuntimeError(f"Could not find ChainedOptimizer grad_norm block in {path}")
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"{path}: applied {sentinel}")
'''
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{patch_code}\nPY", num_nodes=args.num_nodes)


def _patch_te_fused_adam_chunked_step_all_nodes(args: ScriptArgs) -> None:
    # TE's precision-aware Adam unscales fp8/fp16 optimizer states for the
    # whole parameter group before launching the fused update. DSV4 Flash fits
    # the persistent state on one MI355 node, but the full-group fp32 scratch
    # list leaves no headroom. Chunking preserves one optimizer step while
    # keeping the transient fp32 state bounded.
    patch_code = r'''
from pathlib import Path

path = Path("/opt/venv/lib/python3.10/site-packages/transformer_engine/pytorch/optimizers/fused_adam.py")
sentinel = "MILES_TE_FUSED_ADAM_CHUNKED_STEP_PATCH"
text = path.read_text()
if sentinel in text:
    print(f"{path}: {sentinel} already applied")
else:
    patch = r"""

# MILES_TE_FUSED_ADAM_CHUNKED_STEP_PATCH
_MILES_ORIG_FUSED_ADAM_STEP = FusedAdam.step


def _miles_clone_step_value(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _miles_restore_step(group, had_step, value):
    if had_step:
        group["step"] = _miles_clone_step_value(value)
    else:
        group.pop("step", None)


def _miles_param_chunks(params, target_elems):
    chunk = []
    elems = 0
    for param in params:
        param_elems = int(param.numel()) if hasattr(param, "numel") else 0
        if chunk and elems + param_elems > target_elems:
            yield chunk
            chunk = []
            elems = 0
        chunk.append(param)
        elems += param_elems
    if chunk:
        yield chunk


def _miles_chunked_fused_adam_step(self, closure=None, grad_scaler=None):
    import os

    target_elems = int(os.environ.get("MILES_TE_ADAM_CHUNK_ELEMS", "16000000"))
    if target_elems <= 0:
        return _MILES_ORIG_FUSED_ADAM_STEP(self, closure=closure, grad_scaler=grad_scaler)

    loss = closure() if closure is not None else None
    original_groups = self.param_groups
    original_params = {id(group): list(group["params"]) for group in original_groups}
    try:
        for group in original_groups:
            params = original_params[id(group)]
            if not params:
                continue

            had_step = "step" in group
            step_before = _miles_clone_step_value(group.get("step"))
            chunks = list(_miles_param_chunks(params, target_elems))
            if len(chunks) <= 1:
                self.param_groups = [group]
                _miles_restore_step(group, had_step, step_before)
                _MILES_ORIG_FUSED_ADAM_STEP(self, closure=None, grad_scaler=grad_scaler)
                continue

            for chunk in chunks:
                group["params"] = chunk
                self.param_groups = [group]
                _miles_restore_step(group, had_step, step_before)
                _MILES_ORIG_FUSED_ADAM_STEP(self, closure=None, grad_scaler=grad_scaler)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            group["params"] = params
    finally:
        self.param_groups = original_groups
        for group in original_groups:
            group["params"] = original_params[id(group)]
    return loss


FusedAdam.step = _miles_chunked_fused_adam_step
"""
    text = text + patch
    path.write_text(text)
    print(f"{path}: applied {sentinel}")
'''
    with _without_ray_address():
        U.exec_command_all_ray_node(f"python3 - <<'PY'\n{patch_code}\nPY", num_nodes=args.num_nodes)


def _prepare(args: ScriptArgs) -> None:
    _validate_layout(args)

    with _without_ray_address():
        U.exec_command_all_ray_node(f"mkdir -p {args.model_dir} {args.data_dir}", num_nodes=args.num_nodes)

    _patch_sglang_e8m0_nccl_transport_all_nodes(args)
    _patch_megatron_skip_grad_norm_when_unclipped_all_nodes(args)
    _patch_te_fused_adam_chunked_step_all_nodes(args)

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
    if args.num_nodes != 2:
        raise ValueError("This launcher is intentionally scoped to exactly 2 Ray nodes.")
    if args.actor_num_nodes != 2 or args.rollout_num_nodes != 2:
        raise ValueError("This colocate launcher expects actor and rollout to share both nodes.")
    if args.num_gpus_per_node != 8:
        raise ValueError("DeepSeek-V4-Flash MI355 2-node colocate bringup expects exactly 8 GPUs per node.")
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError(
            "Two-node runs must use an already-running Ray cluster. Start Ray on both nodes, "
            "then run this launcher on the head with MILES_SCRIPT_EXTERNAL_RAY=1."
        )
    if args.update_weight_transfer_mode != "broadcast":
        raise ValueError("AMD MI355 2-node colocate bringup uses NCCL broadcast weight sync only.")
    bad_levels = set(args.offload_rollout_level.split()) - {"kv_cache", "weight"}
    if bad_levels:
        raise ValueError(f"Unsupported offload_rollout_level entries: {sorted(bad_levels)}")
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
    rollout_batch_size = _pick(args, "rollout_batch_size", 2, 4)
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
    return {
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


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _prepare(args)
    _execute(args)


if __name__ == "__main__":
    typer.run(main)
