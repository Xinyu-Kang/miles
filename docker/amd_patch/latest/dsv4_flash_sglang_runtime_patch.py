"""Build-time SGLang patches needed by DeepSeek-V4-Flash MI355 rollout.

These were originally applied by scripts/run_deepseek_v4_flash_2node_colocate.py
on every Ray node. Keep them in the image so online weight update behavior is
identical on head and worker before Miles starts.
"""

from pathlib import Path


def replace_once(text: str, old: str, new: str, message: str) -> str:
    if old not in text:
        raise RuntimeError(message)
    return text.replace(old, new, 1)


def patch_model_runner() -> None:
    path = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py")
    e8m0_sentinel = "MILES_E8M0_NCCL_TRANSPORT_PATCH"
    fp4_sentinel = "MILES_FP4_NCCL_TRANSPORT_PATCH"
    bucket_fp4_sentinel = "MILES_BUCKET_FP4_NCCL_TRANSPORT_PATCH"
    ffn_fp4_sentinel = "MILES_FFN_FP4_NCCL_TRANSPORT_PATCH"
    direct_ffn_fp4_sentinel = "MILES_DIRECT_FFN_FP4_NCCL_TRANSPORT_PATCH"
    tensor_bucket_ffn_fp4_sentinel = "MILES_TENSOR_BUCKET_FFN_FP4_NCCL_TRANSPORT_PATCH"
    online_postprocess_sentinel = "MILES_ONLINE_POSTPROCESS_PROCESS_WITHOUT_RESTORE_PATCH"
    online_postprocess_kv_skip_sentinel = "MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_PATCH"
    online_postprocess_kv_method_skip_sentinel = (
        "MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_METHOD_PATCH"
    )
    text = path.read_text()

    if online_postprocess_kv_method_skip_sentinel not in text:
        old_original = """        def supports_online_weight_post_process(quant_method):
            return (
                quant_method is not None
                and hasattr(quant_method, "restore_weights_before_loading")
                and hasattr(quant_method, "process_weights_after_loading")
            )

        if recv_req.restore_weights_before_load:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_post_process(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.restore_weights_before_loading(module)

        if recv_req.post_process_quantization:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_post_process(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
"""
        old_process_without_restore = """        def supports_online_weight_restore(quant_method):
            return quant_method is not None and hasattr(
                quant_method, "restore_weights_before_loading"
            )

        def supports_online_weight_process(quant_method):
            return quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            )

        if recv_req.restore_weights_before_load:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_restore(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.restore_weights_before_loading(module)

        if recv_req.post_process_quantization:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_process(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

        # MILES_ONLINE_POSTPROCESS_PROCESS_WITHOUT_RESTORE_PATCH
"""
        old_kv_class_skip = """        def supports_online_weight_restore(quant_method):
            return quant_method is not None and hasattr(
                quant_method, "restore_weights_before_loading"
            )

        def supports_online_weight_process(quant_method):
            if quant_method is None or not hasattr(
                quant_method, "process_weights_after_loading"
            ):
                return False
            # KV-cache quantization has a post-load hook for static cache
            # scales, but online weight updates should not re-run it.
            if quant_method.__class__.__module__.endswith(".kv_cache"):
                return False
            return True

        if recv_req.restore_weights_before_load:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_restore(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.restore_weights_before_loading(module)

        if recv_req.post_process_quantization:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_process(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

        # MILES_ONLINE_POSTPROCESS_PROCESS_WITHOUT_RESTORE_PATCH
        # MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_PATCH
"""
        new = """        def supports_online_weight_restore(quant_method):
            return quant_method is not None and hasattr(
                quant_method, "restore_weights_before_loading"
            )

        def supports_online_weight_process(quant_method):
            hook = getattr(quant_method, "process_weights_after_loading", None)
            if quant_method is None or hook is None:
                return False
            # KV-cache quantization has a post-load hook for static cache
            # scales, but online weight updates should not re-run it.
            quant_class = quant_method.__class__
            quant_class_module = getattr(quant_class, "__module__", "")
            quant_class_name = getattr(quant_class, "__name__", "")
            hook_func = getattr(hook, "__func__", hook)
            hook_module = getattr(hook_func, "__module__", "")
            if (
                "kv_cache" in quant_class_module
                or "kv_cache" in hook_module
                or "KVCache" in quant_class_name
            ):
                return False
            return True

        if recv_req.restore_weights_before_load:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_restore(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.restore_weights_before_loading(module)

        if recv_req.post_process_quantization:
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if supports_online_weight_process(quant_method):
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

        # MILES_ONLINE_POSTPROCESS_PROCESS_WITHOUT_RESTORE_PATCH
        # MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_PATCH
        # MILES_ONLINE_POSTPROCESS_SKIP_KV_CACHE_METHOD_PATCH
"""
        if old_original in text:
            text = text.replace(old_original, new, 1)
        elif old_process_without_restore in text:
            text = text.replace(old_process_without_restore, new, 1)
        elif old_kv_class_skip in text:
            text = text.replace(old_kv_class_skip, new, 1)
        else:
            raise RuntimeError(f"Could not find SGLang online post_process_weights block in {path}")
        path.write_text(text)
        print(
            f"{path}: applied {online_postprocess_sentinel}, "
            f"{online_postprocess_kv_skip_sentinel}, and "
            f"{online_postprocess_kv_method_skip_sentinel}"
        )
    else:
        print(
            f"{path}: {online_postprocess_sentinel}, "
            f"{online_postprocess_kv_skip_sentinel}, and "
            f"{online_postprocess_kv_method_skip_sentinel} already applied"
        )

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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang update_weights_from_distributed block in {path}",
        )
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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang e8m0 dtype block in {path}",
        )

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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang weights append block in {path}",
        )
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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang bucketed NCCL load block in {path}",
        )

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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang tensor bucket load block in {path}",
        )
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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang FP4 expert suffix checks in {path}",
        )
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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang direct FP4 expert suffix checks in {path}",
        )
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
        text = replace_once(
            text,
            old,
            new,
            f"Could not find SGLang tensor-bucket FP4 expert suffix checks in {path}",
        )
        path.write_text(text)
        print(f"{path}: applied {tensor_bucket_ffn_fp4_sentinel}")
    else:
        print(f"{path}: {tensor_bucket_ffn_fp4_sentinel} already applied")


def patch_deepseek_v4_loader() -> None:
    path = Path("/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py")
    sentinel = "MILES_DSV4_LOADER_FP4_EXPERT_VIEW_PATCH"
    text = path.read_text()
    if sentinel in text:
        print(f"{path}: {sentinel} already applied")
        return

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
    text = replace_once(
        text,
        old,
        new,
        f"Could not find DeepSeek-V4 loader remap block in {path}",
    )
    path.write_text(text)
    print(f"{path}: applied {sentinel}")


def patch_fused_moe_layer() -> None:
    path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py")
    sentinel = "MILES_FUSED_MOE_FP4_EXPERT_COPY_PATCH"
    text = path.read_text()
    if sentinel in text:
        print(f"{path}: {sentinel} already applied")
        return

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
    text = replace_once(
        text,
        old,
        new,
        f"Could not find SGLang fused MoE globals in {path}",
    )

    old = "expert_data.copy_(loaded_weight)"
    new = "expert_data.copy_(_miles_view_fp4_payload_if_needed(expert_data, loaded_weight))"
    if old not in text:
        raise RuntimeError(f"Could not find SGLang fused MoE expert copy sites in {path}")
    text = text.replace(old, new)

    old = "param.data[:, :dim1, :dim2].copy_(loaded_weight)"
    new = (
        "param.data[:, :dim1, :dim2].copy_("
        "_miles_view_fp4_payload_if_needed(param.data[:, :dim1, :dim2], loaded_weight)"
        ")"
    )
    if old not in text:
        raise RuntimeError(f"Could not find SGLang fused MoE static FP4 copy sites in {path}")
    text = text.replace(old, new)

    path.write_text(text)
    print(f"{path}: applied {sentinel}")


def patch_weight_checker() -> None:
    path = Path("/sgl-workspace/sglang/python/sglang/srt/utils/weight_checker.py")
    sentinel = "MILES_WEIGHT_CHECKER_FP4_BYTE_VIEW_PATCH"
    text = path.read_text()
    if sentinel not in text:
        old = """def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"
"""
        new = """def _miles_is_fp4_tensor(t: torch.Tensor) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return fp4_dtype is not None and t.dtype == fp4_dtype


def _miles_byte_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8)


def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"

# MILES_WEIGHT_CHECKER_FP4_BYTE_VIEW_PATCH
"""
        text = replace_once(
            text,
            old,
            new,
            f"Could not find WeightChecker hash helper in {path}",
        )

        old = """    def _reset_tensors(self):
        for name, param in self._model_state():
            if _is_non_persistent_buffer_name(name):
                continue
            param.copy_(_random_like(param))
"""
        new = """    def _reset_tensors(self):
        for name, param in self._model_state():
            if _is_non_persistent_buffer_name(name):
                continue
            if _miles_is_fp4_tensor(param):
                bytes_param = _miles_byte_view(param)
                bytes_param.copy_(
                    torch.randint(
                        low=-128,
                        high=127,
                        size=bytes_param.shape,
                        device=bytes_param.device,
                        dtype=torch.int8,
                    )
                )
                continue
            param.copy_(_random_like(param))
"""
        text = replace_once(
            text,
            old,
            new,
            f"Could not find WeightChecker reset block in {path}",
        )

        old = """        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
"""
        new = """        if name.endswith("weight")
        and name.replace("weight", "weight_scale_inv") in raw
        and not _miles_is_fp4_tensor(raw[name])
"""
        text = replace_once(
            text,
            old,
            new,
            f"Could not find WeightChecker quant_names predicate in {path}",
        )

        old = """    # dequant fp8
    quant_names = [
"""
        new = """    fp4_names = [
        name for name, tensor in raw.items() if _miles_is_fp4_tensor(tensor)
    ]
    skip_compare_names += fp4_names
    for name in fp4_names:
        yield name, True, _miles_byte_view(raw[name])

    # dequant fp8
    quant_names = [
"""
        text = replace_once(
            text,
            old,
            new,
            f"Could not find WeightChecker fp8 dequant section in {path}",
        )

        path.write_text(text)
        print(f"{path}: applied {sentinel}")
    else:
        print(f"{path}: {sentinel} already applied")

    final_sentinel = "MILES_WEIGHT_CHECKER_FP4_FINAL_SKIP_PATCH"
    text = path.read_text()
    if final_sentinel in text:
        print(f"{path}: {final_sentinel} already applied")
        return

    old = """    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]
"""
    new = """    for name in raw:
        if _miles_is_fp4_tensor(raw[name]):
            continue
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]

# MILES_WEIGHT_CHECKER_FP4_FINAL_SKIP_PATCH
"""
    text = replace_once(
        text,
        old,
        new,
        f"Could not find WeightChecker raw tensor yield in {path}",
    )
    path.write_text(text)
    print(f"{path}: applied {final_sentinel}")


def main() -> None:
    patch_model_runner()
    patch_deepseek_v4_loader()
    patch_fused_moe_layer()
    patch_weight_checker()


if __name__ == "__main__":
    main()
