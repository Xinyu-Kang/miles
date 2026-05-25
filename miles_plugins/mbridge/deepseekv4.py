from megatron.core.transformer.enums import AttnBackend

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("deepseek_v4")
class DeepseekV4Bridge(DeepseekV3Bridge):
    _ATTENTION_MAPPING = DeepseekV3Bridge._ATTENTION_MAPPING.copy()

    _ATTENTION_MAPPING.pop("self_attention.linear_q_up_proj.layer_norm_weight", None)
    _ATTENTION_MAPPING.pop("self_attention.linear_kv_up_proj.layer_norm_weight", None)

    _ATTENTION_MAPPING.update(
        {
            "self_attention.wq_a.weight": ["model.layers.{layer_number}.self_attn.wq_a.weight"],
            "self_attention.q_norm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
            "self_attention.wq_b.weight": ["model.layers.{layer_number}.self_attn.wq_b.weight"],
            "self_attention.wkv.weight": ["model.layers.{layer_number}.self_attn.wkv.weight"],
            "self_attention.kv_norm.weight": ["model.layers.{layer_number}.self_attn.kv_norm.weight"],
            "self_attention.wo_a.weight": ["model.layers.{layer_number}.self_attn.wo_a.weight"],
            "self_attention.wo_b.weight": ["model.layers.{layer_number}.self_attn.wo_b.weight"],
            "self_attention.attn_sink": ["model.layers.{layer_number}.self_attn.attn_sink"],
            "self_attention.compressor.ape": ["model.layers.{layer_number}.self_attn.compressor.ape"],
            "self_attention.compressor.wkv.weight": ["model.layers.{layer_number}.self_attn.compressor.wkv.weight"],
            "self_attention.compressor.wgate.weight": [
                "model.layers.{layer_number}.self_attn.compressor.wgate.weight"
            ],
            "self_attention.compressor.norm.weight": ["model.layers.{layer_number}.self_attn.compressor.norm.weight"],
            "self_attention.indexer.linear_wq_b.weight": ["model.layers.{layer_number}.self_attn.indexer.wq_b.weight"],
            "self_attention.indexer.linear_weights_proj.weight": [
                "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
            ],
            "self_attention.indexer.compressor.ape": ["model.layers.{layer_number}.self_attn.indexer.compressor.ape"],
            "self_attention.indexer.compressor.wkv.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wkv.weight"
            ],
            "self_attention.indexer.compressor.wgate.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.wgate.weight"
            ],
            "self_attention.indexer.compressor.norm.weight": [
                "model.layers.{layer_number}.self_attn.indexer.compressor.norm.weight"
            ],
        }
    )

    _OTHER_MAPPING = {
        "hc_attn_fn": ["model.layers.{layer_number}.hc_attn_fn"],
        "hc_attn_base": ["model.layers.{layer_number}.hc_attn_base"],
        "hc_attn_scale": ["model.layers.{layer_number}.hc_attn_scale"],
        "hc_ffn_fn": ["model.layers.{layer_number}.hc_ffn_fn"],
        "hc_ffn_base": ["model.layers.{layer_number}.hc_ffn_base"],
        "hc_ffn_scale": ["model.layers.{layer_number}.hc_ffn_scale"],
    }

    _MLP_MAPPING = DeepseekV3Bridge._MLP_MAPPING.copy()
    _MLP_MAPPING.update(
        {
            "mlp.router.tid2eid": ["model.layers.{layer_number}.mlp.topk.tid2eid"],
        }
    )

    _DIRECT_MAPPING = DeepseekV3Bridge._DIRECT_MAPPING.copy()
    _DIRECT_MAPPING.update(
        {
            "decoder.hc_head_params.hc_head_fn": "model.hc_head_fn",
            "decoder.hc_head_params.hc_head_base": "model.hc_head_base",
            "decoder.hc_head_params.hc_head_scale": "model.hc_head_scale",
        }
    )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        try:
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        except NotImplementedError:
            return self._weight_name_mapping_other(mcore_weights_name)

    def _get_safetensor_io(self, weights_path: str):
        return _RawNameRemapSafeTensorIO(super()._get_safetensor_io(weights_path))

    def _get_transformer_layer_spec(self, vp_stage=None):
        from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec

        self.has_vp_stage = True
        return get_dsv4_spec(None, self.config, vp_stage)

    def _build_config(self):
        # SGLang's temporary HF config loader reuses DeepSeek-V3 config classes for
        # V4, which injects the V3 default first_k_dense_replace=3. Official
        # DeepSeek-V4-Flash is MoE from layer 0, so keep every layer on the MoE
        # path or weight loading will request nonexistent dense MLP weights.
        self.hf_config.first_k_dense_replace = 0
        config = super()._build_config()

        config.attention_backend = AttnBackend.auto

        config.experimental_attention_variant = "dsv4"
        config.dsv4_mode = True
        config.dsa_indexer_n_heads = getattr(self.hf_config, "index_n_heads", 64)
        config.dsa_indexer_head_dim = getattr(self.hf_config, "index_head_dim", 128)
        config.dsa_indexer_topk = getattr(self.hf_config, "index_topk", 512)
        config.vocab_size = self.hf_config.vocab_size

        # SGLang's DSV4 implementation only applies HF scoring_func to the
        # deterministic hash-routed layers. The later expert-bias routed layers
        # go through biased_grouped_topk, which is sigmoid based. Keep the base
        # MBridge/Megatron router score function for those normal MoE layers;
        # the Megatron DSV4 hash-router patch specializes hash layers instead.
        config.dsv4_hash_router_score_function = getattr(self.hf_config, "scoring_func", "sqrtsoftplus")
        config.dsv4_hc_mult = getattr(self.hf_config, "hc_mult", 4)
        config.dsv4_hc_sinkhorn_iters = getattr(self.hf_config, "hc_sinkhorn_iters", 20)
        config.dsv4_hc_eps = getattr(self.hf_config, "hc_eps", 1e-6)

        config.dsv4_compress_ratios = getattr(self.hf_config, "compress_ratios", None)
        config.dsv4_compress_rope_theta = getattr(self.hf_config, "compress_rope_theta", 160000)

        config.dsv4_swiglu_limit = getattr(self.hf_config, "swiglu_limit", 0.0)
        if config.dsv4_swiglu_limit > 0:
            config.bias_activation_fusion = False
            config.activation_func_clamp_value = config.dsv4_swiglu_limit
            if getattr(self.hf_config, "expert_dtype", None) == "fp4":
                config.activation_func_clamp_shared_expert = False

        config.dsv4_o_groups = getattr(self.hf_config, "o_groups", 8)
        config.dsv4_o_lora_rank = getattr(self.hf_config, "o_lora_rank", 1024)
        config.dsv4_n_hash_layers = getattr(
            self.hf_config,
            "num_hash_layers",
            getattr(self.hf_config, "n_hash_layers", 3),
        )
        config.dsv4_window_size = getattr(
            self.hf_config,
            "sliding_window",
            getattr(self.hf_config, "window_size", 128),
        )

        return config


class _RawNameRemapSafeTensorIO:
    def __init__(self, base):
        self._base = base
        self._hf_to_raw = self._build_hf_to_raw_map(base.index)
        self.index = {hf: base.index[raw] for hf, raw in self._hf_to_raw.items()}
        self.hf_dir = getattr(base, "hf_dir", None)

    @staticmethod
    def _build_hf_to_raw_map(raw_index):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        out = {}
        for raw_name in raw_index.keys():
            hf_name = remap(raw_name)
            out[hf_name] = raw_name
            out.setdefault(raw_name, raw_name)
        return out

    def __getattr__(self, item):
        return getattr(self._base, item)

    def load_some_hf_weight(self, hf_weight_names):
        raw_names = []
        hf_for_raw = {}
        raw_scales = {}
        for hf_name in hf_weight_names:
            raw_name = self._hf_to_raw.get(hf_name, hf_name)
            raw_names.append(raw_name)
            hf_for_raw[raw_name] = hf_name
            raw_scale = self._raw_scale_name(raw_name)
            if raw_scale in self._base.index:
                raw_names.append(raw_scale)
                raw_scales[raw_name] = raw_scale

        loaded = self._base.load_some_hf_weight(raw_names)
        out = {}
        for raw_name, hf_name in hf_for_raw.items():
            weight = loaded[raw_name]
            scale_name = raw_scales.get(raw_name)
            if scale_name is not None and weight.element_size() == 1:
                weight = self._dequant_raw_fp8_weight(weight, loaded[scale_name])
            out[hf_name] = weight
        return out

    @staticmethod
    def _raw_scale_name(raw_name):
        if raw_name.endswith(".weight"):
            return raw_name[: -len(".weight")] + ".scale"
        return f"{raw_name}_scale_inv"

    @staticmethod
    def _dequant_raw_fp8_weight(weight, scale):
        import torch

        if weight.dtype == torch.float8_e4m3fn:
            return _dequant_raw_fp8_block_weight(weight, scale)

        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if weight.dtype in (torch.int8, torch.uint8) or (
            fp4_dtype is not None and weight.dtype == fp4_dtype
        ):
            return _dequant_raw_fp4x2_block_weight(weight, scale)

        raise TypeError(
            f"Unsupported V4 quantized weight dtype {weight.dtype} with scale dtype {scale.dtype}"
        )

    def load_one_hf_weight(self, hf_weight_name):
        return self.load_some_hf_weight([hf_weight_name])[hf_weight_name]

    def load_hf_weight_names(self):
        return list(self.index.keys())


def _float8_e8m0_to_float(scale):
    import torch

    if scale.dtype != torch.float8_e8m0fnu:
        return scale.float()

    e = scale.contiguous().view(torch.uint8).to(torch.float32)
    e = torch.clamp(e, max=254)
    out = torch.exp2(e - 127.0)
    return torch.where(e == 0, torch.zeros_like(out), out).view(scale.shape)


def _dequant_raw_fp8_block_weight(weight, scale):
    import torch

    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected V4 e8m0 scale for fp8 weight, got {scale.dtype}")
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(f"expected 2D fp8 weight/scale, got {weight.shape} and {scale.shape}")

    block_m = block_n = 128
    m, n = weight.shape
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(f"fp8 weight shape must be divisible by 128, got {weight.shape}")
    expected_scale_shape = (m // block_m, n // block_n)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"fp8 scale shape {tuple(scale.shape)} does not match weight shape "
            f"{tuple(weight.shape)}; expected {expected_scale_shape}"
        )

    weight_f32 = weight.float().view(m // block_m, block_m, n // block_n, block_n)
    scale_f32 = _float8_e8m0_to_float(scale).view(m // block_m, 1, n // block_n, 1)
    return (weight_f32 * scale_f32).view(m, n).to(torch.bfloat16)


def _dequant_raw_fp4x2_block_weight(weight, scale):
    import torch

    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected V4 e8m0 scale for fp4 weight, got {scale.dtype}")
    if weight.dim() != 2 or scale.dim() != 2:
        raise ValueError(f"expected 2D fp4 weight/scale, got {weight.shape} and {scale.shape}")

    m, packed_n = weight.shape
    n = packed_n * 2
    expected_scale_shape = (m, n // 32)
    if n % 32 != 0 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"fp4 scale shape {tuple(scale.shape)} does not match packed weight shape "
            f"{tuple(weight.shape)}; expected {expected_scale_shape}"
        )

    packed = weight.contiguous().view(torch.uint8)
    lo = torch.remainder(packed, 16).to(torch.long)
    hi = torch.div(packed, 16, rounding_mode="floor").to(torch.long)

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.bfloat16,
        device=weight.device,
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=weight.device)
    out[:, 0::2] = table[lo]
    out[:, 1::2] = table[hi]

    scale_bf16 = _float8_e8m0_to_float(scale).to(torch.bfloat16).view(m, n // 32, 1)
    out.view(m, n // 32, 32).mul_(scale_bf16)
    return out
