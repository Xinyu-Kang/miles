import dataclasses
import re

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.backends.megatron_utils.mbridge_compat import (
    bridge_supports_pr_api,
    create_bridge,
    maybe_transformers_patch_for_bridge,
)
from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._bridge = create_bridge(self.args, self.args.hf_checkpoint)
        self._uses_pr_bridge_api = bridge_supports_pr_api(self._bridge)

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            if self._uses_pr_bridge_api:
                if weight_type == "lora":
                    named_weights = self._bridge.export_adapter_weights(
                        self.model,
                        cpu=False,
                        show_progress=False,
                    )
                elif weight_type == "base":
                    conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                    conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                    named_weights = self._bridge.export_hf_weights(
                        self.model,
                        cpu=False,
                        conversion_tasks=conversion_tasks,
                        merge_adapter_weights=False,
                    )
            else:
                if weight_type == "lora":
                    raise NotImplementedError("LoRA bridge export requires megatron.bridge export_adapter_weights")
                with maybe_transformers_patch_for_bridge(self._bridge):
                    named_weights = (
                        (hf_name, weight, _infer_megatron_name_from_hf_name(hf_name))
                        for hf_name, weight in self._bridge.export_weights(
                            self.model,
                            keep_stacked_experts=False,
                        )
                    )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang). Base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs; LoRA adapters are passed through unchanged.
            named_weights = self._postprocess_and_quantize(named_weights, weight_type)

            if weight_type == "base":
                named_weights = ((n, t) for n, t in named_weights if not is_lora_weight_name(n))
            elif weight_type == "lora":
                named_weights = ((n, t) for n, t in named_weights if is_lora_weight_name(n))

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for hf_param_name, weight, megatron_param_name in named_weights:
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                qmegatron_name = f"module.module.{megatron_param_name}"
                yield from quantize_params(self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config)
            else:
                yield hf_name, weight


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


def _infer_megatron_name_from_hf_name(hf_name: str) -> str:
    layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", hf_name)
    if not layer_match:
        if hf_name == "model.embed_tokens.weight":
            return "module.module.embedding.word_embeddings.weight"
        if hf_name == "lm_head.weight":
            return "module.module.output_layer.weight"
        if hf_name == "model.norm.weight":
            return "module.module.decoder.final_layernorm.weight"
        return hf_name

    layer_idx, rest = layer_match.groups()
    prefix = f"module.module.decoder.layers.{layer_idx}"

    expert_match = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", rest)
    if expert_match:
        expert_idx, proj = expert_match.groups()
        linear = "linear_fc2" if proj == "down_proj" else "linear_fc1"
        return f"{prefix}.mlp.experts.{linear}.weight{expert_idx}"

    shared_match = re.match(r"mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$", rest)
    if shared_match:
        proj = shared_match.group(1)
        linear = "linear_fc2" if proj == "down_proj" else "linear_fc1"
        return f"{prefix}.mlp.shared_experts.{linear}.weight"

    dense_mlp = {
        "mlp.gate_proj.weight": "mlp.linear_fc1.weight",
        "mlp.up_proj.weight": "mlp.linear_fc1.weight",
        "mlp.down_proj.weight": "mlp.linear_fc2.weight",
    }
    if rest in dense_mlp:
        return f"{prefix}.{dense_mlp[rest]}"

    attention = {
        "self_attn.wq_a.weight": "self_attention.wq_a.weight",
        "self_attn.wq_b.weight": "self_attention.wq_b.weight",
        "self_attn.wkv.weight": "self_attention.wkv.weight",
        "self_attn.wo_b.weight": "self_attention.wo_b.weight",
        "self_attn.indexer.wq_b.weight": "self_attention.indexer.linear_wq_b.weight",
        "self_attn.indexer.wk.weight": "self_attention.indexer.linear_wk.weight",
    }
    if rest in attention:
        return f"{prefix}.{attention[rest]}"

    return f"{prefix}.{rest}"


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
