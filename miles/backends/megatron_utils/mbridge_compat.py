from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from typing import Any


def import_auto_bridge():
    try:
        from megatron.bridge import AutoBridge

        return AutoBridge
    except ModuleNotFoundError:
        import miles_plugins.mbridge  # noqa: F401
        from mbridge import AutoBridge

        return AutoBridge


def create_bridge(args: Namespace | None, checkpoint: str):
    AutoBridge = import_auto_bridge()
    if hasattr(AutoBridge, "from_hf_pretrained"):
        return AutoBridge.from_hf_pretrained(checkpoint, trust_remote_code=True)

    from miles.utils.transformers_patch import with_transformers_patch

    bridge_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args is not None:
        vocab_divisor = getattr(args, "make_vocab_size_divisible_by", None)
        if vocab_divisor is not None:
            bridge_kwargs["make_vocab_size_divisible_by"] = vocab_divisor

    with with_transformers_patch():
        return AutoBridge.from_pretrained(checkpoint, **bridge_kwargs)


def bridge_supports_pr_api(bridge) -> bool:
    return all(
        hasattr(bridge, attr)
        for attr in (
            "get_conversion_tasks",
            "export_hf_weights",
        )
    )


def bridge_provider_func(args: Namespace, bridge):
    if hasattr(bridge, "to_megatron_provider"):
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.context_parallel_size = args.context_parallel_size
        provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        provider.variable_seq_lengths = args.variable_seq_lengths
        if hasattr(args, "moe_token_dispatcher_type"):
            provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if getattr(args, "moe_router_bias_update_rate", None) is not None:
            provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
        if getattr(args, "moe_aux_loss_coeff", None) is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
        provider.finalize()

        def provide(pre_process: bool, post_process: bool, vp_stage=None, pg_collection=None):
            if pg_collection is not None:
                provider._pg_collection = pg_collection
            return provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        return provide

    extra_args = {
        "sequence_parallel": args.sequence_parallel,
        "context_parallel_size": args.context_parallel_size,
        "attention_softmax_in_fp32": args.attention_softmax_in_fp32,
        "variable_seq_lengths": args.variable_seq_lengths,
    }
    if hasattr(args, "moe_token_dispatcher_type"):
        extra_args["moe_token_dispatcher_type"] = args.moe_token_dispatcher_type
    if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
        extra_args["num_layers_in_first_pipeline_stage"] = args.decoder_first_pipeline_num_layers
    if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
        extra_args["num_layers_in_last_pipeline_stage"] = args.decoder_last_pipeline_num_layers
    if getattr(args, "moe_router_bias_update_rate", None) is not None:
        extra_args["moe_router_bias_update_rate"] = args.moe_router_bias_update_rate
    if getattr(args, "moe_aux_loss_coeff", None) is not None:
        extra_args["moe_aux_loss_coeff"] = args.moe_aux_loss_coeff

    if hasattr(bridge, "set_extra_args"):
        bridge.set_extra_args(**extra_args)
    else:
        for key, value in extra_args.items():
            setattr(bridge.config, key, value)

    provider = bridge._model_provider([])  # local mbridge API

    def provide(pre_process: bool, post_process: bool, vp_stage=None, pg_collection=None):
        # Local mbridge providers do not accept pg_collection directly; the
        # layer specs use Megatron's initialized process groups.
        return provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    return provide


def maybe_transformers_patch_for_bridge(bridge):
    if bridge_supports_pr_api(bridge):
        return nullcontext()
    from miles.utils.transformers_patch import with_transformers_patch

    return with_transformers_patch()
