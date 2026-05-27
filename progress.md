# DeepSeek V4 Flash on Miles: Progress and Current Blocker

Last updated: 2026-05-27

Goal: enable `deepseek-ai/DeepSeek-V4-Flash` end-to-end in Miles on AMD MI355.

Miles architecture:

- **SGLang** = rollout / inference engine.
- **Megatron** = train engine.
- **MBridge** = HF/SGLang checkpoint -> Megatron model/weights.
- **Miles live sync** = Megatron weights -> SGLang updated rollout weights.

Current state:

- Standalone SGLang loads the original DSV4 Flash checkpoint and generates coherent text.
- Standalone Megatron scoring is close to SGLang after DSV4 router/logprob fixes.
- Single-node Miles can initialize and reach the first rollout path.
- Full single-node train OOMs after rollout because rollout and train are colocated on one 8-GPU node.
- Current blocker: Miles rollout text is garbage after the first Miles weight update into SGLang.

Most important clue:

- Standalone SGLang is good.
- Miles with "skip first update" is good.
- Normal Miles first rollout is bad.
- Therefore the likely bug is in the Megatron/MBridge -> Miles export -> SGLang live-update path, not in SGLang's original checkpoint load.

## Quantization Model

DSV4 Flash is a mixed-quantization checkpoint. This matters because SGLang and Megatron do not use the same in-memory format.

| Component | SGLang checkpoint / rollout | Megatron train | Miles live update |
| --- | --- | --- | --- |
| Dense/shared weights | FP8 E4M3 | dequantized BF16/FP32 | requantized FP8 E4M3 |
| Dense/shared scales | UE8M0, 128x128 blocks | folded into BF16/FP32 tensors | regenerated UE8M0 scales |
| Routed experts | packed FP4 E2M1 | dequantized BF16 | requantized packed FP4 E2M1 |
| Routed expert scales | UE8M0, per 1x32 block | folded into BF16 tensors | regenerated UE8M0 scales |
| Runtime MoE path | SGLang AITER FP4 | Megatron grouped MoE | SGLang post-process rebuilds runtime layout |

Precise wording for AMD FP8/FP4 support: SGLang supports this FP8/FP4 checkpoint on AMD. The risk is that Miles' Megatron load plus Megatron-to-SGLang export/quantization round trip does not yet faithfully reproduce the original checkpoint tensors.

## Bring-Up Changes

### 1. Runtime image and required patches

Files:

- `docker/Dockerfile.rocm_MI350-5_DSV4`
- `docker/amd_patch/latest/megatron.patch`
- `docker/amd_patch/latest/sglang_post_process_weights.patch`

Solved / implementation:

- Built a DSV4-specific ROCm image from the SGLang DSV4 MI35x base.
- Installed ROCm TransformerEngine, MBridge, and Megatron PR 28.
- Patched Megatron for ROCm compatibility and DSV4 hash-router correctness.
- Patched SGLang with a `/post_process_weights` path so Miles can safely overwrite quantized DSV4 tensors.
- The SGLang patch is required because FP4/AITER expert weights are transformed after checkpoint load. Before Miles writes new weights, SGLang must restore loadable buffers; after Miles writes them, SGLang must rebuild runtime buffers.


### 2. Megatron model args and launchers

Files:

- `scripts/models/deepseek-v4-flash.sh`
- `scripts/models/deepseek-v4-flash-4layer.sh`
- `scripts/run_deepseek_v4_flash_fp8_4layer.py`
- `scripts/run_deepseek_v4_flash.py`

Solved / implementation:

- Added the Megatron-side DSV4 Flash model configuration: 43 layers, hidden size 4096, 64 attention heads, 256 routed experts, top-k 6, MoE from layer 0, DSV4 compressed attention, DSV4 indexer, and DSV4 hyper-connection parameters.
- Added a 4-layer model path for smoke testing before full-model bring-up.
- Added the single-node full-model launcher with Megatron TP=8, EP=8, PP=1, CP=1 and SGLang TP=8.
- Enabled GRPO, CPU optimizer offload, activation recompute, SGLang AITER MoE, DSV4 2604B mode, and rollout offload.

The launcher/model files encode the DSV4 shape and plugin entry point directly:

```bash
--num-layers 43
--hidden-size 4096
--num-attention-heads 64
--num-experts 256
--moe-router-topk 6
--experimental-attention-variant dsv4
--dsv4-compress-ratios ...
--spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec
```

The single-node launcher fixes the DSV4/AMD runtime mode with these env settings:

```python
"MILES_DSV4_CKPT_VERSION": "2604",
"MILES_DSV4_2604_SUBMODE": "2604B",
"MEGATRON_USE_KV_QAT": "1",
"SGLANG_DSV4_MODE": "2604",
"SGLANG_DSV4_2604_SUBMODE": "2604B",
"SGLANG_USE_AITER": "1",
"SGLANG_DSV4_FP4_EXPERTS": "true",
```

### 3. Megatron DSV4 model plugin

Files:

- `miles_plugins/models/deepseek_v4/deepseek_v4.py`
- `miles_plugins/models/deepseek_v4/ops/*.py`
- `miles_plugins/models/deepseek_v4/ops/kernel/*.py`

Solved / implementation:

- Implemented `DeepSeekV4Attention` so Megatron can build the same architecture that SGLang serves.
- Added DSV4-specific attention components: `wq_a`, `q_norm`, `wq_b`, `wkv`, `kv_norm`, `wo_a`, `wo_b`, and `attn_sink`.
- Added DSV4 compressed KV, V4 indexer, RoPE/YaRN behavior, hyper-connection modules, sparse MLA kernels, CP-aware top-k helpers, and FP8 activation-QAT simulation.
- Registered the plugin through `--spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec`.

The core Megatron module now mirrors the SGLang DSV4 attention tensor families:

```python
class DeepSeekV4Attention(MegatronModule):
    self.wq_a = TELinear(...)
    self.q_norm = TENorm(...)
    self.wq_b = ColumnParallelLinear(...)
    self.wkv = TELinear(...)
    self.kv_norm = TENorm(...)
    self.wo_a = ColumnParallelLinear(...)
    self.wo_b = RowParallelLinear(...)
    self.attn_sink = nn.Parameter(..., dtype=torch.float32)
```

Why this was necessary:

Without this plugin, Megatron could not reproduce the DSV4 forward path. The comparator would be meaningless because train and rollout would be different models.

### 4. MBridge DSV4 load path

Files:

- `miles_plugins/mbridge/__init__.py`
- `miles_plugins/mbridge/deepseekv4.py`
- `miles/backends/megatron_utils/checkpoint.py`
- `miles/utils/transformers_patch.py`
- `miles/utils/rocm_distributed.py`

Solved / implementation:

- Registered `deepseek_v4` with MBridge.
- Added checkpoint-name mapping from SGLang/HF names into Megatron names.
- Added DSV4 config translation: MoE from layer 0, hash-router layer count, compressor ratios, window size, indexer dimensions, hyper-connection params, `o_groups`, `o_lora_rank`, and SwiGLU clamp behavior.
- Added `--load-hf-with-mbridge` so Miles can load the HF/SGLang checkpoint through MBridge instead of the older bridge path.
- Patched Transformers config loading to delegate temporary DSV4 config classes to SGLang helpers.
- Patched ROCm `dist.scatter` during checkpoint load with a broadcast-loop fallback.

The bridge maps DSV4 checkpoint tensors into Megatron module names:

```python
@register_model("deepseek_v4")
class DeepseekV4Bridge(DeepseekV3Bridge):
    _ATTENTION_MAPPING.update({
        "self_attention.wq_a.weight": ["model.layers.{layer_number}.self_attn.wq_a.weight"],
        "self_attention.wkv.weight": ["model.layers.{layer_number}.self_attn.wkv.weight"],
        "self_attention.wo_a.weight": ["model.layers.{layer_number}.self_attn.wo_a.weight"],
        "self_attention.attn_sink": ["model.layers.{layer_number}.self_attn.attn_sink"],
    })
```

### 5. Checkpoint dequantization for Megatron

File:

- `miles_plugins/mbridge/deepseekv4.py`

Solved / implementation:

- SGLang checkpoint tensors are raw quantized tensors plus scale tensors.
- MBridge now loads both the raw weight and its scale, remaps raw SGLang names into HF-style names, and dequantizes into Megatron train tensors.
- FP8 dense/shared tensors are dequantized from E4M3 with UE8M0 128x128 scales.
- FP4 routed experts are unpacked from E2M1 x2 packed bytes with UE8M0 per-1x32 scales.

The loader dispatches by raw checkpoint dtype:

```python
if weight.dtype == torch.float8_e4m3fn:
    return _dequant_raw_fp8_block_weight(weight, scale)

if weight.dtype in (torch.int8, torch.uint8):
    return _dequant_raw_fp4x2_block_weight(weight, scale)
```

### 6. Megatron-to-SGLang export and requantization

Files:

- `miles/backends/megatron_utils/megatron_to_hf/__init__.py`
- `miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py`
- `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py`

Solved / implementation:

- Added reverse mapping from Megatron parameter names back to SGLang/HF names.
- Split Megatron fused MoE tensors back into SGLang `gate_proj`, `up_proj`, and `down_proj`.
- Exported DSV4 attention, compressor, indexer, router, `tid2eid`, attention sink, and hyper-connection tensors.
- Requantized normal dense/shared tensors to FP8 E4M3 + UE8M0.
- Requantized routed experts to packed FP4 E2M1 + UE8M0 per-1x32 via AITER.
- Fixed a real live-sync bug where `wo_a` needed checkpoint-style FP8 scale ordering for SGLang's DSV4 load/update interpretation.

The export path reverses Megatron names back to SGLang/HF names:

```python
if rest == "mlp.experts.<...>.linear_fc1":
    gate_weight, up_weight = param.chunk(2, dim=0)
    return [("...gate_proj.weight", gate_weight), ("...up_proj.weight", up_weight)]

if rest == "self_attention.wo_a.weight":
    return [(f"model.layers.{layer_idx}.self_attn.wo_a.weight", param)]
```

Routed experts are packed back into the AITER FP4 format expected by SGLang:

```python
qweight, scale = get_hip_quant(QuantType.per_1x32)(weight.contiguous(), shuffle=False)
return [(name, qweight.contiguous()), (name.replace(".weight", ".weight_scale_inv"), scale.contiguous())]
```

### 7. SGLang live-update protocol

Files:

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/common.py`
- `miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`
- `docker/amd_patch/latest/sglang_post_process_weights.patch`

Solved / implementation:

- Added the DSV4 restore-before-load and process-after-load handshake around every base weight update.
- Added update bucketing rules to keep SGLang fusion pairs together, such as `wq_a` with `wkv` and compressor `wkv` with `wgate`.
- Added DSV4 TP/duplicate-param handling for tensors such as `attn_sink`, indexer params, and low-rank duplicated projections.

The update flow now wraps tensor transfer with SGLang restore/post-process calls:

```python
if _should_restore_rollout_weights_before_load(self.quantization_config):
    post_process_weights(..., restore_weights_before_load=True)

for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(...):
    self._send_base_params(hf_named_tensors)

post_process_weights(..., post_process_quantization=True)
```

This is currently the most suspicious area for the garbage rollout blocker.

### 8. Tokenizer, config, and data compatibility

Files:

- `miles/utils/data.py`
- `miles/utils/processing_utils.py`
- `miles/utils/transformers_patch.py`
- `miles/utils/arguments.py`
- `miles/utils/chat_template_utils/tito_tokenizer.py`

Solved / implementation:

- Added DSV4 config loading fallback through SGLang's temporary DeepSeek config helpers.
- Added DeepSeek V4 chat encoding fallback using SGLang's encoder when the HF tokenizer path is not enough.
- Adjusted HF/Megatron config validation to use `moe_intermediate_size` when present.

The tokenizer fallback picks the DSV4 SGLang encoder from the model config:

```python
if config.get("hc_mult") is not None or config.get("compress_ratios") is not None:
    from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
```

This avoids comparing or training on token sequences that differ from SGLang rollout.

### 9. Single-node Miles training flow

Files:

- `scripts/run_deepseek_v4_flash.py`
- Existing flow in `train.py`
- `miles/ray/placement_group.py`
- `miles/backends/megatron_utils/actor.py`
- `miles/backends/sglang_utils/sglang_engine.py`

Solved / implementation:

- Added a reproducible single-node launcher that wires DSV4 model args, MBridge loading, SGLang rollout, GRPO, offload, and AMD/SGLang env knobs.
- Improved rollout offload tagging so colocated runs can offload only selected SGLang memory classes: CUDA graph, KV cache, and weights.
- Improved actor update reconnect behavior and SGLang HTTP error reporting.

The important existing Miles flow is that weight update happens before the first rollout:

```python
# train.py
await actor_model.update_weights()                  # happens before first rollout
rollout_data_ref = await rollout_manager.generate.remote(rollout_id)
await rollout_manager.offload.remote(...)
await actor_model.train(rollout_id, rollout_data_ref)
await actor_model.update_weights()                  # update SGLang for next rollout
```

The train phase is:

1. SGLang generates samples and selected-token rollout logprobs.
2. Miles builds rollout batches and reward/advantage data.
3. Megatron computes current logprobs on the rollout tokens.
4. GRPO/PPO-style policy loss is computed from current logprobs, rollout logprobs, and advantages.
5. Megatron backward + optimizer step updates the actor.
6. Miles exports updated actor weights back to SGLang.


### 10. Standalone SGLang/Megatron comparator and logprob fix

File:

- `tools/compare_sglang_megatron_standalone.py`

Solved / implementation:

- Added a standalone tool to isolate SGLang-vs-Megatron probability parity outside the full RL loop.
- It can generate with SGLang, score fixed sequences with SGLang, score the same sequences with Megatron/MBridge, and compare selected-token logprobs.
- It writes token-level JSONL, top-diff CSV, summary JSON, histogram, and logprob heatmap.
- It found a major DSV4 hash-router mismatch.

The main logprob fix was DSV4 hash routing. SGLang's DSV4 hash-routed layers use deterministic `tid2eid` routing with `sqrtsoftplus` and do not apply the normal routed scaling factor. Megatron's generic router path applied normal scaling, which made early layers diverge. The debug patch pattern was:

```python
def topk_routing_with_score_function(*args, **kwargs):
    if kwargs.get("tid2eid") is not None:
        kwargs = dict(kwargs)
        kwargs["scaling_factor"] = None
        kwargs["score_function"] = "sqrtsoftplus"
    return original(*args, **kwargs)
```

Heatmap interpretation: the heatmap plots SGLang logprob vs Megatron logprob for the same selected tokens. Points near the diagonal mean both systems assign similar probabilities. Current result being mostly diagonal means the standalone model/math path is close, which shifts suspicion toward live weight export/update rather than basic tokenizer/model-forward parity.

## Current Blocker: Garbage Rollout After First Miles Update

Observed behavior:

- Original checkpoint in standalone SGLang: good text.
- Miles normal first rollout: garbage text.
- Miles with first `actor_model.update_weights()` skipped: good text.

Therefore:

- SGLang original load is probably correct.
- Miles corrupts SGLang after exporting Megatron/MBridge weights back into SGLang.

Debug evidence so far:

- Preserving only routed FP4 experts still produced garbage.
- Preserving only attention still produced garbage.
- Preserving only output head still produced garbage.
- Current/next stronger probe: no-op update that skips all known DSV4 tensor groups.

Highest-probability root-cause zones:

- Megatron-to-SGLang tensor naming/mapping mismatch.
- FP8 scale layout/order mismatch.
- FP4 expert packing or scale layout mismatch.
- SGLang restore/post-process interaction for AITER FP4 buffers.
- A DSV4 tensor family not covered by current preserve/skip probes.

## File Inventory

| Area | Files | Technical purpose |
| --- | --- | --- |
| Runtime | `docker/Dockerfile.rocm_MI350-5_DSV4` | ROCm/SGLang/Megatron/MBridge image for MI355 DSV4. |
| Megatron patch | `docker/amd_patch/latest/megatron.patch` | ROCm build guard, DSV4 hash-router fix, memory-saver fix. |
| SGLang patch | `docker/amd_patch/latest/sglang_post_process_weights.patch` | Restore/process hooks for quantized live updates. |
| Model args | `scripts/models/deepseek-v4-flash*.sh` | Megatron DSV4 architecture flags. |
| Single-node launcher | `scripts/run_deepseek_v4_flash.py` | Full single-node DSV4 E2E run config. |
| 4-layer launcher | `scripts/run_deepseek_v4_flash_fp8_4layer.py` | Reduced smoke path. |
| AITER config | `scripts/amd/dsv4_flash_fp4_tp8_fmoe.csv` | SGLang AITER FP4 MoE kernel config. |
| Megatron model | `miles_plugins/models/deepseek_v4/*` | DSV4 attention/compressor/indexer/kernels/QAT. |
| MBridge load | `miles_plugins/mbridge/deepseekv4.py` | SGLang/HF checkpoint -> Megatron weights/config. |
| Checkpoint path | `miles/backends/megatron_utils/checkpoint.py` | `--load-hf-with-mbridge` and MBridge compatibility. |
| Config patch | `miles/utils/transformers_patch.py` | Temporary DSV4 HF config support. |
| ROCm dist patch | `miles/utils/rocm_distributed.py` | Scatter workaround during MBridge load. |
| Export mapping | `miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py` | Megatron weights -> SGLang/HF names. |
| Export quant | `miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py` | BF16 -> FP8/FP4 checkpoint-style tensors. |
| Live update | `miles/backends/megatron_utils/update_weight/*.py` | Bucket, gather, send, restore, post-process SGLang weights. |
| Tokenization | `miles/utils/data.py`, `miles/utils/processing_utils.py` | DSV4 chat encoding and tokenizer compatibility. |
| Offload | `miles/ray/placement_group.py` | More precise rollout memory offload tags. |
| Comparator | `tools/compare_sglang_megatron_standalone.py` | Standalone logprob parity tool. |

