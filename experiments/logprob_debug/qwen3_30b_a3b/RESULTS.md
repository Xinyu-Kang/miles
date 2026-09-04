# Qwen3-30B-A3B A/B/C results

Run date: 2026-09-03

## Outcome

The requested aggregate mean-absolute logprob target was reached on
`crsuse2-m2m-072` with SGLang's prefill-only deterministic inference mode and
an explicit Triton attention backend:

| Comparison | Triton/OFF mean abs | Triton/ON mean abs | Relative change |
|---|---:|---:|---:|
| A-B | 0.016305 | 0.007984 | -51.0% |
| B-C | 0.015165 | 0.008799 | -42.0% |
| A-C | 0.014710 | 0.008446 | -42.6% |
| B0-B1 | 0.017180 | 0.008333 | -51.5% |
| C0-C1 | 0 | 0 | unchanged |

All five ON-mode mean absolute differences are below `0.01`. This threshold is
an aggregate criterion; individual-token tails remain larger (for example,
ON-mode A-C p95 absolute difference is 0.050895).

The OFF and ON response sequences differ, despite identical prompt selection
and seeds, so the table is an aggregate run comparison rather than a paired
cross-run token comparison. A, B, and C are exact-token comparisons within
each run.

## Provenance

The successful node-072 runs used Slurm job `107755`, one 8 x AMD Instinct
MI355X node, and container `logprob-blog-tutorial`.

- Miles: `6b7ac372839606ddee25c36f00e274a1191924c1`, local branch
  `qwen3-logprob-debug`, with the complete dirty patch captured per run.
- Image ID:
  `sha256:7c2e91bb162d633d8313b212989c0674ae5a7de8b7857668f31c2e7c5a2864f2`.
- Immutable repo digest: `rocm/sgl-dev@sha256:7c2e91bb162d633d8313b212989c0674ae5a7de8b7857668f31c2e7c5a2864f2`.
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP: `7.2.26015-fc0010cf6a`.
- SGLang: `0.5.17.dev1371+gc16b821ef`, source SHA
  `c16b821ef3177a688a073c173b44c0ce48b5bf3e`.
- AITER source SHA: `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`.
- Megatron Core: `0.19.0+8c1e05747`, source SHA
  `8c1e05747eb612b382df2632783df5c83a853646`.
- Rollout model: `/root/models/Qwen3-30B-A3B`.
- Trainer checkpoint: `/root/models/Qwen3-30B-A3B_torch_dist`.
- Dataset: `/root/datasets/dapo-math-17k/dapo-math-17k.jsonl`.

The launch-time `run.txt` abbreviated the repo digest to its `sha256:` payload.
The audit-time `docker-image-identity.json` supplement preserves Docker's full
`RepoDigests` value above; the final wrapper now rejects the abbreviated form.

Every full-run artifact directory contains the resolved train command, CLI
override check, runtime environment, image identity, software and source
information, model/dataset manifest and hashes, GPU identity/topology, launcher
help, and checkout-source verification under `provenance/`. The node-072 runs
also contain one complete raw `/server_info` JSON per SGLang worker and a SHA256
manifest for those files. Generated SGLang/AITER worktree entries are recorded
in the dependency status; no SGLang or AITER kernel was patched for this
experiment.

The requested communication environment was active:
`NCCL_IB_GID_INDEX=1`, all eight `ionic_*` HCAs, `ens3` for NCCL/Gloo/TP, and
`NCCL_DMABUF_ENABLE=1`. `SGLANG_RETURN_ORIGINAL_LOGPROB=1` was also active.

## Debugging record

### 1. Baseline on node 229

The first successful baseline retained the AMD launcher's MI355X topology
(PP=2, CP=2, EP=4, TP=1), used 4 prompts, 2 samples per prompt, global batch 8,
response length 128, and stopped after rollout 0 with `num_rollout=3` still in
the resolved command.

Artifact:
`/workspace/logprob-debug-artifacts/qwen3-30b-logprob-fresh-node-107722-20260903`

| Comparison | Count | Signed mean | Mean abs | RMSE | p95 abs | p99 abs | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-B | 1024 | 0.007062 | 0.016149 | 0.063315 | 0.117209 | 0.375701 | 0.624519 |
| B-C | 1024 | 0.000289 | 0.017178 | 0.071150 | 0.116552 | 0.346368 | 0.921656 |
| A-C | 1024 | 0.007352 | 0.016428 | 0.077009 | 0.087475 | 0.280989 | 1.347169 |
| B0-B1 | 1024 | -0.002277 | 0.015355 | 0.057972 | 0.113099 | 0.307367 | 0.562320 |

The existing `train/train_rollout_logprob_abs_diff` was 0.016428, matching the
analyzer's A-C mean absolute difference. Exact token/length/mask checks,
finite-value checks, and a single policy version (`1`) all passed.

### 2. One-request concurrency ablation on node 229

Only `sglang_max_running_requests` changed from 8 to 1. Artifact:
`/workspace/logprob-debug-artifacts/qwen3-30b-logprob-concurrency-1-107722-20260903`.

| Comparison | Baseline mean abs | Concurrency-1 mean abs | Relative change |
|---|---:|---:|---:|
| A-B | 0.016149 | 0.009192 | -43.1% |
| B-C | 0.017178 | 0.010287 | -40.1% |
| A-C | 0.016428 | 0.013382 | -18.5% |
| B0-B1 | 0.015355 | 0.013013 | -15.3% |

B-repeat remained above `0.01`, so B could not yet be used to localize B-C.

### 3. Controlled worker-assignment diagnostic on node 072

The diagnostic reused exact sample 0 from the saved node-229 rollout instead
of sampling again. Source identity:

- dump SHA256:
  `f65cf53f8fb1fd9c01d31b3a5ccad19dd97d443c8ee74500c8979f88a947dcf8`;
- all-token-ID SHA256:
  `a5f7c296f131dab8c23e49d3927e94a1bac26ccb82bb5e80acc0554402c92248`;
- prompt length 127, response length 128, total token count 255.

Five direct repeats went to fixed worker `:15000`, with only that worker's
cache flushed. Four additional scores went directly to workers `:15000`,
`:15003`, `:15006`, and `:15009`, flushing the selected worker each time. Five
router-mediated repeats recorded worker attribution from per-worker request
counters; all five went to `:15006`.

Artifact:
`/workspace/logprob-debug-artifacts/qwen3-30b-logprob-b-worker-diagnostic-attempt3-107755-20260903/b_worker_diagnostic.json`

| Scope | Records | Pairwise token count | Mean abs | RMSE | p95 abs | p99 abs | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Within fixed worker | 5 | 1280 | 0.010828 | 0.041582 | 0.074485 | 0.223246 | 0.339185 |
| Cross worker | 4 | 768 | 0.013921 | 0.059517 | 0.081795 | 0.271028 | 0.587113 |
| Router mediated | 5 | 1280 | 0.012633 | 0.049977 | 0.085540 | 0.234137 | 0.587109 |

This establishes an above-threshold same-worker B floor in the original mode;
worker assignment was controlled before making that claim. Cross-worker
execution adds variability, but routing to different workers cannot explain
the entire effect because the fixed-worker repeats also differ.

### 4. Add the trainer repeatability floor

`--debug-trainer-logprob-repeats 2` scores the same DataIterator-backed batch
twice before `train()`. `forward_only()` resets the iterator before each call,
runs with `torch.no_grad()`, and places the model in eval mode. The analyzer
joins `debug_repeat_1_log_probs` through the existing DumpReader CP/PP/TP
assembly and reports C0-C1.

Every valid node-072 full run produced C0-C1 mean absolute difference exactly
zero across all 1,024 response tokens. This proves the frozen trainer forward
is repeatable for the tested batch and configuration.

### 5. Node-072 original-mode reference

The original AITER-attention configuration was rerun on node 072 after adding
C0-C1. Artifact:
`/workspace/logprob-debug-artifacts/qwen3-30b-logprob-node072-prefilldet-off-107755-20260903`.

| Comparison | Mean abs |
|---|---:|
| A-B | 0.013028 |
| B-C | 0.013131 |
| A-C | 0.015607 |
| B0-B1 | 0.013093 |
| C0-C1 | 0 |

This is a same-node reference, not the control for the final deterministic
comparison, because the successful deterministic configuration required a
different attention backend.

### 6. Deterministic-mode startup capability check

The first ON attempt used only SGLang's high-level deterministic flag. SGLang
selected FA3, but this image's `sgl_kernel` lacks `flash_ops`, so startup ended
with `ImportError: Can not import FA3 in sgl_kernel.` No rollout was produced,
and this failed startup is not treated as an experiment result.

Artifact:
`/workspace/logprob-debug-artifacts/qwen3-30b-logprob-node072-prefilldet-on-107755-20260903`.

Triton is supported in the image, so the final comparison explicitly fixed
attention to Triton in both OFF and ON arms.

### 7. Matched Triton OFF/ON comparison on node 072

Commands:

```bash
VARIANT=prefill_triton \
RUN_ID=qwen3-30b-logprob-node072-prefilldet-off-triton-107755-20260903 \
./experiments/logprob_debug/qwen3_30b_a3b/run.sh

VARIANT=prefill_deterministic \
RUN_ID=qwen3-30b-logprob-node072-prefilldet-on-triton-attempt2-107755-20260903 \
./experiments/logprob_debug/qwen3_30b_a3b/run.sh
```

Artifacts:

- OFF: `/workspace/logprob-debug-artifacts/qwen3-30b-logprob-node072-prefilldet-off-triton-107755-20260903`
- ON: `/workspace/logprob-debug-artifacts/qwen3-30b-logprob-node072-prefilldet-on-triton-attempt2-107755-20260903`

Both strict analyses joined 8 samples and 1,024 active response tokens, found
two B repeats and two C repeats, verified exact rollout/debug/trainer token IDs
and lengths, found only policy version `1`, and rejected no non-finite values.

| Run | A-B | B-C | A-C | B0-B1 | C0-C1 |
|---|---:|---:|---:|---:|---:|
| Triton/OFF mean abs | 0.016305 | 0.015165 | 0.014710 | 0.017180 | 0 |
| Triton/ON mean abs | 0.007984 | 0.008799 | 0.008446 | 0.008333 | 0 |

The ON run's existing Miles A-C metric was 0.0084456, agreeing with the strict
analyzer. Complete p50/p95/p99/max, threshold fractions, position-0 versus
position-1+, per-sample, and per-position statistics are in each run's
`analysis/rollout_0/summary.json`; token-level values are in `tokens.csv`.

The raw server arguments show that both arms resolved to Triton attention,
PyTorch sampling, the same prefill/decode graph configuration, maximum running
requests, cache/scheduler settings, topology, and model. The stable top-level
argument differences are
`enable_deterministic_inference=false/true` and
`enable_prefill_only_deterministic_inference=false/true`. The ON resolved
override history also contains `_deterministic_sampling_backend`, although the
effective sampling backend is PyTorch in both arms.

The high-level mode is still a compound internal intervention. Startup logs
show it installs batch-invariant matrix multiplication behavior and changes an
all-reduce path from AITER custom all-reduce to NCCL even though the serialized
`disable_custom_all_reduce` field remains false. The experiment therefore does
not attribute the improvement to one operator.

## Interpretation

Observation: with assignment controlled, original-mode B repeats on one worker
have mean absolute difference 0.010828. Cross-worker and router-mediated floors
are 0.013921 and 0.012633. C0-C1 is exactly zero in every valid node-072 full
run. In the matched Triton comparison, deterministic ON reduces B0-B1 from
0.017180 to 0.008333 and puts A-B, B-C, and A-C below `0.01` as well.

Inference: the gating variability is on the SGLang side, not in repeated
Megatron scoring. SGLang's high-level deterministic inference mode is an
effective mitigation for this image/configuration. Once both reference floors
are below `0.01`, ON-mode B-C is 0.008799, so this experiment does not show a
remaining above-threshold SGLang-prefill versus Megatron boundary.

Proven cause: same-worker SGLang exact-token prefill variability exists in the
original mode; worker reassignment is not required for it. The tested
deterministic mode causally reduces the aggregate discrepancies below the
requested threshold while the trainer repeat remains exactly stable.

Not proven: which internal deterministic-mode change is responsible, whether
MoE routing is the only contributor, or whether one SGLang/AITER operator is at
fault. No asynchronous HIP synchronization error is used as causal evidence.
TIS was not enabled, no inference kernel was patched, no PR was opened, and the
public blog draft was not edited.

## Node-stall note

Two earlier baseline attempts on `crsuse2-m2m-062` (Slurm job `106687`) made
no progress beyond 0/8 responses while AITER processes waited around a
masked-prefill JIT build lock. The same build completed and the experiment ran
on nodes 229 and 072. This is consistent with node-local or node-state-dependent
startup behavior, but it does not prove a hardware or AITER root cause.

Preserved failed-attempt artifacts:

- `/workspace/logprob-debug-artifacts/qwen3-30b-logprob-baseline-20260903`
- `/workspace/logprob-debug-artifacts/qwen3-30b-logprob-baseline-attempt2-20260903`
