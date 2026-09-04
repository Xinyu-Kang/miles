# Qwen3-30B-A3B A/B/C logprob experiment

This debug-only experiment compares exact response-token logprobs from one
policy snapshot:

- A: SGLang generation-time rollout logprobs.
- B: SGLang teacher-forced exact-token prefill logprobs, repeated after a
  cache flush.
- C: Megatron old-policy forward logprobs computed before training.
- B0-B1: SGLang prefill repeatability.
- C0-C1: frozen-batch Megatron forward repeatability.

The patch does not replace `Sample.rollout_log_probs` in comparison mode, does
not enable TIS, and does not modify an inference kernel.

## Data flow and invariants

`--debug-compare-decode-prefill-logprobs` copies A, extracts the response token
IDs from the existing sample, and uses Miles' existing SGLang prefill scoring
path for B. It stores:

```python
sample.metadata["logprob_debug"] = {
    "response_token_ids": [...],
    "decode_logprobs": [...],
    "prefill_logprobs_repeats": [[...], ...],
}
```

The helper flushes cache before each B score, checks exact returned token IDs,
lengths, and finite values, and leaves A unchanged. The existing production
semantics of `--recompute-logprobs-via-prefill` are unchanged.

Comparison mode requires `--dump-details`, rejects fully asynchronous runs,
and does not require `--true-on-policy-mode`.

`--debug-trainer-logprob-repeats 2` records C0 and C1. The Megatron actor calls
`compute_log_prob` before `train`; `forward_only` resets every `DataIterator`,
runs under `torch.no_grad()`, and puts the model in eval mode. Consequently the
same frozen batch and weights are used twice before an optimizer update.

The rank-sharded train dump is physically written after `train` returns, but
it contains the already-detached pre-update `log_probs` and
`debug_repeat_1_log_probs`. `DumpReader` assembles both across CP/PP/TP into
`TrainRow.log_probs` and `TrainRow.debug_repeat_1_log_probs`.

The analyzer additionally requires:

- exact rollout/debug/trainer token IDs and length metadata;
- identical rollout and trainer response masks;
- A in debug metadata to equal the unchanged `Sample.rollout_log_probs`;
- one non-mixed policy version across all samples and matching rollout/train
  version metadata;
- at least two B repeats and both C repeats;
- finite A, B, C, C1, and difference values.

Any violation is a hard failure.

## Configuration

The wrapper starts from `scripts/amd/run_qwen3_30b_a3b.py` and retains its
MI355X topology:

- 8 GPUs, colocated training and rollout;
- PP=2, CP=2, EP=4, TP=1, expert TP=1;
- 2 rollout GPUs per SGLang engine.

Final full-run overrides are:

- `num_rollout=3`, with `debug_exit_after_rollout=1` as the rollout-0 gate;
- `rollout_batch_size=4`;
- `n_samples_per_prompt=2`;
- `global_batch_size=8`;
- `rollout_max_response_len=128`;
- `sglang_max_running_requests=8`;
- two B repeats and two C repeats;
- seed and rollout seed 1234;
- `SGLANG_RETURN_ORIGINAL_LOGPROB=1`.

Communication settings are:

```ini
NCCL_IB_GID_INDEX=1
NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
GLOO_SOCKET_IFNAME=ens3
NCCL_SOCKET_IFNAME=ens3
TP_SOCKET_IFNAME=ens3
NCCL_DMABUF_ENABLE=1
```

The wrapper invokes the real Miles parser before launch and fails if duplicate
options do not resolve to the intended final values.

## Provenance

For every run, `provenance/` records:

- Miles SHA, status, and complete binary dirty diff, including untracked files;
- image ID and immutable digest passed from the host;
- ROCm, PyTorch, SGLang, AITER, and Megatron versions/source SHAs and statuses;
- GPU identity and topology;
- complete resolved `train.py` command and final override values;
- model/checkpoint/dataset paths, manifests, revisions, and selected hashes;
- launcher `--help` output;
- imported Miles path and `sys.path`, with a hard check that the bind-mounted
  checkout is the import source;
- complete `/server_info` JSON from each of the four SGLang workers and their
  SHA256 manifest.

The wrapper polls `/server_info` in the background while the engines start and
treats missing worker argument captures as a failed otherwise-successful run.

## Run

Run inside the prepared container. Obtain the immutable image fields on the
host and pass them as `CONTAINER_IMAGE_ID` and `CONTAINER_REPO_DIGEST`:

```bash
cd /workspace/miles
CONTAINER_IMAGE_ID='sha256:<image-id>' \
CONTAINER_REPO_DIGEST='<repository>@sha256:<repo-digest>' \
RUN_ID=qwen3-30b-logprob-baseline \
./experiments/logprob_debug/qwen3_30b_a3b/run.sh
```

Artifacts default to `/workspace/logprob-debug-artifacts/<run-id>`, which is
bind-mounted to the host workspace and outside the Miles checkout.

Available variants are:

| Variant | Change |
|---|---|
| `baseline` | Original AMD configuration |
| `concurrency_1` | Maximum running requests 8 to 1 |
| `graph_off` | Disable SGLang graph execution |
| `overlap_off` | Disable SGLang overlap scheduling |
| `radix_cache_off` | Disable the SGLang radix cache |
| `r3` | Enable inference-side rollout routing replay |
| `prefill_triton` | Set SGLang attention backend to Triton |
| `prefill_deterministic` | Triton plus SGLang prefill-only deterministic mode |
| `b_worker_diagnostic` | Reuse one saved sample for direct/router B scoring |

Do not compare `prefill_deterministic` directly with the AITER baseline. Use
`prefill_triton` as its same-node control so the only requested CLI difference
is the high-level deterministic flag. The flag is itself a compound internal
mode, so save and inspect the complete resolved SGLang arguments and startup
logs before drawing an operator-level conclusion.

## Controlled-worker B diagnostic

Set `DIAGNOSTIC_SAMPLE_PATH` to a saved rollout dump. The diagnostic loads its
first sample and preserves the exact complete token list. It performs:

1. five cache-flushed scores sent directly to one fixed worker;
2. one cache-flushed score sent directly to each of four workers;
3. five router-mediated scores with the selected worker attributed through
   per-worker request-counter deltas.

```bash
VARIANT=b_worker_diagnostic \
DIAGNOSTIC_SAMPLE_PATH=/workspace/logprob-debug-artifacts/<source-run>/dump_details/rollout_data/0.pt \
RUN_ID=qwen3-30b-logprob-b-worker-diagnostic \
./experiments/logprob_debug/qwen3_30b_a3b/run.sh
```

`b_worker_diagnostic.json` records source-file and token-ID hashes, every score,
worker assignments, complete worker information, pairwise statistics, and
separate within-worker, cross-worker, and router-mediated aggregates. Do not
label B nondeterministic from router repeats alone.

## Analyze a full run

The wrapper invokes the analyzer automatically after rollout 0. To rerun it:

```bash
python tools/analyze_logprob_abc.py \
  --dump-details /workspace/logprob-debug-artifacts/<run-id>/dump_details \
  --rollout-id 0 \
  --output-dir /workspace/logprob-debug-artifacts/<run-id>/analysis/rollout_0
```

Outputs are:

- `tokens.csv`: one row per response token with identity fields, A, every B
  repeat, C, C1, all five differences, and active mask;
- `summary.json`: overall, position-0, position-1+, per-sample, and per-position
  statistics;
- `summary.md`: compact aggregate and threshold tables.

Every comparison reports count, signed mean, mean absolute difference, RMSE,
p50/p95/p99/maximum absolute difference, and fractions above `1e-3`, `1e-2`,
`5e-2`, and `1e-1`.

## Result

The controlled original-mode diagnostic found mean absolute B repeatability
differences of 0.010828 within one fixed worker, 0.013921 across workers, and
0.012633 through the router. This proves the original-mode variability is not
solely worker reassignment.

On node 072 with Triton fixed in both arms, deterministic OFF versus ON gave:

| Run | A-B | B-C | A-C | B0-B1 | C0-C1 |
|---|---:|---:|---:|---:|---:|
| OFF mean abs | 0.016305 | 0.015165 | 0.014710 | 0.017180 | 0 |
| ON mean abs | 0.007984 | 0.008799 | 0.008446 | 0.008333 | 0 |

All ON aggregate mean absolute differences meet the requested `<0.01` target,
including both reference floors. See `RESULTS.md` for complete provenance,
failed-startup context, the step-by-step record, and the observation/inference/
proven-cause distinction. The public blog draft remains unchanged.
