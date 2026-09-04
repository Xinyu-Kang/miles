# Controlled A/B/C logprob fault-injection tutorial

This directory turns the Qwen3-30B-A3B A/B/C instrumentation into three
deliberate, single-fault debugging exercises. The injected behavior is always
identified as a tutorial fault; it is not evidence of a naturally occurring
Miles, SGLang, or Megatron defect.

The score labels are:

- A: SGLang generation-time rollout logprobs.
- B: exact-token SGLang prefill replay logprobs.
- C: Megatron trainer-forward logprobs captured before the optimizer update.

Every run uses two B scores and two C scores. The analyzer refuses to localize
a boundary unless both B0-B1 and C0-C1 have mean absolute difference strictly
below `0.01`. Token IDs, lengths, masks, policy versions, and finite values are
hard invariants.

## Safety

Faults are off by default. A fault arm requires all of the following:

- `MILES_ENABLE_LOGPROB_FAULT_INJECTION=1`;
- exactly one named active fault;
- `--dump-details`;
- a non-CI process;
- a resolved configuration accepted by `validate_run_config.py`.

Activation prints a loud startup banner. Clean and fixed arms reject leaked
fault variables. Production recompute behavior remains unchanged.

## Shared configuration

The measured runs used Slurm job `107755` on `crsuse2-m2m-072`, one 8 x
MI355X node, and image:

```text
rocm/sgl-dev@sha256:7c2e91bb162d633d8313b212989c0674ae5a7de8b7857668f31c2e7c5a2864f2
```

The wrapper keeps the AMD recipe topology, four prompts, two samples per
prompt, global batch eight, response length 128, seeds 1234, two B repeats,
two C repeats, and `SGLANG_RETURN_ORIGINAL_LOGPROB=1`. It delegates to the
parent `run.sh`, which records the exact resolved train command and all four
complete SGLang `ServerArgs` records under each run's `provenance/` directory.

Inside the prepared container:

```bash
cd /workspace/miles
export CONTAINER_IMAGE_ID=sha256:7c2e91bb162d633d8313b212989c0674ae5a7de8b7857668f31c2e7c5a2864f2
export CONTAINER_REPO_DIGEST=rocm/sgl-dev@sha256:7c2e91bb162d633d8313b212989c0674ae5a7de8b7857668f31c2e7c5a2864f2
```

The parent wrapper exports the node communication configuration documented in
the experiment README, including the eight `ionic_*` HCAs and `ens3` socket
interfaces.

## Reproduce the runs

Case 0 is the shared clean one-rollout control:

```bash
TUTORIAL_CASE=case0 TUTORIAL_ARM=clean \
RUN_ID=qwen3-logprob-tutorial-case0-clean-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh
```

Case 1 applies a `0.10` nat bookkeeping offset only to stored A positions 1+:

```bash
MILES_ENABLE_LOGPROB_FAULT_INJECTION=1 \
MILES_LOGPROB_ACTIVE_FAULT=decode_logprob_offset \
TUTORIAL_CASE=case1 TUTORIAL_ARM=fault \
RUN_ID=qwen3-logprob-tutorial-case1-fault-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh

TUTORIAL_CASE=case1 TUTORIAL_ARM=fixed \
RUN_ID=qwen3-logprob-tutorial-case1-fixed2-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh
```

Case 2 first measures the native temperature mismatch at `0.9` and `0.8`.
Those pilots are retained, but their B-C MAE is too small for the requested
teaching signal. The final fault arm therefore uses the explicitly synthetic,
debug-only trainer normalizer offset allowed by the experiment specification:

```bash
MILES_ENABLE_LOGPROB_FAULT_INJECTION=1 \
MILES_LOGPROB_ACTIVE_FAULT=trainer_normalizer_offset \
TUTORIAL_CASE=case2 TUTORIAL_ARM=fault \
TUTORIAL_CASE2_FAULT_MODE=trainer_offset \
RUN_ID=qwen3-logprob-tutorial-case2-fault-fallback-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh

TUTORIAL_CASE=case2 TUTORIAL_ARM=fixed \
RUN_ID=qwen3-logprob-tutorial-case2-fixed-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh
```

Case 3 runs three measured rollout steps. The fault skips only the update after
rollout 0. All arms use the same `5e-7` learning rate, balanced debug rewards,
fixed worker B replay, deterministic prefill, and Triton MoE runner:

```bash
TUTORIAL_CASE=case3 TUTORIAL_ARM=clean \
TUTORIAL_CASE3_LEARNING_RATE=0.0000005 \
TUTORIAL_CASE3_MOE_RUNNER_BACKEND=triton \
RUN_ID=qwen3-logprob-tutorial-case3-clean-pinned-moetriton-lr5e7-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh

MILES_ENABLE_LOGPROB_FAULT_INJECTION=1 \
MILES_LOGPROB_ACTIVE_FAULT=stale_rollout_weights \
TUTORIAL_CASE=case3 TUTORIAL_ARM=fault \
TUTORIAL_CASE3_LEARNING_RATE=0.0000005 \
TUTORIAL_CASE3_MOE_RUNNER_BACKEND=triton \
RUN_ID=qwen3-logprob-tutorial-case3-fault-pinned-moetriton-lr5e7-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh

TUTORIAL_CASE=case3 TUTORIAL_ARM=fixed \
TUTORIAL_CASE3_LEARNING_RATE=0.0000005 \
TUTORIAL_CASE3_MOE_RUNNER_BACKEND=triton \
RUN_ID=qwen3-logprob-tutorial-case3-fixed-pinned-moetriton-lr5e7-prefilldet-107755-20260904 \
./experiments/logprob_debug/qwen3_30b_a3b/tutorial/run.sh
```

Unset both fault variables before clean or fixed commands. The wrapper rejects
a contaminated environment.

## Build the combined reports

Use `python -m experiments.logprob_debug.qwen3_30b_a3b.tutorial.analyze_tutorial`
with `--case`, `--clean-run`, `--fault-run`, `--fixed-run`, and `--output-dir`.
The Case 3 report was intentionally emitted with
`--allow-signature-failures`: this never relaxes repeatability or identity
checks; it keeps later clean/fixed `<0.01` parity failures visible in the JSON
instead of suppressing the artifact.

Each combined directory contains:

- `table.csv` and `summary.json`/`summary.md`;
- per-position boundary plots and signed-delta histograms;
- a clean/fault/fixed plot and exact-token fault outliers;
- for Case 3, a policy update timeline, logical state labels, functional score
  digests, and finite nonzero trainer gradient norms.

The logical digests hash controller state labels. They are not parameter-tensor
checksums; the report says so explicitly. Full measurements and caveats are in
`TUTORIAL_RESULTS.md`.
