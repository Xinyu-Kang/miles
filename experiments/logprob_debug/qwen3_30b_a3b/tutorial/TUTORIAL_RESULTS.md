# Controlled logprob fault-injection results

Run date: 2026-09-04

## Outcome

All three deliberate faults were localized at their intended A/B/C boundary,
and removing each fault returned that boundary to its clean range. No injected
fault is presented here as a naturally discovered framework bug.

One caveat matters: Case 3 has exact B and C repeatability, but its synchronized
clean/fixed B-C MAE rises to `0.0125-0.0128` after training. That misses the
strict `<0.01` parity criterion. The combined report retains those failed checks
and this tutorial does not pursue their low-level cause, per the switch-of-focus
task.

## Common table

All numbers are mean absolute differences in nats. `r` is rollout index.

| case/arm | r | A-B | B-C | A-C | B0-B1 | C0-C1 | first failed gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Case 0 clean | 0 | 0.009756 | 0.009218 | 0.009337 | 0.009198 | 0 | none |
| Case 1 fault | 0 | 0.100791 | 0.009306 | 0.102758 | 0.008069 | 0 | A vs B |
| Case 1 fixed | 0 | 0.006772 | 0.008432 | 0.007703 | 0.007856 | 0 | none |
| Case 2 fault | 0 | 0.008086 | 0.102164 | 0.102420 | 0.008341 | 0 | B vs C |
| Case 2 fixed | 0 | 0.006334 | 0.007119 | 0.007712 | 0.007535 | 0 | none |
| Case 3 clean | 0 | 0.006738 | 0.009747 | 0.007954 | 0 | 0 | none |
| Case 3 clean | 1 | 0.009614 | 0.012532 | 0.012242 | 0 | 0 | B vs C |
| Case 3 clean | 2 | 0.008622 | 0.012838 | 0.012162 | 0 | 0 | B vs C |
| Case 3 fault | 0 | 0.006738 | 0.009747 | 0.007954 | 0 | 0 | none |
| Case 3 fault | 1 | 0 | 10.951996 | 10.951996 | 0 | 0 | B vs C |
| Case 3 fault | 2 | 0.006548 | 0.012223 | 0.012002 | 0 | 0 | B vs C |
| Case 3 fixed | 0 | 0.006738 | 0.009747 | 0.007954 | 0 | 0 | none |
| Case 3 fixed | 1 | 0.009614 | 0.012532 | 0.012242 | 0 | 0 | B vs C |
| Case 3 fixed | 2 | 0.008622 | 0.012838 | 0.012162 | 0 | 0 | B vs C |

The first gate always checks B repeatability, then C repeatability, then A-B,
B-C, and A-C. Thus B-C localization is used only when both references are
below `0.01`.

## Case 1: decode-logprob bookkeeping

Known injected cause: the rollout postprocessor subtracts `0.10` nat from the
stored A score at response positions 1+, while keeping position 0, tokens, and
model logits unchanged. It stores the clean and faulted values in sample
metadata.

Observation: fault-arm A-B MAE is `0.100791` and A-C is `0.102758`, while B-C
stays at `0.009306`. A-B at position 0 is `0.00000799`; at positions 1+ it is
`0.101585`. B-repeat is `0.008069` and C-repeat is exactly zero.

Inference: exact-token replay and the trainer agree within the accepted floor,
so the first failing boundary is generation-logprob bookkeeping, not model
state or trainer scoring.

Fix verification: removing the postprocessor gives A-B `0.006772`, B-C
`0.008432`, and A-C `0.007703`, all below `0.01`.

The clean/fault/fixed runs use identical prompt tokens and seeds. Some sampled
responses differ across independent reruns, so cross-arm values are aggregate,
not paired-token claims. Within each run all A/B/C comparisons use identical
response token IDs; the postprocessor's no-token-mutation property is tested.

Artifacts:

- clean: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case0-clean-prefilldet-107755-20260904`;
- fault: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case1-fault-prefilldet-107755-20260904`;
- fixed: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case1-fixed2-prefilldet-107755-20260904`;
- combined: `/workspace/logprob-debug-artifacts/tutorial-comparisons/case1`.

## Case 2: probability-definition mismatch

The intended real configuration mismatch was tested first. Source/runtime
captures prove that with `SGLANG_RETURN_ORIGINAL_LOGPROB=1`, A caches raw
pre-temperature SGLang log-softmax, B is an exact-token raw prefill score, and C
divides fp32 logits by `rollout_temperature` in this non-true-on-policy mode.

Observation from the native pilots:

| temperature | A-B | B-C | A-C | B repeat | C repeat |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.007061 | 0.009959 | 0.009756 | 0.007568 | 0 |
| 0.8 | 0.009063 | 0.014414 | 0.012658 | 0.007815 | 0 |

The `0.9` B-C signed/tokenwise effect correlates with token surprisal
(Pearson `r = 0.726`), as expected for a temperature transform, but neither
pilot reaches the requested `0.07` teaching signal. The specification permits
an explicitly synthetic fallback only after reporting those actual semantics.

Known injected cause in the final arm: a debug-only trainer hook subtracts
`0.10` nat from both frozen C repeats. It does not alter A, B, tokens, or
weights.

Observation: A-B remains `0.008086`; B-C becomes `0.102164` and A-C becomes
`0.102420`. B-repeat is `0.008341`; C-repeat remains exactly zero because the
same definition is used for C0 and C1.

Inference: stable references plus A-B agreement localize the first failure to
the SGLang-prefill versus trainer probability definition.

Fix verification: removing the trainer hook gives A-B `0.006334`, B-C
`0.007119`, and A-C `0.007712`.

Artifacts:

- native pilots: `qwen3-logprob-tutorial-case2-fault-t09-prefilldet-107755-20260904`
  and `qwen3-logprob-tutorial-case2-fault-t08-prefilldet-107755-20260904`;
- fault: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case2-fault-fallback-prefilldet-107755-20260904`;
- fixed: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case2-fixed-prefilldet-107755-20260904`;
- combined: `/workspace/logprob-debug-artifacts/tutorial-comparisons/case2`.

## Case 3: stale rollout weights

The existing `--update-weights-interval` was inspected before adding a hook.
In this checkout it gates rollout/old-actor backup behavior when
`keep_old_actor` is active; the synchronous driver still calls the rollout
weight updater after every rollout. Setting it to two would therefore change
algorithm bookkeeping without creating the requested missed broadcast. The
narrow controller hook is used instead.

Known injected cause: the controller deliberately skips the rollout weight
update after rollout 0, without changing collective ordering. The trainer
advances from logical state T0 to T1 while the rollout engines remain at T0.
The next update completes normally, so the rollout side reaches T2 for rollout
2. Clean and fixed arms update after every rollout.

Observation: rollout 0 is identical across all arms. At fault rollout 1, A and
B are identical (`A-B = 0`) at rollout policy version 1, while trainer C uses
the post-update state and both B-C and A-C are `10.951996`. At rollout 2 the
logical states are synchronized again and the gap returns to the clean-scale
range. The stale signal overshoots the desired `0.08-0.15` pedagogical range,
even with a matched `5e-7` learning rate; its boundary signature remains
unambiguous and is reported without rescaling.

All B and C repeats are exactly zero. Trainer gradients are finite and nonzero:
clean/fixed norms are `0.280335`, `0.503984`, and `0.580007`; fault norms are
`0.280335`, `1.810196`, and `0.662455`.

Inference: because A and B agree exactly while their logical rollout state
differs from trainer C only at rollout 1, diagnosis stops at model-state
synchronization. Kernel tracing would be the wrong next step.

Fix verification: the fixed arm exactly reproduces every clean-arm aggregate
and response sequence. Later synchronized B-C values (`0.012532` and
`0.012838`) remain above the strict parity threshold in both clean and fixed
arms. This is a natural cross-runtime floor, not a C-repeatability problem and
not evidence against the stale-weight localization.

The timeline's logical hashes are controller-state witnesses, not parameter
tensor checksums. Functional SHA256 digests bind exact token/score rows, and
the saved rollout policy versions plus update log markers provide the tested
state-transition evidence.

Artifacts:

- clean: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case3-clean-pinned-moetriton-lr5e7-prefilldet-107755-20260904`;
- fault: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case3-fault-pinned-moetriton-lr5e7-prefilldet-107755-20260904`;
- fixed: `/workspace/logprob-debug-artifacts/qwen3-logprob-tutorial-case3-fixed-pinned-moetriton-lr5e7-prefilldet-107755-20260904`;
- combined: `/workspace/logprob-debug-artifacts/tutorial-comparisons/case3`.

## Preserved natural-system baseline

Before fault injection, controlled scoring of one saved sample on node 072
found B pairwise MAE `0.010828` within one fixed worker, `0.013921` across four
workers, and `0.012633` through the router. All router requests happened to hit
worker `:15006`, so router repeats did not independently sample reassignment.
This establishes that the original floor is not solely a router effect.

The matched Triton deterministic-prefill OFF/ON experiment then reduced
B-repeat from `0.017180` to `0.008333`; ON also put A-B, B-C, and A-C below
`0.01` at rollout 0, with C-repeat zero. The full high-level deterministic mode
changes multiple internals, so this only establishes a causal mitigation. A
source/runtime audit further localized the repeatability contributor to the
auto/AITER MoE execution path versus Triton in this image; it did not prove one
specific kernel as the sole root cause. No asynchronous HIP error is used as
causal evidence.

Full natural-baseline provenance and results remain in the parent `RESULTS.md`.
No SGLang/AITER kernel was patched, TIS was not used, no PR was opened, and the
public blog draft was not edited.
