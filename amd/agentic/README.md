# AMD MI355X agentic/multi-turn qualification harness

This directory is the AMD-owned qualification overlay described in the
agentic/multi-turn roadmap. It is intentionally isolated from Miles core and is
not proposed as an upstream API.

## What Phase 0 provides

- A dependency-free preflight manifest with source, image, runtime, GPU, asset,
  launch, allowlisted environment, and service-health evidence.
- Strict launch gates for clean source, pinned image/assets, expected GPU shape,
  and healthy required services.
- An exact-token trajectory sidecar and validator covering prefix inheritance,
  token/logprob/mask alignment, tool observations, status/finish consistency,
  turn/tool counters, and finalize/delete lifecycle.
- Offline run summarization and comparison without WandB.
- Disabled ReTool and SWE launcher scaffolds. They cannot execute until later
  phases pin their artifacts and commands.

The checked-in `.yaml` files use JSON syntax. JSON is a valid YAML subset, so
the preflight remains standard-library-only while still allowing conventional
YAML when PyYAML is installed.

## Directory layout

```text
amd/agentic/
  configs/       workload contracts; disabled until assets are ready
  launch/        preflight and guarded workload launchers
  manifests/     JSON schemas for manifests and trajectory sidecars
  results/       local result bundles (only README is tracked)
  tests/         deterministic fixtures
  tools/         validation, summary, and comparison CLIs
tests/fast/amd_agentic/
  ...            executable tests under Miles' pytest discovery tree
```

## Fixed environment

Use `documents/env_setup_commands.txt` from the parent workspace. Inside the
container:

```bash
cd /workspace/miles
pip install -e . --no-deps
```

The preflight uses a per-command git `safe.directory` override for the
bind-mounted checkout; it does not alter user or repository git configuration.

## Bootstrap collection

The Phase 0 config pins the deterministic fixture and should pass strict
preflight on a clean checkout:

```bash
python -m amd.agentic.launch.preflight \
  --config amd/agentic/configs/phase0_harness_mi355x.yaml \
  --repo-root /workspace/miles \
  --run-id phase0-bootstrap \
  --output amd/agentic/results/phase0-bootstrap/manifest.json
```

The ReTool and SWE configs deliberately fail strict qualification while their
model, dataset, and launch fields are unpinned. Use `--collect-only` with those
configs to record blockers during preparation. Before training, remove
`--collect-only`; a nonzero exit is a hard stop.

## Deterministic two-tool acceptance gate

```bash
python -m amd.agentic.tools.validate_trajectory \
  amd/agentic/tests/fixtures/two_tool_trajectory.json \
  --repeat 100 \
  --output amd/agentic/results/phase0-bootstrap/trajectory_validation.json
```

The command must report `validated=100 valid=100 invalid=0`.

## Offline summary

```bash
python -m amd.agentic.tools.summarize_run \
  --manifest amd/agentic/results/phase0-bootstrap/manifest.json \
  --validation-report amd/agentic/results/phase0-bootstrap/trajectory_validation.json \
  --output amd/agentic/results/phase0-bootstrap/summary.json
```

The summary command returns nonzero while preflight is blocked. This is
intentional: the file is still written for diagnosis, but it cannot be mistaken
for a qualified run.

Compare two qualified summaries with:

```bash
python -m amd.agentic.tools.compare_runs \
  --baseline <baseline-summary.json> \
  --candidate <candidate-summary.json> \
  --output <comparison.json>
```

## Guarded launchers

When a future phase has enabled and pinned a config:

```bash
python -m amd.agentic.launch.run_retool --manifest <manifest.json>
python -m amd.agentic.launch.run_retool --manifest <manifest.json> --execute
```

The first form prints the exact command. The second executes it. Both reject a
failed preflight, dirty source, mismatched workload, disabled config, or empty
command. `run_swe` follows the same contract.

## Security and publication

- Environment collection is allowlist-only; secret-looking variable names are
  redacted even if explicitly listed.
- The source patch is capped at 1 MiB and its full SHA-256 is retained.
- Result bundles may still contain sensitive prompts or tool output. Review
  them before sharing.
- A directory layout hash is not a content checksum. Every required model and
  dataset must declare an immutable revision or SHA-256 before strict launch.
