# AMD agentic qualification results

Generated result bundles belong under a run-ID directory:

```text
results/<run-id>/
  manifest.json
  trajectory_validation.json
  summary.json
  raw/
```

Only this README is tracked. Run outputs can be large and may contain prompts,
tool results, repository content, or environment metadata; review and redact a
bundle before publishing it.

The manifest and summary are WandB-independent. A result is publishable only
when `qualification.passed` and trajectory validation both pass.
