# workflow_failure_annotation.py

## What it does

This helper turns failed GitHub Actions step results into one visible error
annotation. The annotation names the failed step IDs and links to the run.

## How to run it

Workflows set `WORKFLOW_STEPS_JSON` and `WORKFLOW_RUN_URL`, then run:

```bash
python3 workflow_failure_annotation.py \
  --workflow-file .github/workflows/example.yml \
  --workflow-name "Example workflow"
```

Expected output is a GitHub `::error` annotation. This script diagnoses a
failure; it does not retry jobs or place trades.

## Key concepts

- **Step ID:** a stable short name assigned to a workflow step.
- **Outcome:** GitHub's final result for a step, such as success or failure.
- **Annotation:** a prominent message shown on the workflow run summary.
