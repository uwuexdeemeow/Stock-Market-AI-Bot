# model_registry.py — Reproducible Model History

## What It Does

Every successful training run appends one entry to `models/registry.json`.
The entry records the model name, Git commit, research-data fingerprint,
Python version, training command, headline metrics, and a checksum for every
saved artifact.

Plain language: the registry is a receipt showing exactly which code and data
created a model. If a model file is later changed or damaged, verification
detects that its checksum no longer matches.

## How To Run It

Training registers models automatically:

```bash
python3 train.py --ticker AAPL
```

Inspect and verify the newest AAPL run:

```bash
python3 model_registry.py --run-name AAPL
```

Inspect the newest pooled run:

```bash
python3 model_registry.py --run-name pooled
```

Expected output is JSON ending with `"artifacts_valid": true`. A missing or
changed artifact returns a nonzero exit code and lists the problem.

## Key Concepts

- **Registry:** Append-only history of successful training runs.
- **Git commit:** Exact source-code version used for the run.
- **Dataset fingerprint:** Checksum identifying the research input snapshot.
- **Artifact:** Saved model, scaler, calibrator, or training summary.
- **Checksum:** File fingerprint used to detect later changes.
