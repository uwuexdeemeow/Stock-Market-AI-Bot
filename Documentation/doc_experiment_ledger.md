# Experiment Ledger

## What it does

`experiment_ledger.py` appends each research run's settings, metrics, artifacts,
timestamp, and Git commit to a machine-readable history.

## How to use it

Import `append_experiment(name, params, metrics, artifacts, notes)` from a
research script. It writes `logs/experiment_ledger.jsonl` and a flattened CSV
mirror. JSONL is the source of truth; CSV is convenient for a spreadsheet.

## Key terms

- **Append-only:** add new records without rewriting old evidence.
- **Artifact:** an output file produced by an experiment.
- **Git commit:** the exact code version used for the run.
