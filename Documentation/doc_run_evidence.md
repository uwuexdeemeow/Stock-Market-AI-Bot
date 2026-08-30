# `run_evidence.py`

## What it does

This script proves that the broker report, execution scorecard, validation
status, and paper-health report came from the same daily run. It records a run
ID, Git version, safe configuration fingerprint, paper-version fingerprint,
hashed account marker, and checksums. It also stores the rebalance lifecycle.
It cannot connect to Alpaca or submit orders.

## How to run it

```bash
python run_evidence.py
python run_evidence.py --check
python run_evidence.py --json
```

The normal command writes `signals/paper_run_manifest.json`. It exits nonzero
when required files are missing, unreadable, or belong to different runs.

## Key terms

- **Run ID:** one name shared by all outputs from a daily run.
- **Fingerprint:** a checksum that changes when important code changes.
- **Evidence manifest:** the final list of verified files and checksums.
- **Input snapshot:** the exact latest feature rows used to calculate a target.
- **Rebalance lifecycle:** planned, submitted, partially filled, filled,
  rejected/timed out, and finally aligned.
