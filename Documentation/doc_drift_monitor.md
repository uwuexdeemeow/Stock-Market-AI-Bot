# drift_monitor.py — Feature Drift Monitor

## What It Does

After successful training, `train.py` saves a compact baseline describing the
model's input distributions. The baseline stops at the final date used by the
training split, so calibration, test, and newer rows cannot make the baseline
look artificially familiar. The daily workflow compares the latest 60 rows per
ticker with those honest training-time distributions.

Results are appended to `logs/drift.jsonl`. `monitor.py` reads the newest entry
and alerts when PSI or KS crosses its configured threshold. If no baseline
exists yet, the monitor reports `no_data` without blocking paper orders; the
next genuine training run creates the baseline automatically.

## How To Run It

Check the pooled model manually:

```bash
python3 drift_monitor.py --run-name pooled
```

Expected output includes an overall status (`ok`, `caution`, `drift`, or
`no_data`) and per-feature PSI/KS measurements.

## Key Concepts

- **Baseline:** Compact picture of normal feature values from the training era
  only. It does not include the model's later calibration or test data.
- **PSI:** Measures changes in how values fill training-time bins.
- **KS statistic:** Largest distance between current and baseline distributions.
- **Drift:** Inputs changed enough that model predictions deserve caution.
