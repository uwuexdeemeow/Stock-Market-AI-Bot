# refresh_local_research_data.py — Local mirror of the CI research refresh

## What it does (plain English)

The two GitHub Actions workflows (`factor_data_refresh.yml` and
`daily_paper_trading.yml`) keep the live trading bot's data fresh
on Actions's own machines.  But Actions stores its heavy outputs in
`actions/cache` (an Actions-only store), not git, and the daily
workflow only commits trading-side outputs (orders, fills, equity).

That means **your local copy of `data/*.parquet`, the feature quality
report, the factor health JSON, etc. does NOT update by itself**.
If you run any research locally — `nested_walkforward`,
`train.py`, `predict.py`, an ad-hoc backtest — it will use whatever
files were on disk the last time YOU refreshed.

`refresh_local_research_data.py` is the local equivalent of the
`factor_data_refresh.yml` workflow.  It chains the four refresh
scripts in the right order, halts on critical failures, and verifies
that the downstream files actually got written before letting you
move on.

## Why it exists

Before this script, "refresh research locally" meant remembering
four separate commands (research → feature_quality_diagnostic →
feature_research → factor_data_health) in the exact right order,
plus knowing which scripts are slow enough to skip on routine
refreshes and which ones are critical-vs-nice-to-have.  This
captures that runbook as code, so you can just type one command
before any local research session.

## How to run

```bash
# Full refresh (panel + features + IC decay + health gate)
python refresh_local_research_data.py

# Quarterly mode — adds the slow pairwise interaction analysis
python refresh_local_research_data.py --pairs

# Routine refresh — skip the IC decay CSV (it's good for ~3 months)
python refresh_local_research_data.py --skip-feature-research

# Just rebuild the feature reports, leave parquets alone
python refresh_local_research_data.py --skip-research

# Show what would run without executing
python refresh_local_research_data.py --dry-run
```

## Inputs

The script doesn't take any data inputs of its own.  Each step
loads the data it needs from where the prior step wrote it:

| Step | Reads | Produces |
|------|-------|----------|
| `research.py --incremental` | `data/*.parquet` (last bar dates), `signals/watchlist.json` | Updated `data/*.parquet` |
| `feature_quality_diagnostic.py --top 48` | Updated parquets | `signals/feature_quality_report.json`, `signals/feature_quality_summary.csv` |
| `feature_research.py --top 24 --skip-pairs` | Updated parquets, feature specs | `signals/feature_research_summary.csv`, `signals/feature_research_report.json` |
| `factor_data_health.py --strict` | All of the above | `signals/factor_data_health.json` (pass/fail) |

## Outputs

After a successful run you'll have fresh copies of:

- `data/*.parquet` — per-ticker OHLC + factor features
- `signals/feature_quality_report.json` + `feature_quality_summary.csv`
- `signals/feature_research_summary.csv` + `feature_research_report.json`
- `signals/feature_health_profile.json` (built downstream by `alpha_factor_backtest`)
- `signals/factor_data_health.json`

These are exactly the files that downstream research scripts
(`nested_walkforward`, `train.py`, etc.) expect to find.

## Key concepts

- **Panel** — the big tabular dataset of (ticker × date) rows used by
  all the strategy / training / backtest code.  Stored as one
  parquet file per ticker so we can do incremental updates without
  re-reading everything.
- **Incremental refresh** — `research.py --incremental` only
  downloads bars newer than the last bar already in each parquet,
  so a daily refresh is fast (~1 min) even though a from-scratch
  refresh would take an hour.
- **Feature quality** — a small report grading each candidate
  feature on its predictive IC over recent history.  Drives which
  features pass the live trading gate.
- **IC decay CSV** — `feature_research_summary.csv` records each
  feature's full-history IC vs its recent-window IC.  When
  `feature_health.py` decides which features to quarantine, it
  reads this CSV — without it, every feature defaults to "healthy"
  and decayed features dilute the live score.
- **Critical step** — a refresh step where failure aborts the
  whole run.  `research.py` is critical because everything else
  reads from its outputs.  `feature_research.py` is best-effort —
  if it fails, the prior CSV is still usable (just stale).

## Sequencing notes

These steps MUST run in the listed order:

1. `research.py` writes new parquet rows.  Steps 2-4 all read parquets.
2. `feature_quality_diagnostic.py` writes a fresh feature report.
   `factor_data_health.py --strict` (step 4) blocks if this report
   is too old, so step 2 must run before step 4.
3. `feature_research.py` is independent of step 2 but must run
   after step 1 (it needs fresh forward returns from the panel).
4. `factor_data_health.py` is the final gate.  Failure here means
   *something* upstream is off; don't run further research until
   you understand why.

## When to run

| Cadence | Args | Notes |
|---------|------|-------|
| Before any local research session | (default) | Routine: ~5-10 min total |
| Before quarterly retrain | `--pairs` | Adds slow pairwise analysis (~30 min) |
| Already refreshed parquets today | `--skip-research` | ~3 min |
| Already refreshed feature_research this quarter | `--skip-feature-research` | ~2 min |

## Exit codes

- `0` — every requested step succeeded; downstream artifacts present
- `1` — a critical step failed (or a downstream artifact missing);
  later steps were skipped to avoid building on broken inputs
- `2` — argparse / configuration error
