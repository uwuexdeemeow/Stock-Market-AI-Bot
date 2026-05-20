# Monthly Research Runbook

Cadence for the research scripts that don't fire on the daily GitHub
Actions workflows.  Run these locally on the 1st (or close to it) of
each month to keep the live config aligned with the latest data.

## The full monthly ritual

```bash
# Estimated total: 1-2 hours on a 16-32 GB laptop with browser/IDE closed.

# ── Pre-flight: sync with origin ───────────────────────────────────
git fetch origin
git checkout main && git pull --ff-only origin main
git fetch origin signals/latest
git checkout origin/signals/latest -- signals/ logs/

# ── 1. Refresh local research data (~5-10 min) ─────────────────────
python refresh_local_research_data.py

# ── 2. Medium-risk-review trio (~10-15 min) ────────────────────────
#    Each writes a JSON the walkforward reads via medium_risk_review_from_reports
python core_satellite_survivorship_audit.py
python core_satellite_execution_stress.py
python core_satellite_drawdown_throttle.py

# ── 3. Concentration / regime checks (~5 min) ──────────────────────
python concentration_overlay.py
python regime_monitor.py

# ── 4. Nested walkforward (~30-60 min, batched for memory safety) ──
python run_walkforward_batched.py --recent-alpha-grid

# ── 5. Sanity check ────────────────────────────────────────────────
python walkforward_analyzer.py

# ── 6. Publish (only if step 4 produced a new approved family) ─────
#    Skip if step 4 ended with `live_config_approval.approved: True`
#    (the walkforward auto-publishes).  Use this when the approval
#    is False due to an outlier fold but the approved family itself
#    is clean — see commit a27fb0e for the reasoning.
python publish_live_config_from_csv.py --dry-run --source stable_family
# review the dry-run output, then if it looks right:
python publish_live_config_from_csv.py --source stable_family

# ── 7. Regenerate trades + factor decay under the new live config ──
python core_satellite_alpha.py
python factor_decay_monitor.py

# ── 8. Commit + push ───────────────────────────────────────────────
git status   # confirm only expected files changed
git add -f signals/core_satellite_live_configs.json \
          signals/core_satellite_nested_walkforward.json \
          signals/core_satellite_nested_walkforward.csv \
          signals/core_satellite_alpha_metrics.json
git commit -m "monthly research: refresh + walkforward + publish (YYYY-MM-DD)"
git push origin main
```

## Step-by-step rationale

### 1. `refresh_local_research_data.py`

Local mirror of the `factor_data_refresh.yml` workflow.  Actions's
fresh data lives in `actions/cache`, not git — so your local
`data/*.parquet` and feature reports go stale fast.  This wrapper
runs `research.py --incremental` + `feature_quality_diagnostic.py` +
`feature_research.py --top 24 --skip-pairs` + `factor_data_health.py
--strict` in the right order, in separate subprocesses, and verifies
the downstream artifacts.

Add `--pairs` once per quarter for the slow pairwise interaction
sweep that `feature_research.py` can do.

### 2. Medium-risk-review trio

The nested walkforward's `live_config_approval` gate calls
`medium_risk_review_from_reports()` which reads three JSON files:

- `logs/core_satellite_survivorship_audit.json`
- `logs/core_satellite_execution_stress.json`
- `logs/factor_decay_monitor.json`

Old JSONs cause the gate to fail with
`medium_risk_review_failed:<X>_review_missing`.  Refresh these BEFORE
the walkforward so the gate sees current values.

### 3. Concentration + regime checks

`concentration_overlay.py` audits whether the strategy's alpha
concentrates in too few names / sectors.  `regime_monitor.py`
detects whether the live regime classification (risk_on/neutral/
risk_off) has shifted since last month.

These don't gate the walkforward but they help you spot drift
before it's a problem.

### 4. `run_walkforward_batched.py`

Wrapper around `core_satellite_nested_walkforward.py` that runs each
outer fold in a separate Python subprocess.  The walkforward main
process leaks ~1.5-2 GB per outer fold, and on a 31 GB laptop the
straight run OOMs around fold 12.  The batched wrapper resets the
heap between folds.

Auto-detects workers based on `ullAvailPhys`.  On a typical loaded
laptop this picks 2-3 workers; override with `--workers N` if you
have a bigger box.

### 5. `walkforward_analyzer.py`

Independent sanity check on the walkforward result.  Flags four
failure modes that raw return numbers hide:
`score_predictiveness`, `calibration`, `concentration_vulnerability`,
`config_stability`.  Diagnostic warnings, not blockers — they're
useful context for interpreting the headline metrics.

### 6. `publish_live_config_from_csv.py`

The walkforward script auto-publishes only when its run-wide gate
passes (`approved: True`).  If approval fails because of a one-off
outlier fold (e.g. one bad year picked a config that wouldn't ever
trade), the approved FAMILY can still be the right thing to ship.
This script reads the walkforward CSV, applies family-level gates
instead of run-wide, and publishes the most-stable family.

Always `--dry-run` first to confirm what would be published.

### 7. Regenerate trades + factor decay

`core_satellite_alpha.py` rebuilds `signals/core_satellite_alpha_trades.csv`
under the newly-published config.  Then `factor_decay_monitor.py`
reads that trade log to measure recent overlay alpha.  Without this
step the factor decay status reflects the OLD config's trades, not
the new one's.

### 8. Commit + push

Use `-f` because most signals/ files are gitignored except for the
explicit `!`-exception list in `.gitignore`.  The four files above
are the canonical state of "this month's research".

## What to look for in the output

### Walkforward result (`signals/core_satellite_nested_walkforward.json`)

| Field | Pass criterion | Notes |
|-------|----------------|-------|
| `mean_oos_sharpe` | ≥ 0.50 | Lower = the strategy isn't doing real work |
| `mean_oos_alpha_vs_spy_pct` | > 0 | Negative means SPY beats us |
| `worst_oos_max_drawdown_pct` | ≥ -35% | Hard gate; tighter than mean |
| `worst_oos_turnover_pct` | ≤ 600% | Run-wide cap; family-level can be lower |
| `oos_positive_alpha_hit_rate` | ≥ 0.60 | What % of folds beat SPY |
| `selection_bias_gap_sharpe` | ≤ 1.0 | Inner-vs-OOS Sharpe gap; high = overfit |
| `cost_stress_approval_pass` | True | Strategy survives 2x/3x/5x cost stress |
| `medium_risk_review.pass` | True | Trio of inputs from step 2 |

### Factor decay (`logs/factor_decay_monitor.json`)

| Field | Acceptable | Block at |
|-------|-----------|----------|
| `edge_health_status` | `pass` or `advisory` | `block` |
| `real_capital_block` | False | True |
| 60d `overlay_alpha_sum_pct` | > -1.5% with ≥4 trades | Below threshold with sample |
| 120d `top_bucket_excess_return_pct` | > 0% | < 0% |

A `warning` status doesn't halt live trading (`real_capital_block: False`)
but signals the score's recent cross-section ranking has weakened.
If you see it persist for 2+ months, look at the feature decay CSV
(`signals/feature_research_summary.csv`) to see which features are
quarantined and consider a retrain (`train.py`).

## Common failures

### "feature_research_summary.csv missing"

`feature_research.py` was deleted at some point.  Restore from git:

```bash
git show HEAD~10:feature_research.py > feature_research.py
# or find the last commit that had it:
git log --all --diff-filter=D -- feature_research.py
```

Then re-run `refresh_local_research_data.py`.

### "cost_stress_approval_failed"

Some inner fold failed 5x cost stress.  Common cause: high-turnover
candidate won selection.  Either tighten
`MAX_INNER_MEAN_TURNOVER_PCT` in `core_satellite_nested_walkforward.py`
or accept the failure if the published FAMILY itself passes.

### "worst_oos_turnover > 600%"

One outlier fold picked a high-turnover config.  Check which fold
and which family.  If the family is one-of-one (only appeared in
that outlier fold), it won't trade live and you can publish the
approved-family config manually via `publish_live_config_from_csv.py`.

### "factor_decay_review_missing"

`logs/factor_decay_monitor.json` is missing or stale.  Run step 7
above (`python factor_decay_monitor.py`) and the JSON regenerates.

### Walkforward subprocess OOM-killed

The leak.  Use `run_walkforward_batched.py` (which we already
default to in step 4) and/or close Chrome/Discord/IDE before
launching.  See `Documentation/doc_run_walkforward_batched.md` for
the full story.

## Quarterly add-ons

In addition to the monthly steps, every 3 months:

```bash
# Slow pairwise feature interaction analysis
python refresh_local_research_data.py --pairs

# ML model retrain (if you have one)
python train.py
python predict.py
python model_quality.py
python confidence_calibration.py
```

After the retrain, run the full monthly ritual again to validate
that the new model + new IC data still produces an approvable config.
