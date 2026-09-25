# Codebase Audit — 2026-09-26

## What was checked

- **Static scan** of every root script with `ruff` (undefined names, unused
  variables, loop-variable closures, unused imports).
- **Import/reference graph**: which scripts are actually reached from the
  GitHub workflows, the dashboard, the shell/batch helpers and the tests.
- **Manual review** of the math in performance metrics (Sharpe, Sortino,
  CAGR, drawdown, Newey-West t-stat), order sizing and cash fitting, regime
  allocation, overlay weighting, walk-forward fold splitting and purging, and
  cost-model inputs.
- **Test suite** before and after the changes: 813 passed, 24 skipped both
  times. `paper_validation_epoch.py --check-lock` still passes.

## Fixed in this pass

| File | Problem | Fix |
|---|---|---|
| `backtest.py`, `pages/1_Performance.py` | Sortino ratio divided by the plain std of the losing days only. That number shows how *uneven* the losses are, not how *big* they are, so the ratio came out wrong. | Now uses downside deviation: the root-mean-square of `min(return, 0)` over **every** day. |
| `backtest.py` `_newey_west_tstat` | Lag-k autocovariance was averaged over `n-k` points. Standard Newey-West divides by `n`. | Divide by `n`. This keeps the variance estimate positive semi-definite. |
| `backtest.py` `_portfolio_stats_from_equity` | CAGR annualized over `len(equity)` days. N equity points only cover N-1 daily returns. | Uses `(len-1)/252`. |
| `backtest.py` trade simulation | The entry-cost model used the entry day's full volume and closing price, but the order fills at that day's **open** (a small look-ahead in the costs). | ADV now uses only the days before entry. Fill volatility is measured at the signal date. |
| `backtest.py` | The variable `eff_slip` was computed but never used; the same expression was repeated 4 times. | The fill calls now use `eff_slip`. |

## Found but NOT changed (these files are locked by the paper release)

These files are fingerprinted in `paper_version_lock.json`. Editing them makes
`paper_validation_epoch.py --check-lock` fail, and the daily paper run then
stops placing orders until you re-freeze the release. That decision is yours.

1. **`alpaca_paper_trading.py` `snapshot_equity`**: it stamps the row date
   with `datetime.now()`. GitHub runners run on UTC, so a status run after
   8 PM New York time labels today's equity as tomorrow's. Suggested fix:
   ```python
   now = datetime.now(EXECUTION_TIMEZONE).replace(tzinfo=None)
   ```
2. **`paper_health.py` live-vs-backtest drift**: live CAGR uses
   `years = live_days / 252`. N snapshots span N-1 trading days. Suggested fix:
   `years = (live_days - 1) / 252.0`.
3. **`pipeline_shared.py` `add_technical_features`**: `target` is built as
   `(c.shift(-H) > c).astype(int)`, which gives the last H rows a label of
   0 (down) when they should have no label. The later `dropna` on `target`
   can't catch this, because `fillna(0.0)` runs first. Today nothing trains
   on this column (training uses `labels.py`, and every feature list drops
   `target`), so it causes no harm yet. Don't start reading it as a label
   without fixing it first.

## Other observations (low severity, not changed)

- `predict.py` computes a calibrated `p_up` and then throws it away (only the
  raw blend is used). `train.py` does the same with `test_p_up`. Either wire
  calibration in or delete the calibrator loading.
- `core_satellite_alpha._cap_and_rescale`: the final
  `capped / total * min(gross, total)` line does nothing (it multiplies by
  `total / total`). It does no harm.
- Several performance pages annualize with `sqrt(252)` on equity snapshots.
  That is only correct if there is exactly one snapshot per trading day.
- `tests/test_strategies.py` still holds four `@pytest.mark.skip` classes
  (lines ~344-547). They call functions that were deleted with the old
  broker path, which is why ruff reports 16 undefined names. They can be
  deleted.

## Removed as obsolete

- **Scripts nothing imports or runs**: `model.py` (old PyTorch transformer;
  the system is XGBoost/factor based now), `diagnostics.py` (torch/CUDA
  environment check), `ci_check_feature_report.py` (no workflow calls it any
  more).
- **Docs for scripts that no longer exist**: `doc_scanner.md`,
  `doc_ensemble_utils.md`, `doc_tune_xgb_best_tickers.md`,
  `doc_dynamic_weighting.md`, `doc_setup.md`, `doc_model.md`,
  `doc_diagnostics.md`.
- **Old v1–v3 Word documents** (April–May 2026, scanner/LSTM era): all 18
  `Documentation/*.docx`.
- **Old root notes**: `DYNAMIC_WEIGHTING_README.md`, `FIXES_APPLIED.md`,
  `README_PRODUCTION.md`, `ANALYSIS_COVERAGE.md`, and the root
  `WEEKLY_RUNBOOK.md` pointer (the real runbook is
  `Documentation/WEEKLY_RUNBOOK.md`).
- `autoresearch-results.prev-20260526-202246/` (tracked snapshot from May).
- References updated in `CLAUDE.md`, `AGENTS.md`, `dashboard/scripts.py` and
  `Documentation/WEEKLY_RUNBOOK.md`.

## Local clutter (untracked, left in place)

These are git-ignored and exist only on this computer. They were not deleted
because you can't get them back afterwards:

- `research_snapshots/` (~303 MB), `logs/` (~271 MB, includes `logs/pytest_*`)
- `autoresearch-results/` (~0.6 MB)
- 10 `.pytest_tmp*` folders in the project root (Windows denies access to them)

## Follow-up pass (cloud session, 2026-09-26)

Finished every open item from `HANDOFF_2026-09-26.md` that does **not** need a
locked file. Full test suite: 834 passed, 9 skipped (before: 827 passed, 24
skipped; 15 of those skips were the deleted legacy tests).
`paper_validation_epoch.py --check-lock` still passes.

| Item | What changed |
|---|---|
| Handoff task 1: stress a research candidate without publishing it | `core_satellite_execution_stress.py` and `core_satellite_survivorship_audit.py` take `--candidate-json PATH` (walk-forward result or plain config) and optional `--candidate-name`. Output goes only to `logs/research_candidate_execution_stress_<name>.*` / `logs/research_candidate_survivorship_audit_<name>.*`, stamped `research_candidate: true`, `approves_trading: false`. No flag = old behavior. Tests: `tests/test_research_candidate_stress.py`. |
| Bug caught while writing those tests | The execution-stress scenario loop reuses the variable `name`, so a first draft marked official reports as candidates. The candidate variable is now `cand_name`, and a test covers the default path. |
| Handoff task 4: legacy skipped tests | Deleted the four `@pytest.mark.skip` classes in `tests/test_strategies.py` (they called functions that no longer exist) and their unused imports. |
| Handoff task 4: `predict.py` unused calibrated `p_up` | Removed the dead calibrator load (predictions unchanged). Reason: the calibrator was fitted on the 20-day model only, but the live number is the 20-day + 5-day blend. Still open: `train.py` builds per-ticker confidence buckets from **calibrated** probabilities while `predict.py` scores the raw blend. `predict.py` is not in the daily paper workflow. |
| Missing docs | Added `doc_feature_research.md`. Every root script now has a doc. |
| Broader static re-scan | `ruff` (undefined names, loop closures, unused variables): no undefined names; every loop-closure warning is a lambda that runs inside the loop, so it is safe. Remaining unused variables are cosmetic. |

Still waiting on the owner (locked files, not touched): handoff tasks 2
(`snapshot_equity` timezone, `paper_health` years off-by-one) and 3 (stale
embedded stress result in `core_satellite_alpha.py`), and the
`pipeline_shared.py` `target` note above.
