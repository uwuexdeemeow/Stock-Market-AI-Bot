# Plan: unhook and retire the legacy ML layer (2026-10-25)

**Status: Phases 1–4 DONE on branch `legacy-cleanup` (owner: "do all the
phase", 2026-10-25). Merge after the 2026-11-02 V rebalance reconciles,
together with `deps-ci-audit`, which this branch already contains.** See
"What was done" at the end. CLAUDE.md says legacy files are not deleted,
moved or renamed without a plan and the owner's agreement. This is that
plan.

**Key term — legacy layer:** the old XGBoost prediction code (`train.py`,
`predict.py`, `labels.py` and the rest). It was switched off for picks
because model AUC was about 0.51, a coin flip. It still sits underneath
live code, and it costs:
- CI time;
- big dependencies (xgboost, scikit-learn, shap, torch, matplotlib), each one more pip-audit target;
- confusion about what is live.

## What the import map shows (AST scan, 2026-10-25)

- **The rule V daily path does not touch it.**
  - These scripts reach 3 to 22 project modules each, and none of them is
    legacy: `daily_run.py`, `alpaca_paper_trading.py`, `v_rule_signal.py`,
    `broker_truth.py`, `paper_health.py`, `run_evidence.py`,
    `execution_scorecard.py`.
- **One bridge carries it in: `backtest.py`.**
  - It imports `train`, `labels`, `model_quality`, `confidence_calibration`,
    `calibration_stability` and `xgb_feature_engineering` at module load.
  - `train` then pulls in `model_registry`, `nested_cv` and `drift_monitor`, plus matplotlib.
- **Quant code uses only four small names from `backtest.py`** (4,515 lines):

  | Name | Used by |
  |---|---|
  | `INITIAL_CAPITAL` | `alpha_factor_backtest`, `core_satellite_alpha`, `core_satellite_drawdown_throttle`, `core_satellite_tqqq` |
  | `_load_etf_price_frame` | `alpha_factor_backtest`, `core_satellite_alpha` |
  | `_newey_west_tstat` | `alpha_factor_backtest`, `factor_decay_monitor`, `quant_performance_audit` |
  | `DATA_DIR` (from `settings`) | `core_satellite_nested_walkforward` |

- **`forward_test_monitor.py`**, which scores V each month, imports
  `_nyse_sessions` from `core_satellite_alpha`. That one line loads
  `backtest` → `train` → xgboost. `v_rule_signal.py` has the same function already.
- **Never imported by live code:** `predict`, `model_self_check`,
  `shap_feature_reducer`, `leakage_audit`, `alternative_data_features`.
- **Sizes:**
  - Legacy scripts: about 7,400 lines, plus `backtest.py`.
  - Legacy outputs: `models/` 84 MB, `research_snapshots/` 303 MB, `archive/` 5 MB, `autoresearch-results/` 0.6 MB.
  - 11 test files import legacy modules.

## Phases (one PR each; each passes the full suite before merge)

### Phase 1: cut the link (small; can ship with the CI-deps merge)
1. Create `quant_common.py` holding `INITIAL_CAPITAL`, `_load_etf_price_frame`
   and `_newey_west_tstat`, moved verbatim. `backtest.py` re-imports them from
   there, so the legacy code keeps working unchanged.
2. Point the seven quant files above at `quant_common` (and `settings.DATA_DIR`).
3. `forward_test_monitor.py`: take `_nyse_sessions` from `v_rule_signal`.
4. Add a guard test: importing each live entry point must not load any
   legacy module (`sys.modules` check). The link can't come back by accident.
5. Proof that nothing changes:
   - before and after, same data, same output hashes for
     `core_satellite_alpha.py --validation-refresh`, `factor_decay_monitor.py`
     and `forward_test_monitor.py --rule v`;
   - full test suite passes.
6. `core_satellite_alpha.py` is in the paper lock, so this needs a refreeze.
   It can share one refreeze with the `deps-ci-audit` branch.

**Changes nothing that V trades. The FV clock is unaffected.**

### Phase 2: move the code out of the root
- Move the 14 legacy scripts (and `backtest.py`, once Phase 1 leaves it with
  no live users) into `legacy/`.
- Move their docs to `Documentation/legacy/`.
- Split the 11 test files:
  - legacy-only tests move to `tests/legacy/`, run in a separate CI job that does not block;
  - mixed files (`test_sanity`, `test_corrected_audit`, `test_submission_history_guards`, `test_sentiment_fallback`) keep their live parts.
- Update the script lists in CLAUDE.md.
- **Owner decision:** move to `legacy/`, or leave the files in place and only
  take them out of the blocking CI. Moving is clearer, but it breaks old
  command lines such as `python train.py`.

### Phase 3: drop the heavy dependencies
- Once Phase 2 is done, the blocking CI install no longer needs xgboost,
  scikit-learn or matplotlib (`requirements-ci.txt`). Locally, shap, torch,
  torchvision and transformers can also go (`requirements.txt`).
- The legacy job installs a new `requirements-legacy.txt`.
- Expected payoff:
  - a faster CI install;
  - fewer pip-audit failures like the 2026-10 pillow one;
  - a smaller daily-run image.
- Both requirements files are in the paper lock, so this needs a refreeze.

### Phase 4: legacy outputs (owner decision)
- `models/`, `research_snapshots/`, `archive/` and `autoresearch-results/`
  stay in git history either way.
- Options:
  - (a) leave them;
  - (b) tag the current commit `legacy-ml-final`, then remove the folders from `main`;
  - (c) move them to a GitHub release asset.
- Nothing in the quant path reads them.

## What stays (not legacy, still live for core-alpha)

- `pipeline_shared.py` → `sentiment_engine.py`, `social_sentiment.py`,
  `intraday_features.py` and `options_iv_provider.py` stay. `research.py
  --xs-only` (Factor Data Refresh) and `survivorship_audit.py` use them.
- `core_satellite_alpha.py` also makes a lazy call to `sentiment_engine`
  for its news veto.
- All of these feed only core-alpha, which places no orders now. They are a
  separate question, if core-alpha is retired for good.

## Order and timing

1. After the 2026-11-02 V rebalance reconciles: merge `deps-ci-audit` and Phase 1 together, then one refreeze.
2. Phase 2 and Phase 3: when the owner agrees, one PR each.
3. Phase 4: any time, owner's choice.

## What was done (2026-10-25, branch `legacy-cleanup`)

### Phase 1: the link is cut
- `quant_common.py` now holds `INITIAL_CAPITAL`, `_ETF_PRICE_FRAME_CACHE`,
  `_download_yfinance`, `_newey_west_tstat` and `_load_etf_price_frame`,
  copied verbatim. `legacy/backtest.py` re-imports them.
- Repointed to `quant_common`: `alpha_factor_backtest.py`,
  `core_satellite_alpha.py`, `core_satellite_drawdown_throttle.py`,
  `core_satellite_tqqq.py`, `factor_decay_monitor.py` and
  `quant_performance_audit.py`, plus
  `research_evidence/benchmark_20260929/benchmark_check.py` (one import line).
- `core_satellite_nested_walkforward.py` takes `DATA_DIR` from `settings`.
- `forward_test_monitor.py` takes `_nyse_sessions` from `v_rule_signal`.
  It gives the same sessions as core's version: 540 sessions for
  2025-01-01..2027-03-01 and 2,765 for 2016..2026, all equal.
- **Proof of no change:**
  - 122 helper results on real data (3 symbol sets × 4 date grids × 2
    warm-ups, and t-statistics at 3 lags) match the old `backtest.py`
    exactly;
  - `factor_decay_monitor.py` wrote identical JSON (ignoring timestamps)
    and an identical CSV before and after.
- **Guard:** `tests/test_no_legacy_imports.py` checks four things:
  - no legacy script is back in the root;
  - no live file imports a legacy module or xgboost / scikit-learn /
    matplotlib / shap / seaborn (lazy imports included);
  - no live file starts a legacy script except by an explicit
    `legacy/...` path;
  - importing 20 entry points loads none of them.

### Phase 2: moved to `legacy/`
- **Code:** the 14 legacy scripts and `backtest.py` are in `legacy/`
  (`legacy/README.md`). Each gets a 4-line path setup, so
  `python legacy/train.py` still finds `settings.py` and its neighbours.
  `model_registry.PROJECT_ROOT` now points one level up.
- **Docs:** moved to `Documentation/legacy/`, with a banner, and their run
  commands now say `python legacy/<name>.py`.
- **Tests:**
  - 6 legacy-only test files moved to `tests/legacy/`;
  - the legacy parts of 4 mixed files were split out (labels tests from
    `test_sanity` and `test_corrected_audit`, nested_cv tests from
    `test_nested_walkforward_defaults`, predict sentiment-health tests from
    `test_sentiment_fallback`);
  - `test_submission_history_guards` now tests the price loader in
    `quant_common`.
- `pytest.ini` skips `tests/legacy` by default. The new CI job
  `legacy-tests` runs it with `continue-on-error`.
- `daily_run.py`: the `drift_monitor` step is retired. It sat on the
  core-alpha path only and compared ML model inputs with their training
  baseline; nothing read its output.
- `survivorship_audit.py --run-backtest` now calls `legacy/backtest.py`.
- Updated: CLAUDE.md, AGENTS.md, README, `RELIABILITY_FIRST_WORKFLOW.md`,
  the dashboard script map, and a few docs that named `backtest.py`.

### Phase 3: heavy packages dropped
- `requirements-ci.txt`, the daily trading install, no longer has
  scikit-learn, xgboost, joblib, matplotlib or pillow.
- **scipy stays:** in the slim test, pandas needed it for Spearman rank
  correlation, which the factor IC code uses.
- The new `requirements-legacy.txt` is the trading set plus those packages.
- Local `requirements.txt` drops scikit-learn, xgboost, joblib, matplotlib,
  shap and seaborn. Nothing imported shap or seaborn.
- **torch / transformers stay** (correction to the plan above):
  `sentiment_engine.py` uses them for optional FinBERT news sentiment, which
  is not legacy.
- **Proof:** in a clean venv with only `requirements-ci.txt` (no sklearn,
  xgboost, matplotlib or PIL), the full main suite passes. The legacy tests
  pass with the legacy packages installed.

### Phase 4: legacy outputs → option (a), plus a tag
- `models/`, `research_snapshots/`, `autoresearch-results/` and most of
  `archive/` were never in git (they are gitignored or excluded), so there
  is nothing to remove from `main`.
- `archive/paper_epochs/` is tracked and live: `paper_validation_epoch.py`
  writes it. It stays.
- `models/` also stays, because it is `MODEL_DIR` for the core-alpha
  `trade_rules.py`.
- The commit before the move is tagged `legacy-ml-final`.
- Deleting the local folders (for example `research_snapshots/`, 303 MB)
  frees disk space and cannot be undone. That is left to the owner, by hand.

### Release lock
`core_satellite_alpha.py`, `daily_run.py`, `requirements-ci.txt` and
`requirements.txt` are locked, so the branch carries a refreeze for the same
epoch. After merging, run `python paper_validation_epoch.py --check-lock` on
`main`. If `main` moved in between, refreeze again there.
