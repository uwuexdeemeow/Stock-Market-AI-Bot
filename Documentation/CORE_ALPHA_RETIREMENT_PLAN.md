# Plan: retire core-alpha (draft, 2026-10-25)

**Status: PLAN ONLY. No code moves until the owner approves, and not before
the first rule V trade (2026-11-02) has reconciled and the 11/03 monitor has
run.** Each phase is one commit or PR, passes the full suite, and gets a
refreeze when it touches a locked file. Built the same way as
`LEGACY_ML_CLEANUP_PLAN.md`, which went through without a hitch.

**Key term — core-alpha:** the factor stock-picking strategy
(`core_satellite_alpha.py` and everything that feeds it). Its approval
expired on 2026-10-19 and it failed revalidation (8 of 14 years passed the
cost-stress check; the rule needs 60%). It places no orders now. The
research behind it is closed: H-wide-universe showed the edge was mostly the
hand-picked stock list. **Rule V does not use any of it.**

**Why retire it:**
- It is about 16,500 lines of strategy code, plus 43 test files that test
  only core-alpha.
- Three workflows exist only for it: Factor Data Refresh, Shadow Paper
  Journal and Weekly Calendar Robustness.
- About 10 daily-workflow steps, each skipped when `PAPER_MODE` is
  `v_rule`.
- Most of the release lock and the heaviest packages (VADER, feedparser,
  Twitter, finnhub; torch/transformers locally).
- Every audit and every reviewer has to read past it to find what is live.

Retiring it doesn't change what V trades, so the FV forward-test clock is
not affected.

## What the import map shows (AST scan, 2026-10-25)

**The V live path is 38 modules.**
- **Started by** the V-mode daily steps and the V workflows:
  - daily steps: `refresh_etf_data`, `fill_monitor`, `broker_health`,
    `alpaca_paper_trading`, `v_rule_signal`, `regime_monitor`,
    `execution_guard`, `broker_truth`, `paper_health`, `execution_scorecard`,
    `execution_cost_calibration`, `alpaca_paper_gauntlet`,
    `paper_validation_epoch`, `monitor_heartbeat`, `run_evidence`,
    `log_cleanup`;
  - workflows: `forward_test_monitor`, `workflow_watchdog`,
    `workflow_failure_annotation`.
- **Imported by those:**
  - top level: `alpaca_protection`, `alpaca_sdk_adapter`, `broker_interface`,
    `data_manifest`, `data_provider`, `data_validation`, `order_accounting`,
    `risk_sizing`, `safe_io`, `settings`, `signal_freshness`, `v_rule_helper`,
    `factor_data_health`, `ranker_utils`;
  - lazy: `broker_history`, `notifications`, `http_retry`, `paper_policy`.

**Only one real tangle:** `paper_health.py` imports `trading_day_age` from
`factor_data_health.py`, which imports `ranker_utils.py` (factor weights).
That is how two factor modules reach the V path. Moving that small date
helper fixes it.

**Not reached from the V path (60 modules), split into two groups:**

| Group | Modules |
|---|---|
| **Core-alpha strategy and research** (retire) | `core_satellite_alpha`, `core_satellite_calendar_robustness`, `core_satellite_drawdown_throttle`, `core_satellite_execution_stress`, `core_satellite_nested_walkforward`, `core_satellite_survivorship_audit`, `core_satellite_tqqq`, `alpha_factor_backtest`, `factor_decay_monitor`, `factor_data_health`\*, `ranker_utils`\*, `feature_health`, `feature_quality_diagnostic`, `feature_research`, `cross_sectional_features`, `fundamental_features`, `intraday_features`, `options_iv_provider`, `pipeline_shared`, `sentiment_engine`, `social_sentiment`, `research`, `survivorship_audit`, `universe_membership`, `membership_reconciliation`, `robustness_review`, `robustness_scoring`, `robustness_snapshot_gate`, `research_output_guard`, `validation_bundle`, `validate_fixed_live_config`, `publish_live_config_from_csv`, `walkforward_analyzer`, `walkforward_selector_diagnostics`, `run_walkforward_batched`, `prepare_colab_walkforward`, `refresh_local_research_data`, `score_predictiveness_audit`, `edge_ablation`, `edge_evidence`, `quant_performance_audit`, `quant_common`, `portfolio_manager`, `trade_rules`, `execution_model`, `experiment_ledger`, `causal_research`, `corrected_audit`, `corrected_data`, `portfolio_ledger`, `audit_gap_register`, `shadow_paper_journal`, `fractional_shadow_paper`, `paper_shadow_compare`, `paper_scorecard` |
| **Tools, not strategy** (owner's call, see below) | `status`, `monitor`, `dashboard` + `dashboard/` + `pages/`, `config_health`, `disaster_recovery`, `audit_evidence_recovery`, `evidence_audit` |

\* After Phase 1 cuts the tangle.

**Things V reuses on purpose (they stay):**
- **The signal file name.** `v_rule_signal.py` writes
  `signals/core_satellite_alpha_signal.csv`, and `broker_truth`,
  `run_evidence`, `paper_health`, `regime_monitor` and the gauntlet read
  that path. Renaming it touches locked files and the `signals/latest`
  history for no gain. The name stays; a comment says it is rule V's
  signal now.
- **The live-config check.** `alpaca_paper_trading.py` already skips it
  in V mode (`check_live_config_match=not v_rule_mode`, line ~5079).
- **Lock entries for core-alpha files.** `paper_validation_epoch.py` and
  `run_evidence.py` list `core_satellite_alpha.py`, `validation_bundle.py`,
  `trade_rules.py`, `shadow_paper_journal.py` and the two core-alpha JSON
  files. Those entries are removed in Phase 4, with a refreeze.

**Tests (98 files):**
- 43 test only core-alpha modules;
- 11 mix core-alpha and V-path modules: `test_audit_medium_fixes`,
  `test_audit_three_fixes`, `test_corrected_audit`, `test_gap_closure`,
  `test_halt_and_stop_cooldown`, `test_locked_audit_fixes`,
  `test_paper_rebalance_calendar`, `test_reliability_hardening`,
  `test_sanity`, `test_social_sentiment_safety`,
  `test_submission_history_guards`;
- 44 test only the V path or general code.

## Phases

### Phase 1: cut the tangle (small)
1. Move `trading_day_age` from `factor_data_health.py` into
   `data_validation.py`, which is on the V path already, verbatim.
   `factor_data_health` re-imports it from there.
2. Add a guard test, like `test_no_legacy_imports.py`: importing every
   V-path entry point must load no core-alpha module.
3. **Proof:** the full suite passes, and `paper_health.py` writes the same
   JSON before and after (timestamps aside).
4. `paper_health.py` is locked, so this needs a refreeze.

### Phase 2: switch the core-alpha path off
- `daily_run.py`:
  - drop the core-alpha branch of `build_steps` (factor refresh, factor
    data health, `core_satellite_alpha` signal);
  - `PAPER_MODE` other than `v_rule` fails closed with a clear message,
    instead of running core-alpha.
- `daily_paper_trading.yml`: delete the ~10 steps guarded by
  `PAPER_MODE != 'v_rule'`, and the core-alpha files in the cache, upload
  and commit lists.
- **Workflows to delete:** `factor_data_refresh.yml`,
  `shadow_paper_journal.yml`, `weekly_calendar_robustness.yml`.
  **Owner:** delete their cron-job.org jobs; they should already be
  paused.
- `alpaca_paper_trading.py`: **leave its core-alpha branches as they are**
  in this phase. They are dormant, and stripping them is the riskiest edit
  in the plan (see "Owner decisions").
- **Proof:**
  - a dry-run of the daily workflow steps in V mode lists the same steps as
    today;
  - the full suite passes.
- Locked files change (`daily_run.py` and three workflow files), so this
  needs a refreeze.

### Phase 3: move the code to `legacy/core_alpha/`
- Move the strategy and research group from the table above into
  `legacy/core_alpha/`:
  - each file gets the same short path setup as the ML legacy files;
  - docs go to `Documentation/legacy/core_alpha/`;
  - the 43 core-only tests go to `tests/legacy/core_alpha/`, run by the
    non-blocking legacy CI job;
  - the 11 mixed tests are split, keeping their V-path parts.
- `research_evidence/*` scripts that import core-alpha code are frozen
  records. They keep working through the same path setup, or are marked
  "run from `legacy-ml-final` / `core-alpha-final`".
- Update CLAUDE.md, AGENTS.md, README, `RELIABILITY_FIRST_WORKFLOW.md` and
  the dashboard script map.

### Phase 4: lock, requirements and signals
- **Lock:** remove the core-alpha entries from `paper_validation_epoch.py`'s
  file list and from `run_evidence.py`, then refreeze. The lock then covers
  only what V runs, which is the point of a release lock.
- **Requirements:**
  - drop what only core-alpha needed from `requirements-ci.txt`:
    `vaderSentiment`, `feedparser`, `beautifulsoup4`, `finnhub-python`;
  - check `yfinance`, `yahooquery`, `pandas-datareader` and `scipy`
    one by one, because `data_provider.py` (V path) uses some of them;
  - locally, `torch`/`transformers`/`tweepy` move to
    `requirements-legacy.txt`;
  - same proof as last time: the full main suite in a clean venv with only
    the new `requirements-ci.txt`.
- **Tracked core-alpha files:**
  - `signals/core_satellite_live_configs.json`,
    `signals/core_satellite_validation_bundle.json`,
    `signals/core_satellite_nested_walkforward.json`,
    `signals/adaptive_factor_weights.json`, the feature-quality baselines
    and the `logs/` exceptions in `.gitignore`;
  - tag the commit before removal `core-alpha-final`, then remove them
    from `main`. They stay in git history.
- **Not touched:** `archive/paper_epochs/` (live) and the `signals/latest`
  branch (history).

## Owner decisions (needed before Phase 2)

1. **Tools group** (`status`, `monitor`, `dashboard`, `config_health`,
   `disaster_recovery`, `audit_evidence_recovery`, `evidence_audit`):
   (a) keep and repoint at V's files, or (b) retire with core-alpha.
   - Me suggest: keep `status` and `disaster_recovery` (useful for V);
     check what `monitor` and the dashboard show before deciding; retire
     the two audit tools.
2. **Core-alpha branches inside `alpaca_paper_trading.py`:** (a) leave
   them dormant (safe, no change to the order code), or (b) strip them in a
   later, separate PR with its own review.
   - Me suggest (a) until V has a few months of clean fills.
3. **Shadow paper research** (`shadow_paper_journal`,
   `fractional_shadow_paper`, `paper_shadow_compare`): retire with
   core-alpha (me suggest), or keep for a future V shadow such as an
   H-multi-asset M shadow.

## Order and timing

1. **Now:** this plan. No code.
2. **After 11/2 reconciles and the 11/03 monitor has run:** Phase 1, then
   Phase 2, one at a time, each watched for one daily run.
3. **Phases 3 and 4:** when the owner agrees. One PR each.

## What could go wrong

| Risk | Guard |
|---|---|
| A V-path script still reaches a core-alpha file at runtime, not just by import | Phase 2 dry-run step list; a one-day watch after each phase; the guard test |
| The daily workflow loses a step V needs | Phase 2 deletes only steps with an `if: PAPER_MODE != 'v_rule'` guard |
| The lock goes stale and blocks orders | A refreeze in the same PR; `--check-lock` in CI |
| Something old is needed again | Tag `core-alpha-final`; everything stays in git history |
