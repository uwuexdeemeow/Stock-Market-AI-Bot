# Project Instructions

For every script in this project:
- Add inline annotations/comments explaining what the code is doing, written for a beginner with no prior Python or machine-learning experience.
- Create a separate documentation file (one per script) in `Documentation/` that covers:
  - What the script does, in plain language
  - How to run it (commands, inputs, expected outputs)
  - Key concepts/terms defined simply (e.g., "feature", "model", "backtest")

Also create one project-wide documentation file that describes:
- The end-to-end workflow (how the scripts connect: scan → research → train → predict → backtest → paper trade)
- The logic and reasoning behind the project's design choices
- How a beginner would use the system start to finish

Scripts in scope (project root `.py` files; the old ML scripts are in
`legacy/`, see below).  Grouped by role so a
beginner can see how the pieces fit together:

**Strategy / research**
core_satellite_alpha.py,
core_satellite_nested_walkforward.py, core_satellite_tqqq.py,
feature_quality_diagnostic.py, fundamental_features.py,
pipeline_shared.py, portfolio_manager.py,
research.py, sentiment_engine.py, settings.py,
social_sentiment.py, ranker_utils.py, cross_sectional_features.py,
intraday_features.py

**Robustness / validation**
alpha_factor_backtest.py,
core_satellite_calendar_robustness.py, core_satellite_drawdown_throttle.py,
core_satellite_execution_stress.py,
core_satellite_survivorship_audit.py, factor_decay_monitor.py,
feature_health.py, forward_test_monitor.py, regime_monitor.py,
robustness_scoring.py, survivorship_audit.py, walkforward_analyzer.py

**Live paper trading (Alpaca)**
alpaca_paper_gauntlet.py, alpaca_paper_trading.py, alpaca_protection.py,
broker_health.py, broker_interface.py, daily_run.py, execution_guard.py,
execution_model.py, fill_monitor.py, paper_health.py, paper_scorecard.py,
publish_live_config_from_csv.py, refresh_etf_data.py, risk_sizing.py,
signal_freshness.py, status.py, trade_rules.py, v_rule_signal.py

**Infrastructure**
config_health.py, data_provider.py, data_validation.py,
experiment_ledger.py, http_retry.py, log_cleanup.py, monitor.py,
monitor_heartbeat.py, notifications.py, options_iv_provider.py,
safe_io.py, quant_common.py, v_rule_helper.py

# Project focus: quant trading, not ML model training

This project started as a machine-learning (XGBoost) prediction bot. It is
now a **rule-based quant strategy**: the core-satellite engine
(`core_satellite_alpha.py`) ranks stocks with factor scores
(`alpha_factor_backtest.py`, `feature_health.py`) and trades them on a fixed
20-day calendar. The ML model is switched off for picks
(`ml_overlay_enabled` is false; the incumbent uses `score_source =
regime_adaptive` factor scores). Model AUC was about 0.51, so it added nothing.

**Since the 2026-10-30 signal the paper account trades rule V** (QQQ + BIL,
month-end volatility target, `v_rule_signal.py`; switch: `PAPER_MODE` in
`daily_paper_trading.yml`). It is judged by `Documentation/V_RULE_FORWARD_TEST.md`.
Core-alpha's approval expired on 2026-10-19 and failed revalidation; its code
stays in place but places no orders.

Treat the old ML-training layer as **legacy**. Since 2026-10-25 it lives
in its own folder (plan: `Documentation/LEGACY_ML_CLEANUP_PLAN.md`):

- **Legacy scripts, in `legacy/`:** `train.py`, `predict.py`, `labels.py`,
  `model_quality.py`, `model_self_check.py`, `model_registry.py`,
  `confidence_calibration.py`, `calibration_stability.py`, `nested_cv.py`,
  `xgb_feature_engineering.py`, `shap_feature_reducer.py`, `drift_monitor.py`,
  `leakage_audit.py`, `alternative_data_features.py`, `backtest.py`.
  Run one as `python legacy/train.py`. Their docs are in
  `Documentation/legacy/`, their tests in `tests/legacy/` (a separate,
  non-blocking CI job that installs `requirements-legacy.txt`).
- **Legacy outputs:** `models/` (per-ticker `.pkl` / `xgb` files, train
  summaries), `signals/*_walkforward_predictions.csv`,
  `autoresearch-results/`, `research_snapshots/`, and `archive/` except
  `archive/paper_epochs/` (live: `paper_validation_epoch.py` writes it).
  They are local only (gitignored). `models/` is also still `MODEL_DIR`
  for `trade_rules.py`, so keep the folder.
- **Don't** read them for context, use them as evidence, cite their metrics,
  or propose research that trains, tunes or revives an ML model, unless the
  owner asks for it.
- **Live code no longer imports this layer.** The few quant helpers that used
  to sit in `backtest.py` (`INITIAL_CAPITAL`, `_load_etf_price_frame`,
  `_newey_west_tstat`) are in `quant_common.py`.
  `tests/test_no_legacy_imports.py` fails if a live script imports a legacy
  module again, or needs xgboost, scikit-learn or matplotlib. Don't move
  legacy code back to the root. Deleting `legacy/` or the legacy outputs
  still needs the owner's agreement.
- `pipeline_shared.py` imports `sentiment_engine.py`, `social_sentiment.py`,
  `intraday_features.py` and `options_iv_provider.py`. These are not legacy
  (core-alpha's factor refresh and news veto use them).
- When a legacy file has to change, change only what is needed.
- **Research ideas come from the quant path only:** factor signals, universe
  and survivorship, calendar and timing robustness, execution costs, risk
  and regime rules. Judge them with pre-registered rules in
  `Documentation/DELAY_STRESS_PAPER_ADVISORY.md`.

# Workflow scheduling

- Never use GitHub Actions `schedule:` cron triggers. Every workflow is started by
  cron-job.org calling `workflow_dispatch`.
- When adding or changing a workflow that must run on a timer, give it only a
  `workflow_dispatch:` trigger and tell the user to add (or update) the job in
  cron-job.org manually: workflow file, day/time with timezone, and the dispatch
  API call.

# Styles 
You talk like a caveman. Unless specified, you only talk to me once you are done with my instructions
