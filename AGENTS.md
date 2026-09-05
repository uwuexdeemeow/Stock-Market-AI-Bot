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

Scripts in scope (project root `.py` files).  Grouped by role so a
beginner can see how the pieces fit together:

**Strategy / research / training**
backtest.py, confidence_calibration.py, core_satellite_alpha.py,
core_satellite_nested_walkforward.py, core_satellite_tqqq.py,
diagnostics.py, feature_quality_diagnostic.py, fundamental_features.py,
labels.py, leakage_audit.py, model.py, model_quality.py,
model_self_check.py, pipeline_shared.py, portfolio_manager.py,
predict.py, research.py, sentiment_engine.py, settings.py,
social_sentiment.py, train.py, xgb_feature_engineering.py,
ranker_utils.py, calibration_stability.py, cross_sectional_features.py,
intraday_features.py, alternative_data_features.py

**Robustness / validation**
alpha_factor_backtest.py, concentration_overlay.py,
core_satellite_drawdown_throttle.py, core_satellite_execution_stress.py,
core_satellite_survivorship_audit.py, factor_decay_monitor.py,
feature_health.py, nested_cv.py, regime_monitor.py, robustness_scoring.py,
survivorship_audit.py, walkforward_analyzer.py

**Live paper trading (Alpaca)**
alpaca_paper_gauntlet.py, alpaca_paper_trading.py, alpaca_protection.py,
broker_health.py, broker_interface.py, daily_run.py, execution_guard.py,
execution_model.py, fill_monitor.py, paper_health.py, paper_scorecard.py,
publish_live_config_from_csv.py, refresh_etf_data.py, risk_sizing.py,
signal_freshness.py, status.py, trade_rules.py

**Infrastructure**
config_health.py, data_provider.py, data_validation.py,
experiment_ledger.py, http_retry.py, log_cleanup.py, monitor.py,
monitor_heartbeat.py, notifications.py, options_iv_provider.py,
safe_io.py

# Git workflow

- Make code fixes directly on `main` and push the fixes to `main`.
- Do not create new branches or branch-based worktrees unless the user explicitly asks.
- Existing automated publishing to `signals/latest` remains the destination for generated operational evidence, not code fixes.

# Styles 
You talk like a caveman. Unless specified, you only talk to me once you are done with my instructions
