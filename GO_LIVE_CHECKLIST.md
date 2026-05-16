# Go-Live Checklist

**Do not trade real money until every box is checked.** Ordered roughly by blast-radius.

## 1. Security
- [ ] `keys.txt` rotated — both OpenAI and Finnhub keys regenerated at the provider.
- [ ] New keys live in `.env`; `.env` is gitignored; `git log -p .env` returns nothing.
- [ ] `settings.py` loads via `load_dotenv()`; `os.environ.get("OPENAI_API_KEY")` is populated in a fresh shell.
- [ ] Repo permissions: no public forks; read access only to trusted collaborators.

## 2. Reproducibility
- [ ] `requirements.txt` installs cleanly in a fresh venv.
- [ ] `python -m pytest tests/` is green.
- [ ] Every training run writes to `model_registry.py` — `models/registry.json` has at least one entry with git SHA.
- [ ] You can recreate the production model from a git commit + one command.

## 3. Honest Edge (Phase 1 outputs)
- [ ] `python leakage_audit.py` returns 0 for every watchlist ticker.
- [ ] Backtest on ≥100 S&P 500 tickers, 2015–2024, 126-day walk-forward blocks, stress mode on.
- [ ] Net-of-cost Sharpe ≥ 1.0.
- [ ] Newey-West t-stat on alpha vs SPY > 2.0.
- [ ] Backtest reproducible — same seed, same result.

## 4. Model Quality (Phase 2 outputs)
- [ ] Nested walk-forward CV shows hyperparameters stable across folds.
- [ ] Feature count reduced to ~15 most-important via `shap_feature_reducer.py`; OOS Sharpe holds.
- [ ] `calibration_stability.py` reports `max_ks < 0.10`.

## 5. Risk (Phase 3 outputs)
- [ ] Position sizes use `vol_target_size` or `fractional_kelly`, not fixed %.
- [ ] ATR stop-loss is in the live trading loop.
- [ ] Soft de-risking band (8%) and hard halt (15%) verified by a manufactured drawdown test.
- [ ] Execution model accounts for spread + sqrt-impact; capacity warnings log when triggered.

## 6. Ops (Phase 4 outputs)
- [ ] `orchestration.py daily` runs end-to-end under cron/Prefect; exit code 0 on green days.
- [ ] `data_validation.validate_price_frame` called on every feed; bad data raises.
- [ ] `drift_monitor.check_drift` runs daily; snapshot_baseline captured at train time.
- [ ] Monitoring: logs shipped somewhere (Loki / Datadog / CloudWatch) and an alert fires on any `rc != 0`.
- [ ] `broker_interface.Broker` is the only thing live-trading code imports — no direct Moomoo imports outside the adapter.

## 7. Paper Gauntlet (Phase 5)
- [ ] ≥60 trading days of continuous paper trading completed.
- [ ] `python paper_gauntlet.py` → `all_gates_passed = true`.
- [ ] Manual review of 20 random trades: fills, sizing, stops all look sane.

## 8. Business / Personal
- [ ] Capital at risk is money you can afford to lose.
- [ ] You have a written kill-switch procedure (who, how, when to flatten everything).
- [ ] You've decided the max monthly drawdown that halts live trading and reverts to paper.

**When every box is checked:** start live at 10% of your target size for one month, then scale.
