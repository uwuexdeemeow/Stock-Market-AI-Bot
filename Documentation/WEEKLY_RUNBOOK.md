# Stock Bot Operations Runbook

The reference for every script in the bot — what it does, when to run it, how to find it.

## Table of contents

- [Quick reference — by goal](#quick-reference--by-goal)
- [Cadence map — daily / weekly / monthly / quarterly](#cadence-map)
- [Daily routine](#daily-routine)
- [Weekly routine](#weekly-routine)
- [Monthly routine — research + republish](#monthly-routine--research--republish)
- [Quarterly routine — ML retrain](#quarterly-routine--ml-retrain)
- [Script catalog](#script-catalog-alphabetical)
  - [Daily trading scripts](#daily-trading-scripts)
  - [Research & training scripts](#research--training-scripts)
  - [Robustness & validation scripts](#robustness--validation-scripts)
  - [Wrappers & utilities](#wrappers--utilities)
  - [Infrastructure & dashboards](#infrastructure--dashboards)
- [Key files reference](#key-files-reference)
- [Git sync — pulling Actions's work](#git-sync--pulling-actionss-work)
- [Common failures & fixes](#common-failures--fixes)
- [Safety notes](#safety-notes)
- [Real-capital blockers](#real-capital-blockers)

---

## Quick reference — by goal

If you just want one command, find your goal here:

| I want to... | Command |
|---|---|
| Sync today's Actions outputs to my laptop | `pull_daily.bat` (or `bash pull_daily.sh`) |
| See today's status | `python status.py` |
| Run the bot manually today | `python daily_run.py --alpaca` |
| Refresh local data + features before research | `python refresh_local_research_data.py` |
| Run the monthly walkforward | `python run_walkforward_batched.py --recent-alpha-grid` |
| Publish a new live config (after walkforward) | `python publish_live_config_from_csv.py --source stable_family` |
| Generate today's order plan locally (no submit) | `python alpaca_paper_trading.py` |
| See if Actions actually ran today | `git log origin/signals/latest -1 --format="%ai %s"` |
| Verify the bot picks up a new config | `python daily_run.py --dry-run` |
| Open the dashboard | `streamlit run dashboard.py` |
| Diagnose factor decay | `python factor_decay_monitor.py` |
| Sanity-check a walkforward result | `python walkforward_analyzer.py` |

---

## Cadence map

| Cadence | What | Where |
|---|---|---|
| **Every 9:35 AM NY weekday** | Live trading via GitHub Actions | `.github/workflows/daily_paper_trading.yml` (automatic) |
| **Every 7:30 AM NY weekday** | Factor data refresh via Actions | `.github/workflows/factor_data_refresh.yml` (automatic) |
| **Every morning** | `pull_daily.bat` to sync logs to laptop | you |
| **Once per week (Fri)** | Scorecard + paper report | you |
| **Monthly (1st of month)** | Walkforward + medium-risk review + republish | you |
| **Quarterly** | ML retrain + feature_research --pairs | you |

---

## Daily routine

### Morning (5 sec)

```cmd
pull_daily.bat
```

Pulls yesterday's Actions outputs into `signals/` and `logs/`. Refreshes:
- `alpaca_paper_log.csv` — what got submitted
- `alpaca_paper_equity.csv` — equity history
- `alpaca_paper_health.json` — health summary
- `alpaca_daily_status.json` — positions + equity snapshot
- `alpaca_slippage_reversal_report.json` — fill slippage and reversal report
- `monitor_heartbeat.json` — watchdog status
- `logs/daily_run_*.json` — pipeline step results

### Check (anytime)

Open the dashboard:

```cmd
streamlit run dashboard.py
```

The dashboard live-refreshes by default.  Every refresh asks Alpaca for a fresh
paper account snapshot and rewrites `signals/alpaca_daily_status.json` plus
`signals/alpaca_paper_equity.csv`, so the equity card updates without running
`alpaca_paper_trading.py --reconcile`.

Look for:
- 🟢 fresh on Alpaca log, equity, status, health
- Account equity > 0 (was a known bug — fixed)
- Today's trades reflect the published top3 family
- Performance → Execution quality shows recent fill slippage and 5/15/60m
  reversal counts, plus the worst recent tickers by execution risk score
- Telegram alerts arrived for anything red

### Manual run (only if Actions failed)

```bash
python daily_run.py --alpaca
```

This is what Actions runs internally — refreshes data, generates signal, submits orders, reconciles fills, rebuilds health.  Idempotent: refuses to double-submit thanks to `alpaca_paper_log.csv`'s same-day check + Alpaca's deterministic `client_order_id`.

To run without submitting:
```bash
python daily_run.py --alpaca --dry-run
```

---

## Weekly routine

Run every Friday (or whenever convenient):

```cmd
python paper_scorecard.py
python paper_report.py
python alpaca_paper_gauntlet.py --verbose
```

- `paper_scorecard.py` — performance vs walkforward expectation
- `paper_report.py` — human-readable summary of the week
- `alpaca_paper_gauntlet.py` — readiness test for "could this go on real money?"

---

## Monthly routine — research + republish

Run on the 1st of each month (or after any major code/data change).  Total time: ~1–2 hours on a 16-32 GB laptop with browser/IDE closed.

```bash
# ── Pre-flight: sync with origin ───────────────────────────────────
git fetch origin
git checkout main && git pull --ff-only origin main
pull_daily.bat

# ── 1. Refresh local research data (~5-10 min) ─────────────────────
python refresh_local_research_data.py

# ── 2. Medium-risk-review trio (~10-15 min) ────────────────────────
python core_satellite_survivorship_audit.py
python core_satellite_execution_stress.py
python core_satellite_drawdown_throttle.py

# ── 3. Concentration / regime checks (~5 min) ──────────────────────
python concentration_overlay.py
python regime_monitor.py

# ── 4. Nested walkforward (~30-60 min) ─────────────────────────────
python run_walkforward_batched.py --recent-alpha-grid

# ── 5. Sanity check ────────────────────────────────────────────────
python walkforward_analyzer.py

# ── 6. Publish (if walkforward approved an updated family) ─────────
python publish_live_config_from_csv.py --dry-run --source stable_family
# Review output, then if it looks right:
python publish_live_config_from_csv.py --source stable_family

# ── 7. Regenerate trades + factor decay under new live config ──────
python core_satellite_alpha.py
python factor_decay_monitor.py

# ── 8. Commit + push ───────────────────────────────────────────────
git add -f signals/core_satellite_live_configs.json \
          signals/core_satellite_nested_walkforward.json \
          signals/core_satellite_nested_walkforward.csv \
          signals/core_satellite_alpha_metrics.json
git commit -m "monthly research: refresh + walkforward + publish (YYYY-MM-DD)"
git push origin main
```

See `Documentation/MONTHLY_RUNBOOK.md` for the full per-step rationale + output thresholds.

---

## Quarterly routine — ML retrain

Run on the 1st of every 3rd month (after the monthly routine):

```bash
python refresh_local_research_data.py --pairs   # slow ~30 min
python train.py
python predict.py
python model_quality.py
python confidence_calibration.py
```

Then re-run the monthly routine to validate the new model is still approvable.

---

## Script catalog (alphabetical)

### Daily trading scripts

| Script | What it does | Common flags |
|---|---|---|
| `alpaca_paper_gauntlet.py` | Health gate — pass/fail for "should real money trust this?" | `--verbose` |
| `alpaca_paper_trading.py` | Generate and (optionally) submit Alpaca orders | `--submit`, `--status`, `--reconcile`, `--slippage-report`, `--market-order` |
| `broker_health.py` | Pre-flight Alpaca connectivity check | none |
| `core_satellite_alpha.py` | Generate today's signal (which 3 stocks + ETF weights) | `--ignore-stale`, `--walkforward` |
| `daily_paper_check.py` | Quick pass/fail verdict for daily run | none |
| `daily_run.py` | Run the entire daily pipeline (13 steps) | `--alpaca`, `--dry-run`, `--force`, `--health-only` |
| `execution_guard.py` | Repair ETF stop-loss protection, cancel stale orders | `--once`, `--loop` |
| `fill_monitor.py` | Verify yesterday's fills (cancelled, partial, slipped) | `--days N` |
| `monitor_heartbeat.py` | Watchdog — all monitors produced fresh output? | none |
| `notifications.py` | Send Telegram / email alerts | library, not run directly |
| `paper_health.py` | Build deep health dashboard (slippage, drift, risk) | `--broker alpaca` |
| `regime_monitor.py` | Detect risk_on / neutral / risk_off regime shifts | none |
| `risk_sizing.py` | Position sizing helpers | library |
| `signal_freshness.py` | Reject stale signals at trade time | library |
| `status.py` | One-page CLI status snapshot | none |
| `trade_rules.py` | Order generation rules | library |

### Research & training scripts

| Script | What it does | When |
|---|---|---|
| `backtest.py` | Standalone single-config backtest | ad-hoc |
| `calibration_stability.py` | Model calibration over time | after retrain |
| `confidence_calibration.py` | Calibrate predicted → realized return mapping | after retrain |
| `core_satellite_alpha.py` | Live signal generator (also backtests when run alone) | daily / ad-hoc |
| `core_satellite_nested_walkforward.py` | Strategy validation via nested walk-forward | monthly |
| `core_satellite_tqqq.py` | TQQQ overlay variant backtest | research |
| `cross_sectional_features.py` | Cross-sectional rank features | library |
| `diagnostics.py` | Investigate model / signal failures | ad-hoc |
| `feature_quality_diagnostic.py` | Re-grade per-feature live IC | daily (CI) + before walkforward |
| `feature_research.py` | Per-feature analysis (IC trend, sector, decay, pairs) | quarterly |
| `fundamental_features.py` | Sector-relative fundamental z-scores | library |
| `intraday_features.py` | Intraday signal features | library |
| `labels.py` | Forward returns + label engineering | library |
| `leakage_audit.py` | Look-ahead bias check | before any model deploy |
| `model.py` | ML model definitions | library |
| `model_quality.py` | OOS model verification | after retrain |
| `model_self_check.py` | Detect overfit between train and OOS | after retrain |
| `pipeline_shared.py` | Feature-build pipeline (shared helpers) | library |
| `portfolio_manager.py` | Correlation + concentration risk gates | library |
| `predict.py` | Generate model predictions for live | after retrain |
| `ranker_utils.py` | Adaptive factor-weight learning | library |
| `research.py` | Build per-ticker factor parquet panel | daily (CI) + ad-hoc |
| `sentiment_engine.py` | News / social sentiment scoring | library |
| `settings.py` | Watchlist + global config | library |
| `social_sentiment.py` | Reddit / Twitter sentiment | library |
| `train.py` | ML model training | quarterly |
| `xgb_feature_engineering.py` | XGBoost feature prep | library |

### Robustness & validation scripts

| Script | What it does | When |
|---|---|---|
| `alpha_factor_backtest.py` | Per-factor backtest (one factor at a time) | research |
| `concentration_overlay.py` | Position concentration risk audit | monthly |
| `core_satellite_drawdown_throttle.py` | Stress: drawdown circuit breaker | monthly |
| `core_satellite_execution_stress.py` | Stress: execution costs / slippage | monthly |
| `core_satellite_survivorship_audit.py` | Stress: survivorship bias check | monthly |
| `factor_decay_monitor.py` | Recent IC + overlay-α health check | daily (auto) |
| `feature_health.py` | Feature quarantine via IC decay | library |
| `nested_cv.py` | Nested cross-validation helper | library |
| `regime_monitor.py` | Regime change detection | daily (auto) |
| `robustness_scoring.py` | Selection objective for walkforward | library |
| `survivorship_audit.py` | Survivorship audit | research |
| `walkforward_analyzer.py` | Sanity-check a walkforward result (4 failure modes) | after each walkforward |

### Wrappers & utilities

| Script | What it does | When |
|---|---|---|
| `memprofile_walkforward.py` | Reproduce the walkforward memory leak (now fixed) | rare, debugging only |
| `publish_live_config_from_csv.py` | Manual live-config promote | after walkforward |
| `pull_daily.bat` / `pull_daily.sh` | Sync Actions outputs into local repo | every morning |
| `refresh_local_research_data.py` | Local mirror of `factor_data_refresh.yml` | before any local research |
| `run_walkforward_batched.py` | Memory-safe walkforward wrapper | monthly |

### Infrastructure & dashboards

| Script | What it does |
|---|---|
| `config_health.py` | Validate env / settings / required keys |
| `dashboard.py` + `dashboard/` + `pages/` | Streamlit dashboard |
| `data_provider.py` | Multi-source price downloader (yfinance / yahooquery / stooq) |
| `data_validation.py` | Reject malformed price frames |
| `experiment_ledger.py` | Track what's been run for reproducibility |
| `factor_data_health.py` | Strict pass/fail on panel freshness |
| `http_retry.py` | Idempotent HTTP wrapper for data fetches |
| `log_cleanup.py` | Old-log housekeeping |
| `monitor.py` | Generic file watcher |
| `options_iv_provider.py` | Tradier IV data |
| `paper_gauntlet.py` | Generic paper gauntlet (Moomoo) |
| `paper_report.py` | Side-by-side broker performance |
| `paper_scorecard.py` | Weekly performance snapshot |
| `refresh_etf_data.py` | SPY/QQQ/TQQQ refresh (bypasses incremental cache) |
| `safe_io.py` | Atomic writes + utf-8 subprocess helpers |

---

## Key files reference

### Per-broker state (committed to `signals/latest` by daily workflow)

| File | What's in it |
|---|---|
| `signals/alpaca_paper_log.csv` | Every submitted Alpaca paper order + fill status |
| `signals/alpaca_paper_equity.csv` | Daily equity snapshots |
| `signals/alpaca_daily_status.json` | Today's positions + equity + cash (dashboard equity card reads this) |
| `signals/alpaca_paper_health.json` | Slippage, drift, concentration, equity sanity, P&L breakdown |
| `signals/core_satellite_alpha_signal.csv` | Today's target portfolio (tickers + weights) |
| `signals/core_satellite_alpha_orders.csv` | Today's planned orders (BUY / SELL list) |
| `signals/core_satellite_alpha_metrics.json` | Backtest metrics + selected config + paper_ready verdict |
| `signals/monitor_heartbeat.json` | All-monitors-fresh watchdog |
| `signals/factor_data_health.json` | Panel freshness pass/fail |
| `signals/factor_decay_monitor.csv` | Recent IC + overlay-α (60d, 120d) |
| `logs/daily_run_*.json` | Per-step pass/fail/duration for each day |

### Strategy & validation outputs

| File | What's in it |
|---|---|
| `signals/core_satellite_live_configs.json` | The currently-approved live config (what `daily_run.py` reads) |
| `signals/core_satellite_nested_walkforward.json` | Latest walkforward result (aggregate metrics + per-fold details) |
| `signals/core_satellite_nested_walkforward.csv` | Same, in row-per-fold form |
| `signals/feature_health_profile.json` | Which features active / quarantined / strengthening |
| `signals/feature_research_summary.csv` | Per-feature IC stats (drives quarantine logic) |
| `signals/feature_quality_report.json` | Live feature grading (A/B/C/D/F) |
| `signals/walkforward_checkpoint_core_alpha.json` | Resume state during a walkforward run |

### Medium-risk-review inputs (the trio)

| File | What it gates |
|---|---|
| `logs/core_satellite_survivorship_audit.json` | Whether the strategy survives delisted-ticker stress |
| `logs/core_satellite_execution_stress.json` | Whether it survives 2x / 3x / 5x cost stress |
| `logs/factor_decay_monitor.json` | Whether recent overlay-α is positive |

All three must show `pass` or `advisory` for live promotion.

### Local-only state (gitignored, not in signals/latest)

| File | Why |
|---|---|
| `data/*.parquet` | Per-ticker price + factor history (big, refreshed via `research.py`) |
| `signals/walkforward_checkpoint_*.json` | Per-run state |
| Most `signals/*.csv` / `signals/*.json` not in `.gitignore`'s `!` exception list | Per-machine working files |

---

## Git sync — pulling Actions's work

GitHub Actions runs the trading pipeline daily and commits its outputs to a dedicated `signals/latest` branch.  Your local repo doesn't auto-pull those — use the helper:

### Every morning

```cmd
pull_daily.bat
```

Or manually:

```bash
git fetch origin
git checkout main && git pull --ff-only origin main
git fetch origin signals/latest
git checkout origin/signals/latest -- signals/ logs/
git reset HEAD signals/ logs/   # un-stage; files stay on disk for the dashboard
```

### Confirm today's Actions run fired

```bash
git log origin/signals/latest -1 --format="%ai %s"
gh run list --workflow=daily_paper_trading.yml --limit 5
```

If `gh` CLI isn't installed: GitHub web → Actions tab → "Daily Paper Trading".

### After local code changes

```bash
git add path/to/file.py
git commit -m "imperative summary"
git push origin main
```

**Do NOT push to `signals/latest`** — that branch is workflow-owned and force-pushed daily.

### Recovering from a bad merge

```bash
git status                            # see what's modified
git restore path/to/file.py           # discard one file's changes
git fetch origin main
git reset --hard origin/main          # DESTRUCTIVE; only when sure
```

---

## Common failures & fixes

### Dashboard shows $0 equity

The dashboard reads `signals/alpaca_daily_status.json` — make sure it's committed by the workflow.  Fixed in commit `ee57041` (workflow now commits it + fallback to `alpaca_paper_health.json`).

### "Alpaca orders log 🔴 old"

Per-file freshness thresholds — `alpaca_paper_log.csv` is "stale" between 26h and 72h (normal between cron fires).  Fixed in commit `543665d`.

### `factor_data_freshness_pass` fails locally

Local panel is older than Actions's panel.  Refresh:
```cmd
python research.py --incremental
```
If `last_bar` doesn't advance, `fetch_price_data` was hitting the stale-cache bug — fixed in `021d7dc`.

### Nested walkforward OOMs / freezes laptop

Use the batched wrapper instead of running directly:
```cmd
python run_walkforward_batched.py --recent-alpha-grid
```
Or close Chrome/Discord first; auto-detect now sizes against `AvailPhys`.

### "feature_research_summary.csv missing"

Run:
```cmd
python feature_research.py --top 24 --skip-pairs
```
Or:
```cmd
python refresh_local_research_data.py
```

### Walkforward approval False due to one outlier fold

The published FAMILY may still be approvable.  Use:
```cmd
python publish_live_config_from_csv.py --source stable_family --force
```
`--force` overrides single-fold-outlier gates when the family itself is clean.

### Actions hasn't run today

Wait for the cron (13:35 UTC in DST = 9:35 NY).  If past that time and no commit:
- Check GitHub Actions tab for failed runs
- Verify `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` haven't rotated
- Manually trigger via Actions tab → "Daily Paper Trading" → "Run workflow"

---

## Safety notes

- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` belong in `.env`, never committed.
- Use `daily_run.py --dry-run` to inspect any pipeline change before live.
- Never push to `signals/latest` manually.
- `--force` overrides safety checks — use only when you understand the specific safeguard you're bypassing.
- Idempotency is built in: local `_already_submitted_today()` check + Alpaca's deterministic `client_order_id` rejection.  Same-day double-submit is blocked.

---

## Real-capital blockers

Don't promote to real money until ALL of these are green:

| Check | Where |
|---|---|
| Walkforward `live_config_approval.approved: True` | `signals/core_satellite_nested_walkforward.json` |
| Cost stress passes | same JSON, `cost_stress_approval_pass` |
| Medium risk review passes | same JSON, `medium_risk_review.pass` |
| Factor decay `pass` or `advisory` | `logs/factor_decay_monitor.json` |
| 4+ weeks of positive paper alpha | `paper_scorecard.py` output |
| Gauntlet status `paper_ready` | `signals/alpaca_paper_health.json` |
| `walkforward_analyzer.py` shows 0 FAILs | run after each walkforward |
