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
- [Alpaca submit parameters](#alpaca-submit-parameters)
- [Safety notes](#safety-notes)
- [Real-capital blockers](#real-capital-blockers)

---

## Quick reference — by goal

If you just want one command, find your goal here:

| I want to... | Command |
|---|---|
| Sync today's Actions outputs to my laptop | `pull_daily.bat` (or `bash pull_daily.sh`) |
| See today's status | `python status.py` |
| Run the bot manually today | `python daily_run.py --alpaca --timeout 900` |
| Run the Actions-style daily bot locally | `python daily_run.py --alpaca --timeout 900 --skip-factor-refresh` |
| Refresh Alpaca live equity/status only | `python alpaca_paper_trading.py --status` |
| Refresh recent fill-quality stats | `python alpaca_paper_trading.py --slippage-report` |
| Run the shadow paper journal locally | `python shadow_paper_journal.py` |
| Compare Alpaca vs shadow equity | `python paper_shadow_compare.py` |
| Refresh local data + features before research | `python refresh_local_research_data.py` |
| Check restored factor cache before trading | `python factor_data_health.py --strict --no-write` |
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
| **Every 9:55 AM NY weekday** | Shadow config journal via Actions | `.github/workflows/shadow_paper_journal.yml` (automatic) |
| **Every 7:30 AM NY weekday** | Factor data refresh via Actions | `.github/workflows/factor_data_refresh.yml` (automatic) |
| **Every morning** | `pull_daily.bat` to sync logs to laptop | you |
| **Once per week (Fri)** | Scorecard + Alpaca gauntlet | you |
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
- `shadow_paper_journal.csv` — shadow config signal journal
- `shadow_paper_equity.csv` — shadow config simulated equity curve
- `paper_shadow_compare.json` / `.csv` — Alpaca-vs-shadow return comparison
- `workflow_heartbeat_daily.json` / `workflow_heartbeat_shadow.json` — cron proof files
- `monitor_heartbeat.json` — watchdog status
- `logs/daily_run_*.json` — pipeline step results

### Check (anytime)

Open the dashboard:

```cmd
streamlit run dashboard.py
```

The dashboard live-refreshes the account tiles by default without reloading the
whole page.  That refresh asks Alpaca for a fresh paper account snapshot and
rewrites `signals/alpaca_daily_status.json` plus `signals/alpaca_paper_equity.csv`,
so the equity card updates without running `alpaca_paper_trading.py --reconcile`
and without disrupting filters, forms, or scroll position.

Look for:
- 🟢 fresh on Alpaca log, equity, status, health
- Account equity > 0 (was a known bug — fixed)
- Today's trades reflect the published top3 family
- Signal & Orders → Alpaca submission results shows whether each planned order
  was submitted, cash-resized, skipped, filled, or still pending
- Performance → Execution quality shows recent fill slippage and 5/15/60m
  reversal counts, all/limit/market execution segments, plus the worst recent
  tickers by execution risk score
- Home + Performance show Alpaca vs shadow return spread
- Telegram alerts arrived for anything red

### Manual run (only if Actions failed)

```bash
python daily_run.py --alpaca --timeout 900
```

This is what Actions runs internally — refreshes data, generates signal, submits orders, reconciles fills, rebuilds health.  Idempotent: refuses to double-submit by checking Alpaca's live order history first, falling back to `alpaca_paper_log.csv`, and using deterministic `client_order_id` values.

Manual runs no longer suppress the automatic cron later that same trading day.
The workflow may still start on schedule after a manual test, but the broker
duplicate-order checks and deterministic `client_order_id` values prevent
double-submitting the same day's orders.

The Alpaca submit path now has extra no-margin guards:
- default order type is protective day limits (`ALPACA_ORDER_TYPE=limit`)
- closed-market queueing is off by default (`ALPACA_ALLOW_CLOSED_MARKET_QUEUE=0`)
- sells submit before buys
- buys are skipped if sells fail, sells do not fill quickly, or cash is below the no-margin threshold
- cash-limited buys are resized down to the largest whole-share quantity that fits available cash; if even one share cannot fit or the trade falls below `ALPACA_MIN_TRADE_VALUE`, the buy is skipped and logged
- after submit/reconcile, the bot repairs overlay trailing stops from live Alpaca positions and warns if stock value is materially above equity

To run without submitting:
```bash
python daily_run.py --alpaca --dry-run
```

Actions runs the daily Alpaca job with:

```bash
python daily_run.py --alpaca --timeout 900 --skip-factor-refresh
```

That means the daily workflow trades from the latest successful factor-data
cache. If the cache is missing or stale, the workflow first rebuilds it with:

```bash
python research.py --incremental
python feature_quality_diagnostic.py --top 48
python ci_check_feature_report.py --min-features 20
python factor_data_health.py --strict
```

The dedicated factor refresh workflow uses the same core refresh chain:

```bash
python research.py --incremental      # or: python research.py --xs-only
python feature_quality_diagnostic.py --top 48
python factor_data_health.py --strict
```

`feature_quality_diagnostic.py` is the daily factor diagnostic. It writes
`signals/feature_quality_report.json` and `signals/feature_quality_summary.csv`.
`feature_health.py` / `signals/feature_health_profile.json` are downstream
feature-quarantine outputs, not the main scheduled factor-refresh diagnostic.

### Shadow journal

The shadow workflow runs after the Alpaca workflow and records what the shadow
config would have held without submitting broker orders:

```bash
python shadow_paper_journal.py
```

Useful manual flags:

```bash
python shadow_paper_journal.py --ignore-stale
python shadow_paper_journal.py --append-duplicate
python shadow_paper_journal.py --journal-path signals/shadow_paper_journal.csv
python shadow_paper_journal.py --equity-path signals/shadow_paper_equity.csv
python shadow_paper_journal.py --no-restore-signal-artifacts
```

Optional starting equity override:
```bash
SHADOW_PAPER_INITIAL_EQUITY=101161.27 python shadow_paper_journal.py
```

Outputs:
- `signals/shadow_paper_journal.csv`
- `signals/shadow_paper_equity.csv`

---

## Weekly routine

Run every Friday (or whenever convenient):

```cmd
python paper_scorecard.py
python alpaca_paper_gauntlet.py --verbose
```

- `paper_scorecard.py` — performance vs walkforward expectation
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
# Dry-run gates to check:
# - walkforward_analyzer FAIL blocks publish unless --force is used.
# - selection_bias_gap_sharpe must stay under the approval threshold.
# - conc_ov=... overlays are preserved exactly and kept in separate families.

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
| `alpaca_paper_trading.py` | Generate and (optionally) submit Alpaca orders | `--submit`, `--status`, `--reconcile`, `--slippage-report`, `--market-order`, `--limit-order`, `--quote-limit`, `--last-trade-limit`, `--allow-closed-market-queue`, `--allow-stale-signal`, `--max-signal-age-hours H`, `--max-factor-age-trading-days N` |
| `broker_health.py` | Pre-flight Alpaca connectivity check | none |
| `core_satellite_alpha.py` | Generate today's signal (which 3 stocks + ETF weights) | `--ignore-stale`, `--walkforward` |
| `daily_run.py` | Run the entire daily pipeline (13 steps) | `--alpaca`, `--dry-run`, `--force`, `--health-only`, `--skip-refresh`, `--skip-factor-refresh`, `--no-github-sync`, `--timeout N` |
| `execution_guard.py` | Repair ETF stop-loss protection, cancel stale orders | `--once`, `--loop`, `--dry-run` |
| `fill_monitor.py` | Verify yesterday's fills (cancelled, partial, slipped) | `--days N` |
| `monitor_heartbeat.py` | Watchdog — all monitors produced fresh output? | none |
| `notifications.py` | Send Telegram / email alerts | library, not run directly |
| `paper_health.py` | Build deep health dashboard (slippage, drift, risk) | `--broker alpaca` |
| `paper_shadow_compare.py` | Compare Alpaca paper equity vs shadow paper equity | `--alpaca-equity PATH`, `--shadow-equity PATH`, `--csv-out PATH`, `--json-out PATH` |
| `regime_monitor.py` | Detect risk_on / neutral / risk_off regime shifts | none |
| `risk_sizing.py` | Position sizing helpers | library |
| `signal_freshness.py` | Reject stale signals at trade time | library |
| `shadow_paper_journal.py` | Record shadow config signal + simulated equity without sending orders | `--journal-path PATH`, `--equity-path PATH`, `--ignore-stale`, `--append-duplicate`, `--no-restore-signal-artifacts` |
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
| `feature_quality_diagnostic.py` | Re-grade per-feature live IC | daily (CI: `--top 48`) + before walkforward |
| `feature_research.py` | Per-feature analysis (IC trend, sector, decay, pairs) | quarterly (`--top 24 --skip-pairs`, or `--pairs`) |
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
| `research.py` | Build per-ticker factor parquet panel | daily (CI: `--incremental`, optional `--xs-only`) + ad-hoc |
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
| `publish_live_config_from_csv.py` | Manual live-config promote; blocks analyzer FAIL and selection-bias overfit | after walkforward (`--source stable_family`, `--dry-run`, `--force`) |
| `pull_daily.bat` / `pull_daily.sh` | Sync Actions outputs into local repo | every morning |
| `refresh_local_research_data.py` | Local mirror of `factor_data_refresh.yml` plus feature research | `--skip-research`, `--skip-feature-research`, `--pairs`, `--top N`, `--dry-run` |
| `run_walkforward_batched.py` | Memory-safe walkforward wrapper | `--batch-size N`, `--max-batches N`, `--help-wrapper`, plus forwarded walkforward flags such as `--recent-alpha-grid` |

### Infrastructure & dashboards

| Script | What it does |
|---|---|
| `config_health.py` | Validate env / settings / required keys |
| `ci_check_feature_report.py` | CI guard that rejects partial feature-quality rebuilds |
| `dashboard.py` + `dashboard/` + `pages/` | Streamlit dashboard |
| `data_provider.py` | Multi-source price downloader (yfinance / yahooquery / stooq) |
| `data_validation.py` | Reject malformed price frames |
| `experiment_ledger.py` | Track what's been run for reproducibility |
| `factor_data_health.py` | Strict pass/fail on panel freshness (`--strict`, `--ready-only`, `--no-write`) |
| `http_retry.py` | Idempotent HTTP wrapper for data fetches |
| `log_cleanup.py` | Old-log housekeeping |
| `monitor.py` | Generic file watcher |
| `options_iv_provider.py` | Tradier IV data |
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
| `signals/alpaca_slippage_reversal_report.json` | Recent fill slippage and 5/15/30/60 minute post-fill reversal stats |
| `signals/shadow_paper_journal.csv` | Daily shadow config targets, signal metadata, and comparison rows |
| `signals/shadow_paper_equity.csv` | Simulated equity curve for the shadow config |
| `signals/paper_shadow_compare.csv` | Row-by-row Alpaca-vs-shadow return comparison |
| `signals/paper_shadow_compare.json` | Compact Alpaca-vs-shadow summary for dashboard tiles |
| `signals/workflow_heartbeat_daily.json` | Last daily trading workflow event/run proof |
| `signals/workflow_heartbeat_shadow.json` | Last shadow workflow event/run proof |
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
| `signals/feature_quality_summary.csv` | Compact feature-quality grades table |
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

Cron times are UTC inside GitHub:
- Daily Paper Trading: `13:35 UTC` during New York daylight saving time, `14:35 UTC` during standard time.
- Shadow Paper Journal: `13:55 UTC` during New York daylight saving time, `14:55 UTC` during standard time.
- Factor Data Refresh: `11:30 UTC` during New York daylight saving time, `12:30 UTC` during standard time.

In Singapore, that means:
- Daily Paper Trading: `9:35 PM SGT` during New York daylight saving time, `10:35 PM SGT` during standard time.
- Shadow Paper Journal: `9:55 PM SGT` during New York daylight saving time, `10:55 PM SGT` during standard time.
- Factor Data Refresh: `7:30 PM SGT` during New York daylight saving time, `8:30 PM SGT` during standard time.

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
- Confirm the workflow is enabled on GitHub's Actions tab
- Verify `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` haven't rotated
- Manually trigger via Actions tab → "Daily Paper Trading" → "Run workflow"

---

## Alpaca submit parameters

These are the main environment knobs used by `alpaca_paper_trading.py` and the
daily GitHub workflow:

| Variable | Default | What it controls |
|---|---:|---|
| `ALPACA_ORDER_TYPE` | `limit` | Normal rebalance order type. Keep `limit` unless deliberately testing market orders. |
| `ALPACA_LIMIT_REFERENCE` | `last` | Limit anchor. `last` uses planned last trade; `quote` uses live bid/ask when explicitly enabled. |
| `ALPACA_LIMIT_OFFSET_BPS_ETF` | `5` | ETF protective-limit cushion in basis points. |
| `ALPACA_LIMIT_OFFSET_BPS_OVERLAY` | `12` | Overlay-stock protective-limit cushion in basis points. |
| `ALPACA_ALLOW_CLOSED_MARKET_QUEUE` | `0` | Whether non-interactive runs may queue orders while market is closed. Keep `0`. |
| `ALPACA_SKIP_BUYS_UNTIL_SELLS_FILLED` | `1` | Skip buys unless same-run sells fill first. |
| `ALPACA_SKIP_BUYS_WHEN_CASH_BELOW` | `0` | No-margin cash floor. Buys are skipped when cash is below this level. |
| `ALPACA_SELL_FILL_WAIT_SECONDS` | `20` | How long to wait for sell orders to fill before deciding whether buys are safe. |
| `ALPACA_SELL_FILL_POLL_SECONDS` | `2` | How often to poll Alpaca while waiting for sell fills. |
| `ALPACA_MARGIN_WARN_GROSS` | `1.02` | Warn if stock market value / equity is above this level. |
| `ALPACA_TRAILING_STOP` | `1` | Enable overlay trailing stops. |
| `ALPACA_TRAILING_STOP_PCT` | `0.08` | Overlay trailing-stop trail amount. |
| `GUARD_CORE_STOP` | `1` | Enable durable core ETF trailing stops in `execution_guard.py`. |
| `GUARD_CORE_TICKERS` | `SPY,QQQ,TQQQ` | Core ETF symbols protected by guard stops. |
| `GUARD_CORE_TRAIL_PCT` | `0.05` | Core ETF trail amount. |

Daily workflow currently sets:

```yaml
ALPACA_ORDER_TYPE=limit
ALPACA_LIMIT_REFERENCE=last
ALPACA_ALLOW_CLOSED_MARKET_QUEUE=0
GUARD_CORE_STOP=1
GUARD_CORE_TICKERS=SPY,QQQ,TQQQ
GUARD_CORE_TRAIL_PCT=0.05
```

Shadow paper parameter:

| Variable | Default | What it controls |
|---|---:|---|
| `SHADOW_PAPER_INITIAL_EQUITY` | `100000` | Starting fake equity for `shadow_paper_equity.csv` when no prior row exists. |

---

## Safety notes

- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` belong in `.env`, never committed.
- Use `daily_run.py --dry-run` to inspect any pipeline change before live.
- Never push to `signals/latest` manually.
- `--force` overrides safety checks — use only when you understand the specific safeguard you're bypassing.
- Idempotency is built in: the bot checks Alpaca's live order history first,
  falls back to the local order log, and still uses deterministic
  `client_order_id` values.  Same-day double-submit is blocked.
- Do not turn on closed-market queueing or market orders for routine runs.
- If Telegram reports `skipped` buys, inspect the sell status and cash before rerunning.

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
