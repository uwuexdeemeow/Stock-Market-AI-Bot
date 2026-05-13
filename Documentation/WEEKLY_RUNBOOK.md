# Core-Satellite Paper Trading Runbook

This runbook covers **both** strategies:
- **Moomoo**: Core-satellite alpha (`core_satellite_alpha_signal.csv`)
- **Alpaca**: TQQQ-enhanced (`core_satellite_tqqq_signal.csv`)

Both share the same factor data pipeline and regime detection logic.

---

## One-Command Daily Run (Recommended)

The easiest way to run everything is with `daily_run.py`. It chains all steps
in the right order, handles errors gracefully, and sends notifications if
anything breaks:

```bash
python3 daily_run.py              # run everything (16 steps)
python3 daily_run.py --dry-run    # preview what would run without executing
python3 daily_run.py --moomoo     # only run Moomoo steps
python3 daily_run.py --alpaca     # only run Alpaca steps
python3 daily_run.py --stress     # also run stress tests (20 steps)
python3 daily_run.py --report     # also run side-by-side performance report
python3 daily_run.py --skip-refresh  # skip data download (use existing data)
python3 daily_run.py --force      # run even on weekends/holidays
```

### What daily_run.py does (16 steps, in order):

| # | Step | Script | What it does |
|---|------|--------|-------------|
| 1 | refresh_etf_data | `refresh_etf_data.py --refresh` | Download latest ETF prices (SPY, QQQ, TQQQ, etc.) |
| 2 | refresh_factor_data | `research.py` | Refresh factor panel (stock prices + factor scores) |
| 3 | fill_monitor | `fill_monitor.py --days 2` | Verify yesterday's orders filled (catch cancellations) |
| 4 | moomoo_signal | `core_satellite_alpha.py` | Generate core-satellite signal for Moomoo |
| 5 | moomoo_submit | `moomoo_paper_trading.py --submit` | Submit orders to Moomoo (sells first, wait, then buys) |
| 6 | moomoo_status | `moomoo_paper_trading.py --status` | Sync equity/positions and save daily status |
| 7 | moomoo_execution_guard | `moomoo_paper_trading.py --execution-guard` | Repair Moomoo core ETF stop-limit protection |
| 8 | moomoo_health | `paper_health.py` | Build deep health summary (slippage, concentration, risk) |
| 9 | moomoo_gauntlet | `paper_gauntlet.py` | Run Moomoo paper gauntlet health check |
| 10 | moomoo_daily_check | `daily_paper_check.py --skip-status --skip-sync` | Read-only verdict (status/sync already done) |
| 11 | alpaca_signal | `core_satellite_tqqq.py` | Generate TQQQ-enhanced signal for Alpaca |
| 12 | alpaca_submit | `alpaca_paper_trading.py --submit` | Submit orders to Alpaca (auto-snapshots equity) |
| 13 | alpaca_reconcile | `alpaca_paper_trading.py --reconcile` | Reconcile Alpaca order fills |
| 14 | alpaca_execution_guard | `execution_guard.py --once` | Repair ETF stops, cancel stale Alpaca orders, check P&L |
| 15 | alpaca_gauntlet | `alpaca_paper_gauntlet.py` | Run Alpaca paper gauntlet health check |
| 16 | regime_monitor | `regime_monitor.py` | Detect regime changes and alert (risk_on/neutral/risk_off) |

### Built-in safety features:
- **Weekend/holiday guard**: Automatically skips on weekends and US market holidays (use `--force` to override)
- **Data freshness gate**: Warns if factor data > 5 trading days old, blocks if > 10 days old (use `--ignore-stale` to override)
- **Sell-wait-buy phasing**: Moomoo sells execute first, waits for fills + settlement, then buys (prevents cancelled orders from insufficient buying power)
- **Failure notifications**: macOS notification banner if any step fails; optional email alerts via SMTP env vars
- **Regime change alerts**: macOS notification when market regime switches (e.g. risk_on to risk_off)
- **Fill verification**: Checks yesterday's orders before submitting new ones
- **Moomoo ETF protection**: STOP_LIMIT protection is repaired for SPY/QQQ/TQQQ when supported by Moomoo paper trading
- **Alpaca ETF protection**: Broker-side trailing stops are repaired for SPY/QQQ/TQQQ, so basic protection survives laptop sleep/offline time

---

## Manual Step-by-Step (If Not Using daily_run.py)

### Before Market Open

```bash
# 1. Refresh data
python3 refresh_etf_data.py --refresh
python3 research.py

# 2. Check yesterday's fills
python3 fill_monitor.py --days 2

# 3. Generate signals
python3 core_satellite_alpha.py       # Moomoo signal
python3 core_satellite_tqqq.py        # Alpaca signal

# 4. Check regime
python3 regime_monitor.py
```

### Submit Orders (Market Hours)

```bash
# Moomoo
python3 moomoo_paper_trading.py --submit

# Alpaca
python3 alpaca_paper_trading.py --submit
```

### After Orders (Any Time)

```bash
# Moomoo status + health
python3 moomoo_paper_trading.py --status
python3 paper_health.py
python3 daily_paper_check.py

# After fills
python3 daily_paper_check.py --after-fill

# Alpaca reconciliation
python3 alpaca_paper_trading.py --reconcile

# Moomoo safety guard
python3 moomoo_paper_trading.py --execution-guard

# Alpaca safety guard
python3 execution_guard.py --once
```

Do not use `--submit` unless the broker is connected and you intend to send
paper orders. The sell-wait-buy logic handles buying power settlement
automatically, but orders still require market hours.

---

## Weekly Trust Checks

Run these once per week, or after any strategy/data change:

```bash
python3 daily_run.py --stress     # runs all 4 stress tests after normal pipeline
```

Or run stress tests individually:

```bash
python3 config_health.py
python3 refresh_etf_data.py --symbols SPY QQQ TQQQ
python3 leakage_audit.py
python3 core_satellite_execution_stress.py
python3 core_satellite_survivorship_audit.py
python3 core_satellite_drawdown_throttle.py
python3 factor_decay_monitor.py
python3 quant_audit.py
python3 paper_gauntlet.py
python3 alpaca_paper_gauntlet.py
```

What to check:
- `paper_health.py` should show `fill_rate=1.000`, low drift, reasonable slippage
- `quant_audit.py` should show `primary_strategy_ok=True`
- `fill_monitor.py` should show no cancelled or partial orders
- `regime_monitor.py` should log any recent regime transitions
- Survivorship stress should pass (may warn if alpha is survivor-sensitive)
- Execution stress should pass every scenario
- Factor decay must not show negative recent overlay alpha
- Real capital remains blocked until paper gauntlet passes on **both** strategies

---

## Paper Health Checks

Run this any time you want the fast paper-trading dashboard:

```bash
python3 paper_health.py
```

It writes:
- `signals/paper_health.json`
- `logs/paper_health_YYYYMMDD.json`

Key fields to check:
- `freshness_ok`: must be true before any new submit
- `factor_data.latest_data_date` and `factor_data.age_trading_days`: if stale, run `python3 research.py` first
- `paper_equity_days`: must reach at least `20` before real-capital approval
- `submitted_orders`, `filled_orders`, and `fill_rate`: confirms broker fills are being reconciled
- `current_gross_exposure`, `target_gross_exposure`, and `max_drift_abs`: confirms paper account is close to target
- `avg_slippage_bps`, `worst_slippage_bps`: confirms execution is not worse than model assumptions
- `readiness_flags`: separates strategy readiness, signal freshness, broker sync, account alignment, slippage, drawdown, concentration, and real-capital approval
- `concentration`: separates core ETF vs. overlay stock vs. sector concentration
- `open_position_attribution`: splits current open-position P&L into core and overlay sleeves
- `current_order_lifecycle`: lists current-signal orders with filled/unfilled quantity
- `drift_breakdown`: ranks tickers contributing most to account drift
- `stale_open_order_alerts`: flags open/partial orders older than configured threshold
- `go_live_scorecard`: shows progress against real-capital readiness gates

---

## Daily Paper Check

Use this when you want one practical answer while waiting for paper fills:

```bash
python3 daily_paper_check.py
python3 daily_paper_check.py --after-fill
```

It runs read-only Moomoo status, fill reconciliation, paper health, and the
paper gauntlet. It writes:
- `signals/daily_paper_check.json`
- `logs/daily_paper_check_YYYYMMDD.json`
- `logs/paper_snapshots/YYYYMMDDTHHMMSSZ/`

Verdicts:
- `STILL_OPEN`: expected fills are not complete yet
- `NEEDS_REBALANCE`: fills complete but account drift still above threshold
- `ALIGNED`: fills complete and paper account matches target
- `SYNC_OR_WAIT_FOR_FILLS`: strategy and signal are fine, but local logs don't show current-signal fills yet
- `WAIT_FOR_OPEN_ORDERS`: current-signal orders exist, but at least one is still open/partial
- `REBALANCE_PENDING`: broker is synced, but account is still away from target
- `CONTINUE_PAPER`: paper setup healthy, but real-capital gauntlet not approved yet
- `PAPER_GAUNTLET_APPROVED`: all encoded paper gauntlet checks passed

Snapshot cleanup:

```bash
python3 daily_paper_check.py --prune-snapshots --keep-days 30
```

---

## Fill Monitoring

Check if recent orders filled successfully:

```bash
python3 fill_monitor.py              # check last 1 day
python3 fill_monitor.py --days 7     # check last 7 days
python3 fill_monitor.py --quiet      # only print if problems found
```

Sends a macOS notification if any orders are cancelled, partial, or missing.
Automatically runs as step 3 of `daily_run.py` (before new signal generation).

---

## Regime Monitoring

Check if the market regime changed:

```bash
python3 regime_monitor.py            # check both strategies
python3 regime_monitor.py --quiet    # only print if regime changed
```

Regime types:
- **risk_on**: Both QQQ and SPY above their moving averages, low volatility. Heavy QQQ, more overlay stocks.
- **neutral**: SPY above MA but QQQ below, or moderate conditions. Balanced SPY/QQQ mix.
- **risk_off**: SPY below MA or high volatility. Defensive — more SPY, fewer overlay stocks.

Files:
- `signals/regime_history.json` — current and previous regime per strategy
- `signals/regime_changes_log.csv` — log of all regime transitions

---

## Config and Data Health

Before strategy changes, dependency updates, or a weekly run:

```bash
python3 config_health.py
python3 refresh_etf_data.py --symbols SPY QQQ TQQQ
```

`config_health.py` checks key package versions and runs `pip check`.
`refresh_etf_data.py` validates ETF parquet row count, freshness, positive
closes, and flat recent closes. Add `--refresh` only when you want to replace
stale or missing ETF parquet files.

---

## Monthly Robustness Review

Run the calmer comparison strategy:

```bash
python3 core_satellite_robust_mode.py
```

Compare:
- Primary return, Sharpe, and drawdown
- Robust-mode return, Sharpe, and drawdown
- Survivorship-stressed return
- Execution-stressed return

The robust mode is a comparison signal, not the primary signal.

---

## Research-Only Scripts

Run these only when changing strategy logic, score construction, risk controls,
or the universe:

```bash
python3 core_satellite_robust_research.py --stress-top-n 4
python3 core_satellite_robust_research.py --full
python3 survivorship_audit.py --build --report --min-rows 500
```

---

## Notification Setup (Optional)

### macOS Notifications (Built-in)
Works automatically — no setup needed. You get native notification banners for:
- Failed pipeline steps
- Regime changes
- Unfilled orders

### Email Alerts (Optional)
Set these environment variables to receive email alerts on failures:
```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your.email@gmail.com
export SMTP_PASSWORD=your-app-password
export ALERT_EMAIL=your.email@gmail.com  # defaults to SMTP_USER
```

---

## Real-Capital Blockers

Do not move to real capital until **all** of these are true for **both** strategies:
- `paper_gauntlet.py` approves real capital (Moomoo)
- `alpaca_paper_gauntlet.py` approves real capital (Alpaca)
- At least `20` paper equity days exist
- Submitted paper orders exist
- Fill rate is at least `95%`
- Cancel rate is at most `5%`
- Max drift is at most `15%`
- Survivorship and execution stress reports pass
- Factor decay monitor does not block real capital
- `fill_monitor.py` shows no recent cancelled orders

---

## Key Files Reference

| File | What it contains |
|------|-----------------|
| `signals/core_satellite_alpha_signal.csv` | Current Moomoo signal (regime, weights, tickers) |
| `signals/core_satellite_tqqq_signal.csv` | Current Alpaca signal (with TQQQ allocation) |
| `signals/paper_trades.csv` | Trade log with fill statuses |
| `signals/paper_health.json` | Latest health dashboard |
| `signals/paper_daily_status.json` | Latest daily status snapshot |
| `signals/regime_history.json` | Current/previous regime per strategy |
| `signals/regime_changes_log.csv` | Log of all regime transitions |
| `signals/fill_monitor.json` | Latest fill verification report |
| `signals/moomoo_execution_guard_state.json` | Moomoo guard high-water mark state |
| `signals/guard_intraday_state.json` | Execution guard daily alert/debounce state |
| `logs/execution_guard.log` | Execution guard activity and safety actions |
| `logs/daily_run_YYYYMMDD.json` | Daily pipeline run log |
| `data/*.parquet` | Factor panel and ETF price data |
