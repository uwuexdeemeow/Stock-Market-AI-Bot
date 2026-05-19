# Core-Satellite Paper Trading Runbook

This runbook covers the **unified** core-satellite strategy:
- **Signal**: `core_satellite_alpha_signal.csv` (one signal, both brokers)
- **Moomoo**: Reads the same signal for core-satellite alpha
- **Alpaca**: Reads the same signal — TQQQ allocation is included when the
  nested walkforward grid search determines it helps on a risk-adjusted basis

Both brokers share the same factor data pipeline, regime detection, and signal.

---

## One-Command Daily Run (Recommended)

The easiest way to run everything is with `daily_run.py`. It chains all steps
in the right order, handles errors gracefully, and sends notifications if
anything breaks:

```bash
python3 daily_run.py              # run everything (14 steps)
python3 daily_run.py --dry-run    # preview what would run without executing
python3 daily_run.py --moomoo     # only run Moomoo steps
python3 daily_run.py --alpaca     # only run Alpaca steps
python3 daily_run.py --stress     # also run stress tests
python3 daily_run.py --report     # also run side-by-side performance report
python3 daily_run.py --skip-refresh  # skip data download (use existing data)
python3 daily_run.py --force      # run even on weekends/holidays
```

### `daily_run.py` parameters

| Flag | What it does |
|------|-------------|
| `--dry-run` | Print every step that would run, but don't execute anything. Good for checking the pipeline order. |
| `--moomoo` | Only run Moomoo-related steps (signal, submit, status, health, gauntlet). Skips all Alpaca steps. |
| `--alpaca` | Only run Alpaca-related steps (submit, reconcile, guard, gauntlet). Skips all Moomoo steps. |
| `--report` | After the normal pipeline, also run `paper_report.py` for a side-by-side Moomoo vs Alpaca performance comparison. |
| `--stress` | After the normal pipeline, also run 4 stress tests: factor decay, drawdown throttle, execution stress, survivorship audit. |
| `--skip-refresh` | Skip steps 1-2 (ETF data download + factor panel refresh). Use when data is already fresh and you just want to resubmit orders. |
| `--force` | Run even on weekends and US market holidays. Normally the script auto-skips non-trading days. |
| `--timeout N` | Max seconds each step is allowed to run before being killed (default: 300). Increase if `research.py` is slow on your machine. |

### What daily_run.py does (14 steps, in order):

| # | Step | Script | What it does |
|---|------|--------|-------------|
| 1 | refresh_etf_data | `refresh_etf_data.py --refresh` | Download latest ETF prices (SPY, QQQ, TQQQ, etc.) |
| 2 | refresh_factor_data | `research.py` | Refresh factor panel (stock prices + factor scores) |
| 3 | fill_monitor | `fill_monitor.py --days 2` | Verify yesterday's orders filled (catch cancellations) |
| 4 | signal | `core_satellite_alpha.py` | Generate unified signal (both Moomoo and Alpaca) |
| 5 | moomoo_submit | `moomoo_paper_trading.py --submit` | Submit orders to Moomoo (sells first, wait, then buys) |
| 6 | moomoo_status | `moomoo_paper_trading.py --status` | Sync equity/positions and save daily status |
| 7 | moomoo_execution_guard | `moomoo_paper_trading.py --execution-guard` | Repair Moomoo core ETF stop-limit protection |
| 8 | moomoo_health | `paper_health.py` | Build deep health summary (slippage, concentration, risk) |
| 9 | moomoo_gauntlet | `paper_gauntlet.py` | Run Moomoo paper gauntlet health check |
| 10 | moomoo_daily_check | `daily_paper_check.py --skip-status --skip-sync` | Read-only verdict (status/sync already done) |
| 11 | alpaca_submit | `alpaca_paper_trading.py --submit` | Submit orders to Alpaca (reads same signal) |
| 12 | alpaca_reconcile | `alpaca_paper_trading.py --reconcile` | Reconcile Alpaca order fills |
| 13 | alpaca_execution_guard | `execution_guard.py --once` | Repair ETF stops, cancel stale Alpaca orders, check P&L |
| 14 | alpaca_gauntlet | `alpaca_paper_gauntlet.py` | Run Alpaca paper gauntlet health check |

### Built-in safety features:
- **Weekend/holiday guard**: Automatically skips on weekends and US market holidays (use `--force` to override)
- **Data freshness gate**: Warns if factor data > 5 trading days old, blocks if > 10 days old (use `--ignore-stale` to override)
- **Fail-soft sentiment fallback**: Finnhub is preferred for live news, but Yahoo/Google RSS and social sentiment can keep the sentiment gate at `PARTIAL` instead of blocking trades. Trades block only if all fresh sentiment sources are unavailable, or if fallback sentiment finds strongly negative news for a selected stock.
- **Social sentiment safety mode**: StockTwits/X sentiment is live diagnostic/fallback data only (`SOCIAL_SENTIMENT_SAFETY_ENABLED=1`). It is not included as trainable model alpha unless `SOCIAL_SENTIMENT_ALPHA_ENABLED=1` is explicitly enabled after separate validation.
- **Sell-wait-buy phasing**: Moomoo sells execute first, waits for fills + settlement, then buys (prevents cancelled orders from insufficient buying power)
- **Failure notifications**: macOS notification banner if any step fails; optional email alerts via SMTP env vars
- **Regime change alerts**: macOS notification when market regime switches (e.g. risk_on to risk_off)
- **Fill verification**: Checks yesterday's orders before submitting new ones
- **Moomoo ETF protection**: STOP_LIMIT protection is repaired for SPY/QQQ/TQQQ when supported by Moomoo paper trading
- **Alpaca ETF protection**: Broker-side trailing stops are repaired for SPY/QQQ/TQQQ, so basic protection survives laptop sleep/offline time

---

## Script Parameter Reference

### `core_satellite_alpha.py` — Signal Generator

Generates the unified signal CSV that both brokers read. Loads the approved
config from `signals/core_satellite_live_configs.json` (written by the nested
walkforward) and produces today's target weights.

```bash
python3 core_satellite_alpha.py                          # normal daily signal
python3 core_satellite_alpha.py --ignore-stale           # force signal even with old data
python3 core_satellite_alpha.py --walkforward            # run nested validation first, then signal
```

| Flag | What it does |
|------|-------------|
| `--ignore-stale` | Override the data freshness block. Normally the script refuses to generate a signal if factor data is more than 10 trading days old (stale data = stale stock picks = bad trades). Use this only when you know the data is acceptable. |
| `--walkforward` | Before generating the signal, run a one-off nested walkforward validation of the core-alpha strategy. For routine weekly validation, prefer running `core_satellite_nested_walkforward.py` directly. |
| `--no-walkforward` | Legacy no-op. Nested validation is already skipped by default. |
| `--min-train-years N` | Minimum training years before first test fold when using `--walkforward` (default: 4). |

**Output**: `signals/core_satellite_alpha_signal.csv`

---

### `core_satellite_nested_walkforward.py` — Walk-Forward Validation

The **most important** research script. Runs proper nested cross-validation:
the inner loop tunes parameters (including TQQQ weight), the outer loop tests
on unseen future years. Writes approval state that the daily signal generator
reads.

```bash
python3 core_satellite_nested_walkforward.py                    # full run (recommended weekly)
python3 core_satellite_nested_walkforward.py --fast             # smoke test (~10 min instead of hours)
python3 core_satellite_nested_walkforward.py --stable-grid --no-resume  # pinned alpha-decay baseline
python3 core_satellite_nested_walkforward.py --recent-alpha-grid --no-resume  # focused post-2020 alpha grid
python3 core_satellite_nested_walkforward.py --no-resume        # ignore checkpoint, start fresh
python3 core_satellite_nested_walkforward.py --workers 1        # single-threaded (debug)
python3 core_satellite_nested_walkforward.py --no-low-memory    # faster on 32+ GB machines
```

| Flag | Default | What it does |
|------|---------|-------------|
| `--strategy {core-alpha,tqqq,both}` | `core-alpha` | Which strategy to validate. `tqqq` and `both` are deprecated aliases — they all resolve to `core-alpha` now because TQQQ is a grid knob inside the unified strategy. |
| `--min-train-years N` | `3` | Minimum number of calendar years of training data required before the first outer test fold. Smaller = more folds but less training data per fold. |
| `--min-inner-train-years N` | auto | Minimum training years for inner (validation) folds. Defaults to `min_train_years - 1` (minimum 2). Reduce if you have a short data history. |
| `--start-year YYYY` | auto | First outer test year. By default, starts as early as the data allows given `min-train-years`. |
| `--end-year YYYY` | auto | Last outer test year. By default, uses the last full year in the data. |
| `--max-configs N` | all | Cap the number of candidate configs evaluated per fold. Useful for smoke tests. The full grid has 384 configs (96 base × 4 TQQQ weights). |
| `--max-folds N` | all | Cap the number of outer test folds. Useful for quick checks (e.g. `--max-folds 2` tests only the 2 most recent years). |
| `--max-specs N` | `48` | Maximum number of feature specs to load. Reducing this shrinks the factor panel and speeds up each evaluation, but may drop useful features. |
| `--output-prefix NAME` | `core_satellite_nested_walkforward` | Prefix for output files. Changes the JSON/CSV filenames under `signals/`. Using a non-default prefix **disables** automatic live config publishing (safety: debug runs shouldn't overwrite production approvals). |
| `--fast` | off | Use a reduced grid (~32 configs instead of 384) and fewer knob combinations. Good for smoke tests. Still runs proper nested validation — just with fewer candidates. |
| `--stable-grid` | off | Use the consensus baseline grid (24 configs): pins 14-fold winners (h=20, overlay 50%, MA 100, regime-adaptive scoring, risk=off) and explicitly drops overlay=0.7 which had 562% turnover and 0.95 Sharpe in the May 2026 run. Tunes shape (top5/10/15), weighting (sticky_score/risk_parity), vol mode, and TQQQ (0/10%). Auto-publishing is disabled unless `--publish-live-config` is passed. |
| `--recent-alpha-grid` | off | Use the focused post-2020 alpha grid (~48 configs): holding days 20, MA 100, regime-adaptive scoring, risk control off; tunes overlay 50/70%, top3/top5/top15 concentration, sticky-score/risk-parity weighting, high-vol mode, and 0/10% TQQQ. Auto-publishing is disabled unless `--publish-live-config` is passed. |
| `--publish-live-config` | auto | Force writing approval state to `signals/core_satellite_live_configs.json`, even for bounded/debug runs. Normally, runs with `--fast`, `--stable-grid`, `--recent-alpha-grid`, `--max-folds`, `--max-configs`, custom `--output-prefix`, or partial year windows do NOT publish (to prevent debug/research runs from overwriting production approvals). |
| `--no-publish-live-config` | — | Never write approval state. Useful for dry-run full validations where you want to inspect results without affecting production. |
| `--resume` / `--no-resume` | `--resume` | Resume from the last checkpoint if available. After each outer fold completes, progress is saved to `signals/walkforward_checkpoint_core_alpha.json`. If the run crashes or you Ctrl-C, re-running picks up from the last completed fold. Use `--no-resume` to force a clean start (e.g. after changing strategy code). The checkpoint auto-invalidates if you change the config grid or fold range. |
| `--low-memory` / `--no-low-memory` | `--low-memory` | Aggressive memory management — disables eval caching and runs garbage collection after every config. Prevents macOS from OOM-killing the process on 16 GB laptops. Use `--no-low-memory` on machines with 32+ GB RAM for ~2× faster runs. |
| `--workers N` | auto-detected | Number of parallel worker processes. Auto-detected from CPU count and available RAM (each worker needs ~2 GB). Workers use fork-based multiprocessing so the 500 MB factor panel is shared via copy-on-write (no pickling). Set to `1` to disable parallelism for debugging. |

**Key outputs**:
- `signals/core_satellite_nested_walkforward.json` — full results with per-fold metrics
- `signals/core_satellite_nested_walkforward.csv` — fold-level summary table
- `signals/core_satellite_live_configs.json` — approved config for the daily signal generator (only written on full runs)

**What to look for**:
- **Mean OOS Sharpe > 0.5** = strategy has a real edge
- **Mean OOS alpha vs BLEND > 0%** = strategy beats a passive SPY/QQQ mix
- **Selection-bias gap < 1.5** = inner optimizer isn't just fitting noise
- **Config stability > 30%** = the same config wins across different years
- **Winning config tqqq_weight** = tells you if TQQQ helps (0.0 = pure core-alpha)

---

### `alpaca_paper_trading.py` — Alpaca Broker

Reads `signals/core_satellite_alpha_signal.csv`, compares target weights to
current Alpaca positions, and submits rebalance orders.

```bash
python3 alpaca_paper_trading.py                  # show plan (no orders sent)
python3 alpaca_paper_trading.py --submit         # actually submit orders
python3 alpaca_paper_trading.py --status         # show account positions
python3 alpaca_paper_trading.py --reconcile      # check if pending orders filled
python3 alpaca_paper_trading.py --force          # skip safety checks
```

| Flag | What it does |
|------|-------------|
| `--submit` | Actually send orders to Alpaca. Without this flag, the script only shows what it would do (dry run). |
| `--status` | Show current account value, positions, and buying power. No order planning. |
| `--reconcile` | Check if previously submitted orders have filled. Updates the paper log with fill statuses. Run this a few hours after `--submit`. |
| `--force` | Skip both the drift threshold check (normally skips tiny rebalances) and the duplicate-day check (normally blocks re-submitting on the same day). |
| `--market-order` | Use market orders instead of limit orders (this is the default). |

**Environment variables** (set these before first use):
- `ALPACA_API_KEY` — your Alpaca paper trading API key
- `ALPACA_SECRET_KEY` — your Alpaca paper trading secret key
- `ALPACA_BASE_URL` — paper trading URL (default: `https://paper-api.alpaca.markets`)
- `ALPACA_ETF_DRIFT_THRESHOLD` — minimum ETF weight drift before trading (default: `0.03` = 3%)
- `ALPACA_OVERLAY_DRIFT_THRESHOLD` — minimum stock drift before trading (default: `0.01` = 1%)
- `ALPACA_TRAILING_STOP` — enable trailing stops on overlay stocks (default: `1` = on)
- `ALPACA_TRAILING_STOP_PCT` — trailing stop percentage (default: `0.08` = 8%)
- `ALPACA_DD_HALT_PCT` — portfolio drawdown halt threshold (default: `0.12` = 12%)
- `TQQQ_FAST_DD_THRESHOLD` — TQQQ circuit breaker: if TQQQ drops this much from its 5-day high, skip TQQQ buys and close existing TQQQ (default: `-0.15` = -15%)
- `TQQQ_FAST_DD_LOOKBACK_DAYS` — lookback window for the TQQQ circuit breaker (default: `5`)

---

### `moomoo_paper_trading.py` — Moomoo Broker

Reads `signals/core_satellite_alpha_signal.csv`, compares to current Moomoo
positions, and submits rebalance orders via the OpenD desktop app.

```bash
python3 moomoo_paper_trading.py                          # show plan (no orders)
python3 moomoo_paper_trading.py --submit                 # submit orders
python3 moomoo_paper_trading.py --status                 # sync equity + save status
python3 moomoo_paper_trading.py --sync                   # reconcile fill history
python3 moomoo_paper_trading.py --execution-guard        # repair ETF stop-limits
```

| Flag | What it does |
|------|-------------|
| `--submit` | Submit paper orders to Moomoo. Sells execute first, then waits, then buys (prevents buying-power shortfall). |
| `--force` | Override the rebalance guard (still requires `--submit`). |
| `--status` | Sync equity and positions from Moomoo. Writes `signals/paper_daily_status.json` and appends to `signals/paper_equity.csv`. No orders. |
| `--sync` | Reconcile submitted paper orders against Moomoo's order history. Checks what filled, what's still open. |
| `--execution-guard` | Check and repair core ETF (SPY/QQQ/TQQQ) stop-limit protection orders on Moomoo. |
| `--guard-dry-run` | With `--execution-guard`, show what stop changes would be made without actually submitting or cancelling. |
| `--replace-protection` | With `--execution-guard`, cancel and recreate matching ETF protection even if existing orders look fresh. |
| `--allow-closed-market-submit` | Allow submitting orders outside regular US market hours. Usually not recommended. |
| `--allow-stale-signal` | Allow submission even when signal or factor freshness checks fail. Usually not recommended. |
| `--no-sync-after-submit` | Skip the automatic order reconciliation that normally runs after a successful `--submit`. |
| `--lookback-days N` | How many days of Moomoo order history to scan during `--sync` (default: recent). |
| `--min-trade-value N` | Skip rebalance orders smaller than this dollar amount (default from settings). |
| `--limit-offset-bps N` | Limit-order price offset from last price, in basis points. Buys pay up, sells shade down. |
| `--max-signal-age-hours N` | Block submit if the signal CSV is older than this many hours. |
| `--max-factor-age-trading-days N` | Block submit if factor data is older than this many trading days. |
| `--etf-drift-threshold X` | Minimum ETF weight drift needed before trading that ETF. |
| `--overlay-drift-threshold X` | Minimum overlay stock weight drift needed before trading that stock. |
| `--equity N` | Override account equity for position sizing (default: use Moomoo account equity). |
| `--max-submit-gross-exposure X` | Cap submitted target gross exposure (default: `1.00` = no leverage). |
| `--allow-leverage-submit` | Allow submitting even if target gross exposure exceeds the max. |
| `--allow-submit-with-open-orders` | Allow a new submit even when current-signal orders are still open. Usually not recommended. |
| `--reset-paper-log` | Archive the old `signals/paper_trades.csv` before logging this submit attempt. |
| `--tickers T [T ...]` | Only trade these specific tickers. |
| `--exclude-tickers T [T ...]` | Exclude these tickers from trading. |

---

### `execution_guard.py` — Alpaca Safety Guard

Runs one cycle of the Alpaca execution safety guard: repairs ETF trailing stops,
cancels stale orders, and checks intraday P&L against the halt threshold.

```bash
python3 execution_guard.py --once              # one guard cycle
python3 execution_guard.py --once --dry-run    # show actions without executing
```

| Flag | What it does |
|------|-------------|
| `--once` | Run one guard cycle and exit (required — without it, nothing happens). |
| `--dry-run` | Log all actions (stop repairs, order cancellations) without actually doing them. Good for testing. |
| `--force-market-closed` | Allow halt liquidation even when Alpaca reports the market is closed. Normally, liquidation only fires during market hours. |

---

### `paper_gauntlet.py` — Moomoo Health Check

Checks whether the Moomoo paper trading account meets minimum standards for
real-capital deployment. No parameters — just run it:

```bash
python3 paper_gauntlet.py
```

---

### `alpaca_paper_gauntlet.py` — Alpaca Health Check

Same concept as the Moomoo gauntlet, but for the Alpaca account.

```bash
python3 alpaca_paper_gauntlet.py               # normal run
python3 alpaca_paper_gauntlet.py --verbose      # extra detail
python3 alpaca_paper_gauntlet.py --snapshot     # take equity snapshot
python3 alpaca_paper_gauntlet.py --json         # raw JSON output
```

| Flag | What it does |
|------|-------------|
| `--verbose` / `-v` | Show extra detail in the output (per-check breakdown). |
| `--snapshot` | Take an equity snapshot and append it to the tracking CSV. |
| `--json` | Output the raw JSON result instead of the formatted human-readable report. |

---

### `paper_health.py` — Deep Health Dashboard

Builds a comprehensive health summary: slippage, concentration, equity risk,
fill rates, drift breakdown, and go-live readiness scorecard.

```bash
python3 paper_health.py          # formatted output
python3 paper_health.py --json   # raw JSON
```

| Flag | What it does |
|------|-------------|
| `--json` | Print the health summary as JSON instead of the formatted report. |

---

### `daily_paper_check.py` — Quick Daily Verdict

One-command check that gives you a practical answer: is the paper account
aligned, waiting for fills, or drifted?

```bash
python3 daily_paper_check.py                              # full check
python3 daily_paper_check.py --after-fill                 # after expected fills
python3 daily_paper_check.py --skip-status --skip-sync    # read-only (used by daily_run.py)
python3 daily_paper_check.py --prune-snapshots --keep-days 30   # clean old logs
```

| Flag | What it does |
|------|-------------|
| `--after-fill` | After expected fills, sync and print a fill-focused verdict. |
| `--skip-status` | Don't refresh Moomoo status/equity (use cached data). |
| `--skip-sync` | Don't reconcile paper fills from Moomoo (use cached fill data). |
| `--lookback-days N` | Order-history lookback window for sync. |
| `--timeout N` | Max seconds per sub-command (default: reasonable). |
| `--json` | Print the final check result as JSON. |
| `--prune-snapshots` | Delete old snapshot folders from `logs/paper_snapshots/` and exit. |
| `--keep-days N` | With `--prune-snapshots`, how many days of snapshots to keep. |

Verdicts:
- `ALIGNED` — fills complete, account matches target
- `STILL_OPEN` — expected fills not complete yet
- `WAIT_FOR_OPEN_ORDERS` — orders exist but at least one still open/partial
- `NEEDS_REBALANCE` — fills complete but drift still high
- `CONTINUE_PAPER` — paper healthy but gauntlet not approved yet
- `PAPER_GAUNTLET_APPROVED` — all checks passed, ready for real capital

---

### `fill_monitor.py` — Order Fill Verification

Checks if recent orders filled successfully. Sends a macOS notification if
anything is cancelled or partial.

```bash
python3 fill_monitor.py              # check last 1 day
python3 fill_monitor.py --days 7     # check last 7 days
python3 fill_monitor.py --quiet      # only print if problems found
```

| Flag | What it does |
|------|-------------|
| `--days N` | How many days back to check (default: `1`). |
| `--quiet` | Only print output if there are problems (cancelled, partial, or missing fills). |

---

### `regime_monitor.py` — Regime Change Detection

Checks if the market regime changed since the last run and sends alerts.

```bash
python3 regime_monitor.py            # check and report
python3 regime_monitor.py --quiet    # only print if regime changed
```

| Flag | What it does |
|------|-------------|
| `--quiet` | Only print output if a regime change was detected. |

Regime types:
- **risk_on**: QQQ and SPY above their moving averages, low volatility. Heavy QQQ, more overlay stocks, TQQQ if approved.
- **neutral**: SPY above MA but QQQ below, or moderate conditions. Balanced SPY/QQQ mix, no TQQQ.
- **risk_off**: SPY below MA or high volatility. Defensive — more SPY, fewer overlay stocks, no TQQQ.

---

### `paper_report.py` — Side-by-Side Comparison

Compares Moomoo and Alpaca paper trading performance side by side.

```bash
python3 paper_report.py          # formatted report
python3 paper_report.py --json   # raw JSON
```

| Flag | What it does |
|------|-------------|
| `--json` | Output raw JSON instead of the formatted report. |

---

### Data Refresh Scripts

#### `refresh_etf_data.py`

Downloads and validates ETF parquet files used by the strategy.

```bash
python3 refresh_etf_data.py                          # validate only (no download)
python3 refresh_etf_data.py --refresh                # download + replace stale data
python3 refresh_etf_data.py --symbols SPY QQQ TQQQ   # validate specific ETFs
python3 refresh_etf_data.py --json                   # JSON output
```

| Flag | What it does |
|------|-------------|
| `--refresh` | Download and replace stale or missing ETF parquet files. Without this, only validates existing files. |
| `--symbols SYM [SYM ...]` | Only validate/refresh these specific ETF symbols. |
| `--json` | Print results as JSON. |

#### `research.py`

Builds the factor panel parquet files (stock prices + all factor scores).

```bash
python3 research.py                          # rebuild all tickers (default)
python3 research.py --ticker AAPL            # rebuild one ticker only
python3 research.py --tickers AAPL MSFT      # rebuild specific tickers
python3 research.py --incremental            # daily-safe refresh; tail-fills newly added columns
python3 research.py --incremental --backfill-new-columns  # slow historical backfill after feature changes
python3 research.py --xs-only                # only run cross-sectional ranking pass (skip parquet rebuild)
```

| Flag | What it does |
|------|-------------|
| `--ticker T` | Build parquet for one specific ticker. |
| `--tickers T [T ...]` | Build parquets for specific tickers. |
| `--all` | Build parquets for the full watchlist (this is the default). |
| `--incremental` | Refresh only the recent recompute window. If new feature columns were added, the scheduled-safe default fills those columns for the recent tail and leaves older rows blank until a research backfill is run. |
| `--backfill-new-columns` | With `--incremental`, force a full historical rebuild when new feature columns appear. Use this for walkforward/research datasets, not daily GitHub Actions. |
| `--xs-only` | Skip the per-ticker parquet rebuild; only run the cross-sectional rank post-pass on existing parquets. Faster when prices haven't changed. |

---

### Weekly Trust Check Scripts

#### `config_health.py`

Validates local config and dependency health.

```bash
python3 config_health.py                     # full check
python3 config_health.py --skip-pip-check    # skip slow pip dependency check
python3 config_health.py --json              # JSON output
```

#### `feature_quality_diagnostic.py`

Measures which factors actually predict returns and grades them A through F.

```bash
python3 feature_quality_diagnostic.py            # top 48 features
python3 feature_quality_diagnostic.py --top 24   # only top 24
```

| Flag | What it does |
|------|-------------|
| `--top N` | Number of top features to analyze. |

#### `feature_research.py`

Deeper situational analysis — sector-specificity, decay, regime-conditioning,
pairwise interactions.

```bash
python3 feature_research.py                # top 24 features (~5 min)
python3 feature_research.py --top 10       # only top 10 (~2 min)
python3 feature_research.py --skip-pairs   # skip pairwise analysis (~2 min)
```

| Flag | What it does |
|------|-------------|
| `--top N` | Number of top features to analyse (default: `24`). |
| `--pairs N` | Max features to include in pairwise interaction analysis (default: `15`). |
| `--skip-pairs` | Skip pairwise interaction analysis entirely (saves ~3 min). |

---

## Manual Step-by-Step (If Not Using daily_run.py)

### Before Market Open

```bash
# 1. Refresh data
python3 refresh_etf_data.py --refresh
python3 research.py

# 2. Check yesterday's fills
python3 fill_monitor.py --days 2

# 3. Generate signal (one unified signal for both brokers)
python3 core_satellite_alpha.py

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
python3 strict_leakage_audit.py
python3 core_satellite_execution_stress.py
python3 core_satellite_survivorship_audit.py
python3 core_satellite_drawdown_throttle.py
python3 factor_decay_monitor.py
python3 quant_audit.py
python3 paper_gauntlet.py
python3 alpaca_paper_gauntlet.py
```

### Walk-Forward Validation (Weekly or After Strategy Changes)

This is the **most important** trust check. It tells you the TRUE out-of-sample
performance of the strategy — what you'd have actually made running it live:

```bash
# Proper nested validation: inner loop tunes parameters (including TQQQ weight),
# outer yearly loop evaluates true unseen out-of-sample years.
python3 core_satellite_nested_walkforward.py

# Faster smoke/debug version:
python3 core_satellite_nested_walkforward.py --fast --max-folds 2 --max-configs 8 --output-prefix core_satellite_nested_walkforward_smoke

# Full dry run without publishing approval state:
python3 core_satellite_nested_walkforward.py --no-publish-live-config

# Optional one-off validation before generating that signal only:
python3 core_satellite_alpha.py --walkforward --ignore-stale
```

What to check in nested output (`signals/core_satellite_nested_walkforward.json`):
- **Mean OOS Sharpe** and **mean OOS alpha vs BLEND** are the headline numbers
- **Selection-bias gap** shows how much better the inner winner looked than the true outer result
- **Config stability** below 50% means the strategy is fragile across yearly retunes
- The winning config's `tqqq_weight` tells you if TQQQ helps — 0.0 means pure core-alpha won
- Full nested runs write approved or fail-closed live state to
  `signals/core_satellite_live_configs.json` by default. Smoke/debug runs
  (`--fast`, `--max-folds`, `--max-configs`, custom `--output-prefix`,
  partial year windows, or reduced `--max-specs`) do not publish unless
  `--publish-live-config` is passed explicitly.
  Daily signal scripts must load from this file, not from the full-sample grid.

If nested OOS Sharpe is below 0.5, **do not trust the in-sample grid results**.
The strategy needs fundamental rework, not more parameter tuning. The daily
signal script loads from `core_satellite_live_configs.json`.
If no approved config exists, the signal generator fails closed.

### Feature Quality Diagnostic (Weekly)

Measures which factors actually predict returns and which are noise:

```bash
python3 feature_quality_diagnostic.py --top 24
```

Output: `signals/feature_quality_report.json` and `signals/feature_quality_summary.csv`

What to check:
- Grade A/B features are your real edge — keep them
- Grade D/F features are noise — the alpha strategy auto-drops them on next run
- IC decay: features with half-life < 10 days may not survive transaction costs
- Turnover > 60%/period means the feature's rankings are too noisy to trade
- Regime stability: features that only work in bull markets are dangerous

The alpha strategy automatically loads this report and excludes D/F features
from scoring. Run the diagnostic first, then the strategy benefits on next run.

### Feature Research (Weekly)

Deeper situational analysis — WHERE, WHEN, and UNDER WHAT CONDITIONS each
feature works.  Complements `feature_quality_diagnostic.py` (which gives
A/B/C/D/F grades).  This script answers questions like "does ret_5d only
work in tech?" and "is rsi_14 losing its edge?":

```bash
python3 feature_research.py                # analyse top 24 features (~5 min)
python3 feature_research.py --top 10       # only top 10 (~2 min)
python3 feature_research.py --skip-pairs   # skip pairwise interactions (~2 min)
```

Output: `signals/feature_research_report.json` and `signals/feature_research_summary.csv`

What to check:
- **SECTOR-SPECIFIC**: features that only work in some sectors — consider
  sector-conditional weighting
- **DECAYING**: features losing predictive power recently — may need removal
- **HORIZON MISMATCH**: features used at the wrong holding period — IC is
  significantly higher at a different horizon
- **CONDITIONAL**: features whose IC depends on VIX, yield curve, or earnings
  proximity — may need regime-conditional usage
- **SYNERGISTIC PAIRS**: feature pairs that predict better together than alone —
  complementary signals worth combining

### Other Weekly Checks

What to check:
- `paper_health.py` should show `fill_rate=1.000`, low drift, reasonable slippage
- `quant_audit.py` should show `primary_strategy_ok=True`
- `fill_monitor.py` should show no cancelled or partial orders
- `regime_monitor.py` should log any recent regime transitions
- Survivorship stress should pass (may warn if alpha is survivor-sensitive)
- Execution stress should pass every scenario
- Factor decay must not show negative recent overlay alpha
- Real capital remains blocked until paper gauntlet passes on **both** brokers (Moomoo + Alpaca)

---

## Grid Search Details

The nested walkforward grid is deliberately coarse to prevent overfitting:
- 2 holding periods (10, 20 trading days)
- 3 overlay gross levels (25%, 50%, 70%)
- 1 regime preset (the proven winner: cashbuffer)
- 1 MA window (100 days)
- 2 high-vol modes (fixed threshold vs percentile-based)
- 1 score source (regime_adaptive)
- 4 portfolio shapes (top3 + top5 + top10 + top15)
- 2 weighting modes (sticky_score + risk_parity)
- 2 TQQQ risk-on weights (0%, 10%)
- 2 risk-control modes (off + defensive)

**Total: 384 configs per outer fold.** TQQQ is only held during `risk_on`
regime — 0% in neutral/risk_off. The grid search decides whether any TQQQ
allocation helps on a risk-adjusted basis.

For alpha-decay diagnosis, `--stable-grid` pins the repeatedly selected
dimensions (`h=20`, `overlay_gross=0.50`, `ma=100`, `score=regime_adaptive`,
`weighting=sticky_score`, `risk=defensive`) and tests only shape, high-vol
mode, and TQQQ weight. This is an A/B research baseline and will not publish
live approval state unless forced with `--publish-live-config`.

For post-2020 alpha research, `--recent-alpha-grid` pins `h=20`, `ma=100`,
`score=regime_adaptive`, and `risk=off`, then tests the dimensions the latest
run showed actually matter: 50/70% overlay, top3/top5/top15 concentration,
sticky-score vs risk-parity weighting, fixed vs percentile high-vol mode, and
0/10% TQQQ. This mode is also research-only by default and does not publish
unless forced.

Cost stress is an approval gate, not a selector: report base metrics at 1×
costs, then approve only if the same config passes 2×/3×/5× checks.

Fixed parameters (not searched — one correct answer):
- Earnings blackout: 5 days
- Exit rank floor: first option from config
- Adaptive exit mode: first option from config

The winner is selected by **stability-adjusted robustness score** (not CAGR):
mean inner-fold robustness minus 10% of score standard deviation. Only configs
where ≥60% of inner folds pass cost-stress survive to be scored, and configs
with mean inner-fold turnover above 400% are rejected before selection.

**Important**: If you see CAGR > 30% in any output, that's a warning sign of
overfitting. The walk-forward OOS results are the only numbers you should trust.

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

Do not move to real capital until **all** of these are true for **both** brokers:
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
| `signals/core_satellite_alpha_signal.csv` | Unified signal for both brokers (regime, weights, TQQQ, tickers) |
| `signals/core_satellite_live_configs.json` | Approved config from nested walkforward (daily signal reads this) |
| `signals/core_satellite_nested_walkforward.json` | Full nested walkforward results with per-fold OOS metrics |
| `signals/walkforward_checkpoint_core_alpha.json` | Checkpoint for resuming interrupted walkforward runs |
| `signals/paper_trades.csv` | Moomoo trade log with fill statuses |
| `signals/alpaca_paper_log.csv` | Alpaca trade log with fill statuses |
| `signals/paper_health.json` | Latest Moomoo health dashboard |
| `signals/paper_daily_status.json` | Latest Moomoo daily status snapshot |
| `signals/alpaca_paper_equity.csv` | Alpaca daily equity tracking |
| `signals/paper_equity.csv` | Moomoo daily equity tracking |
| `signals/regime_history.json` | Current/previous regime per broker |
| `signals/regime_changes_log.csv` | Log of all regime transitions |
| `signals/fill_monitor.json` | Latest fill verification report |
| `signals/moomoo_execution_guard_state.json` | Moomoo guard high-water mark state |
| `signals/guard_intraday_state.json` | Execution guard daily alert/debounce state |
| `logs/execution_guard.log` | Execution guard activity and safety actions |
| `logs/daily_run_YYYYMMDD.json` | Daily pipeline run log |
| `data/*.parquet` | Factor panel and ETF price data |
