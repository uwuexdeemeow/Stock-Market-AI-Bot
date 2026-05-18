# paper_health.py — Live Health Summary + Drift Detector

## What it does (plain English)

After each daily paper-trading run, this script builds a "health
report" — a single JSON file summarizing how the paper account is
performing.  It's the answer to "is the bot actually working?"

It computes:

1. **Slippage** — how much worse than expected the fills were.
2. **Drift vs walkforward** — is live performance tracking the
   backtest's expectations, or has the strategy drifted?
3. **Drawdown** — recent equity drop from peak.
4. **Concentration risk** — any single ticker or sector over-weighted?
5. **Stale orders** — orders that have been open too long.
6. **Big equity moves** — sanity check for daily P&L over a threshold.
7. **Go-live scorecard** — 11 individual readiness checks.

If anything is off, the script warns via Telegram (and writes the
detail to JSON for later inspection).

## Why it exists

The bot can be running "successfully" (no errors, orders submitted)
while still being broken in subtle ways:

- Filled at bad prices (slippage)
- Doing the right thing but the strategy itself has stopped working
  (drift)
- Concentrated in one stock by accident (sector exposure ran away)

This script catches those.

## How to run

```bash
# Default — evaluates Moomoo paper account (backward compatibility)
python3 paper_health.py

# Evaluate Alpaca paper account (most common now)
python3 paper_health.py --broker alpaca

# Print JSON instead of formatted report
python3 paper_health.py --broker alpaca --json
```

## Inputs (broker-dependent)

When `--broker alpaca`:

| File | What it reads |
|------|---------------|
| `signals/alpaca_paper_log.csv` | Order log with fill status |
| `signals/alpaca_paper_equity.csv` | Daily equity history |
| `signals/alpaca_daily_status.json` | Current positions/equity |
| `signals/core_satellite_nested_walkforward.json` | Backtest expectations to compare against |

When `--broker moomoo` (default):

| File | What it reads |
|------|---------------|
| `signals/paper_trades.csv` | Order log |
| `signals/paper_equity.csv` | Equity history |
| `signals/paper_daily_status.json` | Current state |

## Outputs

| File | What's in it |
|------|--------------|
| `signals/{broker}_paper_health.json` | Full health summary (alpaca or moomoo) |
| `logs/{broker}_paper_health_YYYYMMDD.json` | Dated snapshot |
| Telegram alert | Sent when warnings fire (drift, drawdown, concentration) |

## Drift detection — the headline feature

Drift detection compares **live performance** against the **walkforward
expectations** baked into the approved live config.  Specifically:

- **Sharpe ratio** — if live Sharpe < 30% of backtest mean → warn
- **CAGR** — if live underperforms by > 20pp → warn
- **Drawdown** — if live worst > 1.5x backtest worst → warn

These thresholds are env vars:
```
PAPER_HEALTH_DRIFT_SHARPE_WARN_FRAC=0.3
PAPER_HEALTH_DRIFT_CAGR_UNDERPERFORM_PCT=20
PAPER_HEALTH_DRIFT_DD_WARN_MULT=1.5
PAPER_HEALTH_DRIFT_MIN_LIVE_DAYS=10
```

Drift is only checked once at least 10 live days exist — before that
the comparison is meaningless noise.

## Key concepts

- **Slippage** — difference between expected fill price (signal time)
  and actual fill price (broker execution).  Measured in basis points
  (1 bp = 0.01%).
- **Drawdown** — drop from the highest equity ever seen.  Computed as
  `(peak - current) / peak`.
- **Drift** — when live behavior diverges from backtest expectations.
  May indicate the market regime changed, the strategy decayed, or
  there's a bug.
- **Concentration risk** — having too much money in one ticker or one
  sector.  Thresholds default to 35% per ticker, 50% per sector.
- **Scorecard** — 11 individual go-live readiness checks.  Each passes
  or fails independently; the overall verdict is "ready" only when all
  11 pass.

## When to run

- Automatically every day via `daily_run.py` (recent change — Alpaca
  step added).
- Manually whenever you want a snapshot of how live trading is going.

## What does a passing health report look like?

- All gauntlet gates passed
- Drift detector: still gathering data (< 10 days) OR within thresholds
- Drawdown: less than the warning threshold
- No stale open orders
- No single position over 35%, no sector over 50%
- Slippage average under 10 bps
