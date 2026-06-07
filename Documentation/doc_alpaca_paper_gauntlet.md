# alpaca_paper_gauntlet.py — Go-Live Readiness Check

## What it does (plain English)

This script runs a "gauntlet" — a series of strict checks the strategy
must pass before it would be considered ready for **real capital**
(not just paper money).  Even though the bot is still in paper mode,
running the gauntlet daily gives you an early warning when paper
performance falls outside the bounds you'd require for live deployment.

Eight checks, in order:

1. **Trading days minimum** — at least 20 calendar trading days of
   paper history.
2. **Fill rate** — at least 95% of submitted orders actually filled.
3. **Cancel rate** — fewer than 5% of orders cancelled by the broker.
4. **Sharpe ratio** — annualized Sharpe above the minimum (default 0.5).
5. **Max drawdown** — worst drawdown not deeper than the threshold
   (default -15%).
6. **Signal age** — signal CSV is fresh (under 36 hours old).
7. **Portfolio drift** — actual positions match signal targets within
   tolerance.
8. **Alpaca connectivity** — broker API is reachable.

If any check fails, the gauntlet returns `status=failed`.

## Why it exists

Paper trading "feels safe" because it's not real money — but that means
problems are also easier to ignore.  The gauntlet is a forced
discipline: every day, ask yourself "if this were real money, would I
still trust this?"

## How to run

```bash
# Default — run the full gauntlet
python3 alpaca_paper_gauntlet.py

# Verbose — show extra detail
python3 alpaca_paper_gauntlet.py --verbose

# JSON output (for scripting)
python3 alpaca_paper_gauntlet.py --json

# Just take an equity snapshot (no gauntlet)
python3 alpaca_paper_gauntlet.py --snapshot
```

## Inputs

| File | What it reads |
|------|---------------|
| Alpaca API (live) | Account state, positions, recent orders |
| `signals/alpaca_paper_equity.csv` | Equity history |
| `signals/alpaca_paper_log.csv` | Submitted orders + fill status |
| `signals/core_satellite_alpha_signal.csv` | Target weights |

## Outputs

| File | What's in it |
|------|--------------|
| `logs/alpaca_paper_gauntlet_YYYYMMDD.json` | Full result (all checks + thresholds + reasons) |
| Telegram alert | Sent when status=failed |
| Exit code | 0 = passed OR within 30-day data-accumulation grace period.  1 = failed and accumulation period over. |

The equity CSV and dated JSON report are written atomically, so dashboard and
health checks never read half-written gauntlet data.

## Key concepts

- **Gauntlet** — a series of tests run sequentially.  "Running the
  gauntlet" means surviving all of them.
- **Fill rate** — fraction of submitted orders that actually executed.
  Low fill rate means orders are getting rejected or expiring.
- **Cancel rate** — fraction of orders the broker cancelled (vs orders
  you cancelled).  High cancel rate suggests price slipped too far,
  spread guard caught them, or some other broker rejection.
- **Portfolio drift** — how far current positions are from target
  weights.  Drift > 5% means rebalancing hasn't been keeping up.
- **Signal age** — how old the latest signal CSV is.  The age check
  compares timezone-aware UTC timestamps, so a UTC signal does not look
  stale just because the runner machine is in another timezone.  Older
  than 36 hours means the daily pipeline isn't running.

## When to run

- Daily via `daily_run.py --alpaca` (runs as the `alpaca_gauntlet`
  step).
- Manually whenever you want a snapshot of readiness.

## Recent behavior change (May 19, 2026)

- Now sends a Telegram alert on `status=failed`.  Previously was silent.
- Returns exit code 1 on failure (workflow-visible) **only after** 30
  trading days of data have accumulated.  In the first 30 days,
  failures are expected (insufficient history for Sharpe/drawdown) and
  the gauntlet exits 0 with an informational note.

## Threshold tuning

The thresholds are constants near the top of the file:

```python
MIN_TRADING_DAYS = 20
MIN_FILL_RATE = 0.95
MAX_CANCEL_RATE = 0.05
MIN_SHARPE = 0.5
MAX_DRAWDOWN_PCT = -15.0
MAX_SIGNAL_AGE_HOURS = 36
MAX_DRIFT = 0.10
```

These are tuned for paper trading.  Tighten before considering real
capital (e.g., MIN_SHARPE 1.0+, MAX_DRAWDOWN_PCT -10).
