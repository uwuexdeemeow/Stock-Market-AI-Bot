# execution_guard.py

## What It Does

`execution_guard.py` is the Alpaca safety checker for the paper trading account.
It is built for a laptop workflow where the machine may be asleep for part of
the US trading day.

It has three jobs:

- Repair broker-side trailing stops for core ETFs (`SPY`, `QQQ`, `TQQQ`). These
  orders live at Alpaca, so they remain in place when the laptop is offline.
- Cancel stale open orders that are not trailing stops.
- Monitor intraday account P&L while the laptop is awake. If the loss reaches
  the halt threshold during market hours, it calls the existing Alpaca emergency
  liquidation function.
- Validate broker equity numbers before P&L math. NaN/inf or unreadable equity
  values trigger an alert and skip that P&L cycle instead of writing bad state.

## How To Run

Run one check:

```bash
python3 execution_guard.py --once
```

Preview actions without cancelling or trading:

```bash
python3 execution_guard.py --once --dry-run
```

Run continuously while the laptop is awake:

```bash
python3 execution_guard.py
```

Allow the emergency halt to submit liquidation orders even when Alpaca reports
the market closed:

```bash
python3 execution_guard.py --once --force-market-closed
```

Use `--force-market-closed` carefully. Without it, the guard can still repair
protective stops and cancel stale orders, but it will not send market
liquidation orders while the market is closed.

## Inputs

The script uses the Alpaca paper trading API credentials:

```bash
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Optional guard settings:

```bash
GUARD_CORE_STOP=1
GUARD_CORE_TICKERS=SPY,QQQ,TQQQ
GUARD_CORE_TRAIL_PCT=0.05
GUARD_TQQQ_TRAIL_PCT=0.10
GUARD_STALE_CANCEL=1
GUARD_STALE_MINUTES=90
GUARD_PNL_MONITOR=1
GUARD_PNL_WARN_PCT=-3.0
GUARD_PNL_HALT_PCT=-8.0
GUARD_INTERVAL_SEC=300
GUARD_ALERTS=1
```

## Outputs

The guard writes:

- `logs/execution_guard.log` - timestamped guard activity
- `signals/guard_intraday_state.json` - daily alert/debounce state
- Alpaca trailing stop orders for held core ETFs

It may also send macOS notifications and SMTP email alerts if those settings are
configured.

## How It Fits The Daily Flow

`daily_run.py --alpaca` now runs:

```text
alpaca_submit
alpaca_reconcile
execution_guard.py --once
alpaca_gauntlet
```

`alpaca_paper_trading.py --submit` also clears old core ETF protective stops
before a core ETF rebalance and repairs them afterward. This prevents old sell
stops from reserving shares and blocking rebalance sell orders.

## Key Concepts

- **Broker-side trailing stop**: A stop order stored at Alpaca. It follows the
  price upward and sells if the ETF falls by the configured trail percentage.
- **Local guard**: Code running on the laptop. It only works while the laptop is
  awake and connected.
- **Stale order**: A normal open order that has been waiting longer than the
  configured threshold. Trailing stops are skipped because they are meant to
  stay open.
- **P&L baseline**: The guard prefers Alpaca `last_equity`, which is the prior
  close equity, instead of treating the laptop startup time as the market open.
- **Invalid equity**: A missing, NaN, infinite, or non-positive broker equity
  value. The guard alerts and refuses to calculate P&L from it.
- **Halt sentinel**: Emergency liquidation uses
  `signals/alpaca_halt_active.txt` to avoid firing repeatedly. Remove it only
  after reviewing the halt.

## Failure Modes

- If the laptop is off, only broker-side Alpaca orders remain active.
- Trailing stops do not remove gap risk. A large overnight or premarket gap can
  execute worse than expected after regular market hours begin.
- If Alpaca rejects a protective stop, the guard logs and alerts the failure,
  but it cannot force the broker to accept the order.
- If SMTP is not configured, email alerts are skipped; logs are still written.
