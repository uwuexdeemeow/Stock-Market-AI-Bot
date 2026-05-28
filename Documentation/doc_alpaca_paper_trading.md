# alpaca_paper_trading.py — Alpaca Paper Broker

## What it does (plain English)

This script is the bridge between the strategy's daily signal and the
Alpaca paper-trading account.  Every day it:

1. Reads the latest signal (`core_satellite_alpha_signal.csv`) — which
   tickers to hold, at what weights.
2. Reads current Alpaca positions and equity.
3. Computes the *delta* — what needs to be bought or sold to match
   target weights.
4. Submits those orders to Alpaca (paper mode = no real money).
5. Writes back fresh state files so tomorrow's signal sees the truth:
   - `alpaca_paper_log.csv` — append-only order log with fill status
   - `alpaca_paper_equity.csv` — daily equity snapshots
   - `alpaca_daily_status.json` — current positions + equity (used by
     the signal generator for "sticky" overlay carry-forward)
   - `alpaca_slippage_reversal_report.json` — recent fill slippage and
     post-fill reversal stats for the dashboard

It also wires in safety nets — a portfolio drawdown halt, spread guards
(refuse to trade when bid-ask is too wide), per-ticker fail tracking,
protective day limit orders, market-closed queue blocking, and
broker-side trailing stops on core ETFs.

## Why it exists

Without this script the strategy's signals would just sit in a CSV.
This is the piece that *acts* on them.

## How to run

```bash
# Submit today's orders (the normal mode, runs after signal generation)
python3 alpaca_paper_trading.py --submit

# Reconcile yesterday's pending orders (check what filled vs cancelled)
python3 alpaca_paper_trading.py --reconcile

# Just show current account state, and refresh dashboard snapshot files
python3 alpaca_paper_trading.py --status

# Force submission even when market is closed (orders queue for next open)
python3 alpaca_paper_trading.py --submit --allow-closed-market-queue

# Dry-run — show planned orders without sending to Alpaca
python3 alpaca_paper_trading.py

# Refresh recent fill slippage/reversal stats for the Performance dashboard
python3 alpaca_paper_trading.py --slippage-report

# Emergency override: send market orders instead of protective day limits
python3 alpaca_paper_trading.py --submit --market-order

# Optional experiment: anchor limit prices to live bid/ask instead of last trade
python3 alpaca_paper_trading.py --submit --quote-limit
```

## Inputs

| File | Source | Purpose |
|------|--------|---------|
| `signals/core_satellite_alpha_signal.csv` | `core_satellite_alpha.py` | Target weights for today |
| `signals/core_satellite_live_configs.json` | nested walkforward / publisher | Approved config ID that the signal must match |
| `.env` | local | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |

## Outputs

| File | What's in it |
|------|--------------|
| `signals/alpaca_paper_log.csv` | Every order submitted, with fill status |
| `signals/alpaca_paper_equity.csv` | Daily equity snapshots |
| `signals/alpaca_daily_status.json` | Current positions, cash, equity |
| `signals/alpaca_slippage_reversal_report.json` | Recent fills, slippage vs fill-minute VWAP, and 5/15/30/60 minute reversals |
| `signals/alpaca_halt_active.txt` | Created when drawdown halt fires |

`--status` is read-only for trading, but it still rewrites
`alpaca_paper_equity.csv` and `alpaca_daily_status.json` so the Streamlit
dashboard can show current Alpaca equity without waiting for reconcile.
It also refreshes the execution-quality report.

## Key concepts

- **Paper trading** — fake money but real prices and real fills.  Lets
  you test a strategy with real-time market behavior, no risk.
- **Spread guard** — refuses to trade when the bid-ask spread is wider
  than 1.5%.  Wide spreads mean illiquid markets — you'd get filled at
  bad prices.  Common when market is closed.
- **Marketable limit order** — a limit order placed just above the latest
  price for buys or just below the latest price for sells.  It usually
  fills quickly like a market order, but it caps how far the fill can run
  away from the planned price.
- **Bid/ask quote** — the best current buyer price (bid) and seller price
  (ask).  The script logs these into `alpaca_paper_log.csv` at submission
  time for audit.  Quote-based limit anchoring exists behind
  `--quote-limit` or `ALPACA_LIMIT_REFERENCE=quote`, but the default remains
  last-trade anchoring unless explicitly enabled.
- **Slippage** — the difference between the fill price and a fair reference
  price, here the fill-minute VWAP.  Positive slippage means the fill was
  worse than the reference.
- **Post-fill reversal** — price moving against the trade after the fill.
  For a buy, price dropping after the fill is adverse.  For a sell, price
  rising after the fill is adverse.
- **Drawdown halt** — when account drops 12% from peak, stop submitting
  new buys (sells/stops still allowed).  Auto-clears when account
  recovers past 50% of halt threshold.
- **Sticky holdings** — the strategy carries forward yesterday's picks
  when they still rank well, to reduce churn.  The script reads
  `alpaca_daily_status.json` so it knows what was held.
- **Trailing stop** — broker-side sell order that follows the price up.
  If QQQ goes from $400 to $500, a 5% trailing stop sits at $475.  If
  $500 then drops to $475, it fires.

## Safety rules

The script has multiple layers of protection:

1. **Duplicate submission check** — refuses to run `--submit` twice in
   one day unless `--force` is passed.
2. **Live-config match check** — refuses to trade a signal if it was made
   from an older approved config than the one currently published.  This is
   not bypassed by `--allow-stale-signal`.
3. **Signal sanity check** — parses core ETF columns plus overlay JSON and
   blocks impossible gross exposure, missing weights, shorts, or over-sized
   overlay stock weights.
4. **Numeric order-planning guard** - refuses invalid/non-finite broker
   equity, drops invalid signal weights during gross-exposure scaling, and
   skips tickers with invalid/non-finite prices before converting weights
   into share orders.
5. **Account drawdown halt** — auto-liquidates if portfolio falls 12%
   from peak (configurable via `PORTFOLIO_DRAWDOWN_HALT_PCT`).
6. **Protective day limit orders** — normal submissions use small-cushion
   limit orders by default (`ALPACA_ORDER_TYPE=limit`).  Use
   `--market-order` only when you intentionally want market orders.
7. **Quote audit logging** — every submitted order records current bid,
   ask, midpoint, spread, and which limit reference was used.  This is
   observational by default and does not block or alter trades.
8. **Execution risk scoring** — the slippage report ranks tickers by recent
   bad slippage and post-fill reversal behavior.  This is dashboard-only;
   it does not change position selection or order submission.
9. **Spread guard** — skips individual orders when spread > 1.5%
   (configurable via `MAX_SPREAD_PCT`).
10. **Market-closed fail-closed behavior** — when run non-interactively at
   market close, it aborts instead of queueing orders.  Use
   `--allow-closed-market-queue` or `ALPACA_ALLOW_CLOSED_MARKET_QUEUE=1`
   only when queueing is intentional.
11. **Core ETF trailing stops** — broker-side stops on SPY/QQQ/TQQQ that
   stay active even if the local machine is offline.

## When to run

Typically:
- Once per trading day, ~5 minutes after market open (so spreads are tight)
- Via `daily_run.py --alpaca` which orchestrates the full pipeline
- Or via GitHub Actions cron at 9:35 AM ET on weekdays

You should not run it more than once a day — the duplicate-submission
guard catches that anyway, but it's wasted effort.
