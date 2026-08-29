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
broker-side trailing stops on core ETFs and overlay stocks.
Before sending sell orders, it also checks Alpaca's broker-reported
sellable share count so the plan does not ask to sell shares that are
reserved by another open order or stop.

Paper submission also requires the reviewed `paper_version_lock.json` to match
the current trading files. The script refuses a non-paper Alpaca API endpoint,
skips all buys when account cash cannot be verified, and immediately attempts
to restore protective stops if a pre-submit cancellation sequence aborts.

The execution scorecard can reduce future BUY sizes only when its evidence is
complete, fresh, and `decision_eligible=true`. Missing, unreadable, stale, or
still-collecting scorecards leave size unchanged. Every planned-order row keeps
an explicit `execution_scorecard_reason` so the dashboard and audit log show
why no throttle was applied.

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

# Explicitly allow queueing when market is closed (not for routine runs)
python3 alpaca_paper_trading.py --submit --allow-closed-market-queue

# Dry-run — show planned orders without sending to Alpaca
python3 alpaca_paper_trading.py

# Refresh recent fill slippage/reversal stats for the Performance dashboard
python3 alpaca_paper_trading.py --slippage-report

# Emergency/manual run outside the normal morning window (still capped limits)
python3 alpaca_paper_trading.py --submit --allow-outside-execution-window

# Manual rollback: anchor limit prices to last trade instead of live bid/ask
python3 alpaca_paper_trading.py --submit --last-trade-limit
```

Reconcile checks broker-active statuses such as `pending`, `new`, `open`,
`partially_filled`, and prior `query_failed` rows. This means a partial fill
can later be corrected to `filled` once Alpaca reports the final state.

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
| `signals/alpaca_slippage_reversal_report.json` | Recent fills, slippage vs fill-minute VWAP, 5/15/30/60 minute reversals, and all/limit/market segment summaries |
| `signals/alpaca_halt_active.txt` | Created when drawdown halt fires |

The paper log also records both execution attempts: parent and child IDs,
quote time, bid/ask/midpoint, spread, limits, wait duration, cancel result,
remaining quantity, fill latency, and realized slippage. It also records
execution-planning diagnostics such as the
original requested quantity, Alpaca sellable quantity, and whether a sell
order was clamped to broker-available shares. When a bot-managed trailing stop
reserves shares, planning records `broker_reserved_stop_qty` and counts those
shares because that stop is cancelled immediately before the rebalance sell.
Normal open sell orders remain reserved and are never added back. For buys, it now also records
whether the share quantity was reduced to fit available cash, the original
quantity before that clamp, the reserved cash buffer, broker buying power used
for the cash check, and the cash-clamp reason.

`--status` is read-only for trading, but it still rewrites
`alpaca_paper_equity.csv` and `alpaca_daily_status.json` so the Streamlit
dashboard can show current Alpaca equity without waiting for reconcile.
It also refreshes the execution-quality report.

## Key concepts

- **Paper trading** — fake money but real prices and real fills.  Lets
  you test a strategy with real-time market behavior, no risk.
- **Spread guard** — refuses routine execution above 0.10% for ETFs or
  0.50% for overlay stocks. Wide spreads mean poor or unreliable prices.
- **Two-stage limit** — Stage 1 rests at the bid/ask midpoint for 15 seconds.
  If needed, the bot confirms cancellation and sends only the unfilled shares
  as Stage 2 at the current quote with a 1 bps ETF or 3 bps stock cap. A price-
  quality failure makes Stage 1 more patient and tightens the second cap; a
  fill-rate failure reprices sooner. Timestamped quotes older than five seconds
  are rejected. Neither case changes target quantity.
- **Bid/ask quote** — the best current buyer price (bid) and seller price
  (ask).  The script logs these into `alpaca_paper_log.csv` at submission
  time for audit.  Quote-based limit anchoring is now the default because the
  latest bid/ask is usually fresher than the last trade print.  Use
  `--last-trade-limit` or `ALPACA_LIMIT_REFERENCE=last` only when you need to
  roll back to older behavior.
- **Slippage** — the difference between the fill price and a fair reference
  price, here the fill-minute VWAP.  Positive slippage means the fill was
  worse than the reference.
- **Post-fill reversal** — price moving against the trade after the fill.
  For a buy, price dropping after the fill is adverse.  For a sell, price
  rising after the fill is adverse.
- **Execution segment** — a dashboard-only slice of the same fill report.
  It separates all orders, limit orders, market orders, and trailing stops so
  old market-order behavior does not hide newer limit-order behavior.
- **Eligible versus measured order** — an eligible order is old enough for a
  requested horizon; a measured order also has the required market bar. Each
  horizon reports its own coverage so missing or immature fills cannot dilute
  a bad-result rate.
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
   one day unless `--force` is passed.  It checks Alpaca's live order
   history first, then falls back to the local CSV log.
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
   The halt sentinel is written atomically, so a crash cannot leave a partial
   marker that confuses the next run.
6. **Two-stage capped limits** — normal rebalances never become market orders.
   Stage 1 seeks midpoint price improvement. Stage 2 crosses once with a small
   cap after cancellation is confirmed. Partial fills reprice only the remainder.
7. **Quote audit trail** — every submitted order records current bid, ask,
   midpoint, spread, stage, cancellation result, and which limit was used.
8. **Execution risk scoring** — the slippage report ranks tickers after at
   least five measured rebalance fills. The score changes execution reporting
   and price policy context, never the strategy-approved target quantity.
   The same report also includes all/limit/market summaries so you can compare
   execution quality before and after order-style changes. The portfolio-wide
   scorecard also separates protective-stop behavior from normal rebalances.
   It records whether evidence is `decision_eligible`, but poor execution now
   changes price-policy context rather than silently shrinking approved target
   quantities.
   A separate ATR/volatility sizing cap exists behind
   `ALPACA_LIVE_RISK_CAP_ENABLED=1`. It is paper-only, applies only to overlay
   stock buys, and can reduce but never increase the approved signal target.
   It stays explicitly off in the nightly workflow until its own out-of-sample
   and shadow-paper results prove an improvement. When testing it, configure
   `ALPACA_LIVE_RISK_PER_TRADE` (default 1%), `ALPACA_LIVE_RISK_ATR_MULT`
   (default 2), and `ALPACA_LIVE_RISK_VOL_TARGET` (default 8%). An enabled cap
   fails closed when its local OHLC history is missing or invalid.
   Experimental fixed ATR stops use a separate double gate:
   `ALPACA_OVERLAY_STOP_MODE=atr` and
   `ALPACA_ENABLE_EXPERIMENTAL_ATR_STOPS=1`. They calculate a fixed stop at
   entry minus `ALPACA_LIVE_RISK_ATR_MULT` times the latest 14-day ATR. They
   are refused outside Alpaca paper trading, and missing data fails closed.
   The nightly workflow pins the validated `trailing` mode and disables the
   ATR gate until out-of-sample and shadow-paper evidence justify a switch.
9. **Spread/quote guard** — skips and logs individual orders when spread is
   too wide or Alpaca cannot provide a quote (configurable via
   `MAX_SPREAD_PCT_*` and `ALPACA_REQUIRE_QUOTE_FOR_SUBMIT`).  Skipped rows
   stay in `alpaca_paper_log.csv`, but they do not count as submitted orders.
   Repeated identical spread-guard Telegram alerts are deduped for 20 hours by
   default (`SPREAD_GUARD_ALERT_TTL_HOURS`).
10. **Market-closed fail-closed behavior** — when run non-interactively at
   market close, it aborts instead of queueing orders.  Use
   `--allow-closed-market-queue` or `ALPACA_ALLOW_CLOSED_MARKET_QUEUE=1`
   only when queueing is intentional.
11. **Morning execution window** — routine submissions run only from 09:35 to
   10:30 New York time. Dry runs work anytime. Emergency overrides require
   `--allow-outside-execution-window`.
12. **Core ETF trailing stops** — broker-side stops on SPY/QQQ/TQQQ that
   stay active even if the local machine is offline.
13. **Overlay stop cleanup before sells** — before selling an overlay
   stock, the script cancels that stock's old trailing stop so Alpaca does
   not reject the sell because shares are already reserved.  After the
   rebalance or reconcile step, it recreates a fresh trailing stop for any
   remaining shares unless another sell order is still open.
   Order planning counts shares reserved by these cancellable bot stops, so
   required sells remain in the plan and happen before replacement buys.
14. **Alpaca-authoritative overlay stop repair** — during submit/reconcile,
   the script reads current Alpaca positions and repairs trailing stops for
   every non-ETF holding.  This catches missing or partial stops even when
   the local order log is incomplete.
15. **Sells-before-buys submit guard** — rebalance sells are submitted first.
   Buys are skipped if a sell fails, if a sell does not fill quickly, or if
   cash is still below the no-margin threshold.  If a buy is only too large
   because available cash is tight, the script shrinks the share quantity down
   to what cash can cover after reserving a small cash buffer (default 0.5% of
   equity).  The cash check uses the smaller of raw cash and Alpaca buying
   power, so pending order reservations cannot be double-spent.  If even one
   share would be too small or too expensive, the buy is skipped and logged.
   This avoids accidentally adding leverage when the sell side did not free cash.
15. **Buy-fill wait before overlay stops** — after accepted overlay buys, the
   script waits briefly for fills before trying to attach trailing stops.  This
   avoids the Alpaca rejection where a stop is sent before the shares exist.
   If a buy is still open, the stop is deferred and reconcile/repair handles it
   after fill.
16. **TQQQ pre-trade fail-closed check** — TQQQ buys are blocked if the fast
   drawdown check says unsafe or cannot fetch enough data.  This can be
   loosened with `TQQQ_FAST_DD_FAIL_CLOSED=0`, but the default is safer.
17. **Margin exposure warning** — after submit/reconcile, the script warns if
   stock market value is materially above account equity.

## Telegram order counts

The workflow summary now separates planned trades from Alpaca outcomes:

- **Planned orders** — rows generated in `core_satellite_alpha_orders.csv`.
- **Alpaca accepted** — planned orders that Alpaca accepted for routing.
- **Skipped** — planned orders intentionally not sent, such as cash-limited
  buys that cannot afford one share or orders blocked by a safety guard.
- **Filled** — accepted orders that filled.
- **Failed** — orders rejected or never accepted, such as insufficient
  available quantity.

## When to run

Typically:
- Once per trading day, ~5 minutes after market open (so spreads are tight)
- Via `daily_run.py --alpaca` which orchestrates the full pipeline
- Or via GitHub Actions cron at 9:35 AM ET on weekdays

You should not run it more than once a day — the duplicate-submission
guard catches that anyway, but it's wasted effort.

## Reliability Outcome

Every `--submit` run first writes a fail-closed
`signals/alpaca_submit_outcome.json`. The final status is `executed`,
`no_action`, `blocked`, or `failed`. Counts show every planned order's final
state. A CSV journal keeps one row per run, and deterministic Alpaca client
order IDs stop a retry from creating a duplicate order.
