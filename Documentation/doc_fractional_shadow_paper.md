# fractional_shadow_paper.py - $400 Fractional Shadow Account

## What this script does

This script tests whether the approved active signal can fit inside a small
account that starts with $400. It creates pretend fractional-share fills and
never connects to Alpaca or any other broker.

The normal $100,000 Alpaca paper account remains unchanged. For every run, the
script reads the active signal, verifies its paper gates, loads non-lookahead
prices, marks cash and positions, simulates fractional trades, and records
slippage and regulatory fees.

## How to run it

From the project folder:

```bash
python3 fractional_shadow_paper.py
```

To test another pretend amount, use separate output paths so the $400 history is
not reset or mixed with the new experiment:

```bash
python3 fractional_shadow_paper.py --initial-equity 500 \
  --state-path signals/fractional_500_state.json \
  --orders-path signals/fractional_500_orders.csv \
  --equity-path signals/fractional_500_equity.csv \
  --report-path signals/fractional_500_report.json
```

Expected outputs:

- `signals/fractional_shadow_state.json`: current pretend cash and positions.
- `signals/fractional_shadow_orders.csv`: append-only pretend fills.
- `signals/fractional_shadow_equity.csv`: one account row per price date.
- `signals/fractional_shadow_report.json`: allocation, costs, safety, and blockers.

Running the same signal twice is safe. Its fingerprint is recognized, so no
second set of pretend trades is created.

## GitHub workflow

The existing **Shadow Paper Journal** workflow runs this ledger at 9:55 AM New
York time on trading weekdays under the shared `signals-latest-publisher` lock.

For a manual run, open GitHub Actions, select **Shadow Paper Journal**, choose
**Run workflow**, leave `force=false` and `ignore_stale=false`, and keep the
fractional initial equity at `400`.

The workflow requires a maximum allocation gap of 2%, confirms the requested
starting capital, and proves that zero broker orders were submitted. Evidence
is stored on `signals/latest` and in the downloadable workflow artifact.

## Settings

- `FRACTIONAL_SHADOW_INITIAL_EQUITY` defaults to `400`.
- `FRACTIONAL_SHADOW_MIN_ORDER_NOTIONAL` defaults to `$1`.
- The cash cushion defaults to the greater of `$2` or `0.5%`.
- `FRACTIONAL_SHADOW_SLIPPAGE_BPS` defaults to `10` basis points.
- ETF and stock rebalance bands default to `1%`.
- SEC, FINRA TAF, and CAT fee rates are configurable environment variables.

## Key concepts

**Fractional share:** Less than one full share, such as 0.337 shares of QQQ.

**Notional:** The dollar value of an order.

**Slippage:** The difference between the reference price and simulated fill.

**Regulatory fee:** A small SEC, FINRA, or CAT trading charge, separate from
broker commission.

**Allocation gap:** The difference between target and achieved portfolio weight.

**Shadow-only:** Evidence is written locally; no order can reach a broker.

## Important limitations

- Broker fractionability is not yet checked live for each symbol.
- Fractional protective trailing stops are not implemented.
- Close-price simulation cannot promise a real fill.
- The active validation bundle still says real capital is not approved.
- Promotion needs a separate decision after the required evidence epoch.
