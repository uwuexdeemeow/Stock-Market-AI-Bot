# Options IV Provider

## What it does

`options_iv_provider.py` reads Tradier option-chain data and creates live implied
volatility features. These are snapshots for today's prediction and are not
backfilled into historical training without point-in-time data.

## How to run it

Set `TRADIER_API_TOKEN`, optionally set `TRADIER_USE_SANDBOX=1`, and run
`python3 options_iv_provider.py`. It prints sample ticker IV features. With no
token it prints neutral defaults. This reads quotes only; it does not trade.

## Key terms

- **Implied volatility (IV):** volatility implied by option prices.
- **Option chain:** available option contracts for a ticker.
- **Snapshot:** the current state rather than a historical series.
