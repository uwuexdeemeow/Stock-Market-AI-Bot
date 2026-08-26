# Intraday Features

## What it does

`intraday_features.py` converts Alpaca minute bars into VWAP distance, morning
and last-hour volume share, daily range, and close-position features. Time
windows are converted to New York time so daylight-saving changes are handled.

## How to run it

Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, then run
`python3 intraday_features.py`. It prints features for sample tickers. Without
credentials it prints neutral defaults. It reads market data only and does not
submit orders.

## Key terms

- **VWAP:** average traded price weighted by volume.
- **Minute bar:** one minute of open, high, low, close, and volume.
- **Close position:** where the close sits between the day's low and high.
