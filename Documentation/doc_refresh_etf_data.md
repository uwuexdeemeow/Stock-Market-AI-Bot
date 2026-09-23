# refresh_etf_data.py - What It Does and How to Run It

## What This Script Does

This script checks and refreshes ETF parquet files used by the core-satellite
strategy.  These ETFs include SPY, QQQ, TQQQ, BIL, IEF, and GLD.

It validates that each ETF file:
- has enough rows
- has the required OHLCV columns: `Open`, `High`, `Low`, `Close`, `Volume`
- has a valid positive `Close` column
- has no blank or non-finite closing prices, including in the newest row
- is not flat in recent history
- includes a valid close for the latest completed NYSE session during refresh

Freshness uses real NYSE trading sessions, so weekends and market holidays are
not counted as missing ETF data.

## How to Run It

```bash
python refresh_etf_data.py
python refresh_etf_data.py --refresh
python refresh_etf_data.py --refresh --force
python refresh_etf_data.py --refresh --force --strict
python refresh_etf_data.py --json
```

Expected output:
- a terminal summary for each ETF
- `logs/etf_data_health.json`
- updated `data/<ETF>.parquet` files when refresh succeeds

The health JSON and ETF parquet files are written atomically, so automation does
not read a half-written health report or corrupt ETF cache if the process stops
mid-write.

Use `--strict` in automation. It exits non-zero if any ETF remains missing,
stale, partial, or otherwise unhealthy after validation.

If one source returns a dated row with a blank close, the downloader tries
another adjusted-price source. If both Yahoo paths are incomplete, an optional
Alpaca IEX backup can replace only the latest completed day's missing bar.
The backup uses `adjustment=all` and must agree with the primary source on at
least three recent overlapping closes within 0.5% each. Its single-exchange
price is therefore a guarded fallback, not an exact consolidated close.
Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` for this backup; GitHub passes them
only to the ETF refresh step. If no trustworthy backup exists, the refresh
fails before research uses the bad file. It never guesses tomorrow's price or
silently treats a prior session as yesterday's close.

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| ETF | A fund traded like a stock, such as SPY or QQQ. |
| Parquet | A fast table file format used for price and feature data. |
| Trading session | A real NYSE market day. |
| Force refresh | Download and replace the local ETF file even if it already looks healthy. |

## Provider Safety

ETF prices use the adjusted-OHLCV contract. When a provider changes,
overlapping closes must agree within 0.5% at the median and 2% at the maximum.
An unexplained mismatch is rejected and each successful file gets a provenance
sidecar.
