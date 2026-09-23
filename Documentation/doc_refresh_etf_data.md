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
another adjusted-price source. If none provides a complete bar, the refresh
fails before research can use the bad file; it never guesses tomorrow's price
or silently treats a prior session as yesterday's close.

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
