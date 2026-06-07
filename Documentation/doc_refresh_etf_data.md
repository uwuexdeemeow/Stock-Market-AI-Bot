# refresh_etf_data.py - What It Does and How to Run It

## What This Script Does

This script checks and refreshes ETF parquet files used by the core-satellite
strategy.  These ETFs include SPY, QQQ, TQQQ, BIL, IEF, and GLD.

It validates that each ETF file:
- has enough rows
- has the required OHLCV columns: `Open`, `High`, `Low`, `Close`, `Volume`
- has a valid positive `Close` column
- is not flat in recent history
- is fresh enough for the latest completed NYSE session

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

Use `--strict` in automation. It exits non-zero if any ETF remains missing,
stale, partial, or otherwise unhealthy after validation.

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| ETF | A fund traded like a stock, such as SPY or QQQ. |
| Parquet | A fast table file format used for price and feature data. |
| Trading session | A real NYSE market day. |
| Force refresh | Download and replace the local ETF file even if it already looks healthy. |
