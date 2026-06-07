# factor_data_health.py - What It Does and How to Run It

## What This Script Does

This script checks whether the cached factor data is safe for live paper trading.
Factor data means the saved `data/<TICKER>.parquet` files that contain price,
volume, and model input columns.

It checks:
- every required ticker has a readable parquet file
- every required parquet has the core factor columns needed by live scoring
- the newest saved date is not too old
- feature-quality reports are newer than the factor files
- feature-health profiles exist, are current, and pass the diversification gate
- adaptive factor weights are usable, or a fallback is available

Freshness is counted using NYSE trading sessions, not simple weekdays. That
means weekends and market holidays do not make healthy data look stale.

## How to Run It

```bash
python factor_data_health.py
python factor_data_health.py --strict
python factor_data_health.py --ready-only
python factor_data_health.py --no-write
```

Expected output:
- terminal summary showing ready/not-ready status
- `signals/factor_data_health.json` unless `--no-write` is used

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Factor data | Saved ticker tables used by the model and strategy. |
| Trading session | A real NYSE market day. Weekends and holidays are skipped. |
| Strict mode | Fails the command if the system is not ready to trade. |
| Feature quality | A report showing whether model inputs are still healthy. |
| Feature health | A second gate that checks whether those inputs are diversified enough and not over-reliant on one crowded feature cluster. |
