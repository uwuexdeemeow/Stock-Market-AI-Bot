# data_validation.py - What It Does and How to Run It

## What This Script Does

This script contains safety checks for price and feature tables before the
rest of the system trusts them.

It checks price data for:
- required columns like `Open`, `High`, `Low`, `Close`, and `Volume`
- sorted datetime index with no duplicate rows
- positive prices and non-negative volume
- suspicious one-day moves
- freshness versus the latest completed NYSE session

## How to Run It

This file is usually imported by other scripts.  For a quick check:

```bash
python -m py_compile data_validation.py
python -m pytest tests/test_sanity.py -q
```

Expected output:
- no output when imported successfully
- `ValueError` if a bad table is passed into a validator

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Price frame | A table of daily OHLCV market data. |
| OHLCV | Open, High, Low, Close, and Volume. |
| Feature frame | A table of model input columns. |
| Freshness | Whether the latest row is recent enough to trust. |
