# data_manifest.py

Each manifest records NYSE calendar/timezone details, duplicate calendar days,
adjusted-price columns, and possible corporate-action dates caused by a 20% or
larger move. Candidates are review evidence only: the checker never silently
rewrites a price or changes the active signal.

## What This Script Does

This helper records where each ticker parquet came from. The sidecar manifest
contains the provider, price-adjustment rule, date range, schema, checksum, and
quality warnings.

It also compares overlapping prices when an incremental refresh changes data
provider. A provider change is rejected when the median difference is above
0.5% or any overlap difference is above 2%.

## How It Runs

You normally do not run this file directly. `research.py` and
`refresh_etf_data.py` call it whenever they save a parquet.

Expected output:

```text
data/manifests/AAPL.json
data/manifests/SPY.json
```

## Key Terms

- **Parquet:** The compact file format holding historical market features.
- **Provider:** The service supplying prices, such as Yahoo Finance.
- **Adjusted price:** A price corrected for splits and dividends.
- **Overlap check:** Comparing the same dates from old and new providers.
- **Schema:** The names and data types of columns in a file.
