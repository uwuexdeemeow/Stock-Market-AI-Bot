# core_satellite_survivorship_audit.py - Survivorship Stress Check

## What It Does

`core_satellite_survivorship_audit.py` tests whether the selected core-satellite
strategy still behaves well when locally available failed or delisted tickers are
added back into the research universe.

Plain language: it checks whether the strategy only looked good because it
mostly tested companies that survived.

## How To Run It

```bash
python core_satellite_survivorship_audit.py
```

Inputs:

- Current approved core-satellite config
- Factor data in `data/`
- Failed-name audit profiles from `survivorship_audit.py`

Expected outputs:

- `signals/core_satellite_survivorship_audit.csv`
- `logs/core_satellite_survivorship_audit.json`

Both outputs are written atomically, so approval gates never read a half-written
stress report.

## Key Concepts

- Survivorship bias: backtests can look better if failed companies are missing.
- Audit ticker: a failed or delisted ticker available for stress testing.
- Stressed universe: normal watchlist plus available audit tickers.
- Alpha: return above a benchmark such as QQQ or a blended benchmark.

The report carries strategy and dataset fingerprints. Testing remains partial
until date-effective membership and delisted-name data are complete, so it
cannot authorize real capital.
The fingerprint includes volatility and risk-control modes, preventing results
for one strategy variant from being attached to another.
