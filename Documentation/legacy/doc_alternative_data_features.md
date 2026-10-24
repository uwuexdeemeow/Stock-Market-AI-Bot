> **Legacy (moved 2026-10-25):** this script now lives in `legacy/` and needs
> `pip install -r requirements-legacy.txt`. Live trading does not use it.
> See `Documentation/LEGACY_ML_CLEANUP_PLAN.md`.

# Alternative Data Features

## What it does

`alternative_data_features.py` builds live research features from analyst EPS
revisions, recommendations, short interest, institutional ownership, and
insider activity. Snapshot-only fields are kept out of historical training to
avoid pretending today's information existed in the past.

## How to run it

Run `python3 legacy/alternative_data_features.py`. Optional Finnhub data needs
`FINNHUB_API_KEY`; yfinance calls need internet access. The self-test prints an
AAPL feature dictionary. Missing providers return documented neutral defaults.

## Key terms

- **EPS revision:** an analyst raises or lowers an earnings estimate.
- **Snapshot data:** current information without a trustworthy past timeline.
- **Feature:** one numeric input supplied to a model.
