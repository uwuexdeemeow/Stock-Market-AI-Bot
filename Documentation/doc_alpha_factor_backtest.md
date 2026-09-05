# alpha_factor_backtest.py

## What It Does
`alpha_factor_backtest.py` tests the raw factor ranking system before the model gets credit. It builds factor scores, simulates factor portfolios, compares them with benchmarks, and reports whether the overlay has enough recent edge to be trusted.

It also carries the feature-health gate into the factor research path. That means the backtest and live gate use the same diversification rules:

- At least 6 active feature clusters
- No single cluster above 25% of total factor weight

When `data/universe_membership.csv` passes completeness checks, the loader also
filters every ticker/date row to its effective membership interval. Incomplete
membership data is not partially applied; real-capital validation stays
blocked instead.

## How To Run It
Run the main factor backtest:

```bash
python3 alpha_factor_backtest.py
```

Common output files are written under `signals/`. Exact filenames can vary depending on the command options and current research setup.

## Key Concepts
- Factor score: A combined ranking score built from multiple market features.
- Cross-sectional ranking: Comparing stocks against each other on the same date.
- Trailing score guard: A leak-safe chooser that compares two rankings using
  past rank IC only, shifts the decision past the forward-return horizon, and
  can fall back when the normal score has weaker recent evidence.
- Overlay: The satellite stock-picking sleeve that sits on top of the core ETF exposure.
- Gate: A risk check that can block live capital even when the script still produces research output.
- Feature-health summary: Metadata explaining whether the current feature set is diverse enough for live use.

## September 2026 submission and historical-data repair

The factor-panel loader sorts each ticker's observations before shifting prices and rejects duplicate dates. For each forward-return column it also creates `<return-column>_entry_date`, `_end_date`, `_entry_price`, and `_exit_price`. These refer to exactly the same ticker rows used to calculate the return, including delayed entries. For example, `forward_return_20d_end_date` is the actual twentieth later observed date, which can differ from twenty ordinary weekdays.

Core/satellite callers use `load_factor_panel(specs, require_forward_returns=False)` so future-price availability cannot remove a stock before selection. The loader's default remains unchanged for other consumers. A forward return is a later price gain or loss; a label endpoint is the date when that outcome becomes known. Missing values remain missing until the consuming backtest validates its selected holdings.

Run `python -m pytest tests/test_submission_history_guards.py -q` to check the loader on synthetic parquet inputs. Expected output: passing date and missing-data tests, without market downloads.
