# universe_membership.py

## What This Script Does

This helper prevents survivorship bias by describing when each ticker entered
and left the strategy universe. Historical rows can then be filtered using the
membership that was valid on that date.

Required identity columns are `ticker`, `effective_from`, `effective_to`,
`status`, and `source`. Promotion-quality evidence must also include
`source_url`, `retrieved_at`, `license`, and `access_cost`. `source_url` must be
an HTTP(S) page, `retrieved_at` must be a valid timestamp, the license cannot
be blank, and `access_cost` must say `free`. Delisted names must remain in the
table and retain local price history where available.

The completeness gate also requires a realistically broad historical universe:
at least 400 constituents active at the backtest start, at least 17 removed
members across the period, and local prices for at least 95% of every listed
historical ticker. These defaults stop a current-only watchlist plus one failed
name from being mislabeled as point-in-time evidence.

## How To Use It

Create `data/universe_membership.csv` from a trustworthy historical membership
free, lawful, source-attributed source. Never reconstruct old membership from
today's constituents. The validation bundle automatically reports whether coverage is
complete. Until it is complete, real-money approval remains blocked.

Check it with `python3 universe_membership.py --status`. The JSON output shows
population counts, price-data coverage, missing tickers, and exact blocker
reasons. A community constituent list may help research, but it must not pass
unless its historical breadth and matching delisted price files meet the gate.

`alpha_factor_backtest.load_factor_panel()` now applies the table automatically
once it is complete. A partial table is never applied: research rows remain
unchanged and the validation bundle continues showing the blocker. This avoids
the more subtle bias caused by filtering only the few tickers entered so far.

## Key Terms

- **Point in time:** Using only information known on the historical date.
- **Survivorship bias:** Testing only companies that survived until today.
- **Effective date:** The first or last date a ticker belongs to the universe.
- **Delisted:** A stock that stopped trading on its exchange.
