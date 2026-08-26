# universe_membership.py

## What This Script Does

This helper prevents survivorship bias by describing when each ticker entered
and left the strategy universe. Historical rows can then be filtered using the
membership that was valid on that date.

Required CSV columns are `ticker`, `effective_from`, `effective_to`, `status`,
and `source`. Delisted names must remain in the table and retain local price
history where available.

## How To Use It

Create `data/universe_membership.csv` from a trustworthy historical membership
source. The validation bundle automatically reports whether coverage is
complete. Until it is complete, real-money approval remains blocked.

## Key Terms

- **Point in time:** Using only information known on the historical date.
- **Survivorship bias:** Testing only companies that survived until today.
- **Effective date:** The first or last date a ticker belongs to the universe.
- **Delisted:** A stock that stopped trading on its exchange.
