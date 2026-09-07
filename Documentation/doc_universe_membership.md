
Malformed nonblank membership dates now raise an error. In particular, a typo
in `effective_to` cannot silently turn a removed stock into an indefinitely
active stock. Blank end dates retain their documented open-ended meaning.
Missing ticker identities are rejected. The corrected source gate additionally
requires independent coverage provenance and rejects overlapping intervals.

## What this script does

`universe_membership.py` checks dated stock eligibility and keeps each research
row only while its stock belonged to the historical universe. This prevents
**survivorship bias**: judging the past using only companies still around today.
A membership interval is the inclusive period from `effective_from` through
`effective_to`. **Provenance** means the recorded source of a fact.

## How to run

```sh
python universe_membership.py --status
python universe_membership.py --path data/universe_membership.csv
```

Input is a CSV with ticker, effective_from, effective_to, status, source,
source_url, retrieved_at, license and access_cost. The script prints a JSON
coverage report, including missing tickers and reasons it cannot certify the
universe. An incomplete report is a valid diagnostic output. Nothing is traded
or approved by this command. Corrected research adds stronger raw-price,
corporate-action and membership-provenance checks through corrected_data.py.
