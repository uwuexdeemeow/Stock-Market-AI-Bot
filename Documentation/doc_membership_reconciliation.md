# membership_reconciliation.py

Builds a historical membership review dataset from two saved constituent lists
and manually reviewed primary-source facts. It is offline and never writes the
production membership table, raw manifest, trading settings or freeze.

A **boundary** is the date a company joins or leaves an index. An **interval** is
the whole period between boundaries. A verified boundary does not establish
uninterrupted membership between dates. A **CIK** identifies an SEC filer, not
necessarily a single security or share class. Reused stock symbols require
separate issuer and security histories.

## Run

```sh
python membership_reconciliation.py \
  --candidate data/audit_recovery/20260907T194247901786Z/fja05680_membership.csv \
  --comparison data/audit_recovery/20260907T194247901786Z/hanshof_membership.csv \
  --facts research_evidence/membership_primary_facts.json \
  --start 2012-01-01 --end 2026-09-08
```

The snapshot CSVs contain `date,tickers`. Tickers are comma-separated within each
quoted field. Facts contain source URLs, publisher, announcement date, effective
session and explicit manual review. The curated JSON records downloaded document
hashes when available and identifies sources reviewed through the web reader
when direct downloads are unavailable. No blank hash is presented as verified.
The runner checks structure and conflicts; it does not independently authenticate
manual source reviews or infer missing dates.

Outputs under `signals/corrected_audit/membership_review/`:

- `membership_review_queue.csv`: each candidate interval, matched boundary facts,
  and remaining work. Every full interval/security history remains unverified.
- `primary_membership_events.json`: reviewed facts with candidate match results.
- `membership_identity_report.json` and `.md`: source hashes, conflicting dates,
  priority symbols, issuer observations and blockers.

Conflicting same-date comparison snapshots are reported and excluded from
comparison, never arbitrarily deduplicated. Re-entry intervals stay separate.
The source cutoff is never extended. Date-only announcements do not establish
an intraday availability time for model features. An after-close index change
is recorded using the next trading session's membership convention.

The first collection contains 26 primary-source boundary facts. A Reynolds
American mismatch remains unresolved because the merger/trading cessation and
announced index deletion dates differ; this is not permission to alter raw prices
or to rewrite the source's membership history. Review evidence can be extended,
but full coverage certification remains with the existing corrected-data gates.
