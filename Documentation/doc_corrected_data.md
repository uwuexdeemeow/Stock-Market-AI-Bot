# corrected_data.py

This module imports attributed historical membership, verifies data provenance,
and builds a small raw feature panel. **Point-in-time membership** means a stock
was actually eligible on a particular date, including names later removed.
**Provenance** records where data came from and how it can be checked.

Run `python corrected_audit.py --audit-only` to write
`signals/corrected_audit/data_quality.json`. A blocked report is a useful result;
it does not certify the strategy. Import an independently assembled free-source
membership CSV with
`python corrected_audit.py --import-membership path/to/members.csv`.
The import validates attribution and date intervals but cannot declare the
whole historical universe complete. The subsequent full gate retains existing
400-active-name, removed-name and 95% price-file coverage requirements, and
also requires verified raw files for every relevant constituent and benchmark.

See `doc_corrected_audit.md` for the manifest and action schemas. Missing source,
ticker, checksum, action coverage or required price is a specific gap. Present-day
watchlist data and adjusted ETF caches cannot satisfy raw-accounting validation.
Candidate filtering follows effective membership dates, while prices after
removal remain available to liquidate existing holdings.

`build_raw_features` preserves raw OHLCV execution dollars. A separate causal
return index incorporates splits and ex-date distributions for momentum and
volatility features. It creates 20/60-observation momentum, 20-observation
volatility and dollar volume, plus forward outcomes with actual ticker-row
entry and endpoint dates. Missing future outcomes remain in the selection
panel. External features and dated VIX/sector/earnings context require free
source URLs and publication times no later than the actual exchange close,
including early-close days.

Verified symbol changes connect the real old/new ticker rows for feature and
label continuity while retaining their execution symbols. Conflicting same-date
rows for one renamed business raise an error instead of choosing a convenient
price. The ledger also transfers holdings, stop levels and prior-volume history.

Free verification leads include issuer investor-relations action notices,
[S&P index announcements](https://www.spglobal.com/spdji/en/media-center/news-announcements/)
and [SEC filing APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
These are source leads, not a claim that a complete freely verified historical
dataset has been recovered. Failed or delisted names still need evidence.

## Historical source hardening (September 2026)

A file hash proves which bytes were inspected, not that the file covers every
required trading day. The gate now opens each raw parquet and checks finite
OHLC prices, price ordering, nonnegative volume, unique ordered session dates,
and every required NYSE session. Missing sessions include their count and first
and last dates. Zero volume is allowed as an observation; it is not fabricated
liquidity. Execution rules still control whether a trade can occur.

The raw manifest also needs a `membership` object with `verified: true`, the
membership CSV's `sha256`, coverage `start`/`end`, and `source_url`, `retrieved_at`,
`license`, and `access_cost: "free"`. These fields describe independently checked
evidence, not a way to promote a community reconstruction by changing a flag.
Each raw symbol additionally records `feed` and `symbol_mapping`. Action coverage
has the same source attribution fields. Invalid manifests produce blocked
reports. Directly loaded membership intervals are checked for overlap too.

Actions must use a ledger-supported type, finite nonnegative values (positive
split ratios), actual dates, and nonblank identities/sources. Dividend `date`
means payment date and cannot precede `ex_date`. Missing payment dates cannot be
replaced with processing dates. Unsupported actions block dependent evaluation.

`validate_context_coverage` checks the actual candidate dates, including training
history. A generated `OTHER` sector placeholder is not historical sector evidence;
sector caps, earnings blackout and dynamic regimes require their dated fields.
No complete free historical source is claimed by these checks.

Raw metadata also requires `identity`: a provenance object with `verified: true`,
`security_id` (a stable security identifier) and the raw file's `sha256`, plus URL,
retrieval time, license and free access. This binds identity verification to the
exact prices. `asof=-` alone is insufficient: reused tickers can still combine
unrelated securities. Such files must be reconstructed against the correct
security history before an identity attestation is written.
