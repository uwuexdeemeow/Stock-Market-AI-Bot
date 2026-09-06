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
