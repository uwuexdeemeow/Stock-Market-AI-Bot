# Evidence gap resolution — September 2026

**Code fixes delivered. External evidence remains blocked; all gaps are not closed.**

## Fixed code findings

- Raw parquet content and required-session validation instead of hash/file-existence alone
- Malformed and overlapping membership dates rejected, including ticker-alias overlap at import
- Independent membership coverage bound to the exact source CSV
- Source timestamps, URLs, free access, raw feed and symbol mapping validated
- Raw security identity evidence required to prevent recycled ticker histories passing silently
- Unsupported corporate actions, invalid values and impossible dividend dates rejected
- Complete candidate context required before generated sector defaults can hide missing history
- Market data follows all pagination tokens and rejects duplicate/repeated pages
- Received market-data pages preserved privately even when later recovery fails
- Dividend records with missing payment dates remain explicitly unverified
- No-activity replay scope and fill/fee counts included in evidence reports
- Deprecated SDK model-field access corrected

## Fresh recovery findings

- Updated first membership candidate: 815 historical symbols, cutoff 2026-08-18. It remains unverified; 876 of 889 comparable dates disagree with the second source.
- SPY and QQQ each return 2,684 raw bars, beginning 2016-01-04. The original 2012 start remains unsupported.
- Recovered 1,808 SIVB and 1,843 FRC pre-halt bars. Terminal values and full-period security histories remain unverified.
- SHLD and BBBY symbol-only queries include later history. Reused symbols need stable security identity evidence; turning automatic symbol mapping off is insufficient. [SHLD issuer history](https://www.globalxetfs.com/articles/introducing-shld-the-case-for-defense-tech); [BBBY issuer announcement](https://investors.beyond.com/news-events/press-releases/news-details/2025/Beyond-Inc--Changes-Name-to-Bed-Bath--Beyond-Inc--and-Reclaims-Ticker-Symbol-BBBY/default.aspx).
- Recovered 84 ETF cash dividends, with 30 missing payment dates. No payment dates were inferred from processing dates.
- Current paper replay reconciles, but remains a zero-activity balance-continuity interval. It does not prove trading performance.

## Remaining data blockers

There are now 28 explicit data blockers, versus 21 previously. The increase exposes missing provenance and identity checks; it is not a count of new trading failures. Missing membership still prevents exhaustive enumeration of stock-level requirements.

| Gap | Symbol | Required next action |
| --- | --- | --- |
| membership_table_missing | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| required_ticker_membership_missing | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| membership_provenance_columns_missing | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| historical_universe_too_small | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| inactive_membership_coverage_too_small | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| historical_price_coverage_incomplete | Universe | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| historical_membership_missing | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| membership_coverage_unverified | Universe | Resolve 876 candidate disagreements and duplicate snapshots with dated primary evidence and stable security identities. Cover all dates through the audit end and import with independent hash-bound coverage. |
| raw_price_file_missing | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| source_url_missing | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| retrieved_at_missing | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| license_missing | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| free_raw_source_unverified | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_provenance_invalid | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_feed_or_symbol_mapping_missing | QQQ | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_security_identity_unverified | QQQ | Bind the exact raw file to the correct historical security identifier; reject combined histories from reused tickers, including SHLD and BBBY. |
| corporate_action_coverage_missing | QQQ | Recover issuer-confirmed payment/ex-dates, splits, renames, mergers and delisting outcomes for every historical security; certify coverage only after corroboration. Thirty retrieved ETF dividends still lack payment dates. |
| raw_price_file_missing | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| source_url_missing | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| retrieved_at_missing | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| license_missing | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| free_raw_source_unverified | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_provenance_invalid | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_feed_or_symbol_mapping_missing | SPY | Recover all required 2012-onward sessions for every verified historical member and benchmarks, then record source URL, retrieval time, license, feed, symbol mapping and hashes. Current ETF probes start in 2016. |
| raw_price_security_identity_unverified | SPY | Bind the exact raw file to the correct historical security identifier; reject combined histories from reused tickers, including SHLD and BBBY. |
| corporate_action_coverage_missing | SPY | Recover issuer-confirmed payment/ex-dates, splits, renames, mergers and delisting outcomes for every historical security; certify coverage only after corroboration. Thirty retrieved ETF dividends still lack payment dates. |
| corporate_action_file_unverified | Universe | Recover issuer-confirmed payment/ex-dates, splits, renames, mergers and delisting outcomes for every historical security; certify coverage only after corroboration. Thirty retrieved ETF dividends still lack payment dates. |
| dated_context_missing_or_unverified | Universe | Recover historically published sector and future earnings schedule facts with publication/revision timestamps for every required candidate date. |

## Operational controls

Existing version-lock mismatches and approval/evidence conflicts remain blocked in the companion strategy report. The lock was not reset and no approval flags were edited. Resolving these requires matching validated evidence, not a manufactured pass. The imported September 4 artifact remains dated historical evidence, not the latest completed workflow.

## Verification

Full local suite: 684 passed, 37 skipped. Subsequent affected suite: 188 passed. Final recovery tests: 11 passed. All four remote CI jobs passed: https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34157271981.

No orders submitted, no strategy replacement, no freeze, no period shortening, and no threshold relaxation. Raw account and source responses remain private.
