# Historical-data audit — 7 September 2026

Status: **Blocked. Audit delivered; zero data gates cleared.**

The 21 current gaps are a lower bound. Missing historical membership hides most of the stock-level price and action requirements. Paper trading settings and prospective evidence requirements remain unchanged.

## Measured findings

| Check | Finding | Consequence |
| --- | --- | --- |
| Candidate stock universe | 812 historical symbols; 497 active at the 2012 boundary; 324 removed symbols | Broad enough to investigate, not verified; removal is not the same as delisting. |
| Candidate agreement | 876 disagreements across 889 common, unambiguous dates | Reconcile company identities and primary announcements before import. |
| Duplicate snapshots | Second source repeats 2022-06-21 and 2023-05-14 | Conflicting rows require source review, not arbitrary deduplication. |
| Source cutoff | First source ends 2026-06-30; second 2025-08-23 | Do not extend membership through the present by assumption. |
| SPY and QQQ raw probes | Each has 2,684 sessions, 2016-01-04 through 2026-09-04; 1,006 missing pre-2016 sessions | No missing sessions within observed span; full 2012 coverage still fails. |
| Bar sanity checks | No duplicate sessions, invalid OHLC bounds or negative volumes in the two saved probes | Sanity checks do not establish accuracy or raw provenance. |
| Corporate-action probe | 84 cash dividends; earliest ex-date 2016-03-18; 30 missing payment dates | Successful access, not complete ledger-ready history. |
| Failed-stock probe | SIVB returns seven sessions ending 2023-03-09 | Pre-halt availability demonstrated; terminal recovery and later pricing unresolved. |

Both candidate lists match the SMCI/DECK and WHR/ZION before/after spot check. That is one event, not certification. [Official effective-date announcement](https://press.spglobal.com/2024-03-01-Super-Micro-Computer-and-Deckers-Outdoor-Set-to-Join-S-P-500-Others-to-Join-S-P-100,-S-P-MidCap-400-and-S-P-SmallCap-600).

## Free-source research

Alpaca exposes raw adjustments, feed selection and symbol-mapping controls. Persist all three explicitly; default current symbol mapping can relabel old bars. [Historical bars documentation](https://docs.alpaca.markets/us/reference/stockbars).

Corporate-action query dates refer to processing dates, and availability can lag announcements. Exhausting pages does not prove full coverage or establish when a historical decision could know an event. [Corporate actions documentation](https://docs.alpaca.markets/us/reference/corporateactions-1).

SEC filings can support dated corporate facts. A later filing does not prove an earnings date was announced before a historical trading decision, and a current classification does not establish historical GICS sectors. [SEC public data APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

## Every existing blocker

| ID | Missing evidence | Symbol | Blocks | Next action |
| --- | --- | --- | --- | --- |
| HIST-01 | membership_table_missing | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-02 | required_ticker_membership_missing | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-03 | membership_provenance_columns_missing | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-04 | historical_universe_too_small | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-05 | inactive_membership_coverage_too_small | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-06 | historical_price_coverage_incomplete | Universe | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-07 | historical_membership_missing | Universe | Historical stock eligibility; all stock-overlay comparisons | Reconcile the 876 dated disagreements with original announcements and stable company identifiers; obtain a dated baseline, resolve duplicate snapshots, and cover the interval beyond 2026-06-30. Do not import either candidate as verified. |
| HIST-08 | raw_price_file_missing | QQQ | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-09 | source_url_missing | QQQ | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-10 | retrieved_at_missing | QQQ | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-11 | license_missing | QQQ | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-12 | free_raw_source_unverified | QQQ | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-13 | corporate_action_coverage_missing | QQQ | Cash/share accounting, total-return features and liquidation | Corroborate distributions against issuer records; recover payment and ex-dates, splits, symbol changes, mergers and delisting outcomes. Record per-symbol coverage and hash the normalized action file. An empty API interval cannot certify no actions. |
| HIST-14 | raw_price_file_missing | SPY | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-15 | source_url_missing | SPY | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-16 | retrieved_at_missing | SPY | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-17 | license_missing | SPY | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-18 | free_raw_source_unverified | SPY | Daily returns, momentum/volatility, stops, drawdowns and benchmarks | After membership is verified, enumerate every historical symbol and required session, including post-removal liquidation. Recover the 1,006 pre-2016 ETF sessions from an attributed free raw source; record raw adjustment, feed, symbol mapping, retrieval time, license and file hashes. No verified alternate source yet. |
| HIST-19 | corporate_action_coverage_missing | SPY | Cash/share accounting, total-return features and liquidation | Corroborate distributions against issuer records; recover payment and ex-dates, splits, symbol changes, mergers and delisting outcomes. Record per-symbol coverage and hash the normalized action file. An empty API interval cannot certify no actions. |
| HIST-20 | corporate_action_file_unverified | Universe | Cash/share accounting, total-return features and liquidation | Corroborate distributions against issuer records; recover payment and ex-dates, splits, symbol changes, mergers and delisting outcomes. Record per-symbol coverage and hash the normalized action file. An empty API interval cannot certify no actions. |
| HIST-21 | dated_context_missing_or_unverified | Universe | Sector caps and five-day earnings blackout; corrected candidate evaluation | Recover sector classifications and earnings schedules as known at each decision time, with original publication timestamps and revisions. SEC filing timestamps alone do not prove the future earnings schedule. Leave missing values blocked. |

## Recommended work order

1. Resolve the historical universe: inspect dated disagreements, map corporate identities and corroborate change events and baseline with original sources. Neither community list is approved.
2. Build the complete symbol/session inventory, then source raw prices, liquidation history and corporate actions. Resolve the 2012–2015 gap without changing the experiment period.
3. Reconstruct dated sectors and advance earnings schedules; preserve publication and revision timestamps.
4. Rerun existing source gates; only then evaluate the fixed corrected comparisons.

## Reproduction and limits

The JSON companion records source hashes, request parameters, counts, all 876 disagreements, and each blocker. Saved inputs are from data/audit_recovery/20260907T045716117822Z; new market-data probes are in data/audit_recovery/historical_audit_20260907. Raw files remain private. Compare shared dates after uppercase/dot-to-hyphen normalization, exclude ambiguous duplicate dates, and compare prices against the project NYSE session calendar.

This is a historical-data audit. It does not complete the separate scoring/approval or ablation implementation audits. No strategy change, order, threshold relaxation or freeze occurred.
