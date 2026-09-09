# Daily paper recovery and evidence handoff

The September 9 morning run had all 62 required securities and current prices, but a derived health profile failed a file-time check. That launched approximately 60 minutes of unnecessary research and missed the execution cutoff.

The repair binds the profile to its source content and refreshes only the derived profile when independent price and quality checks pass. The read-only health command also now suppresses implicit loader writes. Existing exposure, alignment and execution-window limits are unchanged.

| Check | Verified outcome |
| --- | --- |
| [Daily rerun](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34370140204) | Cache recovery passed in 2 seconds; signal generation passed; submission refused after 10:30 New York time. Overall run failed. |
| [Release CI](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34370121563) | All four checks passed. Offline regression: 713 passed, 45 skipped. |
| Configuration and version lock | Matching approval evidence and exact code fingerprint pass. |
| Published operational evidence | Complete matching run bundle, including the blocked outcome. |
| Account alignment | 0.026754 exceeds the existing 0.020000 limit. Retain incidents until an eligible normal-session rebalance and independently observed alignment pass. |
| Corrected historical research | Blocked by verified historical membership, prices, actions, context and settlement coverage. No historical period was shortened. |
| Corrected prospective validation | Not started. Separate prerequisites and explicit freeze still required; 252 new sessions and 20 independent matured cohorts remain mandatory. |

Next operational check is the existing 09:35 New York scheduled daily run. No outside-window override was used. Previous paper epochs remain archived; this operational release does not start the corrected prospective clock.

The [complete gap register](gap_register.md) retains all 92 original findings and historical failures. [Code verification](code_verification.json) binds each software closure to the final release and [sanitized test evidence](test_results.xml). Missing inputs remain blockers for dependent research, not excuses to suppress this report.
