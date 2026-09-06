# broker_history.py

This helper reads paginated broker history and reports execution measurement
gaps. It never submits an order. `collect_order_history(api, after=..., until=...)`
returns orders, a completeness flag and errors. The paper execution-quality
report calls it automatically. For offline examples run
`python -m pytest tests/test_corrected_audit.py -q`.

Requests overlap their oldest timestamp and deduplicate order IDs. Both requested
boundaries are included. Repeated pages, failed requests, absent timestamps,
safety limits and more orders at one timestamp than the API can distinguish
produce explicit incompleteness. A full interval can contain more than 100
orders. Incomplete retrieval cannot certify reconciliation.

**Arrival implementation shortfall** compares each fill with its parent's
recorded decision-time midpoint; adverse execution is positive for either side.
It is separate from fill-minute VWAP slippage. Historical arrival quotes are
never reconstructed. Distributions group by side, stage, spread, liquidity,
size and session, retaining unknown categories when source evidence is absent.

`parent_execution_summary` consumes cumulative child-order snapshots, including
zero fills and cancellations, and deduplicates children. Original requested
quantity comes from the decision journal, not a replacement's smaller quantity.
Unfilled shares are reported separately. Opportunity cost remains unknown
without a documented reference price, timestamp and horizon. These snapshots
are execution diagnostics; exact cash replay requires individual fill events
and fees, not rounded historical report averages.
