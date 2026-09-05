# order_accounting.py

## What It Does

This module converts broker-log rows into logical rebalance orders. A passive
Stage 1 and capped Stage 2 are attempts at the same portfolio decision, not two
separate requested trades. It reports honest fill denominators, partial/open
counts, and duplicate attempt chains while excluding protective stops.

## How To Use It

The execution scorecard and paper epoch import `classify_logical_orders()` and
pass it a pandas table from `signals/alpaca_paper_log.csv`. The returned
dictionary contains logical observations and aggregate counts. This is a
library helper; it does not submit orders or write files.

## Key Terms

- **Logical order:** one requested portfolio change.
- **Child attempt:** Stage 1 (`-a1`) or Stage 2 (`-a2`) sent to the broker.
- **Complete fill rate:** fully filled logical orders divided by all accepted logical orders.
- **Duplicate chain:** more than one broker order occupying the same attempt slot.

## Partial fills and cash limits
This module is imported by the scorecard and validation tools; it does not
submit orders and has no standalone command. Run
`python -m pytest tests/test_execution_continuity.py -q` for a saved-data check.
The journal's final quantity, after cash clamping, is the completion target.
For separate children, requested_quantity must describe the whole parent.
Repeated snapshots of a child contribute only their largest cumulative fill.
A child marked filled cannot override a known shortfall in the whole order.
The output separates fully filled and partially filled logical orders.
