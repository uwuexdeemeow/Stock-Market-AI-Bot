# paper_policy.py

This module extracts basic order-planning rules into ordinary functions that
both paper planning and offline accounting can use. It does not contact a
broker. A **target weight** is the desired fraction of account value in an
asset. **Drift** is the difference between that target and today's actual
weight after prices move.

Import `PaperPolicy`, `whole_share_target`, `drift_requires_trade`,
`cash_reserve` or `rebalance_deltas` from Python. The corrected runner supplies
prices, holdings and account value and receives whole-share orders, with sells
first. Run `python -m pytest tests/test_corrected_audit.py -q` for examples.

Defaults preserve the paper rules: 3% ETF drift, 1% stock drift, $25 minimum
trade, 100% gross ceiling, 0.5% cash reserve, 8% stock trailing stop, 12%
drawdown halt, and recovery above half the halt threshold. Whole-share targets
use Python's existing `round` convention, not a newly introduced floor.
The frozen specification records these values. The live strategy is not
automatically changed by creating a different offline policy.
