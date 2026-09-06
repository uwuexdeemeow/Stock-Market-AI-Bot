# edge_evidence.py

This module measures portfolio performance against an exposure-matched SPY/QQQ
alternative. Run it through the corrected runner; its outputs appear in each
trial's `metrics.json` and prospective `edge_monitor.json`. Run
`python -m pytest tests/test_corrected_audit.py -q` for synthetic examples.

**Net return** compounds daily portfolio returns after costs. **Benchmark excess
return** subtracts the alternative's compounded return. Positive raw return can
still be negative excess return. **Regression alpha** is a separate fitted
intercept in `daily_net_return = intercept + beta * benchmark_return`, with a
zero cash-rate assumption. Its annualized value is 252 times the daily
intercept, not a guaranteed annual return.

Paired moving-block bootstrap intervals resample matching strategy/benchmark
days together, using blocks at least as long as the actual label horizon.
This preserves some dependence between neighboring observations. Non-overlapping
rank-IC cohorts must fully mature; an incomplete cohort is excluded whole.
At least 20 cohorts and positive lower 95% bounds for both excess return and
rank IC are required for statistical health. Regression uncertainty is reported
separately. Small or inconclusive samples remain advisory. These intervals do
not undo historical experiment selection or establish prospective success.

The SPY/QQQ ledger follows beginning-of-session exposure and cash conventions,
uses the same costs and terminal-sale convention, and reports residual exposure
differences from whole shares and reserves. It does not run its own independent
drawdown halt. Operational trading safeguards are separate from statistical
edge health.
