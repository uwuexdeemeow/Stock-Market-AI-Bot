# Edge ablation

An ablation removes one ingredient to see how the result changes. This script
runs seven fixed offline comparisons of the unapproved corrected shadow candidate.
It never changes the active strategy or creates a prospective freeze.

```bash
python3 corrected_audit.py --ablations --spec corrected_shadow_spec.json
```

A unique folder under `signals/corrected_audit/ablations/` contains comparison
JSON, CSV and Markdown, plus an append-only trial journal. Missing verified input
records all seven attempts as blocked and lists required data. No old backtest
cache supplies replacement numbers. A successful run saves individual events,
holdings, daily equity, SPY/QQQ benchmark equity/events, metrics and feature
artifacts for every trial.

The variants are full candidate, positive 60-session momentum ranking, removal
of momentum, removal of volatility, removal of dollar-volume, fixed neutral
allocation, and removal of the stock overlay. Removed stock weights become cash
*after* deployment scaling, so ETF allocations are not silently increased.
All learned variants refit inside each inner and outer training window. The
simple momentum baseline has a fixed positive direction and learns nothing from
future returns. The full variant uses the same fitter as the normal corrected run.

The protocol accepts exactly the existing four raw features and one configuration,
with the existing fold windows. There is no parameter search. Inner diagnostics
cannot change outer settings. Each outer artifact is reused unchanged at 1×, 2×,
3× and 5× costs. Baseline policy and deployment gross must match. Each fold starts
in cash and includes terminal liquidation; combined returns reinvest sequentially
across disjoint folds. These are not a continuous live account replay.

Benchmarks passively hold SPY or QQQ with the same maximum gross, cash reserve and
execution-cost assumptions; strategy-specific stops and drawdown halts are disabled
for these passive references. **Excess return** is the difference in compounded
returns, not regression alpha. **Sharpe** measures daily return relative to its
variation. **Drawdown** measures loss from a previous daily peak. **Turnover** is
the total weight traded, including initial and terminal ETF/stock transactions.

Paired 20-session bootstrap blocks compare the same dates; fewer than 400 paired
sessions remain inconclusive. Intervals measure annualized mean return differences,
not CAGR differences. Positive full-minus-removed intervals suggest that ingredient
helped in this inspected history; negative intervals suggest harm. Intervals are
retrospective and unadjusted for multiple comparisons, not profitability proof or
promotion authorization. The existing prospective 252-session and 20 matured
independent-cohort requirements remain unchanged.

Every variant outcome also uses the shared `append_experiment` interface with a
run-local output directory. Outer-fold and combined cost-stress comparisons are
saved separately; the stress results never choose the configuration. Per-fold
full-minus-variant comparisons remain inconclusive when history is too short.

## Evidence closure update

Every attempt now includes source commit, dirty-checkout status, schema version and generation time. This distinguishes regenerated blocked attempts from older candidate/code identities. Run python corrected_audit.py --ablations; all seven variants remain blocked when verified inputs are absent.
