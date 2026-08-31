# Reliability-First Project Workflow

## Canonical daily evidence

All daily paper outputs share one run ID. Broker truth, execution quality,
validation progress, monitor continuity, and incidents are consolidated into
`signals/alpaca_paper_health.json`. Only a complete same-run bundle receives a
`signals/paper_run_manifest.json` status of `complete`; only that bundle may
replace the `signals/latest` branch.

The workflow stays paper-only. Alignment recovery and execution calibration
are review-only, and no readiness result can approve real capital.

## End-To-End Flow

1. `research.py` refreshes ticker data and writes a provider/checksum manifest
   for every parquet.
2. Research and model scripts create possible signals. The factor strategy
   remains champion; ML and sentiment stay shadow challengers.
3. `core_satellite_nested_walkforward.py` tests past-to-future folds, records
   relaxed fallbacks, compares selection with the frozen baseline, and measures
   turnover and cost separately.
4. Execution-stress, factor-decay, and survivorship reports carry matching
   strategy and dataset fingerprints.
5. `validation_bundle.py` combines config, Git commit, dataset, folds, analyzer,
   robustness reports, and approval into one checksummed source of truth.
6. `core_satellite_alpha.py` verifies the bundle and creates paper signals.
7. `daily_run.py` checks data and broker health, then executes each normal
   rebalance with a passive midpoint limit and one capped replacement for the
   confirmed remainder. It keeps sells first, fits buys to cash, restores
   stops, and always reconciles broker truth.
8. `paper_validation_epoch.py` measures a clean operational period. Real-money
   trading remains disabled until a separate explicit human review.

The active August 26 epoch is protected by `paper_version_lock.json`. The lock
fingerprints every file that can alter strategy selection, sizing, submission,
protection, or execution grading. Alpaca paper submission fails closed if a
locked file changes; freezing a reviewed version never changes the epoch's
original start time.

## Beginner Use

Preview the daily route without orders:

```bash
python3 daily_run.py --dry-run --alpaca --skip-refresh --skip-factor-refresh
```

Check the clean paper epoch with `python3 paper_validation_epoch.py --status`.
Scheduled paper runs continue in GitHub Actions and Telegram reports their
classified execution outcome.

For the next walk-forward, commit and push, run
`python3 prepare_colab_walkforward.py`, upload both files to the
`StockBotWalkforward` Drive folder, open the Colab notebook, and run its cells.
Colab uses CPU workers, saves resumable checkpoints, and never gets broker keys.

## Why It Is Designed This Way

Operational truth comes before prediction complexity. A profitable backtest is
not enough when data sources can be mixed, selection does not predict future
alpha, or planned orders can disappear without a final state. Paper trading
stays active to collect execution evidence while real-capital approval stays
false. Execution measurement grades normal rebalances separately from safety
stops, uses only observations that actually have price data, and treats later
market direction as timing advice instead of pretending it is fill quality.

# Post-market execution evidence

The morning paper workflow can finish before 15-minute and 60-minute price
observations exist. A separate GitHub workflow runs at 5:15 PM New York time.
It refreshes account status and execution reports without submitting,
reconciling, cancelling, replacing, or creating orders. It shares the
`signals-latest-publisher` lock with the daily and shadow workflows, then saves
the mature scorecard for the next trading session.

## Independent Quant Performance Audit

`quant_performance_audit.py` is the reference check for historical performance.
It starts with target positions emitted by the strategy, but it does not trust
the strategy's periodic equity curve. It reloads each stock and ETF's raw Open
and Close history, enters after the signal at the next executable Open, marks
positions and cash every day, charges turnover across both ETFs and stocks,
and stitches outer OOS years into one continuous curve.

The JSON report shows gross and transaction-cost net return and alpha versus
QQQ, exact overlap, elapsed-time CAGR, daily Sharpe and information ratio,
daily drawdown, Newey-West alpha evidence, calendar-year bootstrap uncertainty,
and a quantified reconciliation to the old periodic headline. The old 2,701%
headline stays visible only as provisional context; it is not replaced until
the independent reference blockers are cleared.

The same script runs a bounded shadow comparison around the frozen active
configuration: top five, 40% overlay, and stronger sticky weighting. It records
each attempt in the experiment ledger and creates a combined candidate only if
an isolated change passes every predeclared gate. It never writes the active
configuration and never calls Alpaca.

### Small-capital fractional shadow

The 9:55 AM New York Shadow Paper Journal workflow also runs a separate $400
fractional ledger from the restored active signal. It simulates market/day
fractional fills, cash, slippage, and regulatory fees without importing a broker
client. Daily and shadow workflows preserve its state together on
`signals/latest`, so one workflow cannot erase the small-account history.

Shadow restores only caches published by Factor Data Refresh. It no longer
rebuilds research data inside the journal job. A missing or unhealthy factor
cache fails quickly with a visible annotation, while a valid first observation
is labeled `collecting` instead of failing the workflow.

Post-market execution evidence is copied outside the checkout before the
workflow changes to the `signals/latest` publishing branch. Temporary runner
changes are stashed first, preventing a safe read-only audit from failing while
preserving every generated evidence file for publication.

This evidence does not approve real capital. Broker fractionability and a safe
fractional protective-stop design remain explicit promotion blockers.

Point-in-time membership remains fail-closed. A complete table needs effective
dates plus source URL, retrieval timestamp, license, and `access_cost=free`.
Today's constituents are never treated as historical membership. Missing
coverage stays a visible survivorship and promotion blocker.

## Dashboard Evidence Meanings

The dashboard keeps four execution outcomes separate:

- **Measured failure:** a complete, decision-eligible population failed.
- **Insufficient evidence:** observations are collecting or coverage is small.
- **Stale evidence:** measurements exist but are too old for a decision.
- **Operational failure:** the scorecard is missing, unreadable, or errored.

Account alignment is separately shown as pass, fail, or collecting from
canonical broker truth. A missing broker snapshot cannot appear as a measured
alignment failure. The Walkforward page also shows the independent daily audit
beside the periodic fold summary and lists every promotion blocker.

## Shadow Workflow Acceptance

The Shadow Paper Journal defaults remain `force=false` and
`ignore_stale=false`. Each journal row records validation-bundle validity and
an exact config-fingerprint match. After generating the paper-versus-shadow
comparison, the workflow verifies those fields and requires the comparison
artifact to be less than 15 minutes old before it can publish journal evidence.
