# Reliability-First Project Workflow

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
7. `daily_run.py` checks data and broker health, submits sell-first rebalances,
   resizes buys to cash, restores stops, and always reconciles broker truth.
8. `paper_validation_epoch.py` measures a clean operational period. Real-money
   trading remains disabled until a separate explicit human review.

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
false.

# Post-market execution evidence

The morning paper workflow can finish before 15-minute and 60-minute price
observations exist. A separate GitHub workflow runs at 5:15 PM New York time.
It refreshes account status and execution reports without submitting,
reconciling, cancelling, replacing, or creating orders. It shares the
`signals-latest-publisher` lock with the daily and shadow workflows, then saves
the mature scorecard for the next trading session.
