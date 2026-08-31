# paper_validation_epoch.py

Formal validation consumes canonical broker alignment. The 2% per-ticker and
5% gross-exposure limits must both pass, and no critical incident may remain
open. Every status report carries the shared daily run identity.

## What This Script Does

This script starts a clean paper-trading evidence period after an important bot
change. It copies existing logs and reports into a dated archive, then records
the fixed rules the improved bot must satisfy.

Operational files are copied, not removed, because order reconciliation still
needs them. All new measurements are filtered using the epoch start time.
Equity observations use their exact snapshot timestamp when available, so a
snapshot taken later on the epoch's first day is counted correctly. Older files
that only contain a calendar date remain supported.

Execution acceptance now reads average and material bad-slippage rates from
`alpaca_execution_scorecard.json`. That scorecard owns measurement denominators,
so the epoch cannot accidentally grade 14 bad fills out of 25 while the
execution dashboard grades the same evidence out of 19 measured fills. Thin or
incomplete scorecards remain in `collecting` state.

## How To Run It

Start a new epoch:

```bash
python3 paper_validation_epoch.py
```

Check progress:

```bash
python3 paper_validation_epoch.py --status
```

Freeze the reviewed paper code without restarting the active epoch:

```bash
python3 paper_validation_epoch.py --freeze-current
```

This writes `paper_version_lock.json` while keeping the existing epoch ID and
`started_at` value unchanged. Before any Alpaca paper submission, the trader
checks the locked strategy, risk, execution, workflow, live-config, and
validation-bundle files. If one changes, new orders are blocked until a senior
review deliberately freezes the reviewed version again. Documentation and
dashboard-only changes do not break the lock.

The lock also covers `requirements-ci.txt` and `requirements.txt`. Package
version changes can alter model, market-data, or broker behavior even when the
Python files themselves did not change.

The independent watchdog workflow and `workflow_watchdog.py` are locked too,
because their guarded fallback can start a missed paper-trading workflow.
The shadow workflow and paper-vs-shadow comparator are locked because they
decide whether validation evidence is complete or still collecting.

## Key Terms

- **Epoch:** A dated evaluation period for one bot version.
- **Version lock:** Checksums proving the decision and execution files still
  match the reviewed paper version.
- **Rebalance:** Changing positions to match new target weights.
- **Accepted order:** An order Alpaca received successfully.
- **Operational pass:** The bot met the reliability sample requirements.

## Approval Requirements

New epochs started by the script use 30 trading days, 20 accepted orders, three
rebalances, and 10 consecutive classified sessions. Their stricter defaults are
a 95% fill rate, average rebalance slippage no worse than 5 bps, a materially
bad slippage rate no higher than 40%, and no duplicate or unexplained orders.
The frozen August 26 epoch keeps its original thresholds and start time.
