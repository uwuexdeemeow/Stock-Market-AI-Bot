# paper_validation_epoch.py

## What This Script Does

This script starts a clean paper-trading evidence period after an important bot
change. It copies existing logs and reports into a dated archive, then records
the fixed rules the improved bot must satisfy.

Operational files are copied, not removed, because order reconciliation still
needs them. All new measurements are filtered using the epoch start time.
Equity observations use their exact snapshot timestamp when available, so a
snapshot taken later on the epoch's first day is counted correctly. Older files
that only contain a calendar date remain supported.

## How To Run It

Start a new epoch:

```bash
python3 paper_validation_epoch.py
```

Check progress:

```bash
python3 paper_validation_epoch.py --status
```

## Key Terms

- **Epoch:** A dated evaluation period for one bot version.
- **Rebalance:** Changing positions to match new target weights.
- **Accepted order:** An order Alpaca received successfully.
- **Operational pass:** The bot met the reliability sample requirements.

## Approval Requirements

The clean epoch needs 30 trading days, 20 accepted orders, three rebalances,
and 10 consecutive classified sessions. It also requires at least a 95% fill
rate, average rebalance slippage no worse than 5 bps, a materially bad
slippage rate no higher than 40%, and no duplicate or unexplained orders.
