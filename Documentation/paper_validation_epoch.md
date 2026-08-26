# paper_validation_epoch.py

## What This Script Does

This script starts a clean paper-trading evidence period after an important bot
change. It copies existing logs and reports into a dated archive, then records
the fixed rules the improved bot must satisfy.

Operational files are copied, not removed, because order reconciliation still
needs them. All new measurements are filtered using the epoch start time.

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
