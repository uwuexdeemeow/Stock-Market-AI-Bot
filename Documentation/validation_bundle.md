# validation_bundle.py

## What This Script Does

This script creates one trusted validation record for the paper-trading
strategy. The record identifies the exact strategy settings, research data,
Git version, walk-forward results, and robustness reports.

It prevents the bot from accidentally combining a new strategy with old test
reports. It also keeps paper approval separate from real-money approval.

## How To Run It

```bash
python3 validation_bundle.py
```

The script reads `signals/core_satellite_live_configs.json`, writes
`signals/core_satellite_validation_bundle.json`, and marks the existing
strategy as `paper_provisional`.

Expected result: paper trading can continue, but `real_capital_approved` stays
false.

## Key Terms

An old paper configuration with no matching tracked folds is recorded with
`walkforward_source_missing` and `walkforward_folds_missing`. It may remain
paper provisional, but it cannot become real-capital approved.

- **Fingerprint:** A checksum that changes when a configuration or dataset changes.
- **Validation bundle:** One file containing all evidence used to judge a strategy.
- **Provisional:** Allowed for paper testing, but not approved for real money.
- **Atomic write:** Writing a complete replacement file so a crash cannot leave half a file.
