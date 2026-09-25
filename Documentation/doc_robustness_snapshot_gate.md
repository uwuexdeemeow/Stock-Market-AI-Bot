# robustness_snapshot_gate.py

## What it does

This script separates two questions that used to be mixed together:

1. Is the refreshed research snapshot complete and internally consistent?
2. Do its safety reports currently allow paper trading?

A complete snapshot can honestly answer “do not trade.” That is not a broken
data-refresh job. The script records that state as `trading_blocked`, sets
`trading_allowed` to `false`, and makes any existing paper signal inert. It
does not relax or remove the execution-stress, survivorship, factor-decay,
configuration, fingerprint, or freshness checks.

The script writes `signals/robustness_snapshot_status.json`. If trading is
blocked, it also sets the paper signal’s approval flags to false, changes all
target holdings to zero, sets cash to 100%, and clears the stock overlay.

## How to run it

From the project folder:

```powershell
python robustness_snapshot_gate.py
```

Expected outcomes:

- Exit code `0`, status `approved`: evidence matches and trading is allowed.
- Exit code `0`, status `trading_blocked`: evidence matches, but a safety test
  rejected trading. The workflow may finish normally, but no order step runs.
- Exit code `1`, status `invalid`: evidence is missing, mismatched, or cannot be
  trusted. The workflow fails because it cannot safely use the snapshot.

GitHub Actions uses:

```bash
python3 robustness_snapshot_gate.py --github-output
```

`--require-trading-pass` is available for a caller that deliberately wants a
nonzero exit code for either a blocked or invalid result.

## Key terms

- **Snapshot:** One matched set of market data, factor inputs, and safety
  reports from the same refresh.
- **Fingerprint:** A checksum that changes when important inputs change.
- **Evidence identity:** Proof that every report belongs to the exact strategy
  and data snapshot being considered.
- **Safety review:** Execution-stress, survivorship, and factor-decay checks
  that decide whether paper orders are allowed.
- **Fail closed:** When evidence is uncertain or a safety test fails, the bot
  permits no orders.
