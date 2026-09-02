# validation_bundle.py

When `--run-robustness` is used, the bundle builder generates a research-only
signal from the newly walk-forward-approved configuration before rerunning the
execution, survivorship, and factor-decay reports. This breaks the safe
bootstrap cycle without allowing the provisional bundle to reach the broker.

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

Paper approval fails closed unless the walk-forward source, folds, dataset
fingerprint, and all three robustness reports exist and have matching strategy
and dataset fingerprints. A signed bundle with missing or mismatched robustness
evidence cannot authorize new paper orders.

The temporary shadow journal is the only exception. It creates a separate
paper-only bundle without production robustness reports because that path only
records hypothetical results and cannot submit orders or approve real money.

Expected result: paper trading can continue, but `real_capital_approved` stays
false.

If a prior migration dropped folds from an otherwise matching approved
walk-forward, repair the canonical evidence with:

```bash
python3 validation_bundle.py \
  --source-walkforward signals/wf_low_turnover.json \
  --run-robustness
```

The repair refuses evidence for a different live configuration. With
`--run-robustness`, it refreshes the paper signal and the execution-stress,
survivorship, and factor-decay reports before rebuilding the bundle. These are
research-only commands: no broker order is submitted. The resulting source is
stored at `signals/core_satellite_nested_walkforward.json` so GitHub can restore
it without depending on a developer's ignored or absolute scratch path.

## Key Terms

An old paper configuration with no matching tracked folds is recorded with
`walkforward_source_missing` and `walkforward_folds_missing`. It may remain
paper provisional, but it cannot become real-capital approved.

- **Fingerprint:** A checksum that changes when a configuration or dataset changes.
- **Validation bundle:** One file containing all evidence used to judge a strategy.
- **Provisional:** Allowed for paper testing, but not approved for real money.
- **Atomic write:** Writing a complete replacement file so a crash cannot leave half a file.

## Schema 2 Robustness Review

The bundle now stores both report identity and report health. Survivorship and
execution-stress reports may be at most 60 days old; factor-decay evidence may
be at most seven days old. All reports must match the selected config and
dataset, and their current shared review must pass. Factor-decay `warning` and
`block` both reject paper approval. Live signal generation rereads the same
files and verifies their checksums still match the bundle.
## Deployment-exposure identity

The strategy fingerprint includes `deployment_max_gross_exposure`. This means
a historical 1.25x research validation and a 1.00x paper-matched validation
cannot receive the same configuration identity or be treated as equivalent
evidence. Older configurations that lack the field are interpreted as having
no separate deployment scaling limit.

## Survivorship capital eligibility

The deployment record exposes `capital_approval_eligible`. It remains false
unless paper evidence passes, failed-name coverage is complete, and the dated
point-in-time universe is complete. The existing safety rule still forces
`real_capital_approved=false`; eligibility is evidence status, not permission
to trade real money.
