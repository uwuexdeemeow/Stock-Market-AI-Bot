# validation_bundle.py

This script links a selected strategy to the research evidence that supports
it. A bundle is a JSON report containing checksums, the configuration identity,
source coverage and the actual deployment decision. A checksum identifies exact
file contents; a configuration is the set of strategy settings.

## Read evidence without changing configuration

From the project folder:

```sh
python3 corrected_audit.py --evidence-report
```

This reports approval conflicts, missing sources and mismatched identities in
JSON and Markdown. It does not rebuild configurations or start a freeze.

## Rebuild from verified matching evidence

```sh
python3 validation_bundle.py --source-walkforward path/to/matching_research.json --live-config path/to/live_copy.json --output path/to/bundle.json
```

This writes the specified live configuration and bundle and updates the
canonical walk-forward report. Use matching, reviewed evidence and preserve
original reports first. `--run-robustness` additionally refreshes research
reports. Without `--source-walkforward`, migration produces a bundle without
historical folds; that missing evidence cannot create approval.

Daily loading and the unified audit share `validate_live_approval_identity`.
They require matching configuration and bundle fingerprints and consistent
approval states at the top and selected-strategy levels. A rejected record
cannot be overridden by another approval. Rebuilding copies the new bundle's
actual decision to both levels; rejected evidence remains rejected. No lock,
freeze, order submission or real-capital approval follows from this script.
