# Strategy evidence audit

Status: **blocked**

Historical claims require corrected rebuilding. Shadow candidate is separate from deployed factor scoring.

| Source | Complete | Measured fills | Sessions |
| --- | --- | --- | --- |
| local_snapshot | False | 6 | 1 |
| signals/latest | False | 9 | 2 |
| workflow_artifact:33897908550:9946624459:success | True | 9 | 2 |
| imported_workflow_artifact:33897908550:9946624459:success | True | 9 | 2 |

## Blockers and next actions

- completed run has no artifact:33902026497 (data/replay): Restore read-only access and retrieve the latest completed run.
- complete run manifest missing (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca execution scorecard.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca paper health.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:broker truth.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:paper validation epoch status.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca execution scorecard.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca paper health.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:broker truth.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:paper validation epoch status.json (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- snapshot timestamp missing (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment status conflict (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (local_snapshot): Recover matching evidence and rerun existing validation; do not edit approval flags.
- complete run manifest missing (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca execution scorecard.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca paper health.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:broker truth.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:paper validation epoch status.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca execution scorecard.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca paper health.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:broker truth.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:paper validation epoch status.json (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment status conflict (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- config fingerprint missing (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- configuration fingerprint mismatch (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployed configuration missing (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment state missing (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- robustness review missing (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- top bundle reference mismatch (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle hash missing (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle schema outdated (workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- config fingerprint missing (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- configuration fingerprint mismatch (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployed configuration missing (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment state missing (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- robustness review missing (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- top bundle reference mismatch (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle hash missing (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle schema outdated (imported_workflow_artifact:33897908550:9946624459:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- membership table missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- required ticker membership missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- membership provenance columns missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical universe too small (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- inactive membership coverage too small (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical price coverage incomplete (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical membership missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price file missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- source url missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- retrieved at missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- license missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- free raw source unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action coverage missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price file missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- source url missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- retrieved at missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- license missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- free raw source unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action coverage missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action file unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- dated context missing or unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- paper logic fingerprint mismatch (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- recorded replay not certified (data/replay): Recover complete activities, fees and independently verified interval balances; replay to one-cent cash tolerance.
