# Strategy evidence audit

Status: **blocked**

Historical claims require corrected rebuilding. Shadow candidate is separate from deployed factor scoring.

| Source | Complete | Measured fills | Sessions |
| --- | --- | --- | --- |
| local_snapshot | False | None | None |
| signals/latest | False | 9 | 2 |
| workflow_artifact:34129476205:10021497648:success | False | None | None |
| imported_workflow_artifact:33897908550:9946624459:success | True | 9 | 2 |

## Blockers and next actions

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
- complete run manifest missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca execution scorecard.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:alpaca paper health.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:broker truth.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- evidence run mismatch:paper validation epoch status.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca execution scorecard.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:alpaca paper health.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:broker truth.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:paper validation epoch status.json (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- config fingerprint missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- configuration fingerprint mismatch (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployed configuration missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment state missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- robustness review missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- top bundle reference mismatch (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle hash missing (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle schema outdated (workflow_artifact:34129476205:10021497648:success): Recover matching evidence and rerun existing validation; do not edit approval flags.
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
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- paper logic fingerprint mismatch (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.

## Paper interval reconciliation

Status: **Reconciled — no activity**

- Interval (UTC): 2026-09-07T04:57:22.027397+00:00 to 2026-09-07T13:59:54.035671+00:00.
- Independently observed opening and closing cash and shares match exactly.
- Cash difference: $0.00 (allowed tolerance: $0.01).
- Complete activity and order retrieval; 0 interval fills and 0 fee events.
- Saved opening balances match their archived account and positions records.

This US market holiday interval proves balance continuity only. It does not validate execution quality, historical trading results, profitability, or strategy approval. The replay certification flag applies only to this interval's accounting prerequisites. Historical opening balances remain unresolved.

No orders submitted, no freeze started, and no strategy settings changed. Raw broker records remain private. Source hashes appear in the JSON companion.

Next: retain the verified opening snapshot and reconcile again after a normal trading session and broker fee postings. Corrected research remains blocked by missing verified inputs.
