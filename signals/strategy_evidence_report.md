# Strategy evidence audit

Status: **blocked**

Historical claims require corrected rebuilding. Shadow candidate is separate from deployed factor scoring.

| Source | Complete | Measured fills | Sessions |
| --- | --- | --- | --- |
| local_snapshot | False | None | None |
| signals/latest | True | 9 | 2 |
| workflow_artifact:34485337896:10155521944:failure | False | 9 | 2 |

Replay evidence scope: unspecified_in_source.
Replay certification concerns interval accounting only; it does not approve the strategy or prove trading performance.

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
- execution observations insufficient or failed (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:trading days (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:rebalance events (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:accepted orders (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:classified sessions (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:average slippage (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:bad slippage rate (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:stage comparison ready (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:two stage design (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- complete run manifest missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:core satellite alpha orders.csv (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- config fingerprint missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- configuration fingerprint mismatch (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployed configuration missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment state missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment status conflict (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- robustness review missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- top bundle reference mismatch (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle hash missing (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle schema outdated (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- signal freshness failed (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- execution observations insufficient or failed (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:trading days (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:rebalance events (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:accepted orders (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:classified sessions (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:average slippage (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:bad slippage rate (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:stage comparison ready (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:two stage design (workflow_artifact:34485337896:10155521944:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- membership table missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- required ticker membership missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- membership provenance columns missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical universe too small (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- inactive membership coverage too small (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical price coverage incomplete (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- historical membership missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- membership coverage unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price file missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- source url missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- retrieved at missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- license missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- free raw source unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price provenance invalid (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price feed or symbol mapping missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price security identity unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action coverage missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price file missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- source url missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- retrieved at missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- license missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- free raw source unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price provenance invalid (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price feed or symbol mapping missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- raw price security identity unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action coverage missing (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- corporate action file unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- dated context missing or unverified (data/replay): Supply attributed historical membership, raw bars, actions and dated context; rerun source checks.
- current robustness failed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- recorded replay not certified (data/replay): Recover complete activities, fees and independently verified interval balances; replay to one-cent cash tolerance.
- normal session activity reconciliation awaiting observations (data/replay): Reconcile a normal-session interval after broker activity and fee postings; never submit orders to manufacture samples.
- full historical broker accounting uncertified (data/replay): Obtain independently sourced historical opening balances and all subsequent activity; interval arithmetic is insufficient.
- corrected prospective validation requires explicit freeze and new observations (data/replay): After verified inputs, successful evaluation and certified reconciliation, obtain explicit freeze before 252 new sessions and 20 independent matured cohorts.
- review report missing (membership_identity_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review source identity incomplete (membership_identity_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review not fully verified (membership_identity_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review report missing (security_transition_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review source identity incomplete (security_transition_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review not fully verified (security_transition_report): Regenerate the attributed review and resolve its verified-input dependencies.
- review report missing (edge_ablation_comparison): Regenerate the attributed review and resolve its verified-input dependencies.
- review source identity incomplete (edge_ablation_comparison): Regenerate the attributed review and resolve its verified-input dependencies.
- review not fully verified (edge_ablation_comparison): Regenerate the attributed review and resolve its verified-input dependencies.

## Exact version-lock differences


## Independently generated reviews

- membership_identity_report: missing; generated None; source commit None.
- security_transition_report: missing; generated None; source commit None.
- edge_ablation_comparison: missing; generated None; source commit None.
