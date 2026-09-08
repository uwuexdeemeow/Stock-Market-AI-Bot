# Strategy evidence audit

Status: **blocked**

Historical claims require corrected rebuilding. Shadow candidate is separate from deployed factor scoring.

| Source | Complete | Measured fills | Sessions |
| --- | --- | --- | --- |
| local_snapshot | False | None | None |
| signals/latest | False | 9 | 2 |
| workflow_artifact:34256469689:10068248676:failure | False | 9 | 2 |

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
- snapshot stale or future (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment status conflict (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- signal freshness failed (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- execution observations insufficient or failed (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- reported paper version lock invalid (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:trading days (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:rebalance events (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:accepted orders (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:classified sessions (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:average slippage (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:bad slippage rate (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:stage comparison ready (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:two stage design (signals/latest): Recover matching evidence and rerun existing validation; do not edit approval flags.
- complete run manifest missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- required evidence missing:core satellite alpha orders.csv (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- config fingerprint missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- configuration fingerprint mismatch (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployed configuration missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- deployment state missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- paper approval not unanimous (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- robustness review missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- strategy bundle reference mismatch (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- top bundle reference mismatch (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle hash missing (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- validation bundle schema outdated (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- execution observations insufficient or failed (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- reported paper version lock invalid (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:trading days (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:rebalance events (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:accepted orders (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:classified sessions (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:average slippage (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:bad slippage rate (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:stage comparison ready (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
- epoch check failed:two stage design (workflow_artifact:34256469689:10068248676:failure): Recover matching evidence and rerun existing validation; do not edit approval flags.
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
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- locked file changed (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
- paper logic fingerprint mismatch (local_runtime): Review current runtime evidence and version lock; preserve existing freeze.
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

- .github/workflows/daily_paper_trading.yml: 27cbe21ac90afca3becda5972af02b5242bea9987003920c11a48fef4b4f07e0 → 4b903ac819ff04ddfaa670f46a300e06b5908dd420240a15122f1d9604e5b87b
- .github/workflows/post_market_execution_quality.yml: 2fac86db3ad822f05b260bc99e8af8499de3a8064064338a2d2e9ccbe220547c → 8cd45afa24b8d0b6a18da5a68de4a687b58f2a0985623f49219d838fdfd221ac
- .github/workflows/shadow_paper_journal.yml: 4a92d68ab42d75616b5cef9801fc5f8f378f4d69956439d8c4bfcf8d955a8805 → 546f23060b0e2e3f5995bf4214caaaf36e58dad01021b45ae14fff067bf6a939
- alpaca_paper_trading.py: 7cfd5e3b1311a90d38d17bfdd82c6d5eca021aef97cced9c2a4d9ed32c73b247 → 8dc12525a4b33b63ac78270472de4da1338ea6213ecbe1e6551324af7177f1b1
- alpaca_sdk_adapter.py: 18130bf1b5c0e1ed088a7e79ad7d9f03927bbce4b8e291a062df020f78f32b89 → ca9e279443969d5447b24131d58a873d5a97e89a90a871d87c9b2fd734989e94
- core_satellite_alpha.py: d4e334d52ab9a63cfba15cd91cbd249ab4bb56e878ddb26e43231908fcc7c5f5 → 53e8ce2d5bfa34a85e554e626a8e0dc9e1eaa3df0ee48ccf548db0ad5007124f
- daily_run.py: de5027774a89bbd5929b1120a7a322e267c73cf1454b48c62ee5a06ce2a7a977 → 08fb198449888646606e23b1934b43216a3bfd81ee98ac2e4f92d2e1759d1e97
- execution_scorecard.py: 25e1ff0912f9dc4bf3e65876770df78657d5cdd3c68cec1b72a8249fa46db67d → 91d076144906470927f58c1024e1f81d225ba7009f6b333135587645b12ddd47
- order_accounting.py: 58b575704bf842fff95254c95af46cfaacd7361088fe59051c7658a00d3a8a8c → bee3e3f39af187593f9f9bae23df1b4aaf3bcf764ae7a2842119900f5e6d0562
- paper_health.py: 7bc855d683accc030409fbadef5a4455586604f0f18334e48b6d6b0930677baa → 3cc62f26f43116278169cb2f80186f5adee5a6cb5c36700ea3caf9651d3aca4f
- run_evidence.py: f5c531248e04219465a86d4ab4afc89ec204f82570df4ba022a9e9529f948c0f → bf0a92c678c28955f5bb7f4d31c7ccbd4780045cf191e00285288fad3967f7ba
- trade_rules.py: e1c1d0f9d7bb0269205889254c467958cca3f12e651910a75d613453fb1d82db → 0a456248e7fbb20752a04bc980c1336d24371d5db54f32d04fb1f173bca06efc

## Independently generated reviews

- membership_identity_report: missing; generated None; source commit None.
- security_transition_report: missing; generated None; source commit None.
- edge_ablation_comparison: missing; generated None; source commit None.
