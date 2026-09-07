# Evidence audit

This script compares independent copies of the project's evidence. It does not
trade, change approval flags, repair a version lock, or start a new observation
period. Older historical returns remain superseded diagnostics.

Run from the project folder:

```bash
git fetch origin main signals/latest
python3 corrected_audit.py --evidence-report --spec corrected_shadow_spec.json
```

Outputs: `signals/corrected_audit/evidence_report.json` and `evidence_report.md`.
A blocked report is a successful audit delivery: read each blocker and next action.
The report independently labels local files, the fetched `signals/latest` commit,
and artifacts from the latest completed daily workflow. GitHub artifact retrieval
uses existing `gh` authentication (or `GH_TOKEN` in Actions). Unavailable access,
expired artifacts, missing manifests, changed checksums and mixed run IDs remain
gaps. No older successful run is silently substituted for a newer failed run.
The report checks a 96-hour freshness window as an observation warning, not a
market-session trading rule. Fetch first; the command itself does not change Git refs.

Production validation and version-lock checks apply to the local checkout only.
A consistent historical identity is not a present-day paper approval. Conflicting
nested statuses and bundle hashes are reported as blocked; rebuilding matching
validation evidence must resolve them through the existing approval machinery.

The deployed `regime_adaptive` route selects risk-on, neutral and defensive factor
score columns. That does not establish an XGBoost contribution. The corrected
candidate fits a separate raw-feature rank score within training windows.

For a previously recovered verified broker interval:

```bash
python3 corrected_audit.py --evidence-report \
  --reconciliation-report data/audit_recovery/INTERVAL/replay_reconciliation.json
```

Only certification booleans and a source checksum leave the private folder. Raw
balances, order IDs, cash values and broker response bodies are not copied into
published reports. Unknown arrival quotes remain unavailable. The shadow evidence
workflow publishes sanitized reports, including blocked reports, to `signals/latest`.

Terms: a **fingerprint** is a content checksum; a **manifest** lists the files in
one run; **reconciliation** checks that recorded events explain cash and shares;
**provenance** identifies where data came from and when it became available.

When a scheduled guard run has no archive, automatic retrieval records that gap
and inspects up to two older completed daily runs, explicitly retaining each
run's identity. It never relabels an older archive as the latest run.

A connector-downloaded archive can be read without configuring the local CLI:

```bash
python3 corrected_audit.py --evidence-report \
  --workflow-artifact data/audit_recovery/WORKFLOW/artifact.zip \
  --artifact-metadata data/audit_recovery/WORKFLOW/metadata.json \
  --recovery-report data/audit_recovery/INTERVAL/recovery_report.json
```

Metadata is the attributed GitHub API record: `artifact_id`, `run_id`, `head_sha`,
`digest` (the API's `sha256:` digest), `status: "completed"`, `conclusion`,
`updated_at`, and `source_url`. Optional latest-completed-run fields preserve the
fact that a more recent guard run had no artifact. The archive digest is checked
before any content is used; allowed files are read in memory, never extracted.
The recovery summary publishes counts, source fingerprints/cutoffs and price
coverage only. Raw account responses remain private.

When provenance passes, the source audit also checks finite positive raw OHLCV,
unique ticker/date rows, valid high/low ordering and dated context coverage on
all inner/outer decision dates. Gaps remain explicit before expensive trials.

### Detailed source failures and replay scope

Missing raw sessions now retain `count`, `first` and `last` in the sanitized data
report. Empty, malformed or partial files cannot pass through file-existence
checks. Source context validation matches the corrected input loader, including
training history. A replay report carries interval timestamps, fill/fee counts
and `evidence_scope`; `balance_continuity_only_no_activity` is explicitly not
trading-performance evidence. Older reports without a scope are labeled
`unspecified_in_source`, never silently interpreted as successful trading.
