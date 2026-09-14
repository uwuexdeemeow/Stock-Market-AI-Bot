# Daily workflow failure: September 11, 2026

## Evidence and cause

[Run 258](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34605140151)
and [run 259](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34625601725)
both ran commit `d7c846b84b5767c31ae01f6b3226cce4a3784f01` on September 11.
Both completed 16 of 20 pipeline steps. Their first failing step was
`alpaca_submit`, with `paper_version_lock_failed`, before broker submission or
order-plan creation. Neither run submitted an order.

The frozen release predated two commits: `62359f8` changed the daily workflow's
failure reporting, and `d7c846b` regenerated research evidence. Six protected
files no longer matched their reviewed checksums:

- `.github/workflows/daily_paper_trading.yml`
- `logs/core_satellite_execution_stress.json`
- `logs/core_satellite_survivorship_audit.json`
- `logs/factor_decay_monitor.json`
- `signals/core_satellite_live_configs.json`
- `signals/core_satellite_validation_bundle.json`

The same mismatch reproduced locally from the committed release. This was a
stale release lock, not an Alpaca login failure. The regenerated bundle's
checksum and its identity against the live approval both validated. The strategy
configuration stayed unchanged; the approval remains provisional paper only.

The later failures were broker alignment, canonical readiness, and the evidence
manifest. FCX was absent at the broker despite a 2.7918% target; the allowed gap
is 2%. Five alignment incidents remained open. Since submission stopped before
planning, the manifest lacked `core_satellite_alpha_orders.csv`, so publication
to `signals/latest` was correctly refused. Those checks must remain active.

Run 260 was a successful off-season schedule skip; it did not demonstrate a
successful trading recovery. Run 259's scheduled job also started late, outside
the regular execution window. A lock repair alone cannot make a delayed run
eligible to trade outside that window.

## Repair and release procedure

`paper_validation_epoch.py --check-lock` now performs a read-only release check
with a failing exit code on mismatch. Paper Safety CI uses it to expose stale
locks before the daily job. The regression test checks both a matching release
and changed code, and verifies that checking cannot rewrite the approval or epoch.

After reviewing the existing changes and testing the repair, refresh the release
lock deliberately with `--freeze-current`, preserving the epoch and all account
limits. Commit the lock with the release. Do not regenerate a lock automatically
inside the daily workflow: that would silently approve unexpected changes.

The next eligible paper run must independently prove account alignment and a
complete evidence bundle. Historical incident rows are not deleted to make a
dashboard green, and no past failed run is relabeled successful.
