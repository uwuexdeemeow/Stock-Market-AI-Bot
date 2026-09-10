# September 10 paper follow-up

Code repairs are on main, and all four final release CI checks passed. Offline regressions: 715 passed, 45 skipped.

- The September 9 rerun verified cache recovery in 2 seconds instead of about 60 minutes. Its signal generation and lock passed; submission was outside the unchanged execution window.
- Regime monitoring now follows successful signal generation and precedes order submission. A failed signal still blocks recording; a failed submission cannot suppress a valid observation.
- Broker cash and shares reconcile from September 8 16:59 UTC through September 10 04:46 UTC. There were zero fills, so this is balance continuity only.
- Later post-market workflows passed. The green later daily run was an intentional schedule skip. The watchdog correctly retained the failed trading-run alert.
- Runtime approval is now blocked because the factor-decay report exceeded its seven-day limit. The version lock itself passes. 70 of 159 local files differ from the approved research snapshot, so a report cannot safely be regenerated and labeled with that old fingerprint without recovering matching inputs.

Next step: recover the approved research snapshot or rebuild all dependent evidence on verified replacement inputs, then validate a matching bundle and paper lock before a normal-window run. No threshold was relaxed, no outside-window override used, and no corrected prospective freeze started.

[Complete gap register](gap_register.md) · [Unified audit](evidence_report.md) · [Dataset recovery details](dataset_recovery_blocker.json) · [Final CI](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/34438801469)
