# Execution reporting repair — September 5, 2026

Implemented and tested locally. No broker orders or execution-price settings
were changed. The changes have not been pushed or deployed; protected-file
hashes in the paper version lock have not been silently refreshed.

## Findings and changes

- The shadow publisher could fall back to main after a failed checkout and
  force-push a branch missing daily operational records. That fallback is gone.
- All three publishers now retain remote history and push without force.
  Daily publication retains files owned by the shadow and post-market jobs.
- Missing journal denominators produce an unknown overall score and block
  decision eligibility. Broker fills absent from a partly restored journal
  also prevent an overall score. Price-only stage reports no longer claim a
  100% fill rate while omitting canceled, unfilled attempts.
- Completion uses the actual cash-clamped journal quantity. Duplicate child
  snapshots cannot inflate fills, and a filled child cannot conceal an
  incomplete parent.
- Daily manifests require readable signals, plans, journals and submission
  history. Post-market reports share one run ID and evaluate the epoch before
  the health report. Main retains ownership of the epoch definition.

## Historical verification

Recovered original daily evidence under
`archive/execution_recovery_20260905/daily-232/` and `daily-236/` from immutable
commits `364a9e5daa69e70a997313c402a4c7ebae4b8b58` and
`fe2506e56ac0db3ab9dc283215d715d59ba20875`. Thirteen manifest-listed files in
each recovered bundle reproduce their original SHA-256 checksums. Windows
line endings were normalized only where that reproduced the recorded checksum.
These archived copies do not overwrite the current live input files.

September 3 replay connects the original plan, journal and later broker fills:

| Order | Planned | Submitted | Filled | Explanation |
|---|---:|---:|---:|---|
| INTC buy | 205 | 205 | 205 | First attempt 155, second attempt 50 |
| FCX buy | 43 | 30 | 30 | Explicit cash clamp; first attempt canceled at zero fill |

The repaired scorecard finds two accepted and completed logical orders,
four attempts, three child fills, no unmatched broker fills, and 1.03 bps
mean fill-minute slippage. It remains collecting and ineligible because the
sample is small. See `archive/execution_recovery_20260905/replayed_scorecard.json`.

The full test suite passed with 585 tests and 24 skips before one additional
partial-journal regression was added and checked separately. Tests execute the
actual three publication shell blocks against disposable Git repositories,
including untracked-file conflicts, history retention and daily log globs.
This is an offline replay and publication test, not a newly deployed live run.

## Release boundary

Review and commit the repair before updating the protected paper version lock.
Record the reporting change explicitly when refreezing; historical observations
must not be treated as new sessions. Restore any missing operational histories
from original run artifacts with provenance before relying on production
readiness counts. Then run the read-only post-market workflow and verify its
restored inputs and scorecard. Do not trigger a trading run merely to test
report publication.
