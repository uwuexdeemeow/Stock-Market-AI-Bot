# corrected_audit.py

This is the offline entry point for corrected accounting, causal research,
recorded-fill replay and prospective shadow observation. It has no broker-order
connection. A **shadow** strategy is measured without replacing the active
paper strategy. A **freeze** records the exact strategy and evaluation protocol
before new observations are collected.

## Start with the data report

From the project directory:

```powershell
python corrected_audit.py --audit-only
python corrected_audit.py --spec corrected_shadow_spec.json
```

The second command only evaluates when every data gate passes. The supplied
specification is an explicitly unapproved shadow research candidate: the four
raw feature candidates are fitted inside training, and the existing allocation,
sizing and risk defaults are retained. Dated VIX, earnings-calendar and sector
inputs must be supplied before replaying the approved dynamic allocation.
There is no automatic fallback to today's classifications or to a different
strategy. Edit a copy of the specification before freeze; material changes
after freeze require a new cohort and output directory.

## Required input contracts

- `data/universe_membership.csv`: ticker, effective_from, effective_to, status,
  source, source_url, retrieved_at, license and access_cost (`free`). End dates
  are inclusive. Removed and failed names require real coverage.
- `data/raw/TICKER.parquet`: a unique date index with positive finite raw Open,
  High, Low, Close and Volume. Include training/warmup history and all required
  holding/exit sessions. Keep SPY and QQQ too. Missing selected prices stop
  evaluation with ticker/date details.
- `data/raw/manifest.json`: `symbols` maps each ticker to `sha256`, `source_url`,
  `retrieved_at`, `license`, `access_cost: "free"`,
  `adjustment_mode: "raw_ohlcv"`, and `actions_coverage` containing verified,
  source_url, start and end. Top-level `actions_sha256` verifies the action file.
- `data/raw/actions.csv`: event_id, ticker, kind, date, value, source. Supported
  kinds are split (ratio), dividend (cash per entitled share, plus ex_date and
  payment date in date), symbol_change (plus new_ticker), and cash_liquidation
  (cash per share). An empty action file still needs headers and independently
  verified no-action coverage. Never write `verified: true` without evidence.
- Optional `feature_panel` parquet requires `feature_provenance_verified: true`
  and attributed dated inputs; otherwise raw features are built automatically.
  A `dated_context` parquet joins exact date/ticker rows. Both require published_at,
  source_url and access_cost, checked against the actual NYSE close. Context for
  dynamic policy supplies vix_inverted, sector and days_to_next_earnings; unknown
  required observations block evaluation.

The JSON specification contains `features`, `label`, `horizon`, `policy`,
`configurations`, `folds` and optional context paths. Each outer fold has
train_end, start, end and an `inner` list of the same dated intervals. Inner
validation must finish within outer training. Configurations are chosen only
from inner results. Cost evidence, if supplied in a configuration, must contain
attributed pre-window fills; otherwise frozen documented costs apply.

## Outputs and stress checks

Successful runs create unique directories under `signals/corrected_audit/runs/`.
Each trial saves events, daily holdings, daily equity, benchmark equity and
events, metrics and its feature artifact. `trials.jsonl` records every tested
configuration and cutoff; `validation.json` identifies the ledger, code/data
fingerprints and retrospective interpretation. Old result caches are not read.
Cost stress is applied to every fill, not just stock weight changes. Stress
reports must use the same selected configuration and must not select the best
stress level as if it were a strategy parameter.

## Recorded-fill replay

```powershell
python corrected_audit.py --replay-events inputs/events.csv --opening-balances inputs/opening.json --closing-balances inputs/closing.json --marks inputs/marks.csv --history-evidence inputs/broker_history_report.json
```

Balances are JSON objects such as `{"cash": 1000, "holdings": {"SPY": 1}}`.
Events have actual timestamps and kinds. Individual fills need unique event_id,
ticker, signed quantity, price, fee, order_id and decision_id. Do not feed the
same cumulative order snapshot repeatedly as individual fills. Submission and
cancel/reject/expire events keep order IDs; corporate actions use the contract
above and documented dividend entitlement. Daily marks contain timestamp,
ticker, price. Output is replay_reconciliation.json, replay_events.csv and
replay_daily_equity.csv. Missing balances/fees remain gaps; no terminal sale
is fabricated. The recovered September 3 fixture contains known share changes
but insufficient inputs for a one-cent cash reconciliation.
The history report must explicitly have `history_complete: true`; absence or
incompleteness blocks freeze even if the supplied subset balances reconcile.
Opening and closing balance inputs also need `verified: true` and an attributed
`source` to certify freeze. An opening inferred from equity history is not an
independent cash/position statement. Matching arithmetic with that hypothesis
cannot start the prospective clock.

## Freeze and 252 new sessions

```powershell
python corrected_audit.py --spec corrected_shadow_spec.json --freeze
python corrected_audit.py --spec corrected_shadow_spec.json --observe
```

Freeze requires complete source checks, successful corrected evaluation and a
version-matching successful recorded-fill reconciliation. It preserves any
existing freeze. Observation uses the frozen feature artifact, policy and costs
and completed NYSE sessions beginning after freeze. Changed code, protocol,
training data or earlier observed inputs stop the run and require restart.
Newly matured outcomes may arrive without rewriting earlier decisions.

The runner writes prospective daily equity/events, status and paired edge
evidence. The final prospective review requires 252 new observed sessions.
Twenty non-overlapping 20-session cohorts may require longer than 252 sessions;
the review must remain inconclusive if statistical requirements are not met.
Neither command switches the paper strategy or approves real money.

Historical results from the old accounting are superseded diagnostics and need
regeneration. Keep original audits untouched. Code belongs on main; operational
reports belong under the existing signals/latest evidence publisher.
