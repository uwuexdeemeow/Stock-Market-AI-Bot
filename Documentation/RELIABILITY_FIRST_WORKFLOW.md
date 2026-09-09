# Reliability-First Project Workflow

## Corrected audit track: shadow first

The earlier historical accounting and audit results are preserved as superseded
diagnostic evidence. They are not corrected validation and must be regenerated.
The active paper strategy is not switched by the September 2026 code repairs.
Code fixes go directly to `main`; generated operational evidence stays with the
existing `signals/latest` publisher.

For a beginner, the full chain remains scan → research → train → predict →
backtest → paper trade. The scan defines candidate sources; research gathers
dated observations; training learns only from completed past outcomes; prediction
ranks today's eligible names; the corrected backtest records actual shares and
cash through every market session; paper trading supplies separate recorded-fill
evidence. Historical performance never authorizes a strategy switch by itself.

Start the corrected track with `python corrected_audit.py --audit-only`. Read
`signals/corrected_audit/data_quality.json` and obtain the specific missing free
membership, raw-price and corporate-action sources. Do not fill gaps with today's
watchlist, adjusted execution prices or synthetic returns. The existing shadow
journal workflow also writes and publishes the flat
`signals/corrected_audit_data_quality.json` report without starting a freeze.

Next use `python corrected_audit.py --spec corrected_shadow_spec.json`. See
`doc_corrected_audit.md` for input contracts. The example keeps approved paper
allocation/sizing defaults while fitting a small raw feature family inside each
training fold; it is an unapproved shadow candidate. Each fold starts in cash,
uses prior-session decisions and next-open fills, marks daily equity, and charges
all ETF/stock fills once. Inner folds choose settings; outer folds receive frozen
training artifacts. Separate stress runs apply 2×, 3× and 5× costs to the selected
configuration, including ETF purchases and final liquidation. Immutable trials
record the amount of historical experimentation; retrospective confidence
intervals are not adjusted evidence of prospective discovery.

Replay actual individual paper fills with verified opening/closing balances and
fees through the same account recorder. Reconciliation must match shares and
cash within one cent. Missing fees or balances block certification. Recovered
order summaries can confirm known share changes but cannot supply unknown fees,
arrival prices or a fabricated terminal sale. Daily simulation and recorded-fill
replay are different execution measurements and are labeled separately.

Only after complete corrected data/evaluation and recorded-fill reconciliation
can `--freeze` create a protocol. `--observe` measures completed sessions after
freeze without sending orders. Require 252 new NYSE sessions before final
prospective review. Changed strategy code, protocol, training inputs or earlier
observations require a new cohort; no peak reset or historical reuse can shorten
the period. Twenty mature non-overlapping 20-session cohorts may take longer
than 252 sessions. Missing or inconclusive evidence stays inconclusive.

Net compounded return, benchmark-relative excess return, and regression alpha
are separate measurements. Legacy raw overlay sums no longer count as alpha.
Statistical health requires positive lower confidence bounds for both excess
return and rank IC with enough independent cohorts; operational safety gates
remain separate. No result automatically cuts over paper or approves real money.

## Canonical daily evidence

The next evidence-recovery stage is available through
`python audit_evidence_recovery.py --paper --sources --price-probes SPY QQQ`.
It retrieves actual activities and candidate historical sources without trading.
Private responses stay under ignored data/audit_recovery. Separate broker fees
are cash events; recorded margin balances remain visible. A numerical replay
match with an inferred opening is not freeze certification. Independently
verified opening and closing cash/position snapshots are required. The script
saves a current snapshot for future intervals without changing the strategy or
starting the 252-session clock. See doc_audit_evidence_recovery.md.

All daily paper outputs share one run ID. Broker truth, execution quality,
validation progress, monitor continuity, and incidents are consolidated into
`signals/alpaca_paper_health.json`. Only a complete same-run bundle receives a
`signals/paper_run_manifest.json` status of `complete`; only that bundle may
replace the `signals/latest` branch.

The workflow stays paper-only. Alignment recovery and execution calibration
are review-only, and no readiness result can approve real capital.

## End-To-End Flow

1. `research.py` refreshes ticker data and writes a provider/checksum manifest
   for every parquet.
2. Research and model scripts create possible signals. The factor strategy
   remains champion; ML and sentiment stay shadow challengers.
3. `core_satellite_nested_walkforward.py` tests past-to-future folds, records
   relaxed fallbacks, compares selection with the frozen baseline, and measures
   turnover and cost separately.
4. Execution-stress, factor-decay, and survivorship reports carry matching
   strategy and dataset fingerprints.
5. `validation_bundle.py` combines config, Git commit, dataset, folds, analyzer,
   robustness reports, and approval into one checksummed source of truth.
6. `core_satellite_alpha.py` verifies the bundle and creates paper signals.
7. `daily_run.py` checks data and broker health, then executes each normal
   rebalance with a passive midpoint limit and one capped replacement for the
   confirmed remainder. It keeps sells first, fits buys to cash, restores
   stops, and always reconciles broker truth.
8. `paper_validation_epoch.py` measures a clean operational period. Real-money
   trading remains disabled until a separate explicit human review.

The active August 26 epoch is protected by `paper_version_lock.json`. The lock
fingerprints every file that can alter strategy selection, sizing, submission,
protection, or execution grading. Alpaca paper submission fails closed if a
locked file changes; freezing a reviewed version never changes the epoch's
original start time.

## Beginner Use

Preview the daily route without orders:

```bash
python3 daily_run.py --dry-run --alpaca --skip-refresh --skip-factor-refresh
```

Check the clean paper epoch with `python3 paper_validation_epoch.py --status`.
Scheduled paper runs continue in GitHub Actions and Telegram reports their
classified execution outcome.

For the next walk-forward, commit and push, run
`python3 prepare_colab_walkforward.py`, upload both files to the
`StockBotWalkforward` Drive folder, open the Colab notebook, and run its cells.
Colab uses CPU workers, saves resumable checkpoints, and never gets broker keys.

## Why It Is Designed This Way

Operational truth comes before prediction complexity. A profitable backtest is
not enough when data sources can be mixed, selection does not predict future
alpha, or planned orders can disappear without a final state. Paper trading
stays active to collect execution evidence while real-capital approval stays
false. Execution measurement grades normal rebalances separately from safety
stops, uses only observations that actually have price data, and treats later
market direction as timing advice instead of pretending it is fill quality.

# Post-market execution evidence

The morning paper workflow can finish before 15-minute and 60-minute price
observations exist. A separate GitHub workflow runs at 5:15 PM New York time.
It refreshes account status and execution reports without submitting,
reconciling, cancelling, replacing, or creating orders. It shares the
`signals-latest-publisher` lock with the daily and shadow workflows, then saves
the mature scorecard for the next trading session.

## Independent Quant Performance Audit

`quant_performance_audit.py` is the reference check for historical performance.
It starts with target positions emitted by the strategy, but it does not trust
the strategy's periodic equity curve. It reloads each stock and ETF's raw Open
and Close history, enters after the signal at the next executable Open, marks
positions and cash every day, charges turnover across both ETFs and stocks,
and stitches outer OOS years into one continuous curve.

The JSON report shows gross and transaction-cost net return and alpha versus
QQQ, exact overlap, elapsed-time CAGR, daily Sharpe and information ratio,
daily drawdown, Newey-West alpha evidence, calendar-year bootstrap uncertainty,
and a quantified reconciliation to the old periodic headline. The old 2,701%
headline stays visible only as provisional context; it is not replaced until
the independent reference blockers are cleared.

The same script runs a bounded shadow comparison around the frozen active
configuration: top five, 40% overlay, and stronger sticky weighting. It records
each attempt in the experiment ledger and creates a combined candidate only if
an isolated change passes every predeclared gate. It never writes the active
configuration and never calls Alpaca.

### Small-capital fractional shadow

The 9:55 AM New York Shadow Paper Journal workflow also runs a separate $400
fractional ledger from the restored active signal. It simulates market/day
fractional fills, cash, slippage, and regulatory fees without importing a broker
client. Daily and shadow workflows preserve its state together on
`signals/latest`, so one workflow cannot erase the small-account history.

Shadow restores only caches published by Factor Data Refresh. It no longer
rebuilds research data inside the journal job. A missing or unhealthy factor
cache fails quickly with a visible annotation, while a valid first observation
is labeled `collecting` instead of failing the workflow.

Post-market execution evidence is copied outside the checkout before the
workflow changes to the `signals/latest` publishing branch. Temporary runner
changes are stashed first, preventing a safe read-only audit from failing while
preserving every generated evidence file for publication.

This evidence does not approve real capital. Broker fractionability and a safe
fractional protective-stop design remain explicit promotion blockers.

Point-in-time membership remains fail-closed. A complete table needs effective
dates plus source URL, retrieval timestamp, license, and `access_cost=free`.
Today's constituents are never treated as historical membership. Missing
coverage stays a visible survivorship and promotion blocker.

Capital survivorship evidence now requires 100% failed-name coverage plus a
complete point-in-time universe. Strong results from a partial failed-name
sample remain provisional and cannot clear capital eligibility.

## Dashboard Evidence Meanings

The dashboard keeps four execution outcomes separate:

- **Measured failure:** a complete, decision-eligible population failed.
- **Insufficient evidence:** observations are collecting or coverage is small.
- **Stale evidence:** measurements exist but are too old for a decision.
- **Operational failure:** the scorecard is missing, unreadable, or errored.

Account alignment is separately shown as pass, fail, or collecting from
canonical broker truth. A missing broker snapshot cannot appear as a measured
alignment failure. The Walkforward page also shows the independent daily audit
beside the periodic fold summary and lists every promotion blocker.

## Shadow Workflow Acceptance

The Shadow Paper Journal defaults remain `force=false` and
`ignore_stale=false`. Each journal row records validation-bundle validity and
an exact config-fingerprint match. After generating the paper-versus-shadow
comparison, the workflow verifies those fields and requires the comparison
artifact to be less than 15 minutes old before it can publish journal evidence.

## September 2026 Evidence Reset

The research path is now boundary-safe: old rows may warm indicators, but each
walk-forward fold starts in cash and scores only complete in-fold positions.
The validation bundle then combines that fold evidence with fresh, matching,
passing survivorship, execution-stress, and factor-decay reports. Live signal
generation repeats the current report check; a warning blocks paper orders.

Execution validation counts logical rebalances rather than broker child
attempts. The acceptance gate uses complete fills over all accepted parents,
while partial-fill information remains visible separately. Because these rules
change both research and scoring meaning, old epoch evidence must be archived;
a new epoch starts only after regenerated research and robustness gates pass.

## Preserving execution history
Daily, shadow and post-market publishers now build on the existing
signals/latest branch and push without force. Generated files are copied out
before switching branches; a checkout/fetch failure stops publication. Shadow
publication must never fall back to main, where operational files are absent.
Daily publication updates only its owned outputs and retains other jobs' files.
Post-market reports share one run ID and evaluate the epoch before health.
The epoch definition remains owned by main, so a restore cannot revive an old
epoch definition. A missing restored input removes stale checkout copies.

For a beginner: inspect the daily manifest first, then broker alignment and
logical-order fill counts, then price quality. An inexpensive fill cannot prove
that all intended orders were submitted. After collecting at least 20 measured
fills across three sessions, review costs with complete acceptance records.

To verify these changes locally, run `python -m pytest
tests/test_execution_continuity.py tests/test_evidence_publication.py -q` on one
line. The first replays saved September 3 fills; the second runs the actual
publication shell blocks against disposable Git repositories. Neither trades.

Deployment changes protected workflow/reporting files. Commit and review the
repair, then deliberately refresh the paper version lock before enabling the
new release. Preserve the original epoch's evidence and record the reporting
change; do not interpret a refreeze as new trading observations.

## Using the corrected data and submission guards

The beginner workflow remains scan → research → train → predict → backtest → paper trade. Research supplies dated price observations; training and prediction build candidate scores; backtests check historical outcomes; the approved paper path converts validated signals into broker orders.

The September 2026 repair changes how evidence is checked at those handoffs:

1. Keep all decision-date candidates, including stocks whose future prices are missing. Rank first; validate the selected holdings' outcomes second. A missing required price stops the backtest with a ticker/date error. Restore and investigate the source data before retrying.
2. Count NYSE trading sessions, including exchange holidays correctly. Each forward label carries its actual entry and end timestamps. Exclude an unfinished evaluation period as a whole; never discard an individual candidate because its future is unavailable.
3. Confirm trend and volatility changes with a full consecutive-observation window. An isolated opposite observation leaves the previous confirmed state unchanged.
4. Align cached ETF observations separately for each request and use only prices already known by that date. Missing data must not become a made-up flat price or a future price copied backward.
5. Immediately before every rebalance submission stage, check the actual bid/ask and broker timestamp. Skip unsafe first attempts; retain partial fills and record why an unsafe replacement was blocked.

These choices protect the distinction between information known when choosing a trade and outcomes observed later. They do not complete the separate holdings/cash ledger repair or establish corrected profitability. Preserve earlier reports as superseded evidence; regenerate historical evaluations and live signals under the corrected code before relying on them. Code fixes go to `main`; generated operational evidence continues through the existing `signals/latest` publisher.

Offline verification: `python -m pytest tests/test_submission_history_guards.py tests/test_brokers.py tests/test_audit_three_fixes.py -q`, followed by `python -m pytest -q`. Fake brokers and synthetic prices exercise the repaired rules without submitting paper orders.

## Audit first, research next

1. Fetch current operational evidence, then run `python3 corrected_audit.py --evidence-report`.
   Read the JSON/Markdown source identities and blocker actions. Local evidence,
   published evidence and workflow artifacts are separate snapshots.
2. Recover independent interval balances and complete activity/order history using
   the recovery tool. Supply verified historical constituents, raw prices, actions
   and dated context; do not substitute current constituents or invented fees.
3. Rerun corrected evaluation at the actual deployment ceiling. Keep deployed
   factor scoring separate from the unapproved raw-feature shadow candidate.
4. Run `python3 corrected_audit.py --ablations --spec corrected_shadow_spec.json`.
   Review the seven fixed comparisons and uncertainty before proposing indicators.
5. Continue to use existing freeze/observe gates. Neither new command starts an
   epoch, changes paper settings, sends orders or approves capital.

Each script has a separate beginner guide: `doc_evidence_audit.md`,
`doc_edge_ablation.md`, `doc_corrected_audit.md` and `doc_audit_evidence_recovery.md`.
The design favors explicit missing-evidence reports over misleading performance
claims. Code goes to main; sanitized operational reports go to signals/latest.

A daily run on a closed NYSE date is a successful no-action run. Its dedicated
artifact contains only the current skip log and heartbeat; it does not replace
`signals/latest` or refresh old trading reports. The complete trading-manifest
requirement still applies to actual trading runs and unexpected failures.

Both Linux and Windows CI install the YAML test dependency through yamllint,
so workflow publication/holiday-summary regressions run on both platforms.

## Historical evidence recovery and error handling

Before corrected research, run `python corrected_audit.py --audit-only`. Resolve
membership identity and dated coverage first, then enumerate the entire stock
universe and recover raw session prices and corporate actions. Price-file hashes
alone do not prove coverage; the gate checks contents and session dates. A
community list needs corroboration and hash-bound coverage evidence before use.

Read-only recovery supports `--price-probes` and `--action-probes`, follows every
market-data page and keeps candidates private. Payment dates, delisting outcomes
and dated sector/earnings context must remain unknown when sources do not supply
them. Never fill these with today's facts or shorten the agreed evidence period.

The evidence report preserves precise missing-session counts and replay scope.
A zero-fill holiday interval proves only balance continuity. Source hardening
may increase the reported blocker count by exposing previously hidden problems;
that does not indicate new trading losses. No recovery action changes paper
settings, places orders, starts a freeze, or relaxes the 252-session/20-cohort
requirement. Free external evidence that remains unavailable is an explicit
blocker, not an unfinished code exception to conceal.

### Review membership boundaries before importing history

Use `membership_reconciliation.py` with saved candidate snapshots and
`research_evidence/membership_primary_facts.json`. It produces a per-interval
review dataset and a priority list of disputed symbols. Primary announcements
can support individual entry/exit boundaries while the baseline, intervening
changes and security identity remain unresolved. CIK observations and issuer
names are not blanket raw-price identity attestations. Never import the review
queue as approved production membership. Keep merger completion, trading
availability and index deletion as separate dated events when they differ.

### Review complex security changes before correcting history

The membership review now reads `research_evidence/security_transition_facts.json`
and publishes a separate security-transition JSON report plus readable findings
in its Markdown companion. Start with the supported event boundaries, distinguish
issuer/share class from ticker, and separate legal, trading, and index dates.
For Fox, retain the residual old shares and the one-session index overlap. For
IR, retain TT and the new IR distribution as distinct holdings. Confirm raw-price
identifiers and settlement evidence before implementing any ledger corrections.
A matching ticker or reviewed announcement is still not full historical approval;
no freeze, orders, or prospective observations follow from this offline review.

### Track closure without erasing missing evidence

Run recovery first, then membership/transition reviews and the fixed ablation
attempt, and finally the unified evidence report. Use the same explicit saved
review directory while preserving every report's own commit, date and hashes.
The shadow workflow publishes the persistent gap register alongside blocked
reports. The post-market workflow builds a separate execution-observation
manifest; it never borrows a daily run ID. Intentional daily skips produce an
attributed receipt. The frozen version remains unchanged until a separate,
explicitly authorized validation restart. New observations cannot backfill the
252-session or 20-independent-cohort requirement.

Primary payment-date matches are narrower than complete corporate-action
coverage. Raw provider records stay private and immutable; individually reviewed
corrections remain separately attributed until all dependent source gates pass.

Approval repair follows the evidence chain: rebuild from matching source
results, publish the bundle's actual decision consistently at both live-index
levels, and recheck runtime identity. A rejection remains a rejection. Daily
loading and the audit use the same validator; neither changes the frozen lock.

Daily cache recovery first distinguishes stale price/quality inputs from a stale
feature-health profile. If prices and quality already pass, it rebuilds only the
small derived profile and rechecks the strict gate. This prevents a checkout-time
metadata discrepancy from triggering an hour-long price refresh and missing the
09:35–10:30 New York submission window. Genuine input failures retain the full
recovery path; execution-window and portfolio-alignment thresholds are unchanged.
