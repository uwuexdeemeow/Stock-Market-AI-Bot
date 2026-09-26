# broker_truth.py - Broker Truth Reconciliation

## What It Does

`broker_truth.py` builds one truth table for the live Alpaca paper account.
It compares:

- target weights from `signals/core_satellite_alpha_signal.csv`
- planned trades from `signals/core_satellite_alpha_orders.csv`
- submitted/fill status from `signals/alpaca_paper_log.csv`
- live broker positions from `signals/alpaca_daily_status.json`
- live open orders and trailing stops from Alpaca when API keys are available

Plain language: this answers "what does the strategy want, what did we plan,
what did we submit, what actually filled, what does Alpaca hold, and are stops
protecting those positions?"

## How To Run

```bash
python broker_truth.py
```

Prints a short summary and writes:

| Output | Meaning |
|---|---|
| `signals/broker_truth.csv` | Per-ticker reconciliation table |
| `signals/broker_truth.json` | Full machine-readable report |
| `signals/alignment_recovery_plan.csv` | Manual-review correction ideas after a settled alignment failure |
| `signals/alignment_incident_ledger.csv` | Permanent history of detected and resolved alignment incidents |
| `logs/broker_truth_YYYYMMDD.json` | Dated audit snapshot |

Useful options:

```bash
python broker_truth.py --json
python broker_truth.py --strict
python broker_truth.py --offline
python broker_truth.py --require-alignment
```

`--strict` exits nonzero when the report status is `fail`.
`--offline` skips the live Alpaca open-order call and only uses saved files.
`--require-alignment` polls live Alpaca for up to 90 seconds and exits nonzero
unless ordinary buy/sell orders are finished and the settled account passes.
It observes only; it never submits repair orders. Use
`--alignment-wait-seconds` and `--alignment-poll-seconds` to override the wait.

## Key Concepts

| Term | Simple Meaning |
|---|---|
| Target weight | Percent of account equity the strategy wants in one ticker |
| Order plan | The buy/sell instructions calculated before submission |
| Paper log | Local CSV record of what was submitted and what filled |
| Broker position | Shares Alpaca says the account currently owns |
| Trailing stop | Broker-side sell order that follows price upward and protects downside |
| Quantity gap | Difference between expected shares from local fills and broker shares |
| Weight gap | Difference between target weight and actual broker weight |

An empty or malformed signal is never treated as a 0% target for every ticker.
In that case, target comparison is disabled, `maximum_target_weight_gap` is
`null`, and the report records `signal_has_no_target_weights`. This avoids a
false 60% mismatch when the account simply has a 60% QQQ position.

The JSON summary also records the parsed target map, target and broker gross
exposure, and the maximum target-weight gap for quick checking.

## Status Meanings

| Status | Meaning |
|---|---|
| `pass` | No important mismatch found |
| `warning` | Something needs review, but it may be explained by pending fills or offline data |
| `fail` | A serious gap exists, such as failed latest order or missing required stop |
| `collecting` | Not enough broker status data exists yet |

## Environment Variables

| Variable | Default | Meaning |
|---|---:|---|
| `BROKER_TRUTH_QTY_TOLERANCE` | `0.001` | Ignore tiny share-count differences |
| `BROKER_TRUTH_WEIGHT_TOLERANCE` | `0.02` | Warn when broker weight differs from target by more than 2 percentage points |
| `BROKER_TRUTH_GROSS_EXPOSURE_TOLERANCE` | `0.05` | Maximum difference between total target and actual exposure |
| `BROKER_TRUTH_ALIGNMENT_WAIT_SECONDS` | `90` | Maximum enforcing wait for ordinary open orders |
| `BROKER_TRUTH_ALIGNMENT_POLL_SECONDS` | `5` | Delay between live alignment checks |
| `BROKER_TRUTH_REQUIRE_LIVE_ORDERS` | `0` | If `1`, fail when live open/trailing orders cannot be read |

The canonical JSON `summary.alignment` status is `pass`, `pending`, `fail`, or
`collecting`. Protective stop and trailing-stop orders remain visible in the
report, but they do not hide drift or keep alignment pending.

## Safe Alignment Recovery Plan

When settled live positions fail the weight or gross-exposure limits, the
script writes one manual-review row per mismatched ticker to
`signals/alignment_recovery_plan.csv`. It includes buy/sell direction, dollar
correction, target/current weights, a reference price when available, and a
suggested quantity. Quantities from the current matching order plan are marked
as such; otherwise they are estimates from the saved broker snapshot. If no
safe price exists, quantity stays blank and requests a fresh quote.

The plan never calls Alpaca and every row is marked
`manual_review_required_not_submitted`. Pending orders produce no recovery
rows, preventing duplicate trades. A passing report rewrites the file with
headers only so old suggestions cannot be mistaken for current work.

## Alignment Incident Ledger

`signals/alignment_incident_ledger.csv` keeps one durable row per incident.
Repeated polls update the same open row instead of creating duplicates. It
records the signal identity, first/latest reason, initial/maximum/latest gaps,
pending-order count, recovery-plan row count, whether human review was needed,
and whether the incident resolved. A later live `pass` records resolution time
and recovery duration. Missing evidence does not silently close an incident.

The ledger is audit-only. Its `orders_submitted` field is always `False` and no
code in this workflow executes ledger or recovery-plan rows. Only the
post-trade `--require-alignment` check updates this lifecycle; pre-submit and
health-only reports cannot open incidents or clear recovery evidence.

## Daily Pipeline

`daily_run.py --alpaca` runs `broker_truth.py` after `execution_guard.py` and
before `paper_health.py`.  That order matters because the guard may repair
stops, and broker truth should inspect the account after those repairs.

## September 2026: hold days of the tested 20-day calendar

Since audit fix H1, the paper account trades once per 20-day period and then
holds. On hold days the weights drift away from today's targets as prices
move, and today's signal may even name different stocks. That is the tested
strategy, not a fault.

- **Hold day:** the signal's `last_scheduled_rebalance_date` was already
  recorded in `signals/paper_rebalance_period.json` by an **earlier** run.
  The alignment verdict is `pass` with reason
  `hold_day_weights_drift_by_design`. The drift is still reported in
  `maximum_target_weight_gap`, but `weight_gap_enforced` is `false`, and the
  per-ticker target-gap warnings are skipped.
- **Rebalance day:** the period was recorded by **this** run
  (`STOCKBOT_RUN_ID`), or not recorded at all. Weights must match the targets
  within the usual tolerance, as before. A failed rebalance still fails.
- Failed orders, stuck orders and stop checks are handled the same way on
  every day.
- The summary carries `alignment.rebalance_calendar` with the reason.

**Retired stops:** the "stop required" check follows the real switches
(`GUARD_CORE_STOP`, `ALPACA_TRAILING_STOP`, both off by default since
2026-09-26). Before, QQQ/SPY always required a stop, so every day would have
failed with `required_trailing_stop_missing`.

Tests: `python -m pytest tests/test_hold_day_monitors.py -q`.
