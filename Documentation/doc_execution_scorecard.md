# execution_scorecard.py - Alpaca Execution Scorecard

The scorecard keeps samples in `collecting` until 20 measured rebalance fills
exist. It reports median, mean, p75, worst fill, a 95% confidence interval,
coverage, missing-measurement reasons, and advisory breakdowns by symbol, side,
stage/type, spread, quote age, and fill latency. Protective stops stay separate.
Calibration is advisory and cannot change live parameters.

## What This Script Does

`execution_scorecard.py` grades how well Alpaca paper orders are being executed.

Plain meaning: the bot already records planned orders, submitted orders, fills,
slippage, skipped orders, and post-fill reversals. This script turns those raw
records into one JSON scorecard that says whether execution quality is passing,
failing, or still collecting enough data.

Schema version 2 keeps a separate denominator for slippage, 15-minute moves,
and 60-minute moves. A fresh fill with no 60-minute observation is shown as
unmeasured instead of being silently counted as a good outcome. The scorecard
also exposes coverage and a `decision_eligible` flag. Order sizing may use a
failed scorecard only when that flag is true.
It scores only normal rebalance orders. Protective trailing-stop exits are
reported in their own section because a stop intentionally fires during fast
price movement and should not make entry execution look broken.

It checks:
- average slippage in basis points
- bad-slippage rate
- fill rate
- skipped-order rate
- whether enough measured fills and trading sessions exist for a verdict
- Stage 1 versus Stage 2 slippage and latency from individual broker fills

The 15-minute and 60-minute adverse-movement rates remain visible as entry-timing
advice. They do not fail execution because later market direction is different
from the quality of the fill itself. The trading script reads the scorecard for
audit context but no longer silently changes approved portfolio quantities.

## How To Run It

```bash
python execution_scorecard.py
```

Print the full JSON:

```bash
python execution_scorecard.py --json
```

Fail the command when the scorecard status is `fail`:

```bash
python execution_scorecard.py --strict
```

## Inputs

| File | Purpose |
|---|---|
| `signals/alpaca_paper_log.csv` | Submitted, skipped, failed, and filled order rows |
| `signals/alpaca_slippage_reversal_report.json` | Rich slippage and post-fill reversal stats from Alpaca fills |
| `signals/alpaca_execution_scorecard.json` | Previous scorecard, used to compute delta versus the last run |

## Outputs

| File | Purpose |
|---|---|
| `signals/alpaca_execution_scorecard.json` | Latest execution scorecard for dashboard/runbook review |
| `logs/alpaca_execution_scorecard_YYYYMMDD.json` | Dated snapshot for audit history |

## Key Terms

| Term | Simple meaning |
|---|---|
| Fill | Alpaca completed all or part of an order |
| Slippage | Difference between expected price and actual fill price |
| Bad slippage | Slippage worse than the material bad-slippage threshold. Tiny unfavorable fills are still reported as raw bad slippage, but they do not fail the scorecard by themselves |
| Fill rate | Filled orders divided by orders accepted by Alpaca |
| Skipped rate | Orders the bot intentionally did not submit divided by planned log rows |
| Adverse reversal | Price moved against the trade after the fill |
| Protective stop | A broker-side safety exit, scored separately from rebalances |
| Collecting | Fewer than 10 measured rebalance fills or three sessions |
| Warning | Hard gates pass, but average slippage exceeds 5 bps or the material-bad rate exceeds 40% |

## Environment Knobs

| Variable | Default | Meaning |
|---|---:|---|
| `EXECUTION_SCORECARD_MAX_AVG_SLIPPAGE_BPS` | `10` | Max acceptable average slippage |
| `EXECUTION_SCORECARD_MAX_BAD_SLIPPAGE_RATE` | `0.60` | Max share of fills with bad slippage |
| `EXECUTION_SCORECARD_BAD_SLIPPAGE_BPS` | `2` | Minimum unfavorable slippage, in bps, before a fill counts as materially bad |
| `EXECUTION_SCORECARD_MIN_FILL_RATE` | `0.80` | Minimum accepted-order fill rate |
| `EXECUTION_SCORECARD_MAX_SKIPPED_RATE` | `0.35` | Max planned-row skip rate |
| `EXECUTION_SCORECARD_LOOKBACK_DAYS` | `30` | How many recent order-log days to grade |
| `EXECUTION_SCORECARD_MIN_DECISION_ORDERS` | `20` | Minimum measured fills required for each decision metric |
| `EXECUTION_SCORECARD_MIN_DECISION_COVERAGE` | `0.80` | Minimum measured share of eligible fills before throttling is allowed |
| `EXECUTION_SCORECARD_WARN_AVG_SLIPPAGE_BPS` | `5` | Warning level for average slippage |
| `EXECUTION_SCORECARD_WARN_BAD_SLIPPAGE_RATE` | `0.40` | Warning level for materially bad fills |
| `EXECUTION_SCORECARD_MIN_REBALANCE_FILLS` | `20` | Measured fills required before a definitive pass/warning/fail verdict |
| `EXECUTION_SCORECARD_MIN_REBALANCE_SESSIONS` | `3` | Distinct sessions required before pass/warning |
| `ALPACA_EXECUTION_SCORECARD_THROTTLE` | `1` | Preserve scorecard audit metadata; quantity shrinking is retired |
| `ALPACA_EXECUTION_SCORECARD_MAX_AGE_HOURS` | `72` | Ignore stale scorecards older than this many hours |

## How It Fits The Workflow

The daily pipeline runs this after:

1. `alpaca_paper_trading.py --submit`
2. `alpaca_paper_trading.py --reconcile`
3. `paper_health.py`

That order matters because the scorecard needs the latest fill statuses and
the latest slippage report before judging execution quality.

## Logical Order Accounting (Schema 3)

Stage 1 (`-a1`) and Stage 2 (`-a2`) are child attempts of one requested
rebalance, so the scorecard groups them under their deterministic parent.

Logical grouping is used only for the overall fill-rate denominator. The Stage
1/Stage 2 price comparison never reads a grouped paper-log average. It uses
only Alpaca API fill rows whose `client_order_id` ends in `-a1` or `-a2`.
Legacy rows and rows that merely claim an `execution_stage` are excluded.
Pending, canceled, expired, partial, and complete broker-accepted orders all
belong in the denominator. `complete_fill_rate` counts only fully completed
parents; `any_fill_rate` separately shows whether any quantity traded.
Protective stops are excluded. Missing identity fields make the scorecard
decision-ineligible instead of silently improving its rates.

## Execution evidence repair (September 2026)
Run `python execution_scorecard.py` to read the paper journal and broker price
report and write `signals/alpaca_execution_scorecard.json`. It places no orders.
An absent journal means unknown fill/skip rates, a null overall score, and no
decision eligibility. Price-only child reports cannot establish stage fill
rates because canceled zero-fill attempts are absent. Stage slippage still
uses individual broker fills, never a blended parent price.

A denominator is the total population used in a rate. A child attempt is one
broker order within an intended parent trade. A basis point is 0.01 percent.


## Remaining audit repair, September 2026

Reports that explicitly mark broker history incomplete or measurement coverage truncated cannot become decision eligible. Pagination and measurement evidence are supplied by the updated paper report; parent/child snapshots must not be counted as separate independent investment decisions. Existing operational safety checks remain independent of the corrected statistical edge review. Use python -m pytest tests/test_execution_scorecard.py -q for the existing scorecard regressions.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.
