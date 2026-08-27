# execution_scorecard.py - Alpaca Execution Scorecard

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

It checks:
- average slippage in basis points
- bad-slippage rate
- fill rate
- skipped-order rate
- adverse 15-minute and 60-minute reversal rates
- whether execution-risk throttled buys are showing up and filling cleanly

`alpaca_paper_trading.py` also reads this file before planning orders. When
the scorecard status is `fail`, BUY orders are reduced by the configured
scorecard throttle, while SELL orders stay full-size so exits are not blocked.

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
| Throttled buy | A buy order reduced because recent fills for that ticker looked risky |

## Environment Knobs

| Variable | Default | Meaning |
|---|---:|---|
| `EXECUTION_SCORECARD_MAX_AVG_SLIPPAGE_BPS` | `10` | Max acceptable average slippage |
| `EXECUTION_SCORECARD_MAX_BAD_SLIPPAGE_RATE` | `0.60` | Max share of fills with bad slippage |
| `EXECUTION_SCORECARD_BAD_SLIPPAGE_BPS` | `2` | Minimum unfavorable slippage, in bps, before a fill counts as materially bad |
| `EXECUTION_SCORECARD_MIN_FILL_RATE` | `0.80` | Minimum accepted-order fill rate |
| `EXECUTION_SCORECARD_MAX_SKIPPED_RATE` | `0.35` | Max planned-row skip rate |
| `EXECUTION_SCORECARD_MAX_ADVERSE_15M_RATE` | `0.60` | Max adverse 15-minute reversal rate |
| `EXECUTION_SCORECARD_MAX_ADVERSE_60M_RATE` | `0.70` | Max adverse 60-minute reversal rate |
| `EXECUTION_SCORECARD_LOOKBACK_DAYS` | `30` | How many recent order-log days to grade |
| `EXECUTION_SCORECARD_MIN_DECISION_ORDERS` | `20` | Minimum measured fills required for each decision metric |
| `EXECUTION_SCORECARD_MIN_DECISION_COVERAGE` | `0.80` | Minimum measured share of eligible fills before throttling is allowed |
| `ALPACA_EXECUTION_SCORECARD_THROTTLE` | `1` | Let order planning shrink BUY orders when this scorecard fails |
| `ALPACA_EXECUTION_SCORECARD_MAX_AGE_HOURS` | `72` | Ignore stale scorecards older than this many hours |
| `ALPACA_EXECUTION_SCORECARD_FAIL_BUY_SCALE` | `0.75` | BUY quantity multiplier when scorecard status is fail |
| `ALPACA_EXECUTION_SCORECARD_SEVERE_SCORE` | `50` | Score at or below this uses the stronger severe multiplier |
| `ALPACA_EXECUTION_SCORECARD_SEVERE_BUY_SCALE` | `0.50` | BUY quantity multiplier for severe scorecard failures |

## How It Fits The Workflow

The daily pipeline runs this after:

1. `alpaca_paper_trading.py --submit`
2. `alpaca_paper_trading.py --reconcile`
3. `paper_health.py`

That order matters because the scorecard needs the latest fill statuses and
the latest slippage report before judging execution quality.
