# execution_scorecard.py - Alpaca Execution Scorecard

## What This Script Does

`execution_scorecard.py` grades how well Alpaca paper orders are being executed.

Plain meaning: the bot already records planned orders, submitted orders, fills,
slippage, skipped orders, and post-fill reversals. This script turns those raw
records into one JSON scorecard that says whether execution quality is passing,
failing, or still collecting enough data.

It checks:
- average slippage in basis points
- bad-slippage rate
- fill rate
- skipped-order rate
- adverse 15-minute and 60-minute reversal rates
- whether execution-risk throttled buys are showing up and filling cleanly

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
| Bad slippage | Positive slippage, meaning the fill was worse than expected |
| Fill rate | Filled orders divided by orders accepted by Alpaca |
| Skipped rate | Orders the bot intentionally did not submit divided by planned log rows |
| Adverse reversal | Price moved against the trade after the fill |
| Throttled buy | A buy order reduced because recent fills for that ticker looked risky |

## Environment Knobs

| Variable | Default | Meaning |
|---|---:|---|
| `EXECUTION_SCORECARD_MAX_AVG_SLIPPAGE_BPS` | `10` | Max acceptable average slippage |
| `EXECUTION_SCORECARD_MAX_BAD_SLIPPAGE_RATE` | `0.60` | Max share of fills with bad slippage |
| `EXECUTION_SCORECARD_MIN_FILL_RATE` | `0.80` | Minimum accepted-order fill rate |
| `EXECUTION_SCORECARD_MAX_SKIPPED_RATE` | `0.35` | Max planned-row skip rate |
| `EXECUTION_SCORECARD_MAX_ADVERSE_15M_RATE` | `0.60` | Max adverse 15-minute reversal rate |
| `EXECUTION_SCORECARD_MAX_ADVERSE_60M_RATE` | `0.70` | Max adverse 60-minute reversal rate |
| `EXECUTION_SCORECARD_LOOKBACK_DAYS` | `30` | How many recent order-log days to grade |

## How It Fits The Workflow

The daily pipeline runs this after:

1. `alpaca_paper_trading.py --submit`
2. `alpaca_paper_trading.py --reconcile`
3. `paper_health.py`

That order matters because the scorecard needs the latest fill statuses and
the latest slippage report before judging execution quality.
