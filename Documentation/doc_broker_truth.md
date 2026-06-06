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
| `logs/broker_truth_YYYYMMDD.json` | Dated audit snapshot |

Useful options:

```bash
python broker_truth.py --json
python broker_truth.py --strict
python broker_truth.py --offline
```

`--strict` exits nonzero when the report status is `fail`.
`--offline` skips the live Alpaca open-order call and only uses saved files.

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
| `BROKER_TRUTH_REQUIRE_LIVE_ORDERS` | `0` | If `1`, fail when live open/trailing orders cannot be read |

## Daily Pipeline

`daily_run.py --alpaca` runs `broker_truth.py` after `execution_guard.py` and
before `paper_health.py`.  That order matters because the guard may repair
stops, and broker truth should inspect the account after those repairs.
