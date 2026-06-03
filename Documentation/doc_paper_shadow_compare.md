# paper_shadow_compare.py - Paper vs Shadow Comparator

## What It Does

This script compares the real Alpaca paper-trading account against the shadow
paper journal.

It reads:

- `signals/alpaca_paper_equity.csv`
- `signals/shadow_paper_equity.csv`

Then it writes:

- `signals/paper_shadow_compare.csv`
- `signals/paper_shadow_compare.json`

The comparison uses percentage return from the first date both curves can
fairly compare.  This matters because Alpaca and shadow may not start on the
same date or update at the exact same time.

## How To Run

```bash
python3 paper_shadow_compare.py
```

Custom paths:

```bash
python3 paper_shadow_compare.py \
  --alpaca-equity signals/alpaca_paper_equity.csv \
  --shadow-equity signals/shadow_paper_equity.csv \
  --csv-out signals/paper_shadow_compare.csv \
  --json-out signals/paper_shadow_compare.json
```

Expected output:

- A terminal line showing whether Alpaca or shadow is ahead.
- A JSON summary for dashboard metric cards.
- A CSV table for charting both return curves.

## Key Concepts

**Alpaca paper equity**: The account value reported by Alpaca's paper broker.
This reflects the orders the bot actually submitted.

**Shadow paper equity**: A simulated account value for a candidate config. It
does not send orders.

**Common start date**: The later of the two first dates. Starting both return
curves from this date makes the comparison fairer.

**Return spread**: Alpaca return minus shadow return. Positive means Alpaca is
ahead. Negative means shadow is ahead.
