# risk_sizing.py - What It Does and How to Run It

## What This Script Does

This script turns a trading signal into a safe position size.  It helps answer:
"how much should the bot buy?"

It includes:
- volatility-based sizing, so high-volatility names get smaller weights
- Kelly-style confidence sizing, kept conservative by default
- ATR stop sizing, so one stopped trade risks only a chosen slice of equity

Invalid account equity or invalid risk budgets return zero shares instead of
negative share counts.

## How to Run It

This file is normally imported by backtests and trading code.  For a quick
syntax check:

```bash
python -m py_compile risk_sizing.py
```

Related test:

```bash
python -m pytest tests/test_sanity.py -q
```

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Volatility | How much a ticker usually moves. |
| Position size | How much of the portfolio goes into one trade. |
| ATR | Average True Range, a rough daily movement size. |
| Stop loss | A price where the system exits to limit damage. |
| Kelly sizing | A math formula for sizing bets, used here only conservatively. |
