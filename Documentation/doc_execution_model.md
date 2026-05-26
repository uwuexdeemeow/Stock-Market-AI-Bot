# execution_model.py - What It Does and How to Run It

## What This Script Does

`execution_model.py` estimates the real cost of trading. A backtest that buys at
the exact close price is usually too optimistic, so this script adds spread,
slippage, market impact, and commission assumptions.

It is used by research and backtesting code to answer a practical question:
"Would this trade still look good after realistic trading costs?"

## How To Run It

This file is mainly imported by other scripts:

```bash
python -m pytest tests/test_sanity.py -q
```

Useful functions:

- `realistic_fill_price(...)` estimates a buy or sell fill after costs.
- `commission(...)` estimates broker commission by share count.
- `capacity_warning(...)` flags orders that are large versus daily volume.

Expected output from the test command is a passing pytest run.

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Spread | The gap between the best buyer price and best seller price. |
| Slippage | Extra cost from the fill price moving against you. |
| Market impact | Price movement caused by your own order size. |
| ADV | Average daily volume, or how many shares usually trade in one day. |
| Basis point | One-hundredth of one percent. |
