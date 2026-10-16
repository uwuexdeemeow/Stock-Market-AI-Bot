# v_rule_helper.py — this month's QQQ / cash split (rule V)

## What it does, in plain language

H-index-risk tested a simple way to hold QQQ with smaller crashes, and rule
**V** passed (worst fall 2007–2022: −38% instead of −53%, for about 1 point
a year less return). This script does the monthly arithmetic for you:

1. It downloads QQQ's recent daily prices.
2. It measures how jumpy QQQ was over the last 63 trading days
   (volatility, as a yearly %).
3. QQQ share = 20% ÷ that volatility, never more than 100%. The rest is
   cash (BIL, a T-bill fund).

It never trades. Using it with real money is your own decision.

## How to run

From the project root (needs internet):

```bash
python v_rule_helper.py                  # print the split
python v_rule_helper.py --amount 10000   # also in dollars
```

**When:** once a month, on the last trading day after the close (or the
next morning before the open). Then rebalance at the open, as in the test.
Running it mid-day uses a partial day's price; that's fine for a quick
look, but use the month-end run for the real decision.

**Example output:** "QQQ volatility 18.2% a year → QQQ share 100%, cash 0%".
At 25% volatility it would say 80% QQQ, 20% cash.

Tests: `python -m pytest tests/test_v_rule_helper.py -q` (checks it gives
exactly the same split as the tested rule).

## Key terms

- **Volatility:** how much the price jumps around, as a yearly percentage.
  Calm markets: about 12–18%. Stressed markets: 30%+.
- **Volatility target:** holding less of something when it gets jumpier, so
  the swings of your total money stay near the target (here 20%).
- **BIL:** an ETF of 1–3 month US Treasury bills; basically cash that pays
  interest.
- **Rebalance:** trading back to the target split.
