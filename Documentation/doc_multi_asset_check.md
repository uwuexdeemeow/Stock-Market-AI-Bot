# multi_asset_check.py — H-multi-asset, a stocks/bonds/gold mix vs rule V

- **Script:** `research_evidence/multi_asset_20261025/multi_asset_check.py`
- **Rules:** "Hypothesis H-multi-asset" in
  `Documentation/DELAY_STRESS_PAPER_ADVISORY.md`, written before the script ran.

## What it does, in plain language

It asks whether a mix of three funds that often don't fall together
(QQQ for stocks, TLT for long US Treasuries, GLD for gold) beats rule V,
the QQQ + cash rule the paper account trades:

- **V (reference):** at each month end, hold less QQQ when it has been
  jumpy (QQQ = 20% ÷ its recent volatility, at most 100%); the rest goes
  in BIL (a T-bill fund, basically cash).
- **M:** give each fund a weight of 1 ÷ its own volatility, so calmer
  funds get more. Then scale the whole mix so it swings about 20% a year,
  never more than 100% invested.
- **ML:** the same mix, but it may borrow up to 2× the money to reach the
  20% swing. The loan costs BIL's rate + 2 points a year.

Trades happen at the next day's opening price and cost 0.05%. A stress
version trades a day later, pays 0.25%, and borrows at BIL + 4 points.

**Result (2026-10-25):**
- **M passes.** It is much smoother (worst fall −25% vs −38%) but earns
  less (9.0% vs 11.5% a year over 2007–2022).
- **ML fails.** Its extra return came only from 2007–2012 and did not
  survive the stress test.

Details are in the advisory doc.

## How to run

From the project root:

```bash
python research_evidence/multi_asset_20261025/multi_asset_check.py
```

- **Reads:** `data/QQQ.parquet`, `data/TLT.parquet`, `data/GLD.parquet` and
  `data/BIL.parquet` (read only). They were checked against Alpaca's
  adjusted bars first; see "D1" in the advisory doc and `raw/`.
- **Output:** `research_evidence/multi_asset_20261025/multi_asset_check.json`:
  - all numbers, gates, sub-periods, yearly returns and crash years;
  - the exam, for ideas that pass.
- **Engine check:** before any mix number is computed, the script checks
  that its own V matches the older H-index-risk V. If not, it stops.

Tests: `python -m pytest tests/test_multi_asset_check.py -q`.

## Key terms

- **Inverse-volatility weights:** each fund gets a weight proportional to
  1 ÷ how much it swings, so a fund that swings half as much gets twice
  the weight.
- **Volatility target:** the mix is scaled up or down so its expected
  yearly swing is about 20%, the same target V uses.
- **Leverage / gross exposure:** holding more than 100% of your money in
  funds, using a loan. A gross of 1.9 means $1.90 invested for every $1.
- **CAGR, Sharpe, max drawdown:** yearly growth rate; return above cash per
  unit of swing; biggest fall from a high point.
- **Sub-periods:** the decision window cut into three parts, to check
  that a win wasn't just one lucky stretch.
- **Exam:** a final check on 2023–2026, used once and only for an idea
  that passed.
