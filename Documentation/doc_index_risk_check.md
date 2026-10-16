# index_risk_check.py — H-index-risk, simple risk rules on QQQ

Script: `research_evidence/index_risk_20261016/index_risk_check.py`
Rules: "Hypothesis H-index-risk" in `Documentation/DELAY_STRESS_PAPER_ADVISORY.md`
(written before the script ran).

## What it does, in plain language

It compares three ways of holding QQQ from mid-2007 to 2022, then checks
any winner once more on 2023 to today:

- **B, buy and hold:** buy QQQ once and keep it.
- **T, trend:** at each month end, hold QQQ if it is above its 200-day
  average, otherwise hold BIL (a T-bill fund, basically cash).
- **V, volatility target:** at each month end, hold less QQQ when it has
  been jumpy. QQQ weight = 20% ÷ its recent yearly volatility, never more
  than 100%; the rest goes in BIL.

Trades happen at the next day's opening price and cost 0.05%. A stress
version trades a day later and pays 0.25%. A rule passes only if it falls
much less in bad times (at least 10 points less), earns as much per unit
of risk, gives up at most 3 points a year, and still holds in the stress
version and in the 2023–2026 exam.

**Result (2026-10-16):** V passes, T fails the exam. See the advisory doc.

## How to run

From the project root:

```bash
python research_evidence/index_risk_20261016/index_risk_check.py
```

- Uses `data/QQQ.parquet` and `data/BIL.parquet` (read only).
- **Output:** `research_evidence/index_risk_20261016/index_risk_check.json`
  with all numbers, gates, yearly returns and crash years. The console
  prints the main table.

Tests: `python -m pytest tests/test_index_risk_check.py -q`.

## Key terms

- **CAGR:** the average yearly growth rate.
- **Max drawdown:** the biggest fall from a high point to a later low.
- **Sharpe ratio:** return above cash divided by how much it swings.
  Higher means more return per unit of risk.
- **Volatility:** how much the price jumps around, as a yearly percentage.
- **Exam:** a final check on years the rule never saw, used only once.
