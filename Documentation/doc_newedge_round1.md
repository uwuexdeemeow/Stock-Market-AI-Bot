# newedge_round1.py — H-newedge-1, three new-edge ideas in one round

Script: `research_evidence/newedge_20261002/newedge_round1.py`
Tests: `tests/test_newedge_round1.py` (fake data only)
Rules: "Hypothesis H-newedge-1" in `DELAY_STRESS_PAPER_ADVISORY.md`
(pre-registered 2026-10-02, before any run).

## What it does, in plain language

It tests three ideas against the current strategy on the point-in-time
S&P 500 list:

- **W, let winners run:** a held stock stays while its 12-month momentum
  is in the top 30%. Fewer trades, so costs and late fills should hurt less.
- **V, volatility-scaled core:** the ETF part shrinks when QQQ has been
  swinging hard over the last 20 days (never below 30% of normal size, never
  above normal).
- **T, multi-asset trend core:** the ETF part is spread over SPY, QQQ,
  TLT, IEF and GLD. Each asset is held only while its 1-year return beats
  cash (BIL); otherwise that share sits in BIL.

Each idea is switched on by temporarily replacing one helper of the engine
while the script runs. The engine file itself is never edited, so the paper
lock stays valid.

Each idea first has to pass its decision gates on 2013–2022. Only then does
it take the "final exam": the project's normal execution stress test (late
fills, extra costs, the 2023–2026 period), which it must pass with no
exceptions.

## How to run it

```bash
python refresh_etf_data.py --symbols TLT --refresh   # once, T needs TLT
python research_evidence/newedge_20261002/newedge_round1.py
# smoke test only (not a result):
python research_evidence/newedge_20261002/newedge_round1.py --offsets 2 --no-exam
```

It makes 220 engine runs (about 15 minutes), plus up to 3 exams.
Output: `research_evidence/newedge_20261002/newedge_round1.json`. Exam
reports go to `logs/research_candidate_execution_stress_newedge_<idea>.*`.

## Key terms

- **Sharpe ratio:** return divided by how bumpy it was; higher is better.
- **Max drawdown:** the worst drop from a previous high.
- **Trend following:** hold an asset only while it has been going up.
- **Volatility:** how much prices swing from day to day.
- **Decision window / final exam:** ideas are judged on 2013–2022 first.
  2023–2026 is touched only once, at the end, so it can't be tuned to.

## Round 2: `newedge_round2.py` (H-newedge-2)

Script: `research_evidence/newedge_20261002/newedge_round2.py`
(tests: `tests/test_newedge_round2.py`). Round 2 changes the stock score
itself, three ways:

- **Q, low-volatility filter:** each day the most volatile third of stocks
  (by 1-year stock-specific volatility) can't be picked.
- **M, smoothed score:** each stock's score is averaged over its last 5
  days, so one noisy day matters less.
- **N, sector-neutral score:** stocks are ranked within their own sector,
  and at most 1 stock per sector is held.

Run it the same way:
`python research_evidence/newedge_20261002/newedge_round2.py --membership-csv PATH`.
It makes 200 engine runs plus up to 3 exams, and writes
`research_evidence/newedge_20261002/newedge_round2.json`. For the exam, the
score change is applied to the stress test's own panel by swapping
`_ensure_robust_score_columns` while it runs.
