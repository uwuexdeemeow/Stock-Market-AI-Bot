# Stock-Market-AI-Bot

A paper-trading quant system that runs itself on GitHub Actions and trades an
[Alpaca](https://alpaca.markets) paper account. It began as a machine-learning
stock picker. Most of the work since has gone into **testing ideas honestly and
retiring the ones that fail**, including two of its own strategies.

> Paper trading only. Nothing here is investment advice, and no real money is traded.

## The story so far

**1. An XGBoost stock picker that didn't work.** The first version trained
XGBoost models to predict which stocks would rise. Walk-forward tests, where the
model only ever sees data from before the period it's judged on, gave an AUC of
about **0.51**: no better than a coin flip. The model was switched off for picks.

**2. A factor-score strategy, retired by its own rules.** Next came a
"core-satellite" strategy that ranks stocks by factor scores and rebalances on a
fixed 20-day calendar. It paper traded for weeks. Every approval expires after
45 days and must be earned again on fresh data. At its October 2026
revalidation, only 8 of 14 years passed the cost-stress check (the rule needs
60%). Re-running the old code on new data showed why: a top-3 portfolio flips
its picks on tiny price differences, so the result was mostly noise. The gate
was not lowered, and the strategy stopped trading.

**3. Rule V, a volatility-targeted index rule (current).** The paper account now
trades a simple rule: at each month end, hold QQQ at a weight of
min(1, 20% ÷ QQQ's recent 63-day volatility), and put the rest in BIL (T-bills).
It passed a test whose pass/fail rules were written before it ran:

| 2007–2022 | Yearly return | Sharpe | Worst fall |
|---|---|---|---|
| Buy and hold QQQ | 12.7% | 0.61 | −53.4% |
| **Rule V** | 11.6% | 0.64 | **−38.1%** |

It also passed a one-time exam on 2023–2026 data that was held back until the
end (27.5% a year against QQQ's 31.7%, with a similar worst fall). A
trend-following rule tested alongside it protected more in crashes but failed
that exam, and was not re-tested with other settings. Rule V is not crash-proof:
it still lost about 30% in 2008 and 23% in 2022.

## How ideas are judged

- **Rules first, results second.** Every hypothesis is written down with its
  pass/fail gates before any run. Eleven pre-registered hypotheses are logged
  in [`Documentation/DELAY_STRESS_PAPER_ADVISORY.md`](Documentation/DELAY_STRESS_PAPER_ADVISORY.md),
  and most ended in rejection.
- **Walk-forward testing**, with held-back exam periods that are used once.
- **Stress checks:** survivorship bias (stocks that later disappeared),
  execution costs, trading one day late, and rebalance-day luck.
- **A forward test written before the first trade.** Rule V is judged on live
  paper results against buy-and-hold QQQ, with fixed pass/fail rules and a
  checkpoint after 12 monthly rebalances
  ([`Documentation/V_RULE_FORWARD_TEST.md`](Documentation/V_RULE_FORWARD_TEST.md)).
- **Locked code.** The files that place trades are fingerprinted
  (`paper_version_lock.json`), and a CI check flags any change to them that
  wasn't deliberately re-locked.

## How it runs

- **Daily pipeline** (`daily_run.py`): refreshes and validates market data,
  generates the signal, sends two-stage limit orders to Alpaca, then reconciles
  fills and builds a health report. If a critical step fails, trading steps are
  skipped rather than trading on stale data.
- **GitHub Actions** runs the daily trade, data refreshes, execution-quality
  reports, a forward-test monitor and a watchdog that checks the other
  workflows ran. They're started by an external scheduler through
  `workflow_dispatch`.
- **Safety:** drawdown and one-day loss halts, order guards, and checks against
  duplicate or unexplained orders.

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch and Hugging Face
Transformers (FinBERT news sentiment, from the earlier ML stage), the Alpaca API,
GitHub Actions, pytest.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca paper keys
python -m pytest tests/     # run the test suite
python daily_run.py         # run the daily pipeline once
```

Each script has a plain-language guide in [`Documentation/`](Documentation/).
The original ML scripts (`train.py`, `predict.py` and others) are kept because
live code still imports parts of them, but they no longer pick trades.
