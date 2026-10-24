# V rule Alpaca cross-check (`research_evidence/v_rule_alpaca_crosscheck_20261025/`)

## What it does, in plain language

These scripts rebuild rule V from scratch, on a second data source. Rule V
holds QQQ at 20% ÷ its recent volatility, with the rest in BIL, rebalanced
once a month. The rebuild then compares its answer with the project's own
code. If the two agree, the project's V numbers are not caused by a bug or
by bad data.

| File | What it does |
|---|---|
| `fetch.sh` | Downloads daily QQQ and BIL prices (2016 onward) and the NYSE calendar with the **Alpaca CLI**, into `raw/`. |
| `run.py` | A standalone V backtest. It imports no project code. It trades at the next day's open, pays 5 bps a side plus Alpaca's regulatory fees, and writes `equity.csv`, `trades.csv`, `signals.csv`, `summary.json` and the rest. |
| `compare.py` | Runs the project's V test code (`index_risk_check.py`) on both data sources. It compares month-end weights one by one, then whole results, then the live signal's calendar. It writes `compare.json` and `compare_weights.csv`. |
| `report.md` / `notes.md` | The results and every assumption. |

## How to run

```
# 1. Paper keys in ALPACA_API_KEY / ALPACA_SECRET_KEY; Alpaca CLI installed
ALP=alpaca bash research_evidence/v_rule_alpaca_crosscheck_20261025/fetch.sh
# 2. Backtest (run from inside the folder)
cd research_evidence/v_rule_alpaca_crosscheck_20261025 && python run.py
# 3. Cross-check (run from the project root)
python research_evidence/v_rule_alpaca_crosscheck_20261025/compare.py
```

**Expected output:** `run.py` prints V's metrics and the benchmarks.
`compare.py` prints:
- the weight gaps (about 0.0000005 for the code and about 0.0009 for the data);
- three nearly equal performance rows;
- the next signal day (2026-10-30) and trade day (2026-11-02).

## Key terms

- **Backtest:** replaying a rule on old prices to see what it would have done.
- **Volatility:** how jumpy daily returns are, scaled to a yearly number.
- **Next open fill:** the rule decides at a day's close and trades at the next day's opening price. The signal can only use prices that were already known.
- **Slippage:** paying a little worse than the quoted price when you trade.
- **Look-ahead bias:** using a price that wasn't known yet when the decision was made. That would make a backtest look better than reality.
- **Cross-check:** two separate builds getting the same answer, which is evidence that neither has a bug.
- **Data fingerprint:** a count and hash of the input data, so a rerun can prove it used the same prices.

Backtests are hypothetical and are not investment advice.
