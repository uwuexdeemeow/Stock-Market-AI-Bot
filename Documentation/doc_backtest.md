# backtest.py — What It Does and How to Run It

## What This Script Does (Plain English)

`backtest.py` is the **time machine**. It pretends we're back in the past and runs the full trading strategy day-by-day through historical data — but never lets the model see the future.

This is how we check whether the strategy would have made money **before** risking real capital. The results include realistic trading costs (commissions + slippage) and compare against a SPY buy-and-hold benchmark.

**Output:** `signals/backtest_results.csv` and printed metrics (Sharpe, drawdown, hit rate).

---

## How to Run It

```bash
# Run walk-forward backtest on all trained tickers
python backtest.py

# Backtest a single ticker
python backtest.py --ticker AAPL

# Stress test with 2× slippage (conservative scenario)
python backtest.py --stress

# Choose trading mode
python backtest.py --mode long_only          # only buy
python backtest.py --mode long_short         # buy and short
python backtest.py --mode long_only_bear_cash  # go to cash in bear markets
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Walk-forward** | The model is only trained on data before the date it predicts. At each test block, it re-trains from scratch on all available history. No peeking at the future. |
| **Test block** | A 126-trading-day (~6 month) window that the model has never seen during training. |
| **Sharpe ratio** | Risk-adjusted return. A Sharpe of 1.0 means you earned 1 unit of return per unit of risk. Above 1.0 is good; above 1.5 is very good. |
| **Sortino ratio** | Like Sharpe but only penalizes downside volatility (bad surprises), not upside. |
| **Max drawdown** | The worst peak-to-trough loss during the backtest period. -15% means you'd have been down 15% from your peak at some point. |
| **Hit rate** | % of trades that made money. 55%+ with proper sizing can be profitable. |
| **Slippage** | The difference between the price you expected and the price you actually got. Modeled as 0.10% per trade by default. |
| **Commission** | Broker fee: $0.005 per share. |
| **Stress mode** | Doubles slippage to 0.20% to simulate adverse conditions. If the strategy is still profitable, it's more robust. |
| **Embargo** | 5-day gap between train end and test start, matching `train.py` exactly. |
| **Signal quality buckets** | Confidence ranges (e.g., 60–65%, 65–70%, etc.) for which we track empirical precision. Used to set the live trading threshold. |

---

## How It Connects

```
research.py → data/<TICKER>.parquet
                    ↓
               backtest.py (re-trains internally on each fold)
                    ↓
               signals/backtest_results.csv
```

The backtest is the **gate**. If net Sharpe < 1.0 or alpha vs SPY is not statistically significant, do not proceed to live trading.
