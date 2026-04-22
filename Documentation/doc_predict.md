# predict.py — What It Does and How to Run It

## What This Script Does (Plain English)

`predict.py` is the **daily decision maker**. Every trading day, after the market closes, it:
1. Downloads today's fresh price and news data.
2. Loads the trained model for each ticker.
3. Asks: "Given everything we see today, what will this stock do over the next 5 days?"
4. Writes a signal (LONG or SKIP) with a confidence score.

Think of it as the model making its daily picks.

**Output:** `signals/signals.csv` — one row per ticker with direction, confidence, and expected return.

---

## How to Run It

```bash
# Generate predictions for all tickers with trained models
python predict.py

# Predict for one ticker
python predict.py --ticker AAPL

# Show detailed output per ticker
python predict.py --verbose
```

**Expected output:**
- `signals/signals.csv` with columns: `ticker`, `date`, `signal`, `confidence`, `expected_return`, `signal_quality`, `actionable`
- A row with `actionable = True` means the confidence exceeds the threshold and the trade is worth considering.

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Signal** | The model's recommendation: LONG (buy) or SKIP (do nothing). SHORT is disabled by default for safety. |
| **Confidence** | How sure the model is, after calibration. 58% = "slightly above coin flip". 75%+ = strong conviction. |
| **Confidence threshold** | The minimum confidence required to call a signal "actionable". Derived from calibration data (see `confidence_calibration.py`). |
| **Signal quality** | A grade (HIGH / MEDIUM / LOW) based on which confidence bucket the prediction falls into. |
| **Actionable** | True if confidence ≥ threshold AND signal quality is acceptable. Only actionable signals go to paper trading. |
| **Sentiment health** | A check that the news/social sentiment features are actually populated and not all zeros. |
| **Return bin** | The predicted 5-day return category (strong_down / down / flat / up / strong_up). Combined with direction to estimate expected_return. |

---

## How It Connects

```
train.py → models/
               ↓
          predict.py → signals/signals.csv
                              ↓
                    moomoo_paper_trading.py
```
