# dynamic_weighting.py — What It Does and How to Use It

## What This Script Does (Plain English)

`dynamic_weighting.py` adjusts how much to trust each model depending on the current **market regime**. In a calm market (low VIX), XGBoost gets a boost. In a crisis (high VIX, bear trend), the neural models get more weight because they've historically been better at recognizing unusual pattern sequences.

Think of it as the ensemble's "weather forecast": calm = trust the stats model; storm = lean on the pattern-recognition models.

---

## How to Use It (in Code)

```python
from dynamic_weighting import infer_regime_from_row, regime_adjust_weights

# Detect today's regime from a feature row
regime, details = infer_regime_from_row(today_features_row)
# regime = "stable" | "defensive" | "crisis"

# Adjust base weights based on the detected regime
adjusted = regime_adjust_weights(
    base_weights={"xgboost": 0.6, "lstm_attention": 0.2, "transformer": 0.2},
    regime_name=regime,
)
```

---

## Regime Definitions

| Regime | Trigger | Effect |
|---|---|---|
| **Stable** | VIX < 25, SPY + QQQ above 20-day MA | XGBoost weight × 1.15 |
| **Defensive** | VIX 25–35, or bear trend | XGBoost weight × 0.90; neural × 1.10 |
| **Crisis** | VIX > 35, or bear + risk-off signals | XGBoost weight × 0.75; neural × 1.20 |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Regime** | The overall character of the current market: calm / cautious / panic. Detected from VIX, price trends, and credit signals. |
| **VIX** | The "fear index" — how much volatility the options market expects over the next 30 days. |
| **Bear regime** | SPY and QQQ are both below their 20-day moving average AND recent returns are negative. |
| **Risk-off** | Investors fleeing risky assets (stocks) into safe havens (gold, Treasuries). Detected via HYG (credit) and GLD (gold) signals. |
| **Credit stress** | HYG (high-yield bond ETF) dropping — businesses can't borrow cheaply. A leading indicator of equity stress. |
| **Weight floor** | `DYN_WEIGHT_MIN_FLOOR = 0.0001` — no model's weight ever goes exactly to zero from regime adjustment alone. |
