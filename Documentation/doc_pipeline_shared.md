# pipeline_shared.py — What It Does and How to Use It

## What This Script Does (Plain English)

`pipeline_shared.py` is the **feature factory** — the central library that knows how to turn raw price data into a rich table of signals. Both `research.py` (for training) and `predict.py` (for live predictions) call the same functions here, so the features are always computed the same way.

This is important: if `research.py` computed features differently from `predict.py`, the model would be trained on different data than it predicts with, and performance would degrade silently.

You don't run this script directly. Other scripts import from it.

---

## How to Use It (in Code)

```python
from pipeline_shared import build_research_feature_frame, build_live_features_with_latest_news

# Build historical features for training (downloads data back to TRAIN_START)
df = build_research_feature_frame("AAPL", start="2015-01-01", end="2024-01-01")

# Build today's live features for prediction
df = build_live_features_with_latest_news("AAPL")
```

---

## Key Features Built by This Module

### Technical Features (price-based patterns)
| Feature | What It Measures |
|---|---|
| `ret_1d`, `ret_5d`, `ret_20d` | Price return over 1, 5, 20 days |
| `rsi_7`, `rsi_14` | Relative Strength Index (momentum) |
| `macd`, `macd_sig`, `macd_hist` | Moving Average Convergence/Divergence (trend) |
| `atr_14` | Average True Range (volatility) |
| `bb_pos`, `bb_width` | Bollinger Band position and width |
| `vol_ratio` | Today's volume vs 20-day average |
| `tf_alignment` | How aligned short/medium/long-term trends are |

### Macro Features (market context)
| Feature | What It Measures |
|---|---|
| `vix_level` | Fear index level |
| `vix_ratio` | VIX vs its 20-day average |
| `credit_stress` | HYG bond ETF risk signal |
| `spy_above_ma20` | Is SPY above its 20-day moving average? |
| `gld_risk_off` | Gold momentum (risk-off signal) |

### Sentiment Features
| Feature | What It Measures |
|---|---|
| `sent_finbert` | FinBERT news sentiment score |
| `sent_vader` | Lightweight VADER news sentiment |
| `social_bull_minus_bear` | Reddit/StockTwits net bullish sentiment |
| `social_message_volume` | Unusual social media activity |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Feature frame** | A table where each row = one trading day, each column = one signal/feature |
| **Rolling window** | A calculation that looks at the last N days. E.g., `rolling(14).mean()` = 14-day average |
| **EWM (Exponential Weighted Mean)** | Like a rolling average but recent days count more than old ones |
| **ffill** | "Forward fill" — if a data point is missing, use the last known value. Can introduce leakage if not done carefully |
| **Multi-market features** | SPY, QQQ, VIX, GLD etc. give context about the broad market environment |
