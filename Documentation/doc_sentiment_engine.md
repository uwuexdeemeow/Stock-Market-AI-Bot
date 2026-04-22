# sentiment_engine.py — What It Does and How to Use It

## What This Script Does (Plain English)

`sentiment_engine.py` reads financial news headlines and assigns a **sentiment score** — a number from -1.0 (very negative) to +1.0 (very positive). These scores become features the model uses.

It offers three quality levels:

| Level | Tool | Accuracy | Speed | Cost |
|---|---|---|---|---|
| 1 | **FinVADER** | ~65% | Very fast | Free |
| 2 | **FinBERT** | ~89–91% | Slow (GPU speeds it up) | Free |
| 3 | **GPT-4** | Best | Slow | ~$0.002/1000 headlines |

Default is FinBERT if available, FinVADER as fallback.

---

## How to Use It (in Code)

```python
from sentiment_engine import SentimentEngine, score_todays_news

# Create an engine (auto-detects best available level)
engine = SentimentEngine(level="finbert")

# Score a single headline
score = engine.score("Apple beats earnings estimates, raises guidance")
# Returns: 0.85 (strongly positive)

# Score today's news for a ticker and return a feature DataFrame
df_sentiment = score_todays_news("AAPL", lookback_days=7)
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Sentiment score** | A number from -1 to +1 capturing the emotional tone of text. Negative = bad news. Positive = good news. |
| **VADER** | A rule-based sentiment tool. Fast but struggles with financial language (e.g., "beats estimates" scores as neutral). |
| **FinVADER** | VADER extended with 7,300 financial terms. Significantly better for earnings/market news. |
| **FinBERT** | A BERT model fine-tuned on financial news. Deep learning — understands context, not just individual words. |
| **Compound score** | FinVADER's single summary score combining positive, negative, and neutral signals. |
| **Rolling sentiment** | Average sentiment over the last 3, 7, 14 days. Smooths out noisy individual headlines. |
| **Sentiment delta** | Change in average sentiment from last week to this week. A sudden drop can signal an upcoming correction. |

---

## Why VADER Fails on Financial Text

VADER was built for Twitter. It doesn't know that:
- "beats estimates" = positive
- "raised guidance" = positive  
- "near historic lows" (about unemployment) = positive for the economy

FinBERT reads the whole sentence as context and gets these right.
