# social_sentiment.py — What It Does and How to Use It

## What This Script Does (Plain English)

`social_sentiment.py` monitors **StockTwits and X (Twitter)** for what retail traders are saying about a stock right now. Social media often reacts to news hours before it shows up in price data — especially for retail-heavy stocks like TSLA, NVDA, and BTC.

It pulls raw posts/messages, scores them with a finance-specific AI model (FinBERT), and returns a feature table that can be merged into the main feature frame.

---

## How to Use It (in Code)

```python
from social_sentiment import build_social_sentiment_features, get_live_social_signal

# Build historical social features for training
df_social = build_social_sentiment_features("TSLA", dates=price_df.index)

# Get today's live social signal for prediction
signal = get_live_social_signal("TSLA")
# Returns: {"social_combined": 0.42, "social_message_volume": 1.8, ...}
```

---

## Required Environment Variable

Add this to your `.env` file:

```
X_BEARER_TOKEN=your_token_here
```

Get a Bearer Token at https://developer.twitter.com/en/portal/dashboard (free Basic tier is enough).

Without it, X features are skipped and only StockTwits features are used.

---

## Features Produced

| Feature | What It Measures |
|---|---|
| `social_bullish_ratio` | % of StockTwits messages tagged "Bullish" |
| `social_bearish_ratio` | % of StockTwits messages tagged "Bearish" |
| `social_bull_minus_bear` | Net sentiment from -1 (all bearish) to +1 (all bullish) |
| `social_message_volume` | Today's message count vs 30-day average. >1.5 = unusual activity |
| `social_combined` | Composite score: 50% StockTwits + 30% FinBERT text score + 20% X signal |
| `x_mention_count` | Number of recent cashtag tweets found on X |
| `x_sentiment_score` | Engagement-weighted sentiment from X posts (likes + retweets) |

---

## How the Combined Score is Calculated

```
social_combined = 0.50 × StockTwits_bull_minus_bear
                + 0.30 × FinBERT_text_score
                + 0.20 × X_engagement_score
```

- **StockTwits** (50%): Most reliable — users explicitly tag BULLISH or BEARISH
- **FinBERT** (30%): AI reads the actual text of posts and rates them positive/negative
- **X** (20%): Engagement signal (likes + retweets on cashtag posts)

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **StockTwits** | A financial social network where users tag posts BULLISH or BEARISH. Free API, no key needed. |
| **X (Twitter)** | General social network searched for cashtag posts (e.g. `$TSLA`). Requires a Bearer Token. |
| **FinBERT** | A version of the BERT AI model fine-tuned on financial news. Understands finance language better than general sentiment tools. |
| **Cashtag** | A stock ticker prefixed with `$` (e.g. `$AAPL`) — standard notation on X and StockTwits. |
| **Message volume spike** | Sudden surge in social posts about a ticker. Even without sentiment direction, volume alone predicts volatility. |
| **Retail-driven stock** | A stock heavily influenced by small individual investors (vs. institutional funds). Social signals matter more here. |

---

## API Keys

- **StockTwits**: No key needed. Rate limit: ~60 requests/min.
- **X (Twitter)**: Free Bearer Token needed. Get one at https://developer.twitter.com/en/portal/dashboard

Without an X token, the X features are skipped with a debug log message. The model still works — X features get filled with zeros.
