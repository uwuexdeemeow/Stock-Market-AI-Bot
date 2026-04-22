# social_sentiment.py — What It Does and How to Use It

## What This Script Does (Plain English)

`social_sentiment.py` monitors **Reddit and StockTwits** for what retail traders are saying about a stock right now. Social media often reacts to news hours before it shows up in price data — especially for retail-heavy stocks like TSLA, NVDA, and GME.

It pulls raw posts/messages, scores them, and returns a feature table that can be merged into the main feature frame.

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

## Features Produced

| Feature | What It Measures |
|---|---|
| `social_bullish_ratio` | % of StockTwits messages tagged "Bullish" |
| `social_bearish_ratio` | % of StockTwits messages tagged "Bearish" |
| `social_bull_minus_bear` | Net sentiment from -1 (all bearish) to +1 (all bullish) |
| `social_message_volume` | Today's message count vs 30-day average. >1.5 = unusual activity |
| `social_combined` | Composite score combining volume and direction |
| `reddit_score` | Upvote/comment-weighted sentiment from relevant subreddits |
| `reddit_mention_volume` | Normalised Reddit mention count vs baseline |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **StockTwits** | A financial social network where users tag posts BULLISH or BEARISH. Free API, no key needed. |
| **Reddit (r/wallstreetbets, r/stocks)** | Communities where retail traders discuss stocks. High activity often precedes short squeezes or viral momentum. |
| **Retail-driven stock** | A stock heavily influenced by small individual investors (vs. institutional funds). Social signals matter more here. |
| **Message volume spike** | Sudden surge in social posts about a ticker. Even without sentiment direction, volume alone predicts volatility. |
| **PRAW** | Python Reddit API Wrapper — the library used to fetch Reddit posts. Requires a free API key. |

---

## API Keys

- **StockTwits**: No key needed. Rate limit: ~60 requests/min.
- **Reddit**: Free key needed. Takes 2 minutes to get at https://www.reddit.com/prefs/apps

Without a Reddit key, only StockTwits features are populated. The model still works — Reddit features get filled with zeros.
