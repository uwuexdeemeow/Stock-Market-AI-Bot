"""
social_sentiment.py — Social Media Sentiment for Stock Signals
==============================================================
WHAT THIS FILE DOES:
  Pulls real-time sentiment from StockTwits and X (Twitter/formerly Twitter)
  for any ticker. Social media often LEADS the news by hours, especially on
  retail-driven stocks like TSLA, NVDA, and crypto.

WHY SOCIAL MEDIA MATTERS:
  - Financial news (Finnhub, Yahoo Finance) = what professionals SAY
  - Social media = what millions of traders FEEL right now

  StockTwits is especially useful because users explicitly tag posts BULLISH
  or BEARISH — no guesswork needed about sentiment direction.

  X (Twitter) is useful because:
    - Cashtag searches ($NVDA, $GS) surface real-money traders
    - High engagement (likes + retweets) = signal vs noise filter
    - FinBERT can score tweet text for sentiment direction

SIGNALS PROVIDED:
  social_bull_ratio      : % of StockTwits messages tagged Bullish (centred at 0)
  social_bear_ratio      : % tagged Bearish (centred at 0)
  social_net_sentiment   : bull minus bear (-1 to +1)
  social_message_volume  : normalised message activity (>1 = unusual)
  social_text_score      : FinBERT score on social post text
  social_combined        : weighted blend (50% ST tags + 30% text + 20% X)
  x_mention_count        : normalised X mention volume
  x_sentiment_score      : FinBERT score on X tweet text
  social_agrees_news     : +1 if social and news sentiment agree, -1 if conflict

HOW TO GET A FREE X API KEY:
  1. Go to developer.twitter.com → create a project + app
  2. Generate a Bearer Token (read-only access, no user auth needed)
  3. Set: X_BEARER_TOKEN="your_token" in your .env file
  Rate limits: 60 search requests / 15 min, 500k tweets / month (free tier)
  — plenty for 9 tickers at 1-2 live fetches per day.
"""

import logging
import os
import time

import numpy as np
import pandas as pd

log = logging.getLogger("social_sentiment")

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS (from .env — never hardcode)
# ─────────────────────────────────────────────────────────────────────────────

X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STOCKTWITS (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stocktwits_sentiment(ticker: str, max_messages: int = 100) -> dict:
    """
    Fetch the latest StockTwits messages and return sentiment breakdown.

    StockTwits users explicitly tag their posts BULLISH or BEARISH, which is
    more reliable than inferring sentiment from text alone.

    Returns a dict with:
      total_messages   : how many messages we fetched (0 if API fails)
      bullish_count    : messages tagged BULLISH
      bearish_count    : messages tagged BEARISH
      bullish_ratio    : bullish / (bullish + bearish), 0.5 if untagged
      net_sentiment    : bullish_ratio - bearish_ratio  (-1 to +1)
      sample_texts     : up to 20 post bodies (for FinBERT scoring)
    """
    # StockTwits doesn't accept hyphens — BTC-USD → BTC
    st_ticker = ticker.replace("-USD", "").replace("-", "").upper()
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{st_ticker}.json"

    try:
        import requests
        resp = requests.get(url, params={"limit": max_messages}, timeout=10)

        # 429 = rate limited; 403 = ticker not found or API changed — both are
        # non-fatal, just return empty rather than logging a scary warning.
        if resp.status_code in (403, 429):
            log.debug("[%s] StockTwits status %s — returning empty", ticker, resp.status_code)
            return _empty_stocktwits_result()

        if resp.status_code != 200:
            log.warning("[%s] StockTwits returned %s", ticker, resp.status_code)
            return _empty_stocktwits_result()

        messages = resp.json().get("messages", [])
        if not messages:
            return _empty_stocktwits_result()

        bullish = bearish = 0
        texts = []
        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment", {})
            if sentiment:
                basic = sentiment.get("basic", "")
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1
            body = msg.get("body", "")
            if body:
                texts.append(body)

        tagged = bullish + bearish
        if tagged > 0:
            bull_ratio = bullish / tagged
            bear_ratio = bearish / tagged
        else:
            bull_ratio = bear_ratio = 0.5

        return {
            "total_messages": len(messages),
            "bullish_count":  bullish,
            "bearish_count":  bearish,
            "bullish_ratio":  bull_ratio,
            "bearish_ratio":  bear_ratio,
            "net_sentiment":  bull_ratio - bear_ratio,
            "sample_texts":   texts[:20],
        }

    except ImportError:
        log.warning("requests not installed — pip install requests")
        return _empty_stocktwits_result()
    except Exception as e:
        log.debug("[%s] StockTwits fetch failed: %s", ticker, e)
        return _empty_stocktwits_result()


def _empty_stocktwits_result() -> dict:
    return {
        "total_messages": 0, "bullish_count": 0, "bearish_count": 0,
        "bullish_ratio": 0.5, "bearish_ratio": 0.5,
        "net_sentiment": 0.0, "sample_texts": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — X / TWITTER (requires free Bearer Token from developer.twitter.com)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_x_sentiment(ticker: str, max_tweets: int = 100) -> dict:
    """
    Search recent X (Twitter) posts for a ticker cashtag and return raw data.

    Uses the v2 API search endpoint with app-only auth (Bearer Token).
    Filters to English, non-retweet posts so we get original opinions.

    Cashtag format: $GS, $NVDA, $BTC (the leading $ is the financial marker
    that traders use on X to indicate they're talking about a specific stock).

    Returns:
      total_tweets    : number of tweets found
      texts           : list of tweet texts (for FinBERT scoring)
      total_likes     : total likes across all fetched tweets
      total_retweets  : total retweets
      engagement_score: (likes + 2×retweets) / n_tweets (>10 = viral)
    """
    if not X_BEARER_TOKEN:
        log.debug("[%s] X_BEARER_TOKEN not set — X signals disabled", ticker)
        return _empty_x_result()

    try:
        import tweepy

        client = tweepy.Client(bearer_token=X_BEARER_TOKEN, wait_on_rate_limit=False)

        # Build cashtag query — filter out retweets and non-English posts
        st_ticker = ticker.replace("-USD", "").upper()
        # For crypto use both the cashtag and the coin name
        if "BTC" in st_ticker:
            query = "($BTC OR bitcoin) -is:retweet lang:en"
        else:
            query = f"${st_ticker} -is:retweet lang:en"

        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_tweets, 100),  # API max per request
            tweet_fields=["text", "public_metrics"],
        )

        if not response.data:
            return _empty_x_result()

        texts, likes, retweets = [], 0, 0
        for tweet in response.data:
            if tweet.text:
                texts.append(tweet.text)
            metrics = tweet.public_metrics or {}
            likes    += metrics.get("like_count", 0)
            retweets += metrics.get("retweet_count", 0)

        n = len(response.data)
        return {
            "total_tweets":    n,
            "texts":           texts[:30],
            "total_likes":     likes,
            "total_retweets":  retweets,
            "engagement_score": (likes + retweets * 2) / max(n, 1),
        }

    except ImportError:
        log.debug("tweepy not installed — X signals disabled (pip install tweepy to enable)")
        return _empty_x_result()
    except Exception as e:
        # Rate limit, auth error, or network problem — degrade gracefully
        log.debug("[%s] X fetch failed: %s", ticker, e)
        return _empty_x_result()


def _empty_x_result() -> dict:
    return {
        "total_tweets": 0, "texts": [],
        "total_likes": 0, "total_retweets": 0, "engagement_score": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TEXT SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_social_texts(texts: list[str]) -> float:
    """
    Score a list of social media posts with FinBERT (or FinVADER fallback).
    Returns a composite score: -1.0 (very bearish) to +1.0 (very bullish).
    """
    if not texts:
        return 0.0
    try:
        from sentiment_engine import SentimentEngine
        try:
            engine = SentimentEngine(level="finbert")
        except Exception:
            engine = SentimentEngine(level="finvader")
        scores = engine.score_batch(texts)
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        log.warning("Text scoring failed: %s", e)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HISTORICAL DAILY FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_social_sentiment_features(
    ticker: str,
    dates: pd.DatetimeIndex,
    *,
    use_stocktwits: bool = True,
    use_x: bool = True,
    score_texts: bool = True,
) -> pd.DataFrame:
    """
    Build daily social sentiment features aligned to trading dates.

    IMPORTANT LIMITATION:
    Neither StockTwits nor X provides free historical data going back years.
    For historical training data, social features will be 0 for all dates
    except the most recent one (today's live reading).

    This is still valuable:
      1. Live predictions improve significantly (social adds timely signal)
      2. The model learns 0 = "no social data" = neutral (safe default)
      3. When social IS available (live), it differentiates well

    Parameters:
      ticker        : stock symbol e.g. "GS"
      dates         : trading day index from your price DataFrame
      use_stocktwits: set False to skip StockTwits
      use_x         : set False to skip X (or if no bearer token)
      score_texts   : run FinBERT on post text (adds ~3s, improves quality)

    Returns:
      DataFrame indexed to `dates` with columns:
        social_bull_ratio, social_bear_ratio, social_net_sentiment,
        social_message_volume, social_text_score, social_combined,
        x_mention_count, x_sentiment_score, social_agrees_news
    """
    result = pd.DataFrame(index=dates, data={
        "social_bull_ratio":    0.0,
        "social_bear_ratio":    0.0,
        "social_net_sentiment": 0.0,
        "social_message_volume": 0.0,
        "social_text_score":    0.0,
        "social_combined":      0.0,
        "x_mention_count":      0.0,
        "x_sentiment_score":    0.0,
        "social_agrees_news":   0.0,
    })

    if not len(dates):
        return result

    latest = dates[-1]

    # ── StockTwits ─────────────────────────────────────────────────────────
    st = _empty_stocktwits_result()
    if use_stocktwits:
        log.info("[%s] Fetching StockTwits...", ticker)
        st = fetch_stocktwits_sentiment(ticker)
        if st["total_messages"] > 0:
            log.info("[%s] StockTwits: %d msgs | bull=%.0f%% bear=%.0f%%",
                     ticker, st["total_messages"],
                     st["bullish_ratio"] * 100, st["bearish_ratio"] * 100)
            result.at[latest, "social_bull_ratio"]    = st["bullish_ratio"] - 0.5
            result.at[latest, "social_bear_ratio"]    = st["bearish_ratio"] - 0.5
            result.at[latest, "social_net_sentiment"] = st["net_sentiment"]
            # Normalise volume: 50 msgs/day ≈ baseline for most tickers
            result.at[latest, "social_message_volume"] = min(st["total_messages"] / 50.0, 5.0)

    # ── X (Twitter) ────────────────────────────────────────────────────────
    x_result = _empty_x_result()
    if use_x:
        log.info("[%s] Fetching X posts...", ticker)
        x_result = fetch_x_sentiment(ticker)
        if x_result["total_tweets"] > 0:
            log.info("[%s] X: %d tweets | likes=%d retweets=%d",
                     ticker, x_result["total_tweets"],
                     x_result["total_likes"], x_result["total_retweets"])
            # Normalise: 20 tweets with good engagement = above-average signal
            norm_mentions = min(x_result["total_tweets"] / 20.0, 5.0)
            result.at[latest, "x_mention_count"] = norm_mentions

    # ── Text Scoring (FinBERT on combined posts) ────────────────────────────
    text_score = 0.0
    if score_texts:
        all_texts = st.get("sample_texts", []) + x_result.get("texts", [])
        if all_texts:
            text_score = score_social_texts(all_texts[:30])
            result.at[latest, "social_text_score"] = text_score
            if x_result["total_tweets"] > 0:
                result.at[latest, "x_sentiment_score"] = text_score  # X posts dominate the blend

    # ── Combined Score ──────────────────────────────────────────────────────
    # 50% StockTwits explicit tags (most reliable — users state their position)
    # 30% FinBERT text sentiment (captures nuance beyond binary bullish/bearish)
    # 20% X activity (engagement-weighted mention volume as sentiment proxy)
    st_net = float(result.at[latest, "social_net_sentiment"])
    x_engagement = float(result.at[latest, "x_mention_count"])
    # Scale X mention count to [-1, +1] using text score as direction
    x_signal = text_score * min(x_engagement / 2.5, 1.0) if x_result["total_tweets"] > 0 else 0.0

    has_st = st["total_messages"] > 0
    has_x  = x_result["total_tweets"] > 0

    if has_st and has_x:
        combined = 0.50 * st_net + 0.30 * text_score + 0.20 * x_signal
    elif has_st:
        combined = 0.65 * st_net + 0.35 * text_score
    elif has_x:
        combined = 0.75 * text_score + 0.25 * x_signal
    else:
        combined = 0.0

    result.at[latest, "social_combined"] = float(np.clip(combined, -1.0, 1.0))

    # ── social_agrees_news ─────────────────────────────────────────────────
    # +1 if social and news sentiment point the same direction (confirming signal)
    # -1 if they conflict (uncertainty — reduce position sizing or skip)
    #  0 if either is near-zero (no agreement possible)
    # This is only meaningful for live prediction (historical social = 0).
    if abs(combined) > 0.05:
        # news_sentiment from the main feature frame is not available here,
        # so we set a placeholder that pipeline_shared.py can overwrite live.
        # For now, 0 = "no news/social conflict info" (safe neutral default).
        result.at[latest, "social_agrees_news"] = 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — LIVE SIGNAL FOR PREDICTOR (predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_social_signal(ticker: str, use_text_scoring: bool = True) -> dict:
    """
    Get the CURRENT social sentiment for use in live predictions.

    Returns:
      combined         : overall social score -1 (bearish) to +1 (bullish)
      bullish_pct      : % bullish on StockTwits (0-100)
      bearish_pct      : % bearish on StockTwits
      message_volume   : normalised StockTwits activity level
      x_tweets         : X tweet count fetched
      x_engagement     : likes + 2×retweets normalised
      text_score       : FinBERT score on social text
      social_available : True if we got data from at least one source
    """
    st      = fetch_stocktwits_sentiment(ticker)
    x_data  = fetch_x_sentiment(ticker)

    text_score = 0.0
    if use_text_scoring:
        all_texts = st.get("sample_texts", []) + x_data.get("texts", [])
        if all_texts:
            text_score = score_social_texts(all_texts[:30])

    st_net     = st["net_sentiment"]
    x_eng      = x_data["engagement_score"]
    x_signal   = text_score * min(x_eng / 10.0, 1.0) if x_data["total_tweets"] > 0 else 0.0
    has_st     = st["total_messages"] > 0
    has_x      = x_data["total_tweets"] > 0

    if has_st and has_x:
        combined = 0.50 * st_net + 0.30 * text_score + 0.20 * x_signal
    elif has_st:
        combined = 0.65 * st_net + 0.35 * text_score
    elif has_x:
        combined = 0.75 * text_score + 0.25 * x_signal
    else:
        combined = 0.0

    return {
        "combined":        float(np.clip(combined, -1.0, 1.0)),
        "bullish_pct":     st["bullish_ratio"] * 100,
        "bearish_pct":     st["bearish_ratio"] * 100,
        "message_volume":  min(st["total_messages"] / 50.0, 5.0),
        "x_tweets":        x_data["total_tweets"],
        "x_engagement":    min(x_data["engagement_score"] / 10.0, 5.0),
        "text_score":      text_score,
        "social_available": has_st or has_x,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — python social_sentiment.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("\n" + "=" * 60)
    print(" SOCIAL SENTIMENT ENGINE TEST")
    print("=" * 60)

    for ticker in ["NVDA", "GS", "BTC-USD"]:
        print(f"\n  Testing {ticker}...")
        result = get_live_social_signal(ticker, use_text_scoring=False)
        print(f"  StockTwits: {result['bullish_pct']:.1f}% bull | {result['bearish_pct']:.1f}% bear")
        print(f"  X tweets  : {result['x_tweets']}  engagement={result['x_engagement']:.2f}")
        print(f"  Combined  : {result['combined']:+.3f}")
        print(f"  Available : {result['social_available']}")
        time.sleep(1)

    print("\n" + "=" * 60)
    print("X Bearer Token:", "SET" if X_BEARER_TOKEN else "NOT SET (X signals disabled)")
    if not X_BEARER_TOKEN:
        print("  Get a free token at: developer.twitter.com")
        print("  Then add X_BEARER_TOKEN=your_token to your .env file")
    print("=" * 60 + "\n")
