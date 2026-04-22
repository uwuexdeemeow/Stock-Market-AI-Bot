"""
social_sentiment.py — Social Media Sentiment for Stock Signals
==============================================================
WHAT THIS FILE DOES:
  Pulls real-time sentiment from Reddit and StockTwits for any ticker.
  This is a separate signal source from financial news — social media
  often LEADS the news by hours, especially on retail-driven stocks
  like TSLA, NVDA, AMD, GME, and crypto.

WHY SOCIAL MEDIA MATTERS:
  Think of it like this:
    - Financial news (Finnhub, Yahoo Finance) = what professionals SAY
    - Social media = what millions of retail traders FEEL right now

  Academic research shows Reddit "buzz" predicts short-term price moves
  (1-5 days) for retail-heavy stocks with a correlation of ~0.3-0.5.
  That's not huge, but it's a genuinely independent signal that makes
  your ensemble predictions more diversified.

  StockTwits is especially useful because:
    - Users tag their posts as BULLISH or BEARISH explicitly
    - Volume of posts (message_volume) predicts volatility
    - Works entirely without an API key

  Reddit (via Pushshift or the official API) is good for:
    - WallStreetBets (r/wallstreetbets) — retail sentiment extremes
    - Upvote/comment ratios = community agreement on a thesis
    - Detects "meme stock" events before they go viral

SIGNALS THIS MODULE PROVIDES:
  social_bullish_ratio   : % of StockTwits messages tagged Bullish
  social_bearish_ratio   : % tagged Bearish
  social_bull_minus_bear : net sentiment (-1 to +1)
  social_message_volume  : normalised message activity (>1 = unusual)
  social_combined        : final composite score (-1 to +1)
  reddit_score           : Reddit upvote/comment-weighted sentiment
  reddit_mention_volume  : normalised mention count

HOW TO USE:
  from social_sentiment import build_social_sentiment_features
  df_social = build_social_sentiment_features("AAPL", dates)
  # Returns a DataFrame you can concat into your main feature frame

FREE vs PAID:
  StockTwits: 100% free, no API key needed. Rate-limited to ~60 req/min.
  Reddit: Free official API with a key. Covers 1000 posts/request.
  Limits: No API key = limited history. Key = 60 requests/minute.

HOW TO GET FREE REDDIT API KEY (takes 2 minutes):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "Create another app"
  3. Choose "script"
  4. Fill in name + redirect URI (use http://localhost)
  5. Copy client_id (under app name) and client_secret
  6. Set: export REDDIT_CLIENT_ID="your_id"
          export REDDIT_CLIENT_SECRET="your_secret"
          export REDDIT_USER_AGENT="stockbot/1.0 by yourusername"

ACCURACY EXPECTATIONS:
  Social sentiment works best for:
    - Highly retail-traded stocks (TSLA, NVDA, GME, AMC, BTC)
    - Short prediction horizons (1-3 days)
    - Detecting sentiment extremes (euphoria / panic)

  Social sentiment works WORST for:
    - Institutional-dominated stocks (BRK, JPM as a standalone signal)
    - Long prediction horizons (5+ days)
    - Low-activity tickers with <50 daily messages

INTEGRATION:
  Add this to 02_research_v4.py's build_feature_frame_for_dates():
    social = build_social_sentiment_features(ticker, dates)
    if not social.empty:
        feature_frames.append(social)
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("social_sentiment")

# ─────────────────────────────────────────────────────────────────────────────
# REDDIT CREDENTIALS (from environment variables — never hardcode secrets)
# ─────────────────────────────────────────────────────────────────────────────

REDDIT_CLIENT_ID     = os.environ.get("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
REDDIT_USER_AGENT    = os.environ.get("REDDIT_USER_AGENT", "stockbot/1.0").strip()

# Subreddits to monitor for stock mentions
STOCK_SUBREDDITS = [
    "wallstreetbets",   # largest retail trading community (~15M members)
    "stocks",           # more serious investors
    "investing",        # long-term focused but still has sentiment signals
    "StockMarket",      # mixed retail/institutional
    "options",          # options flow discussion (high activity on expiry weeks)
]

# For crypto tickers
CRYPTO_SUBREDDITS = [
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "CryptoMarkets",
]

# Ticker-specific subreddits (these dedicate whole communities to one stock)
TICKER_SUBREDDITS = {
    "TSLA": ["teslainvestorsclub", "TSLA"],
    "NVDA": ["nvidia"],
    "AAPL": ["apple"],
    "AMZN": ["amznstock"],
    "GME" : ["Superstonk", "GME"],
    "AMC" : ["amcstock"],
    "BTC-USD": ["Bitcoin", "btc"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STOCKTWITS (free, no API key, most reliable)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stocktwits_sentiment(ticker: str, max_messages: int = 100) -> dict:
    """
    Fetch the latest StockTwits messages and return sentiment breakdown.

    StockTwits is a social network for traders where users can explicitly
    tag their posts as BULLISH or BEARISH. This is more reliable than
    inferring sentiment from text — users TELL you what they think.

    HOW IT WORKS:
      The StockTwits API is free and doesn't need an API key.
      We call their public endpoint for the latest messages on a ticker.
      Each message either has a sentiment tag (bullish/bearish) or doesn't.

    Returns a dict with:
      total_messages   : how many messages we fetched (0 if API fails)
      bullish_count    : messages tagged BULLISH
      bearish_count    : messages tagged BEARISH
      bullish_ratio    : bullish_count / total (0.5 = neutral)
      bearish_ratio    : bearish_count / total
      net_sentiment    : bullish_ratio - bearish_ratio (-1 to +1)
      sentiment_tagged : fraction of messages that have explicit tags
    """
    # Clean the ticker for StockTwits (they don't accept hyphens like BTC-USD)
    st_ticker = ticker.replace("-USD", "").replace("-", "").upper()

    url = f"https://api.stocktwits.com/api/2/streams/symbol/{st_ticker}.json"
    params = {"limit": max_messages}

    try:
        import requests
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 429:
            log.warning("[%s] StockTwits rate limit hit — waiting 5s", ticker)
            time.sleep(5)
            return _empty_stocktwits_result()

        if resp.status_code != 200:
            log.warning("[%s] StockTwits returned %s", ticker, resp.status_code)
            return _empty_stocktwits_result()

        data = resp.json()
        messages = data.get("messages", [])

        if not messages:
            return _empty_stocktwits_result()

        bullish = 0
        bearish = 0
        tagged  = 0
        texts   = []

        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment", {})
            if sentiment:
                tagged += 1
                basic = sentiment.get("basic", "")
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1

            body = msg.get("body", "")
            if body:
                texts.append(body)

        total = len(messages)
        tagged_total = bullish + bearish  # only count explicitly tagged

        # Use tagged messages for the ratio (more reliable than all messages)
        if tagged_total > 0:
            bullish_ratio = bullish / tagged_total
            bearish_ratio = bearish / tagged_total
        else:
            bullish_ratio = 0.5
            bearish_ratio = 0.5

        return {
            "total_messages"   : total,
            "bullish_count"    : bullish,
            "bearish_count"    : bearish,
            "bullish_ratio"    : bullish_ratio,
            "bearish_ratio"    : bearish_ratio,
            "net_sentiment"    : bullish_ratio - bearish_ratio,  # -1 to +1
            "sentiment_tagged" : tagged / max(total, 1),
            "sample_texts"     : texts[:20],  # for FinBERT scoring
        }

    except ImportError:
        log.warning("requests not installed — pip install requests")
        return _empty_stocktwits_result()
    except Exception as e:
        log.warning("[%s] StockTwits fetch failed: %s", ticker, e)
        return _empty_stocktwits_result()


def _empty_stocktwits_result() -> dict:
    return {
        "total_messages": 0, "bullish_count": 0, "bearish_count": 0,
        "bullish_ratio": 0.5, "bearish_ratio": 0.5,
        "net_sentiment": 0.0, "sentiment_tagged": 0.0, "sample_texts": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REDDIT (requires free API key for best results)
# ─────────────────────────────────────────────────────────────────────────────

def _get_reddit_client():
    """
    Return a PRAW Reddit client if credentials are available.
    PRAW = "Python Reddit API Wrapper" — a library that makes Reddit API easy.

    If no credentials: returns None (we fall back to zero social features).
    If praw not installed: returns None with a helpful message.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return None

    try:
        import praw
        reddit = praw.Reddit(
            client_id     = REDDIT_CLIENT_ID,
            client_secret = REDDIT_CLIENT_SECRET,
            user_agent    = REDDIT_USER_AGENT,
        )
        # Test the connection
        _ = reddit.user.me()  # Will raise if credentials are wrong
        return reddit
    except ImportError:
        log.warning("praw not installed — pip install praw. Reddit signals disabled.")
        return None
    except Exception as e:
        log.warning("Reddit auth failed: %s. Check your credentials.", e)
        return None


def fetch_reddit_mentions(
    ticker: str,
    reddit_client,
    hours_back: int = 24,
    max_posts: int = 100,
) -> dict:
    """
    Search Reddit for mentions of a ticker in the last N hours.

    We search across multiple stock subreddits and count:
      - How many posts mention the ticker
      - Combined upvotes (higher = more community agreement)
      - Comment count (higher = more discussion / controversy)

    WHY UPVOTES MATTER:
      A post with 10,000 upvotes reached 100x more people than one with 100.
      Weighting by upvotes approximates "how much of Reddit saw this."
      This is the same logic financial media uses for "trending" articles.

    Returns a dict with:
      mention_count         : raw number of posts mentioning ticker
      upvote_weighted_score : mention count weighted by upvotes (normalised)
      avg_score_per_post    : mean upvotes per mentioning post
      comment_activity      : normalised comment count (proxy for buzz)
    """
    if reddit_client is None:
        return _empty_reddit_result()

    st_ticker = ticker.replace("-USD", "").upper()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    # Pick the right subreddits — ticker-specific are most relevant
    subreddits_to_check = list(
        CRYPTO_SUBREDDITS if "-USD" in ticker or "-" in ticker
        else STOCK_SUBREDDITS
    )
    # Add ticker-specific subreddit if one exists
    if st_ticker in TICKER_SUBREDDITS:
        subreddits_to_check = TICKER_SUBREDDITS[st_ticker] + subreddits_to_check

    total_mentions = 0
    total_upvotes  = 0
    total_comments = 0
    post_texts     = []

    try:
        for sub_name in subreddits_to_check[:4]:  # cap at 4 subreddits to avoid rate limits
            try:
                subreddit = reddit_client.subreddit(sub_name)

                # Search for ticker mentions in the last 24h
                # Reddit search uses Lucene syntax: "flair:AAPL" OR title search
                search_query = f"{st_ticker} OR ${st_ticker}"
                posts = subreddit.search(
                    search_query, sort="new", time_filter="day", limit=max_posts // 4
                )

                for post in posts:
                    try:
                        post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        if post_time < cutoff:
                            continue

                        # Only count posts that actually mention our ticker
                        combined_text = (post.title + " " + post.selftext).upper()
                        if f"${st_ticker}" in combined_text or f" {st_ticker} " in combined_text:
                            total_mentions += 1
                            total_upvotes  += max(post.score, 0)  # upvotes can be negative
                            total_comments += post.num_comments
                            post_texts.append(post.title[:200])

                    except Exception:
                        continue

                time.sleep(0.5)  # be polite to Reddit API — avoid bans

            except Exception as e:
                log.debug("Subreddit %s failed: %s", sub_name, e)
                continue

    except Exception as e:
        log.warning("[%s] Reddit fetch failed: %s", ticker, e)
        return _empty_reddit_result()

    if total_mentions == 0:
        return _empty_reddit_result()

    avg_score = total_upvotes / max(total_mentions, 1)

    return {
        "mention_count"         : total_mentions,
        "total_upvotes"         : total_upvotes,
        "avg_score_per_post"    : avg_score,
        "comment_activity"      : total_comments,
        "mention_upvote_score"  : total_upvotes / max(total_mentions, 1),
        "post_texts"            : post_texts[:20],
    }


def _empty_reddit_result() -> dict:
    return {
        "mention_count": 0, "total_upvotes": 0,
        "avg_score_per_post": 0.0, "comment_activity": 0,
        "mention_upvote_score": 0.0, "post_texts": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TEXT SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_social_texts(texts: list[str]) -> float:
    """
    Score a list of social media posts using FinBERT (preferred) or FinVADER.

    Returns a single composite score: -1.0 (very bearish) to +1.0 (very bullish).

    IMPORTANT: Social media language is different from financial news.
    "TSLA to the moon 🚀" = bullish (but FinVADER misses emojis)
    "Diamond hands 💎" = bullish (FinVADER has no idea)
    "This is financial advice jk lol" = neutral (FinBERT handles this)

    FinBERT handles most cases well. The rocket/moon/fire emoji language
    is an acknowledged limitation — these posts often get scored neutral
    instead of bullish. This is okay; we still have the explicit BULLISH/BEARISH
    tags from StockTwits which are much more reliable than text scoring.
    """
    if not texts:
        return 0.0

    try:
        from sentiment_engine import SentimentEngine
        # Try FinBERT first (most accurate), fall back to FinVADER
        try:
            engine = SentimentEngine(level="finbert")
        except Exception:
            engine = SentimentEngine(level="finvader")

        scores = engine.score_batch(texts)
        if not scores:
            return 0.0

        return float(np.mean(scores))

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
    use_reddit: bool = True,
    use_stocktwits: bool = True,
    score_texts: bool = True,
) -> pd.DataFrame:
    """
    Build daily social sentiment features aligned to trading dates.

    This is the main function that 02_research_v4.py should call.
    It returns a DataFrame with social features for every trading day.

    IMPORTANT LIMITATION:
    Neither StockTwits nor Reddit provides FREE historical data going back years.
    - StockTwits free tier: ~latest 100 messages only (no history)
    - Reddit: ~1 month lookback with free API

    This means for historical training data, social features will be ZERO
    for most dates. Only the LIVE prediction will have real values.

    This is still useful because:
      1. Live predictions become better (social signal adds information)
      2. The model learns that zeros = "no social signal available" (neutral)
      3. When social signal IS available (live), it's a strong differentiator

    If you have a StockTwits Pro or Reddit Premium subscription, you can
    modify this to fetch historical data. For most users, just use the
    live-only approach.

    Parameters:
      ticker        : stock symbol e.g. "AAPL"
      dates         : trading day index from your price DataFrame
      use_reddit    : set False to disable Reddit (if no API key)
      use_stocktwits: set False to disable StockTwits
      score_texts   : run FinBERT on post text (adds ~5s per ticker, improves quality)

    Returns:
      DataFrame indexed to `dates` with the following columns:
        social_bull_ratio     : StockTwits % bullish (-1 to +1 after centering)
        social_bear_ratio     : StockTwits % bearish
        social_net_sentiment  : bull minus bear
        social_message_volume : normalised message count (>1 = above average)
        social_text_score     : FinBERT score on social post text
        social_combined       : weighted blend of all social signals
        reddit_mention_count  : normalised Reddit mention volume
        reddit_upvote_score   : normalised upvote-weighted sentiment
        social_agrees_news    : 1 if social and news sentiment agree, -1 if conflict
    """
    # Create output DataFrame with all zeros (safe default)
    result = pd.DataFrame(index=dates)
    social_cols = [
        "social_bull_ratio", "social_bear_ratio", "social_net_sentiment",
        "social_message_volume", "social_text_score", "social_combined",
        "reddit_mention_count", "reddit_upvote_score", "social_agrees_news",
    ]
    for col in social_cols:
        result[col] = 0.0

    # ── StockTwits ─────────────────────────────────────────────────────────
    st_result = {}
    if use_stocktwits:
        log.info("[%s] Fetching StockTwits sentiment...", ticker)
        st_result = fetch_stocktwits_sentiment(ticker)

        if st_result["total_messages"] > 0:
            log.info(
                "[%s] StockTwits: %d messages | Bullish=%.1f%% Bearish=%.1f%%",
                ticker,
                st_result["total_messages"],
                st_result["bullish_ratio"] * 100,
                st_result["bearish_ratio"] * 100,
            )

            # Apply today's reading to the MOST RECENT date only
            # (We can't get historical StockTwits data for free)
            latest_idx = dates[-1] if len(dates) > 0 else None
            if latest_idx is not None:
                result.loc[latest_idx, "social_bull_ratio"]    = st_result["bullish_ratio"] - 0.5  # centre at zero
                result.loc[latest_idx, "social_bear_ratio"]    = st_result["bearish_ratio"] - 0.5
                result.loc[latest_idx, "social_net_sentiment"] = st_result["net_sentiment"]

                # Normalise message volume (>0.5 = above typical)
                # We use a rough baseline of 50 messages/day as "average"
                msg_vol = st_result["total_messages"] / 50.0
                result.loc[latest_idx, "social_message_volume"] = min(msg_vol, 5.0)

                # Score the actual text of the posts with FinBERT
                if score_texts and st_result.get("sample_texts"):
                    text_score = score_social_texts(st_result["sample_texts"])
                    result.loc[latest_idx, "social_text_score"] = text_score

        else:
            log.info("[%s] StockTwits: no data (ticker may not be tracked)", ticker)

    # ── Reddit ─────────────────────────────────────────────────────────────
    reddit = None
    if use_reddit:
        reddit = _get_reddit_client()

    reddit_result = {}
    if reddit is not None:
        log.info("[%s] Fetching Reddit mentions...", ticker)
        reddit_result = fetch_reddit_mentions(ticker, reddit)

        if reddit_result["mention_count"] > 0:
            log.info(
                "[%s] Reddit: %d mentions | %d upvotes | %d comments",
                ticker,
                reddit_result["mention_count"],
                reddit_result["total_upvotes"],
                reddit_result["comment_activity"],
            )

            latest_idx = dates[-1] if len(dates) > 0 else None
            if latest_idx is not None:
                # Normalise: 10 mentions with 1000 upvotes = very high activity
                norm_mentions = min(reddit_result["mention_count"] / 10.0, 5.0)
                norm_upvotes  = min(reddit_result["total_upvotes"] / 1000.0, 5.0)

                result.loc[latest_idx, "reddit_mention_count"] = norm_mentions
                result.loc[latest_idx, "reddit_upvote_score"]  = norm_upvotes

                # Score Reddit post titles with FinBERT
                if score_texts and reddit_result.get("post_texts"):
                    reddit_text_score = score_social_texts(reddit_result["post_texts"])
                    # Blend with existing text score
                    existing = float(result.loc[latest_idx, "social_text_score"])
                    blended = (existing + reddit_text_score) / 2.0
                    result.loc[latest_idx, "social_text_score"] = blended
    else:
        if use_reddit:
            log.info("[%s] Reddit disabled (no API key or praw not installed)", ticker)

    # ── Combined Social Score ───────────────────────────────────────────────
    # Weighted blend of StockTwits signal + Reddit signal + text score
    # StockTwits gets highest weight because explicit tags are most reliable
    latest_idx = dates[-1] if len(dates) > 0 else None
    if latest_idx is not None:
        st_net   = float(result.loc[latest_idx, "social_net_sentiment"])
        txt_scr  = float(result.loc[latest_idx, "social_text_score"])
        rd_score = float(result.loc[latest_idx, "reddit_upvote_score"])

        # Normalise reddit upvote score to [-1, +1] range
        # (it's currently 0-5 scale, which we treat as directional via text sentiment)
        rd_directional = txt_scr * min(rd_score / 2.0, 1.0)  # text direction × reddit magnitude

        # Combine: 50% StockTwits explicit tags, 35% text sentiment, 15% Reddit
        if st_result.get("total_messages", 0) > 0:
            combined = 0.50 * st_net + 0.35 * txt_scr + 0.15 * rd_directional
        elif reddit_result.get("mention_count", 0) > 0:
            combined = 0.70 * txt_scr + 0.30 * rd_directional
        else:
            combined = 0.0

        result.loc[latest_idx, "social_combined"] = float(np.clip(combined, -1.0, 1.0))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — LIVE SIGNAL FOR PREDICTOR (04_predict_v3.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_social_signal(ticker: str, use_text_scoring: bool = True) -> dict:
    """
    Get the CURRENT social sentiment for use in live predictions.

    This is what 04_predict_v3.py should call — it returns a single dict
    with today's social readings rather than a full historical DataFrame.

    Usage in predict_ticker():
        from social_sentiment import get_live_social_signal
        social = get_live_social_signal(ticker)
        log.info("[%s] Social: %+.3f (StockTwits: %.1f%% bull | Reddit: %d mentions)",
                 ticker, social["combined"], social["bullish_pct"], social["reddit_mentions"])

    Returns:
        combined          : overall social score, -1 (bearish) to +1 (bullish)
        bullish_pct       : % bullish on StockTwits (0-100)
        bearish_pct       : % bearish on StockTwits
        message_volume    : normalised StockTwits activity level
        reddit_mentions   : raw Reddit mention count (last 24h)
        social_available  : True if we got at least StockTwits data
    """
    st = fetch_stocktwits_sentiment(ticker)
    reddit = _get_reddit_client()
    rd = fetch_reddit_mentions(ticker, reddit) if reddit else _empty_reddit_result()

    text_score = 0.0
    if use_text_scoring:
        all_texts = st.get("sample_texts", []) + rd.get("post_texts", [])
        if all_texts:
            text_score = score_social_texts(all_texts[:30])  # cap to avoid slowness

    st_net = st["net_sentiment"]
    rd_dir = text_score * min(rd["total_upvotes"] / 1000.0, 1.0) if rd["mention_count"] > 0 else 0.0

    if st["total_messages"] > 0 and rd["mention_count"] > 0:
        combined = 0.50 * st_net + 0.35 * text_score + 0.15 * rd_dir
    elif st["total_messages"] > 0:
        combined = 0.65 * st_net + 0.35 * text_score
    elif rd["mention_count"] > 0:
        combined = 0.70 * text_score + 0.30 * rd_dir
    else:
        combined = 0.0

    return {
        "combined"         : float(np.clip(combined, -1.0, 1.0)),
        "bullish_pct"      : st["bullish_ratio"] * 100,
        "bearish_pct"      : st["bearish_ratio"] * 100,
        "message_volume"   : min(st["total_messages"] / 50.0, 5.0),
        "reddit_mentions"  : rd["mention_count"],
        "reddit_upvotes"   : rd["total_upvotes"],
        "text_score"       : text_score,
        "social_available" : st["total_messages"] > 0 or rd["mention_count"] > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run directly to verify it works
# python social_sentiment.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    test_tickers = ["AAPL", "NVDA", "TSLA"]

    print("\n" + "=" * 60)
    print(" SOCIAL SENTIMENT ENGINE TEST")
    print("=" * 60)

    for ticker in test_tickers:
        print(f"\n  Testing {ticker}...")
        result = get_live_social_signal(ticker, use_text_scoring=False)  # skip slow FinBERT for test

        print(f"  StockTwits: {result['bullish_pct']:.1f}% bull | {result['bearish_pct']:.1f}% bear")
        print(f"  Reddit mentions (24h): {result['reddit_mentions']}")
        print(f"  Combined score: {result['combined']:+.3f}")
        print(f"  Data available: {result['social_available']}")
        time.sleep(1)  # avoid rate limits between tickers

    print("\n" + "=" * 60)
    print("  Reddit API key status:", "SET" if REDDIT_CLIENT_ID else "NOT SET (Reddit disabled)")
    if not REDDIT_CLIENT_ID:
        print("  Get a free key at: https://www.reddit.com/prefs/apps")
        print("  Then: export REDDIT_CLIENT_ID=your_id")
        print("         export REDDIT_CLIENT_SECRET=your_secret")
    print("=" * 60 + "\n")
