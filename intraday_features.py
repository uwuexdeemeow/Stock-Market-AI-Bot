"""
intraday_features.py -- VWAP and intraday volume features from Alpaca minute bars.

WHY THIS FILE EXISTS
--------------------
Daily OHLCV data tells you WHAT happened, but not HOW.  Two stocks can both
close up 2%, but one did it smoothly while the other whipsawed all day.
Intraday data reveals:
  - VWAP deviation: is the stock trading above or below its volume-weighted
    average price?  Institutional traders use VWAP as a benchmark -- stocks
    above VWAP have buying pressure, below have selling pressure.
  - Volume profile: is volume concentrated in the morning (smart money) or
    afternoon (retail)?  Heavy morning volume with price moves = informed flow.
  - Intraday range vs close: did the stock close near its high (bullish) or
    low (bearish) relative to the day's range?

DATA SOURCE
-----------
Alpaca Markets API (free with any account, including paper trading).  You
already have ALPACA_API_KEY and ALPACA_SECRET_KEY configured for paper trading.
This uses the same credentials to fetch 1-minute bars for the most recent
trading day.

FEATURES PRODUCED
-----------------
  vwap_dist          -- distance from close to VWAP as % of price
                        positive = close above VWAP (buying pressure)
  vwap_dist_5d       -- 5-day average of vwap_dist (trend in VWAP positioning)
  volume_am_ratio    -- morning volume (9:30-12:00 ET) / regular-session volume
                        high = institutional activity front-loaded
  volume_last_hour   -- last-hour volume (15:00-16:00 ET) / regular-session volume
                        high = end-of-day positioning / MOC orders
  intraday_range_pct -- (high - low) / open as percentage
                        high = volatile day, low = calm day
  close_position     -- where close sits in day's range: (close-low)/(high-low)
                        1.0 = closed at high (bullish), 0.0 = at low (bearish)

SETUP
-----
Uses the same ALPACA_API_KEY / ALPACA_SECRET_KEY env vars as paper trading.
No extra setup needed.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, time as dt_time, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Alpaca credentials (reuse from paper trading setup) ──────────────────────
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "").strip()
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
MARKET_TIMEZONE = "America/New_York"
REGULAR_SESSION_START = dt_time(9, 30)
MORNING_SESSION_END = dt_time(12, 0)
LAST_HOUR_START = dt_time(15, 0)
REGULAR_SESSION_END = dt_time(16, 0)


def _bars_in_market_time(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of bars indexed in New York market time.

    PLAIN ENGLISH: Alpaca timestamps are UTC.  New York switches between EST
    and EDT during the year, so fixed UTC hour checks drift around daylight
    saving transitions.  We convert timestamps to America/New_York first, then
    compare against market-clock times like 9:30 AM and 4:00 PM.
    """
    if bars.empty:
        return bars.copy()

    out = bars.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(MARKET_TIMEZONE)
    return out


def _alpaca_get_bars(
    ticker: str,
    timeframe: str = "1Min",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch minute bars from Alpaca's data API.

    PLAIN ENGLISH: This calls Alpaca's historical bar endpoint to get
    minute-by-minute price and volume data.  We use this to compute
    intraday features like VWAP.

    Args:
        ticker: stock symbol (e.g. "AAPL")
        timeframe: bar size ("1Min", "5Min", "15Min", "1Hour", "1Day")
        start: start datetime in RFC3339 format
        end: end datetime in RFC3339 format
        limit: max bars to return (up to 10000)

    Returns a DataFrame with columns: open, high, low, close, volume, vwap
    indexed by timestamp.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()

    import requests

    url = f"{ALPACA_DATA_URL}/stocks/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {
        "timeframe": timeframe,
        "limit": limit,
        "adjustment": "raw",
        "feed": "sip",   # use SIP (consolidated) feed for best data
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            logger.debug("Alpaca bars %s returned %d: %s", ticker, resp.status_code, resp.text[:200])
            return pd.DataFrame()

        data = resp.json()
        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        # Alpaca returns: t (timestamp), o, h, l, c, v, vw (VWAP), n (trade count)
        rename = {"t": "timestamp", "o": "open", "h": "high", "l": "low",
                  "c": "close", "v": "volume", "vw": "vwap", "n": "trades"}
        df = df.rename(columns=rename)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

        return df

    except Exception as e:
        logger.debug("Alpaca bars error for %s: %s", ticker, e)
        return pd.DataFrame()


def _compute_intraday_features_from_bars(bars: pd.DataFrame) -> dict[str, float]:
    """Compute intraday features from a single day's minute bars.

    PLAIN ENGLISH: Given all the 1-minute bars for one trading day, this
    computes features that describe HOW the price moved during the day:
    - Where did it close relative to VWAP?
    - Was regular-session volume heavier in the morning or afternoon?
    - How wide was the intraday range?
    - Did it close near the high or the low?
    All volume split windows are evaluated in America/New_York so daylight
    saving time changes do not shift the morning or last-hour buckets.
    """
    result = {
        "vwap_dist": 0.0,
        "volume_am_ratio": 0.50,
        "volume_last_hour": 0.15,
        "intraday_range_pct": 0.0,
        "close_position": 0.50,
    }

    if bars.empty or len(bars) < 10:
        return result

    # VWAP distance: how far is the close from VWAP?
    # Alpaca provides VWAP per bar (vw field = volume-weighted average price)
    day_close = float(bars["close"].iloc[-1])
    if "vwap" in bars.columns and bars["vwap"].notna().any():
        # Use the last VWAP value (it's cumulative throughout the day)
        # Actually compute true VWAP: sum(price * volume) / sum(volume)
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        vol = bars["volume"].astype(float)
        cumulative_vwap = (typical_price * vol).sum() / max(vol.sum(), 1)
        if cumulative_vwap > 0 and day_close > 0:
            # Express as percentage distance from VWAP
            # Positive = close above VWAP (buying pressure)
            result["vwap_dist"] = round(
                float(np.clip((day_close - cumulative_vwap) / cumulative_vwap, -0.05, 0.05)),
                5,
            )

    # Volume profile: morning and last-hour volume as a share of the regular
    # 9:30-16:00 ET session.  Convert timestamps to New York time first so DST
    # transitions do not move the split windows.
    market_bars = _bars_in_market_time(bars)
    if isinstance(market_bars.index, pd.DatetimeIndex):
        times = market_bars.index.time
        regular_mask = (times >= REGULAR_SESSION_START) & (times < REGULAR_SESSION_END)
        total_vol = float(market_bars.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 0.0
        if total_vol > 0:
            am_mask = regular_mask & (times < MORNING_SESSION_END)
            am_vol = float(market_bars.loc[am_mask, "volume"].sum()) if am_mask.any() else 0.0
            result["volume_am_ratio"] = round(
                float(np.clip(am_vol / total_vol, 0.0, 1.0)), 3
            )

            last_hour_mask = regular_mask & (times >= LAST_HOUR_START)
            last_hour_vol = float(market_bars.loc[last_hour_mask, "volume"].sum()) if last_hour_mask.any() else 0.0
            result["volume_last_hour"] = round(
                float(np.clip(last_hour_vol / total_vol, 0.0, 1.0)), 3
            )

    # Intraday range: (day_high - day_low) / open
    day_high = float(bars["high"].max())
    day_low = float(bars["low"].min())
    day_open = float(bars["open"].iloc[0])
    if day_open > 0:
        result["intraday_range_pct"] = round(
            float(np.clip((day_high - day_low) / day_open, 0.0, 0.20)), 4
        )

    # Close position: where does close sit in the day's range?
    # 1.0 = closed at the high, 0.0 = closed at the low
    day_range = day_high - day_low
    if day_range > 0.001:
        result["close_position"] = round(
            float(np.clip((day_close - day_low) / day_range, 0.0, 1.0)), 3
        )

    return result


def fetch_intraday_features(ticker: str, lookback_days: int = 5) -> dict[str, float]:
    """Fetch intraday features for a ticker using Alpaca minute bars.

    PLAIN ENGLISH: This fetches the last N trading days of minute bars
    and computes intraday features for each day.  The final output
    includes both today's features and a 5-day average for VWAP distance
    (to capture multi-day trends in institutional flow).

    Args:
        ticker: stock symbol
        lookback_days: how many days of history for the VWAP trend

    Returns a dict of feature_name -> value.
    """
    result = {
        "vwap_dist": 0.0,
        "vwap_dist_5d": 0.0,
        "volume_am_ratio": 0.50,
        "volume_last_hour": 0.15,
        "intraday_range_pct": 0.0,
        "close_position": 0.50,
        "has_intraday_data": 0.0,
    }

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return result

    try:
        # Fetch minute bars for the last N+2 calendar days (to cover weekends)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days + 4)
        start_str = start_dt.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_dt.strftime("%Y-%m-%dT23:59:59Z")

        bars = _alpaca_get_bars(
            ticker,
            timeframe="1Min",
            start=start_str,
            end=end_str,
            limit=5000,  # ~390 bars/day × 5 days + buffer
        )

        if bars.empty:
            return result

        # Group bars by New York market date. Alpaca timestamps are UTC, so a
        # simple UTC date can put extended-hours bars on the wrong trading day.
        market_bars = _bars_in_market_time(bars)
        market_bars["_date"] = market_bars.index.date
        daily_groups = dict(list(market_bars.groupby("_date")))

        if not daily_groups:
            return result

        # Compute features for each day
        daily_features = []
        for date, day_bars in sorted(daily_groups.items()):
            if len(day_bars) >= 10:  # need at least 10 minute bars
                feats = _compute_intraday_features_from_bars(day_bars)
                daily_features.append(feats)

        if not daily_features:
            return result

        # Latest day's features
        latest = daily_features[-1]
        result["vwap_dist"] = latest["vwap_dist"]
        result["volume_am_ratio"] = latest["volume_am_ratio"]
        result["volume_last_hour"] = latest["volume_last_hour"]
        result["intraday_range_pct"] = latest["intraday_range_pct"]
        result["close_position"] = latest["close_position"]
        result["has_intraday_data"] = 1.0

        # 5-day average VWAP distance (trend in institutional positioning)
        recent_vwaps = [f["vwap_dist"] for f in daily_features[-lookback_days:]]
        if recent_vwaps:
            result["vwap_dist_5d"] = round(float(np.mean(recent_vwaps)), 5)

    except Exception as e:
        logger.debug("fetch_intraday_features(%s) error: %s", ticker, e)

    return result


def batch_fetch_intraday_features(
    tickers: list[str], delay: float = 0.3
) -> dict[str, dict]:
    """Fetch intraday features for multiple tickers with rate limiting.

    PLAIN ENGLISH: Alpaca's free tier allows 200 requests/minute.
    With a 0.3s delay between tickers, we do ~200 requests/minute
    which is right at the limit.  For 147 tickers this takes ~45 seconds.
    """
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_intraday_features(ticker)
        if i < len(tickers) - 1 and delay > 0:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            logger.info("Intraday fetch progress: %d/%d tickers", i + 1, len(tickers))
    return results


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    if not ALPACA_API_KEY:
        print("No ALPACA_API_KEY set. Set it to test:")
        print("  export ALPACA_API_KEY='your-key'")
        print("  export ALPACA_SECRET_KEY='your-secret'")
        print()
        print("Showing neutral defaults:")
        result = fetch_intraday_features("AAPL")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
    else:
        print("Using Alpaca API")
        print()

        for ticker in ["AAPL", "NVDA", "SPY", "TQQQ"]:
            print(f"--- {ticker} ---")
            result = fetch_intraday_features(ticker)
            for k, v in sorted(result.items()):
                print(f"  {k}: {v}")
            print()
