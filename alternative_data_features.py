"""
alternative_data_features.py — Earnings revisions, analyst estimates,
short interest, and institutional ownership features.

WHY THIS FILE EXISTS
--------------------
Price/volume technical features give AUC ≈ 0.51 for cross-sectional stock
ranking — essentially coin-flip.  Academic research shows that the strongest
single predictor of individual stock returns is EARNINGS REVISIONS: when
analysts revise EPS estimates upward, the stock tends to outperform over the
next 1-3 months (the "post-earnings-announcement drift" / "earnings momentum"
anomaly).

Short interest and institutional ownership provide complementary signals:
  - High short interest + declining → bearish (informed traders betting down)
  - Rising institutional ownership → bullish (smart money accumulating)

DATA SOURCES
------------
  - yfinance: eps_trend (current vs 7d/30d/60d/90d ago), recommendations,
              upgrades_downgrades, info (short interest, institutional %)
  - finnhub:  monthly recommendation trends (free tier)

LIMITATIONS
-----------
  - yfinance eps_trend is SNAPSHOT data — it shows the current consensus
    and how it changed over recent windows, NOT historical revisions at
    each past date.  For backtesting, we can only use this for LIVE
    predictions, not for training.  Historical EPS revision data requires
    a paid data vendor (Estimize, Refinitiv, Bloomberg).
  - Short interest from yfinance.info is also snapshot-only (current + prior month).
  - For training, we use finnhub monthly recommendation trends which have
    date stamps, plus upgrades_downgrades from yfinance which are dated.

PLAIN ENGLISH
-------------
  - "EPS revision" = analysts changed their earnings forecast. If they raised it,
    the stock usually goes up. If they cut it, it usually goes down.
  - "Short interest" = how many shares are being borrowed and sold by people
    betting the stock will go down. High short interest = lots of pessimism.
  - "Institutional ownership" = what fraction of shares are held by big funds.
    Rising institutional ownership = smart money is buying.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from settings import FINNHUB_API_KEY
except ImportError:
    FINNHUB_API_KEY = ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Live EPS revision features (snapshot → features for today's prediction)
# ─────────────────────────────────────────────────────────────────────────────

def build_live_eps_revision_features(ticker: str) -> dict[str, float]:
    """Extract EPS revision momentum from yfinance's eps_trend snapshot.

    PLAIN ENGLISH: We look at the current quarter's EPS estimate and compare
    it to what analysts thought 7, 30, 60, and 90 days ago.  If estimates
    have been rising → positive revision momentum → bullish signal.

    Returns a dict of feature_name → value (for a single date: today).
    """
    import yfinance as yf
    result = {
        "alt_eps_revision_7d": 0.0,
        "alt_eps_revision_30d": 0.0,
        "alt_eps_revision_60d": 0.0,
        "alt_eps_revision_90d": 0.0,
        "alt_eps_surprise_pct": 0.0,
        "alt_n_analysts": 0.0,
        "alt_has_estimates": 0.0,
    }

    try:
        tk = yf.Ticker(ticker)

        # EPS trend: current estimate vs past snapshots
        eps_trend = tk.eps_trend
        if eps_trend is not None and not eps_trend.empty:
            # Use current quarter (row "0q") for revision signals
            if "0q" in eps_trend.index:
                row = eps_trend.loc["0q"]
                current = float(row.get("current", 0.0) or 0.0)
                if current != 0.0:
                    # Revision = (current - past) / |past|
                    # Positive means estimates have been raised
                    for period, col in [
                        ("7d", "7daysAgo"), ("30d", "30daysAgo"),
                        ("60d", "60daysAgo"), ("90d", "90daysAgo"),
                    ]:
                        past = float(row.get(col, 0.0) or 0.0)
                        if past != 0.0:
                            revision = (current - past) / abs(past)
                            result[f"alt_eps_revision_{period}"] = float(
                                np.clip(revision, -1.0, 1.0)
                            )
                    result["alt_has_estimates"] = 1.0

        # Earnings estimate: number of analysts covering
        ee = tk.earnings_estimate
        if ee is not None and not ee.empty and "0q" in ee.index:
            n = float(ee.loc["0q"].get("numberOfAnalysts", 0) or 0)
            result["alt_n_analysts"] = min(n, 60.0)

            # Year-ago EPS vs current estimate → surprise proxy
            year_ago = float(ee.loc["0q"].get("yearAgoEps", 0) or 0)
            avg_est = float(ee.loc["0q"].get("avg", 0) or 0)
            if year_ago != 0.0 and avg_est != 0.0:
                result["alt_eps_surprise_pct"] = float(
                    np.clip((avg_est - year_ago) / abs(year_ago), -2.0, 5.0)
                )

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live analyst recommendation features
# ─────────────────────────────────────────────────────────────────────────────

def build_live_recommendation_features(ticker: str) -> dict[str, float]:
    """Extract analyst recommendation momentum from yfinance.

    PLAIN ENGLISH: We look at how many analysts say "buy" vs "sell" and
    whether the consensus has been shifting.  A stock going from mostly
    "hold" to mostly "buy" is a bullish signal.

    Returns a dict of feature_name → value.
    """
    import yfinance as yf
    result = {
        "alt_rec_mean": 3.0,       # 1=strong buy, 5=strong sell, 3=hold
        "alt_rec_buy_pct": 0.0,    # fraction of analysts saying buy/strongBuy
        "alt_rec_change_1m": 0.0,  # change in buy% from prior month
        "alt_n_upgrades_90d": 0.0, # net upgrades in last 90 days
    }

    try:
        tk = yf.Ticker(ticker)

        # Recommendation summary (monthly snapshots)
        recs = tk.recommendations
        if recs is not None and not recs.empty and len(recs) >= 1:
            latest = recs.iloc[0]
            total = sum(
                float(latest.get(k, 0) or 0)
                for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]
            )
            if total > 0:
                buy_count = float(latest.get("strongBuy", 0) or 0) + float(latest.get("buy", 0) or 0)
                result["alt_rec_buy_pct"] = buy_count / total

                # Weighted mean: 1=strongBuy, 2=buy, 3=hold, 4=sell, 5=strongSell
                weighted = (
                    1 * float(latest.get("strongBuy", 0) or 0)
                    + 2 * float(latest.get("buy", 0) or 0)
                    + 3 * float(latest.get("hold", 0) or 0)
                    + 4 * float(latest.get("sell", 0) or 0)
                    + 5 * float(latest.get("strongSell", 0) or 0)
                ) / total
                result["alt_rec_mean"] = round(float(np.clip(weighted, 1.0, 5.0)), 2)

            # Month-over-month change in buy percentage
            if len(recs) >= 2:
                prev = recs.iloc[1]
                prev_total = sum(
                    float(prev.get(k, 0) or 0)
                    for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]
                )
                if prev_total > 0:
                    prev_buy = (float(prev.get("strongBuy", 0) or 0) + float(prev.get("buy", 0) or 0)) / prev_total
                    result["alt_rec_change_1m"] = float(
                        np.clip(result["alt_rec_buy_pct"] - prev_buy, -0.5, 0.5)
                    )

        # Upgrades/downgrades: count net upgrades in last 90 days
        ud = tk.upgrades_downgrades
        if ud is not None and not ud.empty:
            cutoff = datetime.now() - timedelta(days=90)
            # Index is GradeDate (datetime)
            if hasattr(ud.index, 'tz'):
                ud.index = ud.index.tz_localize(None) if ud.index.tz is None else ud.index.tz_convert(None)
            recent = ud[ud.index >= pd.Timestamp(cutoff)]
            if not recent.empty and "Action" in recent.columns:
                actions = recent["Action"].str.lower()
                n_up = int((actions.str.contains("upgrade", na=False)).sum())
                n_down = int((actions.str.contains("downgrade", na=False)).sum())
                result["alt_n_upgrades_90d"] = float(np.clip(n_up - n_down, -20, 20))

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Live short interest & institutional ownership features
# ─────────────────────────────────────────────────────────────────────────────

def build_live_short_interest_features(ticker: str) -> dict[str, float]:
    """Extract short interest and institutional ownership from yfinance.info.

    PLAIN ENGLISH:
    - Short ratio = days to cover all short positions at average volume.
      High short ratio (>5) means lots of bearish bets that would take a
      week to unwind → potential short squeeze.
    - Short % of float = what fraction of tradable shares are sold short.
      Above 10% is notable, above 20% is extreme.
    - Short change = are shorts increasing or decreasing vs prior month?
      Rising shorts = growing pessimism. Falling shorts = bears closing out.
    - Institutional % = fraction held by big funds. Low institutional
      ownership means the stock is under-covered and may be mispriced.

    Returns a dict of feature_name → value.
    """
    import yfinance as yf
    result = {
        "alt_short_ratio": 0.0,
        "alt_short_pct_float": 0.0,
        "alt_short_change_mom": 0.0,
        "alt_institutional_pct": 0.0,
        "alt_insider_pct": 0.0,
        "alt_target_upside": 0.0,
        "alt_has_short_data": 0.0,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info

        # Short interest
        short_ratio = float(info.get("shortRatio", 0) or 0)
        result["alt_short_ratio"] = float(np.clip(short_ratio, 0.0, 30.0))

        short_pct = float(info.get("shortPercentOfFloat", 0) or 0)
        result["alt_short_pct_float"] = float(np.clip(short_pct, 0.0, 1.0))

        # Month-over-month change in shares short
        shares_short = float(info.get("sharesShort", 0) or 0)
        shares_short_prior = float(info.get("sharesShortPriorMonth", 0) or 0)
        if shares_short_prior > 0:
            change = (shares_short - shares_short_prior) / shares_short_prior
            result["alt_short_change_mom"] = float(np.clip(change, -1.0, 2.0))
            result["alt_has_short_data"] = 1.0

        # Institutional & insider ownership
        inst_pct = float(info.get("heldPercentInstitutions", 0) or 0)
        result["alt_institutional_pct"] = float(np.clip(inst_pct, 0.0, 1.0))

        insider_pct = float(info.get("heldPercentInsiders", 0) or 0)
        result["alt_insider_pct"] = float(np.clip(insider_pct, 0.0, 1.0))

        # Analyst target price vs current price → upside potential
        target_mean = float(info.get("targetMeanPrice", 0) or 0)
        current = float(info.get("currentPrice", 0) or info.get("previousClose", 0) or 0)
        if target_mean > 0 and current > 0:
            upside = (target_mean - current) / current
            result["alt_target_upside"] = float(np.clip(upside, -0.5, 2.0))

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Historical recommendation trends from Finnhub (free tier, dated)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_finnhub_recommendation_history(ticker: str) -> pd.DataFrame:
    """Fetch monthly analyst recommendation trends from Finnhub.

    Unlike yfinance's snapshot data, Finnhub returns dated monthly records
    that can be used for backtesting.  Each record has the date and counts
    of strongBuy/buy/hold/sell/strongSell.

    Returns a DataFrame with columns: date, buy_pct, rec_mean, n_analysts.
    Empty DataFrame if no data or API key missing.
    """
    if not FINNHUB_API_KEY:
        return pd.DataFrame()

    try:
        import requests
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data:
            return pd.DataFrame()

        rows = []
        for rec in data:
            total = sum(rec.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
            if total == 0:
                continue
            buy_ct = rec.get("strongBuy", 0) + rec.get("buy", 0)
            weighted = (
                1 * rec.get("strongBuy", 0) + 2 * rec.get("buy", 0)
                + 3 * rec.get("hold", 0) + 4 * rec.get("sell", 0)
                + 5 * rec.get("strongSell", 0)
            ) / total
            rows.append({
                "date": pd.Timestamp(rec["period"]),
                "alt_rec_buy_pct": round(buy_ct / total, 3),
                "alt_rec_mean": round(weighted, 2),
                "alt_n_analysts": total,
            })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        # Compute month-over-month change in buy_pct
        df["alt_rec_change_1m"] = df["alt_rec_buy_pct"].diff().clip(-0.5, 0.5).fillna(0.0)
        return df

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Historical upgrades/downgrades from yfinance (dated events)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_upgrade_downgrade_history(ticker: str) -> pd.DataFrame:
    """Fetch dated analyst upgrades/downgrades and compute rolling features.

    Returns a daily-frequency DataFrame with:
      alt_net_upgrades_90d: rolling 90-day net upgrades (upgrades - downgrades)
      alt_upgrade_momentum: is the trend improving or worsening?
    """
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        ud = tk.upgrades_downgrades
        if ud is None or ud.empty or "Action" not in ud.columns:
            return pd.DataFrame()

        # Parse actions into +1 (upgrade), -1 (downgrade), 0 (other)
        actions = ud["Action"].str.lower()
        ud = ud.copy()
        ud["score"] = 0
        ud.loc[actions.str.contains("upgrade", na=False), "score"] = 1
        ud.loc[actions.str.contains("downgrade", na=False), "score"] = -1

        # Ensure datetime index without timezone
        if hasattr(ud.index, 'tz') and ud.index.tz is not None:
            ud.index = ud.index.tz_convert(None)
        ud.index = pd.to_datetime(ud.index).normalize()

        # Aggregate to daily net score
        daily = ud.groupby(ud.index)["score"].sum()
        daily = daily.sort_index()

        # Rolling 90-day sum
        result = pd.DataFrame(index=daily.index)
        result["alt_net_upgrades_90d"] = daily.rolling(90, min_periods=1).sum().clip(-20, 20)
        # Momentum: is the 30-day sum increasing vs the prior 30 days?
        sum_30 = daily.rolling(30, min_periods=1).sum()
        sum_60_30 = daily.rolling(60, min_periods=1).sum() - sum_30
        result["alt_upgrade_momentum"] = (sum_30 - sum_60_30).clip(-10, 10)

        return result.fillna(0.0)

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Combined live feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_all_live_alternative_features(ticker: str) -> dict[str, float]:
    """Build all alternative data features for live prediction.

    Calls the EPS revision, recommendation, and short interest builders
    and merges their outputs into a single dict.  Missing data gets
    neutral defaults (0.0 for most, 3.0 for rec_mean).

    This is meant to be called from predict.py for each ticker, then
    the features are added to the live prediction row.
    """
    features = {}
    features.update(build_live_eps_revision_features(ticker))
    features.update(build_live_recommendation_features(ticker))
    features.update(build_live_short_interest_features(ticker))
    return features


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Alternative Data Features Self-Test ===\n")

    ticker = "AAPL"
    print(f"--- Live EPS Revisions ({ticker}) ---")
    eps = build_live_eps_revision_features(ticker)
    for k, v in eps.items():
        print(f"  {k}: {v}")

    print(f"\n--- Live Recommendations ({ticker}) ---")
    recs = build_live_recommendation_features(ticker)
    for k, v in recs.items():
        print(f"  {k}: {v}")

    print(f"\n--- Live Short Interest ({ticker}) ---")
    si = build_live_short_interest_features(ticker)
    for k, v in si.items():
        print(f"  {k}: {v}")

    print(f"\n--- Finnhub Recommendation History ({ticker}) ---")
    hist = fetch_finnhub_recommendation_history(ticker)
    if not hist.empty:
        print(f"  Records: {len(hist)}")
        print(hist.tail(3))
    else:
        print("  No data (API key missing or endpoint unavailable)")

    print(f"\n--- Upgrade/Downgrade History ({ticker}) ---")
    ud = fetch_upgrade_downgrade_history(ticker)
    if not ud.empty:
        print(f"  Records: {len(ud)}")
        print(ud.tail(5))
    else:
        print("  No data")

    print(f"\n--- Combined Live Features ({ticker}) ---")
    all_feats = build_all_live_alternative_features(ticker)
    for k, v in sorted(all_feats.items()):
        print(f"  {k}: {v}")

    print("\n=== Self-test complete ===")
