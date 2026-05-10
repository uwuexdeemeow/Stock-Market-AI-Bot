"""
core_satellite_alpha.py — test factor alpha as a satellite around SPY/QQQ.

The prior alpha_factor_backtest asks whether factors can replace ETF beta. This
script asks a more realistic question: can a lower-turnover factor sleeve add
active return on top of a SPY/QQQ core without exceeding 1.25x gross exposure?
It is research-only and does not enable paper trading.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    HORIZON_DAYS,
    MAX_GROSS_EXPOSURE,
    attach_scores,
    benchmark_equity,
    compare_to_benchmarks,
    gate_metrics,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
    portfolio_stats,
    subperiod_metrics,
)
from backtest import INITIAL_CAPITAL, _load_etf_price_frame
from settings import SIGNAL_DIR, SLIPPAGE_BASE_PCT


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT VETO — filter out stocks with strongly negative news sentiment
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: Before finalizing the live signal, we check each selected
# overlay stock's recent news sentiment.  If a stock has very negative news
# (e.g., fraud allegations, major lawsuit, earnings disaster), we remove it
# from the signal and let the next-best candidate take its spot.
# This only runs during LIVE signal generation, not during backtesting.

SENTIMENT_VETO_ENABLED = True      # set False to skip sentiment checking entirely
SENTIMENT_VETO_THRESHOLD = -0.25   # compound score below this → vetoed
SENTIMENT_BOOST_WEIGHT = 0.10      # multiply factor score by (1 + sentiment * this)


def _fetch_live_sentiment(tickers: list[str], timeout_per_ticker: float = 5.0) -> dict[str, float]:
    """
    Fetch real-time news sentiment for a list of tickers.

    PLAIN ENGLISH: For each ticker, grab recent news headlines from RSS feeds
    (Yahoo Finance, Google News, etc.), run them through FinVADER/FinBERT, and
    return a sentiment score between -1 (very negative) and +1 (very positive).
    Score of 0 means neutral or no news found.

    Returns dict mapping ticker → compound sentiment score.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "Stock picking scripts"))
    try:
        from sentiment_engine import get_sentiment_engine, SENTIMENT_RSS_SOURCES, headline_mentions_ticker
    except ImportError:
        # If sentiment engine not available, return empty (no veto)
        return {}

    # Local RSS fetch with SSL workaround (macOS Python often lacks certs)
    def _safe_fetch_rss(url: str, ticker: str) -> list[str]:
        """Fetch RSS headlines with SSL verification disabled for reliability."""
        import feedparser
        import requests
        import warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS")
        filled = url.format(ticker=ticker, ticker_lower=ticker.lower())
        try:
            resp = requests.get(
                filled,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36"},
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            return [e.get("title", "").strip() for e in feed.entries[:15] if e.get("title", "").strip()]
        except Exception:
            return []

    engine = get_sentiment_engine("finvader")  # fast, no GPU needed
    results: dict[str, float] = {}

    for ticker in tickers:
        all_headlines: list[str] = []
        try:
            for src in SENTIMENT_RSS_SOURCES[:3]:  # limit to top 3 sources for speed
                headlines = _safe_fetch_rss(src["url"], ticker)
                if src.get("filter_headlines"):
                    headlines = [h for h in headlines if headline_mentions_ticker(ticker, h)]
                all_headlines.extend(headlines[:10])  # max 10 per source
        except Exception:
            pass  # network errors → skip this ticker, no veto

        if all_headlines:
            scores = engine.score_batch(all_headlines[:20])  # max 20 headlines total
            results[ticker] = float(np.mean(scores)) if scores else 0.0
        else:
            results[ticker] = 0.0  # no news = no veto

    return results


def _apply_sentiment_veto(
    selected: pd.DataFrame,
    day: pd.DataFrame,
    *,
    score_col: str,
    shape: str,
    exit_rank_floor: float,
    max_per_sector: int,
    earnings_blackout_days: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Check sentiment for selected overlay stocks and veto strongly negative ones.

    PLAIN ENGLISH: After the factor model picks the top stocks, we ask:
    "Is there terrible news about any of these stocks RIGHT NOW?"
    If yes, we drop that stock and let the next-best candidate in.

    Returns (updated_selected, sentiment_scores_dict).
    """
    if not SENTIMENT_VETO_ENABLED or selected.empty:
        return selected, {}

    tickers = selected["ticker"].tolist()
    # Also fetch sentiment for a few backup candidates in case we need replacements
    backup_tickers = []
    ranked_all = day.dropna(subset=[score_col]).copy()
    ranked_all["_rank"] = ranked_all[score_col].rank(pct=True)
    ranked_all = ranked_all.sort_values("_rank", ascending=False)
    for _, row in ranked_all.iterrows():
        t = str(row["ticker"])
        if t not in tickers and len(backup_tickers) < 3:
            backup_tickers.append(t)

    print(f"  Fetching sentiment for {len(tickers)} selected + {len(backup_tickers)} backup tickers...")
    sentiments = _fetch_live_sentiment(tickers + backup_tickers)

    # Report sentiment scores
    vetoed = []
    for t in tickers:
        score = sentiments.get(t, 0.0)
        status = "✗ VETO" if score < SENTIMENT_VETO_THRESHOLD else "✓ OK"
        print(f"    {t:6s} sentiment={score:+.3f}  {status}")
        if score < SENTIMENT_VETO_THRESHOLD:
            vetoed.append(t)

    if not vetoed:
        return selected, sentiments

    # Remove vetoed stocks and re-select to fill spots
    print(f"  ⚠ Vetoed {len(vetoed)} stocks: {', '.join(vetoed)}")
    # Mark vetoed tickers in day so they won't be re-selected
    day_filtered = day[~day["ticker"].isin(vetoed)].copy()
    new_selected = _select_sticky_holdings(
        day_filtered,
        set(t for t in tickers if t not in vetoed),  # keep non-vetoed as "held"
        score_col=score_col,
        return_col=None,
        shape=shape,
        exit_rank_floor=exit_rank_floor,
        max_per_sector=max_per_sector,
        earnings_blackout_days=earnings_blackout_days,
    )
    return new_selected, sentiments


# ─────────────────────────────────────────────────────────────────────────────
# DATA FRESHNESS GATE
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: Before we generate any signal, check how old the factor data
# is.  If the newest data is more than 5 trading days old, the signal might be
# stale and we warn loudly.  If it's more than 10 trading days old, we refuse
# to write a signal at all — trading on ancient data is dangerous.

STALE_WARN_TRADING_DAYS = 5    # warn if factor data older than this
STALE_BLOCK_TRADING_DAYS = 10  # refuse to generate signal if older than this


def check_factor_freshness(
    panel: pd.DataFrame,
    warn_days: int = STALE_WARN_TRADING_DAYS,
    block_days: int = STALE_BLOCK_TRADING_DAYS,
    ignore_stale: bool = False,
) -> dict:
    """
    Check how old the factor panel data is.

    PLAIN ENGLISH: Looks at the most recent date in the factor panel and
    compares it to today.  Returns a dict with the age info and whether
    the data is fresh enough to trade on.

    Returns:
        dict with keys:
            latest_date: str — most recent date in the panel
            age_trading_days: int — how many trading days old the data is
            fresh: bool — True if data is fresh enough (no warning)
            blocked: bool — True if data is too old to generate signal
            message: str — human-readable status
    """
    latest = pd.Timestamp(panel["date"].max())
    today = pd.Timestamp.now().normalize()

    # Count trading days between latest data and today using business day range
    # (excludes weekends, but not holidays — close enough for a safety check)
    if latest >= today:
        age = 0
    else:
        age = len(pd.bdate_range(latest + pd.tseries.offsets.BDay(1), today))

    result = {
        "latest_date": str(latest.date()),
        "age_trading_days": age,
        "fresh": age <= warn_days,
        "blocked": age > block_days and not ignore_stale,
        "ignore_stale": ignore_stale,
    }

    if age <= warn_days:
        result["message"] = f"Factor data is fresh ({age} trading days old, latest={latest.date()})"
    elif age <= block_days:
        result["message"] = (
            f"⚠ WARNING: Factor data is {age} trading days old (latest={latest.date()}). "
            f"Signal will be generated but gates_all_pass will be set to False."
        )
    else:
        if ignore_stale:
            result["message"] = (
                f"⚠ OVERRIDE: Factor data is {age} trading days old (latest={latest.date()}). "
                f"--ignore-stale flag used — proceeding anyway with gates_all_pass=False."
            )
            result["blocked"] = False
        else:
            result["message"] = (
                f"✗ BLOCKED: Factor data is {age} trading days old (latest={latest.date()}). "
                f"Too stale to generate a signal (limit={block_days} trading days). "
                f"Run your data pipeline to refresh, or use --ignore-stale to override."
            )

    return result


CORE_PRESETS = {
    "qqq_tilt_40_60": {"SPY": 0.40, "QQQ": 0.60},
    "qqq_heavy_25_75": {"SPY": 0.25, "QQQ": 0.75},
}
REGIME_PRESETS = {
    "qqq_trend_switch": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.00,
            "overlay_gross": 0.25,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay35": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.90,
            "overlay_gross": 0.35,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay50": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_fast_core110_overlay15": {
        "ma_window": 75,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.10,
            "overlay_gross": 0.15,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.85,
            "overlay_gross": 0.40,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay70_core55": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.65,
            "overlay_gross": 0.35,
        },
    },
    # --- CASH BUFFER VARIANT ---
    # PLAIN ENGLISH: Same as overlay70_core55 but holds 20% cash in risk_off
    # and 10% cash in neutral.  The idea is that when the market is stressed,
    # you want LESS total exposure, not just different ETFs.  The grid search
    # will compare this variant to the original and pick whichever is better.
    "qqq_trend_switch_overlay70_core55_cashbuffer": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            # Risk_on stays fully invested — bull market, no reason for cash
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            # Neutral: reduce overlay slightly → ~10% cash buffer
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.50,
            "overlay_gross": 0.60,
        },
        "risk_off": {
            # Risk_off: pull back to 80% gross → 20% cash buffer
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.55,
            "overlay_gross": 0.25,
        },
    },
    # --- ADAPTIVE VOL + CASH BUFFER ---
    # PLAIN ENGLISH: Combines the cash buffer with percentile-based vol
    # detection.  Instead of a fixed 0.30 vol threshold, it uses the top 20%
    # of trailing 1-year vol.  This adapts to changing market conditions —
    # avoids whipsawing between regimes when overall vol shifts higher/lower.
    "qqq_trend_switch_overlay70_core55_adaptive": {
        "ma_window": 100,
        "high_vol": 0.30,           # fallback, but percentile mode overrides
        "high_vol_mode": "percentile",  # ← the new adaptive threshold
        # Blend risk_on and risk_off scores proportionally (no hard flip)
        "score_blend": True,
        # Trigger early rebalance when regime changes mid-holding-period
        "early_rebalance_on_regime_change": True,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.50,
            "overlay_gross": 0.60,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.55,
            "overlay_gross": 0.25,
        },
    },
}
CORE_OVERLAY_COMBOS = (
    (1.00, 0.25),
    (0.75, 0.25),
    (0.75, 0.50),
)
SCORE_SOURCES = ("factor_walkforward", "regime_adaptive", "regime_adaptive_low_vol", "regime_adaptive_consensus")
# PLAIN ENGLISH: "top3" picks 3 best stocks, "top5" picks 5.  More stocks
# means less concentration risk but potentially weaker alpha (diluted by
# lower-ranked picks).  Grid search tests both and cost-stress will catch
# if extra turnover from 5 names hurts net returns.
SHAPES = ("top3", "top5")
WEIGHTING_MODES = ("sticky_score", "risk_parity", "sticky_risk_parity")
EXIT_RANK_FLOORS = (0.80,)
ADAPTIVE_EXIT_MODES = ("fixed",)
# PLAIN ENGLISH: max_per_sector controls how many stocks from the same
# sector can be in the overlay at once.  (2) = up to 2 tech stocks,
# (1) = force cross-sector diversification (one per sector max).
# Grid search tests both and picks whichever gives better risk-adjusted return.
MAX_PER_SECTOR_OPTIONS = (1, 2)
EARNINGS_BLACKOUT_DAY_OPTIONS = (0, 5)
COST_STRESS_MULTIPLIERS = (2.0, 3.0, 5.0)
HOLDING_DAY_OPTIONS = (10, 20)
# PLAIN ENGLISH: Drawdown circuit breaker.  If portfolio equity drops more than
# this % from its all-time peak, we cut ALL exposure to zero (100% cash) until
# equity recovers above the re-entry threshold.  This protects against catastrophic
# losses during market crashes.
# (0.0) = disabled (no circuit breaker), (0.25) = go to cash if down 25% from peak.
# Grid search tests both to see if the protection is worth the missed recovery.
DRAWDOWN_CIRCUIT_BREAKER_OPTIONS = (0.0,)  # DD breaker hurts regime-switching configs (redundant), disabled from grid
# PLAIN ENGLISH: Portfolio volatility targeting.  We track how volatile our
# portfolio returns have been recently (rolling 20-day annualized stdev).
# If realized vol is HIGHER than our target, we shrink exposure (smaller positions).
# If realized vol is LOWER than our target, we increase exposure (bigger positions).
# This keeps risk roughly constant across calm and turbulent markets.
# (0.0) = disabled (no vol scaling).  (0.15) = target 15% annualized vol.
# Cap at 1.5x to prevent excessive leverage during very calm periods.
VOL_TARGET_OPTIONS = (0.0,)  # 0.15 tested — reduces DD but hurts Sharpe/returns, disabled
VOL_TARGET_LOOKBACK = 20        # how many past returns to use for realized vol
VOL_TARGET_MAX_SCALE = 1.5      # never scale exposure above 1.5x base
VOL_TARGET_MIN_SCALE = 0.3      # never scale exposure below 0.3x base
# PLAIN ENGLISH: When True, blend risk_on and risk_off scores proportionally
# based on regime strength (0-1).  When False, hard-switch between score types
# like before.  Grid search tests both and picks whichever works better.
# NOTE: score_blend and early_rebalance_on_regime_change are baked into
# specific preset configs (the "adaptive" variant) rather than grid-searched
# across ALL presets — this keeps the total config count manageable.
# PLAIN ENGLISH: No single overlay stock can be more than this fraction of
# the overlay sleeve.  0.25 = max 25% in one name (was 0.35).  Lower cap
# spreads risk more evenly across overlay picks, reducing blowup risk from
# one stock cratering.
MAX_SINGLE_NAME_WEIGHT = 0.25
# Broker/paper-trading safety: research can evaluate up to MAX_GROSS_EXPOSURE,
# but submitted paper signals default to no leverage unless the paper-trading
# script is deliberately run with --allow-leverage-submit.
PAPER_MAX_GROSS_EXPOSURE = 1.00
PAPER_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))

MAX_POSITIVE_YEAR_ALPHA_SHARE = 0.35
MAX_TOP_TICKER_CONTRIB_SHARE = 0.50
ROBUST_COST_STRESSES = (2.0, 3.0, 5.0)
PERIODS_PER_YEAR = 252.0 / HORIZON_DAYS
_PANEL_DAY_CACHE: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {}
_ETF_PRICE_CACHE: dict[tuple[tuple[pd.Timestamp, ...], tuple[str, ...]], pd.DataFrame] = {}


def _panel_day_map(panel: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    cache_key = id(panel)
    cached = _PANEL_DAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mapped = {pd.Timestamp(dt): day for dt, day in panel.groupby("date", sort=True)}
    _PANEL_DAY_CACHE[cache_key] = mapped
    return mapped


def _cached_etf_prices(price_index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    normalized_dates = tuple(pd.Timestamp(dt) for dt in price_index)
    key = (normalized_dates, tuple(tickers))
    cached = _ETF_PRICE_CACHE.get(key)
    if cached is not None:
        return cached
    prices = _load_etf_price_frame(pd.DatetimeIndex(normalized_dates), tickers)
    if len(prices) > 1:
        synthetic = [
            ticker
            for ticker in tickers
            if ticker in prices.columns
            and float(pd.to_numeric(prices[ticker], errors="coerce").diff().abs().fillna(0.0).sum()) == 0.0
        ]
        if synthetic:
            raise RuntimeError(
                f"ETF price data unavailable for {synthetic}; refusing to run core alpha with synthetic flat prices."
            )
    _ETF_PRICE_CACHE[key] = prices
    return prices


def _low_vol_rank(panel: pd.DataFrame) -> pd.Series:
    for col in (
        "hvol_20d",
        "factor_idio_vol_252_spy",
        "xs_rank_sector_hvol_20d",
        "xs_rank_sector_factor_idio_vol_252_spy",
    ):
        if col not in panel.columns:
            continue
        raw = pd.to_numeric(panel[col], errors="coerce")
        if raw.notna().any():
            return raw.groupby(panel["date"]).rank(pct=True)
    return pd.Series(0.50, index=panel.index)


def _rank_existing_score(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(0.50, index=panel.index)
    raw = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return raw.groupby(panel["date"]).rank(pct=True).fillna(0.50)


def _consensus_score(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    ranks = pd.concat([_rank_existing_score(panel, col).rename(col) for col in cols], axis=1)
    avg_rank = ranks.mean(axis=1, skipna=True)
    floor_rank = ranks.min(axis=1, skipna=True)
    disagreement = ranks.std(axis=1, skipna=True).fillna(0.0)
    raw = 0.70 * avg_rank + 0.25 * floor_rank - 0.10 * disagreement
    return raw.groupby(panel["date"]).rank(pct=True).fillna(avg_rank).fillna(0.50)


def _ensure_robust_score_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    low_vol_bonus = (1.0 - _low_vol_rank(out)).fillna(0.50)
    if "factor_risk_on_low_vol_score" not in out.columns:
        out["factor_risk_on_low_vol_score"] = (
            0.80 * pd.to_numeric(out["factor_risk_on_score"], errors="coerce")
            + 0.20 * low_vol_bonus
        ).fillna(out["factor_risk_on_score"])
    if "factor_walkforward_low_vol_score" not in out.columns:
        out["factor_walkforward_low_vol_score"] = (
            0.80 * pd.to_numeric(out["factor_walkforward_score"], errors="coerce")
            + 0.20 * low_vol_bonus
        ).fillna(out["factor_walkforward_score"])
    if "factor_risk_on_consensus_score" not in out.columns:
        out["factor_risk_on_consensus_score"] = _consensus_score(
            out,
            ["factor_risk_on_score", "factor_walkforward_score", "factor_risk_on_low_vol_score"],
        )
    if "factor_walkforward_consensus_score" not in out.columns:
        out["factor_walkforward_consensus_score"] = _consensus_score(
            out,
            ["factor_walkforward_score", "factor_score", "factor_walkforward_low_vol_score"],
        )
    if "factor_defensive_consensus_score" not in out.columns:
        out["factor_defensive_consensus_score"] = _consensus_score(
            out,
            ["factor_defensive_score", "factor_walkforward_score", "factor_walkforward_low_vol_score"],
        )
    return out


def _load_regime_indicators(rebalance_dates: pd.DatetimeIndex, exit_dates: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    """
    Compute trend and volatility indicators used for regime detection.

    PLAIN ENGLISH: This function decides whether we're in a bull market
    (risk_on), a mixed market (neutral), or a stressed/bear market (risk_off).
    It looks at two things:
      1. TREND: Are SPY and QQQ above their moving averages? (uptrend = good)
      2. VOLATILITY: Is QQQ moving around a lot? (high vol = bad)

    The high_vol_mode config controls how we measure "high" volatility:
      - "fixed" (default): volatility above a hard number (e.g. 0.30 = 30%)
      - "percentile": volatility in the top 20% of its trailing 1-year range
        This adapts to changing market conditions — what's "high" in a calm
        year is different from "high" in a volatile year.
    """
    start = pd.Timestamp(rebalance_dates.min()) - pd.tseries.offsets.BDay(260)
    end = pd.Timestamp(exit_dates.max())
    timeline = pd.date_range(start, end, freq="B")
    prices = _cached_etf_prices(pd.DatetimeIndex(timeline), ["SPY", "QQQ"])
    ma_window = int(config.get("regime_ma_window", 100))
    high_vol_threshold = float(config.get("regime_high_vol", 0.30))
    high_vol_mode = str(config.get("high_vol_mode", "fixed"))
    out = pd.DataFrame(index=prices.index)
    out["SPY"] = prices["SPY"]
    out["QQQ"] = prices["QQQ"]
    out["spy_trend_ok"] = prices["SPY"] >= prices["SPY"].rolling(200, min_periods=50).mean()
    out["qqq_trend_ok"] = prices["QQQ"] >= prices["QQQ"].rolling(ma_window, min_periods=50).mean()
    out["qqq_realized_vol"] = prices["QQQ"].pct_change().rolling(20, min_periods=10).std().mul(np.sqrt(252))

    if high_vol_mode == "percentile":
        # PLAIN ENGLISH: Instead of a fixed number, compare today's vol to
        # the last ~1 year of vol.  If today's vol is in the top 20%
        # (i.e. percentile rank > 0.80), that's "high."  This adapts —
        # in a calm year, 20% vol might be high; in a crazy year, 35%
        # might only be moderate.
        vol_pctile = out["qqq_realized_vol"].rolling(252, min_periods=60).rank(pct=True)
        out["high_vol"] = vol_pctile > 0.80
    else:
        # Fixed threshold — original behavior
        out["high_vol"] = out["qqq_realized_vol"] > high_vol_threshold

    return out.ffill().bfill()


def _resolve_allocation(dt: pd.Timestamp, config: dict, regime_indicators: pd.DataFrame | None) -> tuple[str, dict[str, float], float, float]:
    regime_mode = str(config.get("regime_mode", "static"))
    if regime_mode in REGIME_PRESETS and regime_indicators is not None:
        row = regime_indicators.loc[pd.Timestamp(dt)]
        if bool(row["qqq_trend_ok"]) and bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "risk_on"
        elif bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "neutral"
        else:
            regime = "risk_off"
        preset = REGIME_PRESETS[regime_mode][regime]
        return (
            regime,
            dict(preset["core_weights"]),
            float(preset["core_gross"]),
            float(preset["overlay_gross"]),
        )
    forced_regime = str(config.get("current_regime", "") or "")
    if regime_mode in REGIME_PRESETS and forced_regime in REGIME_PRESETS[regime_mode]:
        preset = REGIME_PRESETS[regime_mode][forced_regime]
        return (
            forced_regime,
            dict(preset["core_weights"]),
            float(preset["core_gross"]),
            float(preset["overlay_gross"]),
        )
    return (
        "static",
        dict(config["core_weights"]),
        float(config["core_gross"]),
        float(config["overlay_gross"]),
    )


def _top_count(n_names: int, shape: str) -> int:
    if shape == "top3":
        return 3
    if shape == "top5":
        return 5
    if shape == "top10":
        return 10
    return max(1, int(np.ceil(n_names * 0.10)))


def _score_col(source: str) -> str:
    if source == "factor_plus_model":
        return "factor_plus_model_score"
    if source == "factor_walkforward":
        return "factor_walkforward_score"
    if source == "regime_adaptive":
        return "factor_walkforward_score"
    return "factor_score"


def _compute_regime_strength(dt: pd.Timestamp, regime_indicators: pd.DataFrame | None) -> float:
    """
    Compute a continuous 0-to-1 regime strength score.

    PLAIN ENGLISH: Instead of hard-switching between "risk_on" and "risk_off"
    scores, this gives a number between 0 and 1 that says HOW MUCH we're in
    risk_on territory.

    The score is based on 3 binary signals:
      - QQQ above its moving average (+1)
      - SPY above its moving average (+1)
      - Volatility NOT high (+1)

    So: 3/3 = 1.0 (full risk_on), 0/3 = 0.0 (full risk_off),
    1/3 or 2/3 = in between.  This lets us blend offensive and defensive
    scores proportionally rather than flipping all at once.
    """
    if regime_indicators is None:
        return 1.0  # No indicators → assume risk_on (default behavior)
    try:
        row = regime_indicators.loc[pd.Timestamp(dt)]
        signals = [
            bool(row["qqq_trend_ok"]),
            bool(row["spy_trend_ok"]),
            not bool(row["high_vol"]),
        ]
        return sum(signals) / len(signals)
    except (KeyError, IndexError):
        return 1.0


def _blended_score_col(
    day: pd.DataFrame,
    source: str,
    regime: str,
    regime_strength: float,
) -> str:
    """
    Create a blended score column in `day` that mixes risk_on and risk_off scores.

    PLAIN ENGLISH: When regime_strength is 1.0, you get pure risk_on scores
    (momentum).  When it's 0.0, you get pure risk_off scores (defensive).
    In between, you get a weighted average.  This prevents the portfolio from
    flipping overnight from all-momentum to all-defensive.

    Returns the name of the (possibly new) column in `day` that contains
    the blended scores.
    """
    # If strength is all-or-nothing, just use the existing column (no blend needed)
    if regime_strength >= 1.0:
        return _score_col_for_regime(source, "risk_on")
    if regime_strength <= 0.0:
        return _score_col_for_regime(source, "risk_off")

    risk_on_col = _score_col_for_regime(source, "risk_on")
    risk_off_col = _score_col_for_regime(source, "risk_off")

    # If both columns are the same (e.g. non-regime-adaptive source), no blend
    if risk_on_col == risk_off_col:
        return risk_on_col

    # Check both columns exist
    if risk_on_col not in day.columns or risk_off_col not in day.columns:
        return _score_col_for_regime(source, regime)

    # Blend: strength * risk_on + (1-strength) * risk_off
    # NOTE: We use a unique column name per strength level to avoid
    # overwriting the cached day DataFrame when multiple configs share it.
    # Rounding strength to 2 decimals keeps the number of columns manageable.
    strength_key = f"{regime_strength:.2f}".replace(".", "_")
    blend_col = f"_blended_regime_score_{strength_key}"
    if blend_col not in day.columns:
        day[blend_col] = (
            regime_strength * day[risk_on_col].fillna(0)
            + (1 - regime_strength) * day[risk_off_col].fillna(0)
        )
    return blend_col


def _score_col_for_regime(source: str, regime: str) -> str:
    if source == "regime_adaptive_consensus":
        if regime == "risk_on":
            return "factor_risk_on_consensus_score"
        if regime == "risk_off":
            return "factor_defensive_consensus_score"
        return "factor_walkforward_consensus_score"
    if source == "regime_adaptive_low_vol":
        if regime == "risk_on":
            return "factor_risk_on_low_vol_score"
        if regime == "risk_off":
            return "factor_defensive_score"
        return "factor_walkforward_low_vol_score"
    if source != "regime_adaptive":
        return _score_col(source)
    if regime == "risk_on":
        return "factor_risk_on_score"
    if regime == "risk_off":
        return "factor_defensive_score"
    return "factor_walkforward_score"


def _select_sticky_holdings(
    day: pd.DataFrame,
    held: set[str],
    *,
    score_col: str,
    return_col: str | None,
    shape: str,
    exit_rank_floor: float,
    max_per_sector: int,
    earnings_blackout_days: int = 0,
) -> pd.DataFrame:
    required_cols = [score_col] if return_col is None else [score_col, return_col]
    ranked = day.dropna(subset=required_cols).copy()
    if ranked.empty:
        return ranked
    ranked["_rank_score"] = ranked[score_col].rank(pct=True)
    if earnings_blackout_days > 0 and "days_to_next_earnings" in ranked.columns:
        days_to_earnings = pd.to_numeric(ranked["days_to_next_earnings"], errors="coerce")
        ranked["_earnings_blackout"] = days_to_earnings.between(0, earnings_blackout_days, inclusive="both")
    else:
        ranked["_earnings_blackout"] = False
    ranked = ranked.sort_values("_rank_score", ascending=False)
    target_n = _top_count(len(ranked), shape)

    keep = ranked[(ranked["ticker"].isin(held)) & (ranked["_rank_score"] >= exit_rank_floor)]
    selected_rows = []
    sector_counts: dict[str, int] = {}

    def _can_add(row: pd.Series) -> bool:
        if max_per_sector <= 0:
            return True
        sector = str(row.get("sector", "OTHER"))
        return sector_counts.get(sector, 0) < max_per_sector

    def _add(row: pd.Series) -> None:
        selected_rows.append(row)
        sector = str(row.get("sector", "OTHER"))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for _idx, row in keep.sort_values("_rank_score", ascending=False).iterrows():
        if len(selected_rows) >= target_n:
            break
        if _can_add(row):
            _add(row)
    selected_tickers = {str(row["ticker"]) for row in selected_rows}

    for _idx, row in ranked.iterrows():
        if len(selected_rows) >= target_n:
            break
        ticker = str(row["ticker"])
        if ticker in selected_tickers:
            continue
        if bool(row.get("_earnings_blackout", False)):
            continue
        if _can_add(row):
            _add(row)
            selected_tickers.add(ticker)

    if not selected_rows:
        return ranked.iloc[0:0]
    return pd.DataFrame(selected_rows)


def _cap_and_rescale(weights: pd.Series, gross: float, cap: float) -> pd.Series:
    if weights.empty or cap <= 0:
        return weights
    capped = weights.clip(upper=cap)
    for _ in range(10):
        remaining = gross - float(capped.sum())
        if remaining <= 1e-9:
            break
        free = capped < cap - 1e-12
        if not bool(free.any()):
            break
        free_sum = float(capped[free].sum())
        if free_sum <= 0:
            capped.loc[free] = capped.loc[free] + remaining / int(free.sum())
        else:
            capped.loc[free] = capped.loc[free] + remaining * capped.loc[free] / free_sum
        capped = capped.clip(upper=cap)
    total = float(capped.sum())
    if total > 0 and total < gross - 1e-6:
        capped = capped / total * min(gross, total)
    return capped


def _overlay_weights(
    selected: pd.DataFrame,
    overlay_gross: float,
    weighting: str,
    *,
    max_single_name_weight: float = MAX_SINGLE_NAME_WEIGHT,
) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    if weighting == "score":
        raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
        raw_sum = float(raw.sum())
        weights = raw / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    elif weighting == "vol_score":
        raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
        if "hvol_20d" in selected.columns:
            vol = pd.to_numeric(selected["hvol_20d"], errors="coerce")
        elif "factor_idio_vol_252_spy" in selected.columns:
            vol = pd.to_numeric(selected["factor_idio_vol_252_spy"], errors="coerce")
        else:
            vol = pd.Series(0.20, index=selected.index)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median()).clip(lower=0.05, upper=2.0)
        adjusted = raw / vol
        raw_sum = float(adjusted.sum())
        weights = adjusted / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    elif weighting == "risk_parity":
        # PLAIN ENGLISH: Equal Risk Contribution (ERC) approximation.
        # Instead of weighting by score (which ignores risk), we weight
        # inversely by each stock's recent volatility.  A stock with 40%
        # annualized vol gets HALF the position size of one with 20% vol.
        # This way each stock contributes roughly equal risk to the portfolio.
        # With only 3-5 stocks, inverse-vol is the standard ERC approximation
        # (full covariance-based ERC is too noisy with so few names).
        if "hvol_20d" in selected.columns:
            vol = pd.to_numeric(selected["hvol_20d"], errors="coerce")
        elif "factor_idio_vol_252_spy" in selected.columns:
            vol = pd.to_numeric(selected["factor_idio_vol_252_spy"], errors="coerce")
        else:
            vol = pd.Series(0.20, index=selected.index)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median()).clip(lower=0.05, upper=2.0)
        inv_vol = 1.0 / vol
        raw_sum = float(inv_vol.sum())
        weights = inv_vol / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    else:
        weights = pd.Series(overlay_gross / len(selected), index=selected.index)
    weights.index = selected["ticker"].astype(str).to_numpy()
    weights = weights.astype(float)
    if max_single_name_weight > 0 and float(weights.max()) > max_single_name_weight:
        weights = _cap_and_rescale(weights, overlay_gross, max_single_name_weight)
    return weights


def _sticky_overlay_weights(
    selected: pd.DataFrame,
    overlay_gross: float,
    weighting: str,
    prev_overlay: pd.Series,
    *,
    max_single_name_weight: float = MAX_SINGLE_NAME_WEIGHT,
    sticky_blend: float = 0.65,
) -> pd.Series:
    # PLAIN ENGLISH: Map the "sticky_X" wrapper name to the base weighting method.
    # "sticky_score" → base "score", "sticky_risk_parity" → base "risk_parity".
    # The stickiness logic below blends the new weights with previous weights to
    # reduce unnecessary turnover (and thus trading costs).
    STICKY_MAP = {"sticky_score": "score", "sticky_risk_parity": "risk_parity"}
    is_sticky = weighting in STICKY_MAP
    current_weighting = STICKY_MAP.get(weighting, weighting)
    current = _overlay_weights(
        selected,
        overlay_gross,
        current_weighting,
        max_single_name_weight=max_single_name_weight,
    )
    if current.empty or prev_overlay.empty or not is_sticky:
        return current
    retained = current.index.intersection(prev_overlay.index)
    if retained.empty:
        return current
    prev = prev_overlay.reindex(current.index).fillna(0.0)
    if float(prev.abs().sum()) > 0:
        prev = prev / float(prev.abs().sum()) * overlay_gross
    blended = (1.0 - sticky_blend) * current + sticky_blend * prev
    missing = current.index.difference(retained)
    if len(missing) > 0:
        blended.loc[missing] = current.loc[missing]
    total = float(blended.abs().sum())
    if total > 0:
        blended = blended / total * overlay_gross
    if max_single_name_weight > 0 and float(blended.max()) > max_single_name_weight:
        blended = _cap_and_rescale(blended, overlay_gross, max_single_name_weight)
    return blended.astype(float)


def _exit_floor_for_regime(config: dict, regime: str) -> float:
    base = float(config.get("exit_rank_floor", 0.80))
    if str(config.get("adaptive_exit_mode", "fixed")) != "regime":
        return base
    if regime == "risk_on":
        return min(base, 0.70)
    if regime == "risk_off":
        return max(base, 0.90)
    return base


def run_core_satellite(panel: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame, dict]:
    holding_days = int(config.get("holding_days", HORIZON_DAYS))
    entry_delay_days = int(config.get("entry_delay_days", 0))
    if entry_delay_days > 0:
        return_col = f"forward_return_delay{entry_delay_days}_{holding_days}d"
    else:
        return_col = "forward_return" if holding_days == HORIZON_DAYS else f"forward_return_{holding_days}d"
    if return_col not in panel.columns:
        raise ValueError(f"Missing return column for core-satellite run: {return_col}")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebalance_dates = dates[::holding_days]
    entry_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days) for dt in rebalance_dates])
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days) for dt in rebalance_dates])
    price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(entry_dates) | set(exit_dates)))
    etf_prices = _cached_etf_prices(price_index, ["SPY", "QQQ"])
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in REGIME_PRESETS:
        regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)

    # --- EARLY REGIME-CHANGE REBALANCE ---
    # PLAIN ENGLISH: Normally we only rebalance every N days (e.g. every 10
    # trading days).  But if the regime changes in the middle of a holding
    # period (e.g. risk_on → risk_off on day 3), we're stuck with the wrong
    # positioning for 7 more days.
    #
    # When early_rebalance_on_regime_change is True, we check every trading
    # day for regime changes and insert extra rebalance dates when the regime
    # flips.  This means the portfolio adapts faster to market shifts.
    #
    # Risk: More rebalances = more turnover = more cost.  The grid search
    # will test both settings and pick whichever gives better net returns.
    if config.get("early_rebalance_on_regime_change", False) and regime_indicators is not None:
        extra_dates = []
        prev_regime = None
        for dt in dates:
            ts = pd.Timestamp(dt)
            if ts not in regime_indicators.index:
                continue
            row = regime_indicators.loc[ts]
            if bool(row["qqq_trend_ok"]) and bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
                cur_regime = "risk_on"
            elif bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
                cur_regime = "neutral"
            else:
                cur_regime = "risk_off"

            if prev_regime is not None and cur_regime != prev_regime:
                # Regime changed — add this date as a rebalance point
                if ts not in rebalance_dates:
                    extra_dates.append(ts)
            prev_regime = cur_regime

        if extra_dates:
            # Merge extra dates into rebalance schedule and re-sort
            combined = sorted(set(rebalance_dates) | set(extra_dates))
            rebalance_dates = pd.DatetimeIndex(combined)
            # Recompute entry/exit for all dates
            entry_dates = pd.DatetimeIndex([
                pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days)
                for dt in rebalance_dates
            ])
            exit_dates = pd.DatetimeIndex([
                pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
                for dt in rebalance_dates
            ])
            price_index = pd.DatetimeIndex(sorted(
                set(rebalance_dates) | set(entry_dates) | set(exit_dates)
            ))
            etf_prices = _cached_etf_prices(price_index, ["SPY", "QQQ"])
            # Reload regime indicators with expanded date range
            regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)
    day_map = _panel_day_map(panel)

    equity = INITIAL_CAPITAL
    held: set[str] = set()
    prev_overlay = pd.Series(dtype=float)
    rows = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0
    ticker_contrib: dict[str, float] = {}

    # ── Drawdown circuit breaker state ─────────────────────────────────
    # PLAIN ENGLISH: Track the highest equity we've ever had.  If current
    # equity drops too far below that peak, go to 100% cash.  We stay in
    # cash until the REGIME turns back to risk_on — because cash doesn't
    # grow, so we can't wait for portfolio equity to "recover" on its own.
    # Instead we wait for the market to show strength again (regime = risk_on)
    # before re-entering.
    dd_threshold = float(config.get("drawdown_circuit_breaker", 0.0))
    peak_equity = equity       # highest equity seen so far
    circuit_breaker_active = False  # True = we're in cash, waiting for risk_on

    # ── Volatility targeting state ─────────────────────────────────────
    # PLAIN ENGLISH: Keep a rolling window of recent portfolio returns so
    # we can compute realized volatility.  If a target is set, we scale
    # exposure each period to keep vol near the target.
    vol_target = float(config.get("vol_target", 0.0))
    recent_returns: list[float] = []  # rolling window of strategy returns

    for dt in rebalance_dates:
        day = day_map[pd.Timestamp(dt)]

        regime, core_weights, core_gross, overlay_gross = _resolve_allocation(pd.Timestamp(dt), config, regime_indicators)

        # ── Check circuit breaker ──────────────────────────────────────
        # PLAIN ENGLISH: If the circuit breaker threshold is set (> 0):
        #   1. If breaker is OFF: check if we've dropped too far from peak.
        #      If yes, go to 100% cash (activate breaker).
        #   2. If breaker is ON: check if regime is back to risk_on.
        #      If yes, deactivate breaker and re-enter the market.
        #      We use regime (not equity recovery) because cash doesn't grow —
        #      we'd be stuck forever waiting for equity to recover while in cash.
        if dd_threshold > 0.0:
            drawdown_from_peak = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
            if circuit_breaker_active:
                # Re-enter when regime turns risk_on (market showing strength)
                if regime == "risk_on":
                    circuit_breaker_active = False
                    # Reset peak to current equity so we don't immediately re-trigger
                    peak_equity = equity
            else:
                # Trigger breaker if drawdown exceeds threshold
                if drawdown_from_peak >= dd_threshold:
                    circuit_breaker_active = True

        # If circuit breaker is active, override to zero exposure (100% cash)
        if circuit_breaker_active:
            core_gross = 0.0
            overlay_gross = 0.0

        # ── Volatility targeting ──────────────────────────────────────────
        # PLAIN ENGLISH: If vol targeting is enabled, compute how volatile
        # our recent returns have been.  If we're running hotter than target,
        # shrink positions.  If calmer, grow them (up to 1.5x cap).
        # Need at least VOL_TARGET_LOOKBACK periods of history before scaling.
        if vol_target > 0.0 and len(recent_returns) >= VOL_TARGET_LOOKBACK:
            recent_arr = np.array(recent_returns[-VOL_TARGET_LOOKBACK:])
            realized_vol = float(np.std(recent_arr, ddof=1)) * np.sqrt(252.0 / holding_days)
            if realized_vol > 1e-6:
                vol_scale = float(np.clip(
                    vol_target / realized_vol,
                    VOL_TARGET_MIN_SCALE,
                    VOL_TARGET_MAX_SCALE,
                ))
                core_gross *= vol_scale
                overlay_gross *= vol_scale

        # PLAIN ENGLISH: If score blending is enabled, we mix risk_on and
        # risk_off scores based on how "risk_on" the market really is (0-1).
        # This avoids the jarring overnight flip from all-momentum to all-defensive.
        if config.get("score_blend", False):
            strength = _compute_regime_strength(pd.Timestamp(dt), regime_indicators)
            score_col = _blended_score_col(day, str(config["score_source"]), regime, strength)
        else:
            score_col = _score_col_for_regime(str(config["score_source"]), regime)
        exit_floor = _exit_floor_for_regime(config, regime)
        selected = _select_sticky_holdings(
            day,
            held,
            score_col=score_col,
            return_col=return_col,
            shape=config["shape"],
            exit_rank_floor=exit_floor,
            max_per_sector=int(config["max_per_sector"]),
            earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
        )
        overlay = _sticky_overlay_weights(
            selected,
            overlay_gross,
            config["weighting"],
            prev_overlay,
            max_single_name_weight=float(config.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
        )
        held = set(overlay.index.astype(str))

        aligned = pd.concat([prev_overlay.rename("prev"), overlay.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        extra_cost = turnover * float(config.get("extra_turnover_cost_bps", 0.0)) / 10_000.0
        cost = turnover * SLIPPAGE_BASE_PCT * float(config.get("cost_stress", 1.0)) + extra_cost
        total_turnover += turnover
        total_cost += cost

        # ── Compute exit date and holding period scaling ────────────────
        # PLAIN ENGLISH: If early rebalance is enabled, the ACTUAL holding period
        # might be shorter than holding_days (because we rebalance early on regime
        # change).  We use the actual next rebalance date as exit, not the full
        # holding_days.  This prevents double-counting returns across overlapping
        # periods.
        entry_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days)
        try:
            dt_idx = rebalance_dates.get_loc(dt)
        except KeyError:
            dt_idx = -1
        if isinstance(dt_idx, int) and dt_idx >= 0 and dt_idx + 1 < len(rebalance_dates):
            next_rebal = pd.Timestamp(rebalance_dates[dt_idx + 1])
            exit_dt = next_rebal + pd.tseries.offsets.BDay(entry_delay_days)
        else:
            exit_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
        # Compute actual days held for scaling factor returns
        actual_bdays = max(1, len(pd.bdate_range(entry_dt, exit_dt)) - 1)
        # PLAIN ENGLISH: factor_scale adjusts pre-computed forward returns
        # (which assume full holding_days) to the actual shorter period.
        # E.g. if expected 10 days but held only 3, multiply returns by 3/10.
        factor_scale = actual_bdays / max(1, holding_days)

        factor_ret = 0.0
        max_sector_weight = 0.0
        effective_overlay_names = 0.0
        if not overlay.empty:
            selected_by_ticker = selected.set_index("ticker")
            ticker_returns = selected_by_ticker.loc[overlay.index, return_col]
            ticker_period_contrib = ticker_returns * overlay * factor_scale
            factor_ret = float(ticker_period_contrib.sum())
            for ticker, value in ticker_period_contrib.items():
                ticker_contrib[str(ticker)] = ticker_contrib.get(str(ticker), 0.0) + float(value)
            sector_weights = selected_by_ticker.loc[overlay.index, "sector"].to_frame().join(overlay.rename("weight"))
            max_sector_weight = float(sector_weights.groupby("sector")["weight"].sum().abs().max())
            norm_w = overlay.abs() / max(float(overlay.abs().sum()), 1e-9)
            effective_overlay_names = float(1.0 / max(float((norm_w ** 2).sum()), 1e-9))

        spy_ret = float(etf_prices.loc[exit_dt, "SPY"] / etf_prices.loc[entry_dt, "SPY"] - 1.0)
        qqq_ret = float(etf_prices.loc[exit_dt, "QQQ"] / etf_prices.loc[entry_dt, "QQQ"] - 1.0)
        core_ret = core_gross * (float(core_weights["SPY"]) * spy_ret + float(core_weights["QQQ"]) * qqq_ret)
        strategy_ret = core_ret + factor_ret - cost
        equity *= 1.0 + strategy_ret
        # Track returns for vol targeting (add BEFORE updating peak)
        recent_returns.append(strategy_ret)
        # Update peak equity for circuit breaker tracking
        if equity > peak_equity:
            peak_equity = equity

        rows.append({"date": exit_dt, "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "exit_date": exit_dt,
            "regime": regime,
            "circuit_breaker_active": circuit_breaker_active,
            "score_col": score_col,
            "adaptive_exit_mode": str(config.get("adaptive_exit_mode", "fixed")),
            "exit_rank_floor_used": exit_floor,
            "earnings_blackout_days": int(config.get("earnings_blackout_days", 0)),
            "entry_delay_days": entry_delay_days,
            "extra_turnover_cost_bps": float(config.get("extra_turnover_cost_bps", 0.0)),
            "n_overlay_positions": int(len(overlay)),
            "core_spy_weight": float(core_weights["SPY"]),
            "core_qqq_weight": float(core_weights["QQQ"]),
            "core_gross": core_gross,
            "overlay_gross": overlay_gross,
            "gross_exposure": core_gross + float(overlay.abs().sum()),
            "top_overlay_weight": float(overlay.abs().max()) if not overlay.empty else 0.0,
            "effective_overlay_names": effective_overlay_names,
            "max_sector_overlay_weight": max_sector_weight,
            "overlay_tickers": ",".join(overlay.index.astype(str).tolist()),
            "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
            "turnover": turnover,
            "cost": cost,
            "core_return": core_ret,
            "factor_overlay_return": factor_ret,
            "period_return": strategy_ret,
        })
        prev_overlay = overlay

    equity_series = pd.DataFrame(rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    trades = pd.DataFrame(trade_rows)
    positive_contrib = {k: v for k, v in ticker_contrib.items() if v > 0}
    positive_contrib_sum = sum(positive_contrib.values())
    if positive_contrib and positive_contrib_sum > 0:
        top_ticker, top_value = max(positive_contrib.items(), key=lambda kv: kv[1])
        top_ticker_share = float(top_value / positive_contrib_sum)
    else:
        top_ticker, top_ticker_share = "", 1.0
    extra = {
        "turnover_pct": round(total_turnover * 100.0, 2),
        "estimated_cost_pct": round(total_cost * 100.0, 4),
        "avg_gross_exposure": round(float(trades["gross_exposure"].mean()), 3) if not trades.empty else 0.0,
        "avg_overlay_positions": round(float(trades["n_overlay_positions"].mean()), 2) if not trades.empty else 0.0,
        "max_single_name_weight": round(float(trades["top_overlay_weight"].max()), 4) if not trades.empty else 0.0,
        "avg_effective_overlay_names": round(float(trades["effective_overlay_names"].mean()), 3) if not trades.empty else 0.0,
        "max_sector_overlay_weight": round(float(trades["max_sector_overlay_weight"].max()), 4) if not trades.empty else 0.0,
        "top_ticker_overlay_contributor": top_ticker,
        "top_ticker_overlay_contribution_share": round(top_ticker_share, 3),
        "n_rebalances": int(len(trades)),
    }
    if "regime" in trades.columns:
        extra["regime_counts"] = {str(k): int(v) for k, v in trades["regime"].value_counts().sort_index().items()}
    return equity_series, trades, extra


def evaluate(panel: pd.DataFrame, config: dict) -> tuple[dict, pd.Series, pd.DataFrame]:
    equity, trades, extra = run_core_satellite(panel, config)
    periods_per_year = 252.0 / int(config.get("holding_days", HORIZON_DAYS))
    stats = portfolio_stats(equity, periods_per_year)
    bench = benchmark_equity(pd.DatetimeIndex(equity.index))
    bench_stats = {symbol: portfolio_stats(bench[symbol], periods_per_year) for symbol in bench.columns}
    comps = compare_to_benchmarks(equity, bench)
    subs = subperiod_metrics(equity, bench)
    holdout = _holdout_comparisons(equity, bench, start="2023-01-01", end="2026-12-31")
    strat_rets = equity.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity.index.year).sum() * 100.0
    metrics = {
        **config,
        **stats,
        **extra,
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "holdout_2023_2026": holdout,
        "yearly_alpha_pct": {str(k): round(float(v), 2) for k, v in yearly_alpha.items()},
    }
    gates = gate_metrics(metrics, bench_stats, subs, yearly_alpha)
    gates.update(_core_robust_gate_overrides(metrics, holdout, yearly_alpha))
    gates["all_pass"] = all(v for k, v in gates.items() if k.endswith("_pass"))
    metrics["core_satellite_gate_results"] = gates
    metrics["paper_ready"] = False
    return metrics, equity, trades


def _holdout_comparisons(equity: pd.Series, bench: pd.DataFrame, *, start: str, end: str) -> dict:
    eq = equity.loc[(equity.index >= start) & (equity.index <= end)]
    if len(eq) < 3:
        return {"data_available": False}
    out: dict[str, object] = {"data_available": True}
    eq_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    out["strategy_return_pct"] = round(eq_ret * 100.0, 2)
    for symbol in ("SPY", "QQQ", "BLEND"):
        b = bench[symbol].reindex(eq.index).ffill().bfill()
        bm_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
        out[f"{symbol.lower()}_return_pct"] = round(bm_ret * 100.0, 2)
        out[f"alpha_vs_{symbol.lower()}_pct"] = round((eq_ret - bm_ret) * 100.0, 2)
    return out


def _core_robust_gate_overrides(metrics: dict, holdout: dict, yearly_alpha: pd.Series) -> dict:
    positive_alpha_years = yearly_alpha[yearly_alpha > 0]
    if not positive_alpha_years.empty:
        year_share = float(positive_alpha_years.max() / max(float(positive_alpha_years.sum()), 1e-9))
    else:
        year_share = 1.0
    top_ticker_share = float(metrics.get("top_ticker_overlay_contribution_share", 1.0) or 1.0)
    max_weight = float(metrics.get("max_single_name_weight", 1.0) or 1.0)
    holdout_available = bool(holdout.get("data_available", False))
    return {
        "year_alpha_concentration_pass": bool(year_share <= MAX_POSITIVE_YEAR_ALPHA_SHARE),
        "holdout_2023_2026_vs_qqq_pass": bool(holdout_available and float(holdout.get("alpha_vs_qqq_pct", -999.0)) > 0),
        "holdout_2023_2026_vs_blend_pass": bool(holdout_available and float(holdout.get("alpha_vs_blend_pct", -999.0)) > 0),
        "single_name_weight_cap_pass": bool(max_weight <= MAX_SINGLE_NAME_WEIGHT + 1e-9),
        "top_ticker_contribution_pass": bool(top_ticker_share <= MAX_TOP_TICKER_CONTRIB_SHARE),
        "max_positive_year_alpha_share": round(year_share, 3),
    }



def _scale_paper_targets_to_gross(
    *,
    target_spy: float,
    target_qqq: float,
    overlay: pd.Series,
    max_gross: float = PAPER_MAX_GROSS_EXPOSURE,
) -> tuple[float, float, pd.Series, float, float, bool]:
    """
    Convert a research allocation into a broker-safe paper allocation.

    Backtests may intentionally test 1.25x gross exposure, but paper order
    submission should default to <= 1.00x gross so Moomoo does not reject/cancel
    buy orders because of insufficient buying power.
    """
    raw_gross = float(abs(target_spy) + abs(target_qqq) + float(overlay.abs().sum()))
    if raw_gross <= 0 or raw_gross <= float(max_gross) + 1e-9:
        return target_spy, target_qqq, overlay.copy(), raw_gross, 1.0, False
    scale = float(max_gross) / raw_gross
    return target_spy * scale, target_qqq * scale, overlay * scale, raw_gross, scale, True


def _paper_signal_timestamp() -> str:
    try:
        tz = ZoneInfo(str(PAPER_SIGNAL_TIMEZONE))
    except Exception:
        tz = datetime.now().astimezone().tzinfo
    return datetime.now(tz).isoformat(timespec="minutes")


def write_paper_signal(panel: pd.DataFrame, metrics: dict) -> Path:
    holding_days = int(metrics.get("holding_days", HORIZON_DAYS))
    latest_date = pd.Timestamp(panel["date"].max())
    regime_indicators = None
    if str(metrics.get("regime_mode", "static")) in REGIME_PRESETS:
        try:
            regime_indicators = _load_regime_indicators(
                pd.DatetimeIndex([latest_date]),
                pd.DatetimeIndex([latest_date]),
                metrics,
            )
        except Exception as exc:
            if str(metrics.get("current_regime", "")) not in REGIME_PRESETS[str(metrics.get("regime_mode", "static"))]:
                raise
            print(f"Warning: using last known regime {metrics.get('current_regime')} because regime indicator refresh failed: {exc}")
    current_regime, core_weights, core_gross, overlay_gross = _resolve_allocation(latest_date, metrics, regime_indicators)
    score_col = _score_col_for_regime(str(metrics["score_source"]), current_regime)
    day = panel[panel["date"] == latest_date]
    selected = _select_sticky_holdings(
        day,
        set(),
        score_col=score_col,
        return_col=None,
        shape=str(metrics["shape"]),
        exit_rank_floor=float(metrics["exit_rank_floor"]),
        max_per_sector=int(metrics["max_per_sector"]),
        earnings_blackout_days=int(metrics.get("earnings_blackout_days", 0)),
    )

    # ── Sentiment veto: remove stocks with very negative recent news ──────
    # PLAIN ENGLISH: Check if any selected stock has terrible news right now.
    # If so, drop it and pick the next-best candidate from the factor rankings.
    sentiment_scores = {}
    if SENTIMENT_VETO_ENABLED and not selected.empty:
        try:
            selected, sentiment_scores = _apply_sentiment_veto(
                selected, day,
                score_col=score_col,
                shape=str(metrics["shape"]),
                exit_rank_floor=float(metrics["exit_rank_floor"]),
                max_per_sector=int(metrics["max_per_sector"]),
                earnings_blackout_days=int(metrics.get("earnings_blackout_days", 0)),
            )
        except Exception as exc:
            print(f"  ⚠ Sentiment veto failed (proceeding without): {exc}")

    overlay = _sticky_overlay_weights(
        selected,
        overlay_gross,
        str(metrics["weighting"]),
        pd.Series(dtype=float),
        max_single_name_weight=float(metrics.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
    )
    raw_target_spy = core_gross * float(core_weights["SPY"])
    raw_target_qqq = core_gross * float(core_weights["QQQ"])
    raw_gross = core_gross + float(overlay.abs().sum())
    target_spy, target_qqq, paper_overlay, raw_paper_gross, paper_scale, paper_scaled = _scale_paper_targets_to_gross(
        target_spy=raw_target_spy,
        target_qqq=raw_target_qqq,
        overlay=overlay,
        max_gross=PAPER_MAX_GROSS_EXPOSURE,
    )
    gross = abs(target_spy) + abs(target_qqq) + float(paper_overlay.abs().sum())
    row = {
        "paper_signal_type": "core_satellite_alpha",
        "paper_ready": bool(metrics.get("paper_ready", False)),
        "core_preset": metrics["core_preset"],
        "regime_mode": str(metrics.get("regime_mode", "static")),
        "current_regime": current_regime,
        "score_source": metrics["score_source"],
        "target_spy_weight": round(target_spy, 4),
        "target_qqq_weight": round(target_qqq, 4),
        "target_cash_weight": round(1.0 - gross, 4),
        "gross_exposure": round(gross, 4),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "paper_max_gross_exposure": PAPER_MAX_GROSS_EXPOSURE,
        "raw_research_gross_exposure": round(raw_gross, 4),
        "paper_weight_scale": round(paper_scale, 8),
        "paper_weights_scaled": bool(paper_scaled),
        "overlay_gross": round(float(paper_overlay.abs().sum()), 4),
        "raw_overlay_gross": round(float(overlay.abs().sum()), 4),
        "core_gross": round(abs(target_spy) + abs(target_qqq), 4),
        "raw_core_gross": round(core_gross, 4),
        "max_single_name_weight": round(float(metrics.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)), 4),
        "robust_cost_stress_pass": bool(metrics.get("robust_cost_stress_pass", False)),
        "adaptive_exit_mode": str(metrics.get("adaptive_exit_mode", "fixed")),
        "earnings_blackout_days": int(metrics.get("earnings_blackout_days", 0)),
        "holding_days": holding_days,
        "overlay_tickers": ",".join(paper_overlay.index.astype(str).tolist()),
        "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in paper_overlay.items()}, sort_keys=True),
        "raw_overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
        "single_name_stock_picker_enabled": False,
        "ml_overlay_enabled": bool(metrics["score_source"] == "factor_plus_model"),
        "factor_overlay_enabled": True,
        "latest_factor_date": str(latest_date.date()),
        "cost_stress": float(metrics.get("cost_stress", 1.0)),
        "gates_all_pass": bool(metrics.get("core_satellite_gate_results", {}).get("all_pass", False)),
        "reason": "hardened core-satellite gates pass" if metrics.get("paper_ready") else "hardened core-satellite gates have not passed",
        "predicted_at": _paper_signal_timestamp(),
        "sentiment_veto_enabled": SENTIMENT_VETO_ENABLED,
        "sentiment_scores_json": json.dumps(
            {k: round(v, 4) for k, v in sentiment_scores.items()}, sort_keys=True
        ) if sentiment_scores else "{}",
    }
    out = Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    return out


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "n/a"


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "n/a"


def print_run_summary(grid: pd.DataFrame, metrics: dict, signal_path: Path) -> None:
    comps = metrics["benchmark_comparisons"]
    gates = metrics["core_satellite_gate_results"]
    subperiods = metrics["subperiods"]

    print("\nCore-Satellite Alpha Summary")
    print("=" * 30)
    print(f"Status:        {'PAPER READY' if metrics.get('paper_ready') else 'NOT READY'}")
    print(f"Selected:      {metrics.get('core_preset')} ({metrics.get('regime_mode', 'static')})")
    print(f"Holding days:  {metrics.get('holding_days')}")
    print(f"Cost stress:   {_fmt_num(metrics.get('cost_stress'), 1)}x")
    vt = metrics.get("vol_target", 0.0)
    if vt and float(vt) > 0:
        print(f"Vol target:    {float(vt)*100:.0f}%")
    print(f"Regime counts: {metrics.get('regime_counts', {})}")

    print("\nPerformance")
    print(f"  Total return: {_fmt_pct(metrics.get('total_return_pct'))}")
    print(f"  CAGR:         {_fmt_pct(metrics.get('cagr_pct'))}")
    print(f"  Sharpe:       {_fmt_num(metrics.get('sharpe'))}")
    print(f"  Max DD:       {_fmt_pct(metrics.get('max_drawdown_pct'))}")
    print(f"  Turnover:     {_fmt_pct(metrics.get('turnover_pct'))}")
    print(f"  Est. costs:   {_fmt_pct(metrics.get('estimated_cost_pct'))}")
    print(f"  Max name wt:  {_fmt_pct(float(metrics.get('max_single_name_weight', 0.0)) * 100.0)}")
    print(f"  Eff. names:   {_fmt_num(metrics.get('avg_effective_overlay_names'))}")

    print("\nBenchmark Alpha")
    for symbol in ("SPY", "QQQ", "BLEND"):
        comp = comps.get(symbol, {})
        print(
            f"  vs {symbol:<5} alpha {_fmt_pct(comp.get('alpha_pct'))}"
            f" | benchmark {_fmt_pct(comp.get('benchmark_return_pct'))}"
            f" | t-stat {_fmt_num(comp.get('nw_tstat_vs_benchmark'))}"
        )

    holdout = metrics.get("holdout_2023_2026", {})
    if holdout.get("data_available"):
        print("\nHoldout 2023-2026")
        print(
            f"  strategy {_fmt_pct(holdout.get('strategy_return_pct'))}"
            f" | alpha vs QQQ {_fmt_pct(holdout.get('alpha_vs_qqq_pct'))}"
            f" | alpha vs BLEND {_fmt_pct(holdout.get('alpha_vs_blend_pct'))}"
        )

    stress = metrics.get("robust_cost_stress_summary", {})
    if stress:
        print("\nCost Stress")
        print(
            f"  levels {stress.get('cost_levels', 'n/a')}"
            f" | min alpha vs QQQ {_fmt_pct(stress.get('min_alpha_vs_qqq_pct'))}"
            f" | pass {'YES' if metrics.get('robust_cost_stress_pass') else 'NO'}"
        )

    print("\nSubperiod Alpha vs 60/40")
    for name, row in subperiods.items():
        if not row.get("data_available"):
            print(f"  {name:<9} n/a")
            continue
        print(
            f"  {name:<9} alpha {_fmt_pct(row.get('alpha_pct'))}"
            f" | strategy {_fmt_pct(row.get('return_pct'))}"
            f" | benchmark {_fmt_pct(row.get('benchmark_return_pct'))}"
        )

    failed = [name for name, ok in gates.items() if name.endswith("_pass") and not bool(ok)]
    print("\nGates")
    print(f"  Result: {'PASS' if gates.get('all_pass') else 'FAIL'}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  Failed: none")

    display_cols = [
        "paper_ready",
        "core_preset",
        "regime_mode",
        "holding_days",
        "earnings_blackout_days",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "robust_cost_stress_pass",
        "subperiod_stability_pass",
    ]
    top = grid.head(5).copy()
    print("\nTop Grid Rows")
    print(top[[c for c in display_cols if c in top.columns]].to_string(index=False))

    print("\nSaved Outputs")
    print(f"  metrics: signals/core_satellite_alpha_metrics.json")
    print(f"  grid:    signals/core_satellite_alpha_grid.csv")
    print(f"  signal:  {signal_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Core-satellite alpha signal generator")
    parser.add_argument("--ignore-stale", action="store_true",
                        help="Override stale data block — generate signal even if factor data is very old")
    args = parser.parse_args()

    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    specs = load_feature_specs()
    ml_scores = load_prediction_scores()
    panel = _ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, ml_scores))

    # ── Data freshness gate ────────────────────────────────────────────
    # PLAIN ENGLISH: Before running hundreds of grid search configs, check if
    # the factor data is fresh.  If it's too old, refuse to generate a signal
    # because stale data leads to stale stock picks → bad trades.
    freshness = check_factor_freshness(panel, ignore_stale=args.ignore_stale)
    print(f"\n  {freshness['message']}")
    if freshness["blocked"]:
        raise SystemExit(f"Aborting: {freshness['message']}")
    # We'll pass this flag down so write_paper_signal can set gates_all_pass=False
    _FACTOR_DATA_FRESH = freshness["fresh"]

    candidate_configs: list[dict] = []
    for core_preset, core_weights in CORE_PRESETS.items():
        for score_source in SCORE_SOURCES:
            for shape in SHAPES:
                for weighting in WEIGHTING_MODES:
                    for exit_rank_floor in EXIT_RANK_FLOORS:
                        for adaptive_exit_mode in ADAPTIVE_EXIT_MODES:
                            for max_per_sector in MAX_PER_SECTOR_OPTIONS:
                                for core_gross, overlay_gross in CORE_OVERLAY_COMBOS:
                                    for earnings_blackout_days in EARNINGS_BLACKOUT_DAY_OPTIONS:
                                        for cost_stress in COST_STRESS_MULTIPLIERS:
                                            for holding_days in HOLDING_DAY_OPTIONS:
                                                for dd_breaker in DRAWDOWN_CIRCUIT_BREAKER_OPTIONS:
                                                    for vt in VOL_TARGET_OPTIONS:
                                                        if core_gross + overlay_gross > MAX_GROSS_EXPOSURE + 1e-9:
                                                            continue
                                                        candidate_configs.append({
                                                            "core_preset": core_preset,
                                                            "regime_mode": "static",
                                                            "core_weights": core_weights,
                                                            "score_source": score_source,
                                                            "shape": shape,
                                                            "weighting": weighting,
                                                            "exit_rank_floor": float(exit_rank_floor),
                                                            "adaptive_exit_mode": str(adaptive_exit_mode),
                                                            "max_per_sector": int(max_per_sector),
                                                            "earnings_blackout_days": int(earnings_blackout_days),
                                                            "core_gross": float(core_gross),
                                                            "overlay_gross": float(overlay_gross),
                                                            "max_gross_exposure": MAX_GROSS_EXPOSURE,
                                                            "max_single_name_weight": MAX_SINGLE_NAME_WEIGHT,
                                                            "cost_stress": float(cost_stress),
                                                            "holding_days": int(holding_days),
                                                            "drawdown_circuit_breaker": float(dd_breaker),
                                                            "vol_target": float(vt),
                                                        })

    # PLAIN ENGLISH: Skip presets that historically never compete.  They still
    # exist in REGIME_PRESETS for research/manual runs, but the grid search
    # ignores them to avoid wasting compute on configs that max at Sharpe ~1.2.
    GRID_SKIP_PRESETS = {"qqq_heavy_25_75", "qqq_tilt_40_60", "qqq_trend_switch", "qqq_trend_switch_fast_core110_overlay15"}
    for regime_name, regime_preset in REGIME_PRESETS.items():
        if regime_name in GRID_SKIP_PRESETS:
            continue
        risk_on = regime_preset["risk_on"]
        for score_source in SCORE_SOURCES:
            for shape in SHAPES:
                for weighting in WEIGHTING_MODES:
                    for exit_rank_floor in EXIT_RANK_FLOORS:
                        for adaptive_exit_mode in ADAPTIVE_EXIT_MODES:
                            for max_per_sector in MAX_PER_SECTOR_OPTIONS:
                                for earnings_blackout_days in EARNINGS_BLACKOUT_DAY_OPTIONS:
                                    for cost_stress in COST_STRESS_MULTIPLIERS:
                                        for holding_days in HOLDING_DAY_OPTIONS:
                                            for dd_breaker in DRAWDOWN_CIRCUIT_BREAKER_OPTIONS:
                                                for vt in VOL_TARGET_OPTIONS:
                                                    candidate_configs.append({
                                                        "core_preset": regime_name,
                                                        "regime_mode": regime_name,
                                                        "regime_ma_window": int(regime_preset["ma_window"]),
                                                        "regime_high_vol": float(regime_preset["high_vol"]),
                                                        # Pass through adaptive vol mode if set in preset
                                                        "high_vol_mode": str(regime_preset.get("high_vol_mode", "fixed")),
                                                        # Pass through score blend and early rebalance from preset
                                                        "score_blend": bool(regime_preset.get("score_blend", False)),
                                                        "early_rebalance_on_regime_change": bool(regime_preset.get("early_rebalance_on_regime_change", False)),
                                                        "core_weights": dict(risk_on["core_weights"]),
                                                        "score_source": score_source,
                                                        "shape": shape,
                                                        "weighting": weighting,
                                                        "exit_rank_floor": float(exit_rank_floor),
                                                        "adaptive_exit_mode": str(adaptive_exit_mode),
                                                        "max_per_sector": int(max_per_sector),
                                                        "earnings_blackout_days": int(earnings_blackout_days),
                                                        "core_gross": float(risk_on["core_gross"]),
                                                        "overlay_gross": float(risk_on["overlay_gross"]),
                                                        "max_gross_exposure": MAX_GROSS_EXPOSURE,
                                                        "max_single_name_weight": MAX_SINGLE_NAME_WEIGHT,
                                                        "cost_stress": float(cost_stress),
                                                        "holding_days": int(holding_days),
                                                        "drawdown_circuit_breaker": float(dd_breaker),
                                                        "vol_target": float(vt),
                                                    })

    rows: list[dict] = []
    for config in candidate_configs:
        metrics, equity, trades = evaluate(panel, config)
        comps = metrics["benchmark_comparisons"]
        gates = metrics["core_satellite_gate_results"]
        core_weights = config["core_weights"]
        rows.append({
            "core_preset": config["core_preset"],
            "regime_mode": config.get("regime_mode", "static"),
            "regime_ma_window": config.get("regime_ma_window", np.nan),
            "regime_high_vol": config.get("regime_high_vol", np.nan),
            "high_vol_mode": config.get("high_vol_mode", "fixed"),
            "regime_counts": json.dumps(metrics.get("regime_counts", {}), sort_keys=True),
            "core_spy_weight": core_weights["SPY"],
            "core_qqq_weight": core_weights["QQQ"],
            "score_source": config["score_source"],
            "shape": config["shape"],
            "weighting": config["weighting"],
            "exit_rank_floor": float(config["exit_rank_floor"]),
            "adaptive_exit_mode": str(config.get("adaptive_exit_mode", "fixed")),
            "max_per_sector": int(config["max_per_sector"]),
            "earnings_blackout_days": int(config.get("earnings_blackout_days", 0)),
            "core_gross": float(config["core_gross"]),
            "overlay_gross": float(config["overlay_gross"]),
            "cost_stress": float(config["cost_stress"]),
            "holding_days": int(config["holding_days"]),
            "score_blend": bool(config.get("score_blend", False)),
            "early_rebalance": bool(config.get("early_rebalance_on_regime_change", False)),
            "drawdown_circuit_breaker": float(config.get("drawdown_circuit_breaker", 0.0)),
            "vol_target": float(config.get("vol_target", 0.0)),
            "max_gross_exposure": MAX_GROSS_EXPOSURE,
            "max_single_name_weight": float(config.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
            "total_return_pct": metrics["total_return_pct"],
            "cagr_pct": metrics["cagr_pct"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
            "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
            "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
            "nw_tstat_vs_blend": comps["BLEND"]["nw_tstat_vs_benchmark"],
            "turnover_pct": metrics["turnover_pct"],
            "estimated_cost_pct": metrics["estimated_cost_pct"],
            "avg_gross_exposure": metrics["avg_gross_exposure"],
            "avg_overlay_positions": metrics["avg_overlay_positions"],
            "max_single_name_weight_realized": metrics["max_single_name_weight"],
            "avg_effective_overlay_names": metrics["avg_effective_overlay_names"],
            "max_sector_overlay_weight": metrics["max_sector_overlay_weight"],
            "top_ticker_overlay_contributor": metrics["top_ticker_overlay_contributor"],
            "top_ticker_overlay_contribution_share": metrics["top_ticker_overlay_contribution_share"],
            "holdout_alpha_vs_qqq_pct": metrics["holdout_2023_2026"].get("alpha_vs_qqq_pct", np.nan),
            "holdout_alpha_vs_blend_pct": metrics["holdout_2023_2026"].get("alpha_vs_blend_pct", np.nan),
            "subperiod_stability_pass": gates["subperiod_stability_pass"],
            "year_alpha_concentration_pass": gates["year_alpha_concentration_pass"],
            "single_name_weight_cap_pass": gates["single_name_weight_cap_pass"],
            "top_ticker_contribution_pass": gates["top_ticker_contribution_pass"],
            "holdout_2023_2026_vs_qqq_pass": gates["holdout_2023_2026_vs_qqq_pass"],
            "holdout_2023_2026_vs_blend_pass": gates["holdout_2023_2026_vs_blend_pass"],
            "paper_ready": gates["all_pass"],
        })

    grid = pd.DataFrame(rows)
    robust_key_cols = [
        "core_preset",
        "regime_mode",
        "regime_ma_window",
        "regime_high_vol",
        "core_spy_weight",
        "core_qqq_weight",
        "score_source",
        "shape",
        "weighting",
        "exit_rank_floor",
        "adaptive_exit_mode",
        "max_per_sector",
        "earnings_blackout_days",
        "core_gross",
        "overlay_gross",
        "holding_days",
        "vol_target",
        "max_single_name_weight",
    ]
    stress = (
        grid.groupby(robust_key_cols, dropna=False)
        .agg(
            stress_cost_levels=("cost_stress", lambda s: ",".join(str(float(v)) for v in sorted(set(s)))),
            stress_min_alpha_vs_spy_pct=("alpha_vs_spy_pct", "min"),
            stress_min_alpha_vs_qqq_pct=("alpha_vs_qqq_pct", "min"),
            stress_min_alpha_vs_blend_pct=("alpha_vs_blend_pct", "min"),
            stress_min_holdout_alpha_vs_qqq_pct=("holdout_alpha_vs_qqq_pct", "min"),
            stress_min_holdout_alpha_vs_blend_pct=("holdout_alpha_vs_blend_pct", "min"),
            stress_all_gates_pass=("paper_ready", "all"),
        )
        .reset_index()
    )
    required_costs = {float(v) for v in ROBUST_COST_STRESSES}
    stress["stress_has_required_costs"] = stress["stress_cost_levels"].apply(
        lambda value: required_costs.issubset({float(v) for v in str(value).split(",") if v})
    )
    stress["robust_cost_stress_pass"] = (
        stress["stress_has_required_costs"]
        & stress["stress_all_gates_pass"]
        & (stress["stress_min_alpha_vs_spy_pct"] > 0)
        & (stress["stress_min_alpha_vs_qqq_pct"] > 0)
        & (stress["stress_min_alpha_vs_blend_pct"] > 0)
        & (stress["stress_min_holdout_alpha_vs_qqq_pct"] > 0)
        & (stress["stress_min_holdout_alpha_vs_blend_pct"] > 0)
    )
    grid = grid.merge(
        stress[robust_key_cols + [
            "stress_cost_levels",
            "stress_min_alpha_vs_spy_pct",
            "stress_min_alpha_vs_qqq_pct",
            "stress_min_alpha_vs_blend_pct",
            "stress_min_holdout_alpha_vs_qqq_pct",
            "stress_min_holdout_alpha_vs_blend_pct",
            "robust_cost_stress_pass",
        ]],
        on=robust_key_cols,
        how="left",
    )
    grid = grid.sort_values(
        ["paper_ready", "robust_cost_stress_pass", "alpha_vs_blend_pct", "sharpe", "max_drawdown_pct"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    grid_path = Path(SIGNAL_DIR) / "core_satellite_alpha_grid.csv"
    grid.to_csv(grid_path, index=False)
    if grid.empty:
        raise SystemExit("No core-satellite configs evaluated.")

    hardened = grid[
        (grid["score_source"].isin(["factor_walkforward", "regime_adaptive", "regime_adaptive_low_vol", "regime_adaptive_consensus"]))
        & (grid["cost_stress"] == 2.0)
        & (grid["robust_cost_stress_pass"])
    ]
    hardened_pass = hardened[hardened["paper_ready"]]
    if not hardened_pass.empty:
        selected = hardened_pass.iloc[0]
        selection_reason = "best_hardened_walkforward_passing_row"
    elif not hardened.empty:
        selected = hardened.iloc[0]
        selection_reason = "best_hardened_walkforward_row"
    else:
        selected = grid.iloc[0]
        selection_reason = "best_available_row"

    selected_config = {
        "core_preset": str(selected["core_preset"]),
        "regime_mode": str(selected.get("regime_mode", "static")),
        "core_weights": {"SPY": float(selected["core_spy_weight"]), "QQQ": float(selected["core_qqq_weight"])},
        "score_source": str(selected["score_source"]),
        "shape": str(selected["shape"]),
        "weighting": str(selected["weighting"]),
        "exit_rank_floor": float(selected["exit_rank_floor"]),
        "adaptive_exit_mode": str(selected.get("adaptive_exit_mode", "fixed")),
        "max_per_sector": int(selected["max_per_sector"]),
        "earnings_blackout_days": int(selected.get("earnings_blackout_days", 0)),
        "core_gross": float(selected["core_gross"]),
        "overlay_gross": float(selected["overlay_gross"]),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "max_single_name_weight": float(selected.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
        "cost_stress": float(selected["cost_stress"]),
        "holding_days": int(selected["holding_days"]),
        "score_blend": bool(selected.get("score_blend", False)),
        "early_rebalance_on_regime_change": bool(selected.get("early_rebalance", False)),
        "drawdown_circuit_breaker": float(selected.get("drawdown_circuit_breaker", 0.0)),
        "vol_target": float(selected.get("vol_target", 0.0)),
            }
    if str(selected_config["regime_mode"]) in REGIME_PRESETS:
        preset = REGIME_PRESETS[str(selected_config["regime_mode"])]
        selected_config["regime_ma_window"] = int(preset["ma_window"])
        selected_config["regime_high_vol"] = float(preset["high_vol"])
        # Pass through adaptive vol mode if the preset uses it
        selected_config["high_vol_mode"] = str(preset.get("high_vol_mode", "fixed"))
    # ── Save winning config to JSON for easy inspection / auto-loading ─────
    # PLAIN ENGLISH: Save the best config parameters to a standalone JSON file.
    # This makes it easy to see what config won without digging through the
    # full metrics JSON, and could be loaded by other tools or scheduled jobs.
    winning_config_path = Path(SIGNAL_DIR) / "core_satellite_alpha_winning_config.json"
    winning_config_data = {
        **selected_config,
        "selection_reason": selection_reason,
        "grid_sharpe": float(selected.get("sharpe", 0)),
        "grid_return_pct": float(selected.get("total_return_pct", 0)),
        "grid_max_dd_pct": float(selected.get("max_drawdown_pct", 0)),
        "grid_holdout_pct": float(selected.get("holdout_return_pct", 0)),
        "selected_at": datetime.now().isoformat(),
    }
    with open(winning_config_path, "w") as f:
        json.dump(winning_config_data, f, indent=2, default=str)

    best_metrics, best_equity, best_trades = evaluate(panel, selected_config)
    best_metrics["selected_features"] = specs
    best_metrics["grid_rows"] = int(len(grid))
    best_metrics["best_config_source"] = selection_reason
    best_metrics["robust_cost_stress_pass"] = bool(selected.get("robust_cost_stress_pass", False))
    best_metrics["robust_cost_stress_summary"] = {
        "cost_levels": str(selected.get("stress_cost_levels", "")),
        "min_alpha_vs_spy_pct": float(selected.get("stress_min_alpha_vs_spy_pct", np.nan)),
        "min_alpha_vs_qqq_pct": float(selected.get("stress_min_alpha_vs_qqq_pct", np.nan)),
        "min_alpha_vs_blend_pct": float(selected.get("stress_min_alpha_vs_blend_pct", np.nan)),
        "min_holdout_alpha_vs_qqq_pct": float(selected.get("stress_min_holdout_alpha_vs_qqq_pct", np.nan)),
        "min_holdout_alpha_vs_blend_pct": float(selected.get("stress_min_holdout_alpha_vs_blend_pct", np.nan)),
    }
    best_metrics["paper_ready"] = bool(best_metrics["core_satellite_gate_results"]["all_pass"])

    # If factor data is stale, override paper_ready to False — don't trade on old data
    if not _FACTOR_DATA_FRESH:
        best_metrics["paper_ready"] = False
        best_metrics["core_satellite_gate_results"]["all_pass"] = False
        best_metrics["factor_data_stale"] = True
        best_metrics["factor_data_freshness"] = freshness
        print(f"  ⚠ gates_all_pass forced to False because factor data is stale")

    pd.DataFrame({"equity": best_equity}).to_csv(Path(SIGNAL_DIR) / "core_satellite_alpha_equity.csv")
    best_trades.to_csv(Path(SIGNAL_DIR) / "core_satellite_alpha_trades.csv", index=False)
    signal_panel = _ensure_robust_score_columns(
        attach_scores(load_factor_panel(specs, require_forward_returns=False), specs, ml_scores)
    )
    signal_path = write_paper_signal(signal_panel, best_metrics)
    with open(Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print_run_summary(grid, best_metrics, signal_path)


if __name__ == "__main__":
    main()
