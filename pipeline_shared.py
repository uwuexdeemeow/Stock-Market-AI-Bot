"""
pipeline_shared.py — Single source of truth for research/live feature engineering.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from settings import (
    MULTI_MARKET, SECTOR_MAP,
    USE_MULTI_TIMEFRAME, USE_VIX_TERM, USE_OPTIONS_DATA,
    RETURN_HORIZON_DAYS, SOCIAL_SENTIMENT_ENABLED, USE_EARNINGS_DATA, USE_NEWS_SENTIMENT,
    SENTIMENT_ENGINE_LEVEL,
)
from sentiment_engine import build_sentiment_feature_dataframe, SentimentEngine, score_todays_news
from social_sentiment import build_social_sentiment_features, get_live_social_signal
from fundamental_features import (
    build_pead_features,
    build_iv_rank_features,
    build_sector_strength_features,
    build_market_breadth_features,
    build_gap_features,
    build_volume_features,
)

def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = flatten_yf(df)
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna().copy()

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for lag in [1,3,5,10,20]:
        df[f"ret_{lag}d"] = c.pct_change(lag)
    for p in [5,10,20,50,200]:
        ma = c.rolling(p).mean()
        df[f"ma{p}"] = ma
        df[f"dist_ma{p}"] = (c - ma)/(ma + 1e-9)
    df["ma_cross_5_20"] = np.sign(df["ma5"] - df["ma20"])
    df["ma_cross_20_50"] = np.sign(df["ma20"] - df["ma50"])
    for period in [7,14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        df[f"rsi_{period}"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["rsi_divergence"] = df["ret_5d"] - df["rsi_14"].pct_change(5)
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_norm"] = df["macd"] / (c + 1e-9)
    prev_c = c.shift(1)
    tr = pd.concat([h-lo, (h-prev_c).abs(), (lo-prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_norm"] = df["atr_14"] / (c + 1e-9)
    df["atr_ratio"] = df["atr_14"] / (df["atr_14"].rolling(20).mean() + 1e-9)
    std20 = c.rolling(20).std()
    bb_u = df["ma20"] + 2*std20; bb_l = df["ma20"] - 2*std20
    df["bb_pos"] = (c - bb_l) / (bb_u - bb_l + 1e-9)
    df["bb_width"] = (bb_u - bb_l) / (df["ma20"] + 1e-9)
    df["vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)
    df["vol_ratio_5"] = v / (v.rolling(5).mean() + 1e-9)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv.pct_change(5)
    typical = (h + lo + c) / 3
    df["vwap_dist"] = (c - (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)) / (c + 1e-9)
    lr = np.log(c / c.shift(1))
    df["hvol_5d"] = lr.rolling(5).std() * np.sqrt(252)
    df["hvol_20d"] = lr.rolling(20).std() * np.sqrt(252)
    df["hvol_ratio"] = df["hvol_5d"] / (df["hvol_20d"] + 1e-9)
    df["vol_imbalance"] = (c - df["Open"]) / (h - lo + 1e-9)
    df["vol_imbalance_5d"] = df["vol_imbalance"].rolling(5).mean()
    df["spread_proxy"] = (h - lo) / ((h + lo)/2 + 1e-9)
    df["uptick_ratio"] = (c.diff() > 0).rolling(5).mean()
    vol_norm = v / (v.rolling(20).mean() + 1e-9)
    df["dark_pool_signal"] = (df["ret_1d"].abs() / (vol_norm + 1e-9)).clip(0,5)/5
    var1 = lr.rolling(20).var() * 5
    var5 = lr.rolling(5).var()
    df["variance_ratio"] = (var5 / (var1 + 1e-9)).clip(0,3)
    df["hl_range"] = (h - lo) / (c + 1e-9)
    df["hl_range_5d"] = df["hl_range"].rolling(5).mean()
    df["roc_5"] = c.pct_change(5) / (c.pct_change(20).rolling(4).mean() + 1e-9)
    df["roc_10"] = c.pct_change(10) / (c.pct_change(40).rolling(4).mean() + 1e-9)
    df["drawdown_20d"] = c / (c.rolling(20, min_periods=5).max() + 1e-9) - 1.0
    df["drawdown_60d"] = c / (c.rolling(60, min_periods=20).max() + 1e-9) - 1.0
    df["volume_chg_1d"] = v.pct_change(1)
    df["volume_chg_5d"] = v.pct_change(5)
    # FIX: align direction target with return horizon
    df["target"] = (c.shift(-RETURN_HORIZON_DAYS) > c).astype(int)
    return df

def _completed_period_to_daily(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Expose period-level values only after the period has fully completed."""
    known = series.dropna().copy()
    if known.empty:
        return pd.Series(0.0, index=dates)
    known.index = pd.DatetimeIndex(known.index) + pd.offsets.BDay(1)
    return known.reindex(dates, method="ffill").fillna(0.0)


def build_multi_timeframe(close: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(index=dates)
    if not USE_MULTI_TIMEFRAME:
        return result
    wk = close.resample("W-FRI").last()
    wk_ret = wk.pct_change(1)
    result["weekly_ret"] = _completed_period_to_daily(wk_ret, dates).values
    result["weekly_vol"] = _completed_period_to_daily(wk_ret.rolling(8).std(), dates).values
    mo = close.resample("ME").last()
    result["monthly_ret"] = _completed_period_to_daily(mo.pct_change(1), dates).values
    result["monthly_trend"] = np.sign(_completed_period_to_daily(mo - mo.rolling(3).mean(), dates))
    result["weekly_trend"] = np.sign(result["weekly_ret"])
    result["tf_alignment"] = result["weekly_trend"] + np.sign(result["weekly_ret"]) + result["monthly_trend"]
    return result

def build_vix_features(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    result = pd.DataFrame(index=dates)
    if not USE_VIX_TERM:
        return result
    try:
        vix = flatten_yf(yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True))["Close"]
        vix = vix.reindex(dates, method="ffill").bfill()   # FIX: pandas deprecation
        result["vix_level"] = vix.values
        result["vix_high"] = (vix > 25).astype(int).values
        result["vix_extreme"] = (vix > 35).astype(int).values
        result["vix_spike"] = (vix.pct_change(1) > 0.15).astype(int).values
        result["vix_percentile"] = vix.rolling(252, min_periods=20).rank(pct=True).values
        try:
            vix3m = flatten_yf(yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=True))["Close"]
            vix3m = vix3m.reindex(dates, method="ffill").fillna(vix)
            result["vix3m_level"] = vix3m.values
            result["vix_ratio"] = (vix / (vix3m + 1e-9)).values
            result["vix_inverted"] = (result["vix_ratio"] > 1.0).astype(int)
        except Exception:
            result["vix3m_level"] = result["vix_level"]
            result["vix_ratio"] = 1.0
            result["vix_inverted"] = 0
    except Exception:
        for col in ["vix_level","vix_high","vix_extreme","vix_spike","vix_percentile","vix3m_level","vix_ratio","vix_inverted"]:
            result[col] = 0.0
    return result

def build_multi_market(ticker: str, dates: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    result = pd.DataFrame(index=dates)
    symbols = dict(MULTI_MARKET)
    sector = SECTOR_MAP.get(ticker.upper())
    if sector:
        symbols["sector"] = sector

    spy_close = None  # kept for regime feature below
    vix_close = None

    for name, sym in symbols.items():
        try:
            raw = flatten_yf(yf.download(sym, start=start, end=end, progress=False, auto_adjust=True))
            close = raw["Close"].reindex(dates, method="ffill")
            result[f"{name}_ret1d"] = close.pct_change(1).fillna(0)
            result[f"{name}_ret5d"] = close.pct_change(5).fillna(0)
            result[f"{name}_ret20d"] = close.pct_change(20).fillna(0)
            # Horizon-matched return for the target function — must equal RETURN_HORIZON_DAYS
            # so make_direction_target can shift it forward to get the correct benchmark window.
            if name in ("spy", "sector") and RETURN_HORIZON_DAYS not in (5, 20):
                result[f"{name}_ret{RETURN_HORIZON_DAYS}d"] = close.pct_change(RETURN_HORIZON_DAYS).fillna(0)
            if name in ("spy", "qqq", "dia"):
                ma20 = close.rolling(20).mean()
                result[f"{name}_above_ma20"] = (close > ma20).astype(int).fillna(0)
                result[f"{name}_vol10"] = close.pct_change(1).rolling(10).std().fillna(0)
            if name == "spy":
                spy_close = close
                ma200 = close.rolling(200, min_periods=50).mean()
                result["spy_dist_ma200"] = ((close - ma200) / (ma200 + 1e-9)).clip(-0.3, 0.3).fillna(0)
                ma50 = close.rolling(50, min_periods=20).mean()
                result["spy_dist_ma50"] = ((close - ma50) / (ma50 + 1e-9)).clip(-0.2, 0.2).fillna(0)
            if name == "vix":
                vix_close = close
            if name == "gld":
                result["gld_risk_off"] = (close.pct_change(5) > 0.02).astype(int).fillna(0)
            if name == "hyg":
                result["credit_stress"] = (close.pct_change(5) < -0.02).astype(int).fillna(0)
        except Exception:
            continue

    # Regime label: combines SPY trend + VIX level into a single -1/0/+1 signal.
    # The model can condition all other features on this regime context.
    #   +1 = bull  (SPY above 200d MA and VIX calm < 20)
    #   -1 = bear  (SPY below 200d MA  or  VIX stressed > 28)
    #    0 = neutral / transitional
    if spy_close is not None and vix_close is not None:
        spy_above_200 = (result.get("spy_dist_ma200", pd.Series(0, index=dates)) > 0)
        vix_bull = vix_close < 20
        vix_bear = vix_close > 28
        regime = pd.Series(0, index=dates, dtype=float)
        regime[spy_above_200 & vix_bull] = 1.0
        regime[~spy_above_200 | vix_bear] = -1.0
        result["regime"] = regime.values
        # Continuous version: how far into bull/bear territory (0–1 scale)
        result["regime_strength"] = result.get("spy_dist_ma200", pd.Series(0, index=dates)).values
    else:
        result["regime"] = 0.0
        result["regime_strength"] = 0.0

    return result.fillna(0)

def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(index=dates)
    result["dow_monday"] = (dates.dayofweek == 0).astype(int)
    result["dow_friday"] = (dates.dayofweek == 4).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    result["month_end"] = 0
    result["opex_week"] = 0
    for i, date in enumerate(dates):
        month_dates = dates[dates.month == date.month]
        if len(month_dates) >= 3 and date in month_dates[-3:]:
            result.iloc[i, result.columns.get_loc("month_end")] = 1
    return result


CONSERVATIVE_BASE_COLUMNS = {"Open", "High", "Low", "Close", "Volume", "target"}
CONSERVATIVE_FEATURE_PREFIXES = (
    "ret_",
    "hvol_",
    "rsi_",
    "dist_ma",
    "ma_cross_",
    "macd",
    "atr_",
    "bb_",
    "vol_ratio",
    "volume_chg_",
    "hl_range",
    "spread_proxy",
    "roc_",
    "drawdown_",
    "weekly_",
    "monthly_",
    "tf_alignment",
    "spy_",
    "qqq_",
    "vix_",
    "regime",
    "ret_vs_spy",
    "ret_vs_qqq",
    "breadth_",
    "pct_above_",
)
CONSERVATIVE_FEATURE_EXACT = {
    "obv_slope",
    "vwap_dist",
    "uptick_ratio",
    "variance_ratio",
}
RISKY_FEATURE_KEYWORDS = (
    "social",
    "iv_",
    "put_call",
    "option",
    "earn",
    "eps_",
    "analyst",
    "recommend",
    "short_interest",
    "dark_pool",
)

RAW_SENTIMENT_COLUMNS = (
    "news_sentiment",
    "sent_pos",
    "sent_neg",
    "sent_premarket",
    "sent_afterhours",
    "sentiment_disagreement",
    "headline_volume",
    "headline_vol_spike",
    "sent_decay_1d",
    "sent_decay_3d",
    "sent_neg_decay",
    "sent_pos_decay",
    "sentiment_3d",
    "sentiment_7d",
    "sentiment_delta",
    "sentiment_accel",
)


def sentiment_raw_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in RAW_SENTIMENT_COLUMNS if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]


def fit_sentiment_zscore_stats(frame: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in columns or sentiment_raw_columns(frame):
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_sentiment_distribution_matching(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = frame.copy()
    fitted = stats or fit_sentiment_zscore_stats(out)
    for col, st in fitted.items():
        if col not in out.columns:
            out[col] = 0.0
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = float(st.get("mean", 0.0))
        std = float(st.get("std", 1.0)) or 1.0
        if not np.isfinite(std) or abs(std) < 1e-6:
            std = 1.0
        out[f"sent_z_{col}"] = ((s - mean) / std).clip(-5.0, 5.0)
    return out, fitted


def add_relative_strength_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"ret_5d", "spy_ret5d"}.issubset(out.columns):
        out["ret_vs_spy_5d"] = out["ret_5d"] - out["spy_ret5d"]
    if {"ret_20d", "spy_ret20d"}.issubset(out.columns):
        out["ret_vs_spy_20d"] = out["ret_20d"] - out["spy_ret20d"]
    if {"ret_5d", "qqq_ret5d"}.issubset(out.columns):
        out["ret_vs_qqq_5d"] = out["ret_5d"] - out["qqq_ret5d"]
    if {"ret_20d", "qqq_ret20d"}.issubset(out.columns):
        out["ret_vs_qqq_20d"] = out["ret_20d"] - out["qqq_ret20d"]
    return out


def keep_conservative_feature_set(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only point-in-time price, volume, relative-strength, and regime features."""
    out = add_relative_strength_features(frame)
    keep = []
    for col in out.columns:
        col_l = col.lower()
        if col in CONSERVATIVE_BASE_COLUMNS:
            keep.append(col)
            continue
        if any(k in col_l for k in RISKY_FEATURE_KEYWORDS):
            continue
        if col.startswith("sent_z_"):
            keep.append(col)
            continue
        if col in CONSERVATIVE_FEATURE_EXACT or any(col.startswith(prefix) for prefix in CONSERVATIVE_FEATURE_PREFIXES):
            keep.append(col)
    return out.loc[:, keep]

def build_options_features_context(ticker: str, dates: pd.DatetimeIndex, live: bool = False) -> pd.DataFrame:
    """Return neutral options features until point-in-time options history is available."""
    result = pd.DataFrame(index=dates)
    result["iv_atm"] = 0.25
    result["put_call_ratio"] = 1.0
    result["iv_skew"] = 0.0
    return result


def build_earnings_features_context(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Neutral earnings features unless point-in-time earnings data is explicitly enabled."""
    if USE_EARNINGS_DATA:
        return build_pead_features(ticker, dates)
    return pd.DataFrame(index=dates, data={
        "eps_surprise_pct": 0.0,
        "days_since_earnings": 60.0,
        "days_to_next_earnings": 60.0,
    })

def apply_feature_lag(df: pd.DataFrame, keywords: list[str], lag_days: int = 5) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        col_l = col.lower()
        if any(k in col_l for k in keywords):
            out[col] = out[col].shift(lag_days)
    return out


def build_sentiment_features(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if not USE_NEWS_SENTIMENT:
        return pd.DataFrame(index=dates)
    return build_sentiment_feature_dataframe(ticker, dates, finnhub_client=None, engine_level=SENTIMENT_ENGINE_LEVEL, sleep_rss=0.1, logger=None)

def build_research_feature_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_price_data(ticker, start, end)
    if df.empty:
        return df
    df = add_technical_features(df)
    dates = df.index

    # Build multi-market frame first (sector_ret5d/20d needed by sector strength)
    mm_frame = build_multi_market(ticker, dates, start, end)

    frames = [
        build_multi_timeframe(df["Close"], dates),
        build_vix_features(dates, start, end),
        mm_frame,
        build_gap_features(df),
        build_volume_features(df),
        build_market_breadth_features(dates, start, end),
    ]
    if USE_NEWS_SENTIMENT:
        try:
            frames.append(build_sentiment_features(ticker, dates))
        except Exception:
            pass
    if SOCIAL_SENTIMENT_ENABLED:
        try:
            frames.append(build_social_sentiment_features(ticker, dates))
        except Exception:
            pass

    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    # Sector strength is derived from the combined frame (needs sector_ret5d/20d + ret_5d/20d)
    sector_strength = build_sector_strength_features(ticker, out)
    out = pd.concat([out, sector_strength], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    out, _ = apply_sentiment_distribution_matching(out)
    raw_sent = out[sentiment_raw_columns(out)].copy()
    out = keep_conservative_feature_set(out)
    if not raw_sent.empty:
        out = pd.concat([out, raw_sent], axis=1)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.dropna(subset=["Close", "target"], inplace=True)
    return out

def build_live_features_with_latest_news(
    ticker: str,
    feature_cols: list[str],
    sentiment_zscore_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame | None:
    end = datetime.utcnow().date()
    start = end - timedelta(days=270)
    start_str = start.isoformat()
    end_str = (end + timedelta(days=1)).isoformat()

    df = fetch_price_data(ticker, start_str, end_str)
    if df.empty:
        return None
    df = add_technical_features(df)
    dates = df.index

    mm_frame = build_multi_market(ticker, dates, start_str, end_str)

    frames = [
        build_multi_timeframe(df["Close"], dates),
        build_vix_features(dates, start_str, end_str),
        mm_frame,
        build_gap_features(df),
        build_volume_features(df),
        build_market_breadth_features(dates, start_str, end_str),
    ]

    diagnostic_frames: list[pd.DataFrame] = []
    if USE_NEWS_SENTIMENT:
        try:
            sent = build_sentiment_feature_dataframe(
                ticker,
                dates,
                finnhub_client=None,
                engine_level=SENTIMENT_ENGINE_LEVEL,
                sleep_rss=0.1,
                logger=None,
            )
            if sent is None or sent.empty:
                sent = pd.DataFrame(index=dates)
            else:
                sent = sent.reindex(dates)

            for col in [
                "news_sentiment",
                "headline_volume",
                "live_news_headline_count",
                "live_news_breaking",
                "live_news_score_abs",
                "live_news_has_signal",
            ]:
                if col not in sent.columns:
                    sent[col] = 0.0

            try:
                engine = SentimentEngine(level=SENTIMENT_ENGINE_LEVEL)
                live_news = score_todays_news(ticker, engine) or {}
                latest = dates[-1]
                composite_score = float(live_news.get("composite_score", 0.0) or 0.0)
                headline_count = float(live_news.get("headline_count", 0) or 0.0)
                breaking_count = float(live_news.get("breaking_count", 0) or 0.0)
                score_abs = float(live_news.get("avg_abs_score", 0.0) or 0.0)
                has_signal = 1.0 if (
                    headline_count > 0 or abs(composite_score) > 1e-9 or score_abs > 1e-9
                ) else 0.0

                sent.loc[latest, "news_sentiment"] = composite_score
                sent.loc[latest, "headline_volume"] = headline_count
                sent.loc[latest, "live_news_headline_count"] = headline_count
                sent.loc[latest, "live_news_breaking"] = breaking_count
                sent.loc[latest, "live_news_score_abs"] = score_abs
                sent.loc[latest, "live_news_has_signal"] = has_signal
            except Exception as e:
                print(f"[{ticker}] live news refresh failed: {e}")

            diagnostic_frames.append(sent)
        except Exception as e:
            print(f"[{ticker}] sentiment feature builder failed: {e}")

    if SOCIAL_SENTIMENT_ENABLED:
        try:
            social_df = build_social_sentiment_features(ticker, dates)
            social_live = get_live_social_signal(ticker)
            if not social_df.empty and social_live.get("social_available"):
                latest = social_df.index[-1]
                social_df.loc[latest, "social_combined"] = float(social_live.get("combined", 0.0))
                social_df.loc[latest, "social_message_volume"] = float(social_live.get("message_volume", 0.0))
            diagnostic_frames.append(social_df)
        except Exception:
            pass

    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    sector_strength = build_sector_strength_features(ticker, out)
    out = pd.concat([out, sector_strength], axis=1)
    if diagnostic_frames:
        diagnostics_all = pd.concat(diagnostic_frames, axis=1).reindex(out.index)
        diagnostics_all = diagnostics_all.loc[:, ~diagnostics_all.columns.duplicated()]
        out = pd.concat([out, diagnostics_all], axis=1)
    out, _ = apply_sentiment_distribution_matching(out.loc[:, ~out.columns.duplicated()], sentiment_zscore_stats)
    out = keep_conservative_feature_set(out)
    if diagnostic_frames:
        diagnostics = pd.concat(diagnostic_frames, axis=1).reindex(out.index)
        diagnostics = diagnostics.loc[:, ~diagnostics.columns.duplicated()]
        keep_diagnostics = [c for c in diagnostics.columns if c in feature_cols or c.startswith("live_")]
        if "news_sentiment" in diagnostics.columns:
            keep_diagnostics.append("news_sentiment")
        keep_diagnostics = list(dict.fromkeys(keep_diagnostics))
        if keep_diagnostics:
            out = pd.concat([out, diagnostics[keep_diagnostics]], axis=1)
    missing_cols = [c for c in feature_cols if c not in out.columns]
    if missing_cols:
        out = pd.concat([out, pd.DataFrame(0.0, index=out.index, columns=missing_cols)], axis=1)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
