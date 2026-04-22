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
    RETURN_HORIZON_DAYS, SOCIAL_SENTIMENT_ENABLED,
)
from sentiment_engine import build_sentiment_feature_dataframe, SentimentEngine, score_todays_news
from social_sentiment import build_social_sentiment_features, get_live_social_signal

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
    # FIX: align direction target with return horizon
    df["target"] = (c.shift(-RETURN_HORIZON_DAYS) > c).astype(int)
    return df

def build_multi_timeframe(close: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(index=dates)
    if not USE_MULTI_TIMEFRAME:
        return result
    wk = close.resample("W").last().ffill()
    result["weekly_ret"] = wk.pct_change(1).reindex(dates, method="ffill").fillna(0.0).values
    result["weekly_vol"] = wk.pct_change(1).rolling(8).std().reindex(dates, method="ffill").fillna(0.0).values
    mo = close.resample("ME").last().ffill()
    result["monthly_ret"] = mo.pct_change(1).reindex(dates, method="ffill").fillna(0.0).values
    result["monthly_trend"] = np.sign((mo - mo.rolling(3).mean()).reindex(dates, method="ffill").fillna(0.0))
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
    for name, sym in symbols.items():
        try:
            raw = flatten_yf(yf.download(sym, start=start, end=end, progress=False, auto_adjust=True))
            close = raw["Close"].reindex(dates, method="ffill")
            result[f"{name}_ret1d"] = close.pct_change(1).fillna(0)
            result[f"{name}_ret5d"] = close.pct_change(5).fillna(0)
            if name in ("spy","qqq","dia"):
                ma20 = close.rolling(20).mean()
                result[f"{name}_above_ma20"] = (close > ma20).astype(int).fillna(0)
                result[f"{name}_vol10"] = close.pct_change(1).rolling(10).std().fillna(0)
            if name == "gld":
                result["gld_risk_off"] = (close.pct_change(5) > 0.02).astype(int).fillna(0)
            if name == "hyg":
                result["credit_stress"] = (close.pct_change(5) < -0.02).astype(int).fillna(0)
        except Exception:
            continue
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

def build_options_features_context(ticker: str, dates: pd.DatetimeIndex, live: bool = False) -> pd.DataFrame:
    """Historical research uses neutral options features to avoid leakage. Live prediction can use current snapshot."""
    result = pd.DataFrame(index=dates)
    result["iv_atm"] = 0.25
    result["put_call_ratio"] = 1.0
    result["iv_skew"] = 0.0
    if not live or not USE_OPTIONS_DATA:
        return result
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return result
        chain = tk.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return result
        price = float(tk.fast_info.get("last_price", 0) or 0)
        if price <= 0:
            return result
        calls["dist"] = (calls["strike"] - price).abs()
        puts["dist"] = (puts["strike"] - price).abs()
        atm_c = calls.loc[calls["dist"].idxmin()]
        atm_p = puts.loc[puts["dist"].idxmin()]
        iv_c = float(atm_c.get("impliedVolatility", 0.25) or 0.25)
        iv_p = float(atm_p.get("impliedVolatility", 0.25) or 0.25)
        iv_atm = (iv_c + iv_p) / 2
        call_vol = float(calls["volume"].sum()); put_vol = float(puts["volume"].sum())
        pc_ratio = put_vol / (call_vol + 1)
        otm_p = puts[puts["strike"].between(price * 0.93, price * 0.97)]
        otm_c = calls[calls["strike"].between(price * 1.03, price * 1.07)]
        iv_op = float(otm_p["impliedVolatility"].mean()) if not otm_p.empty else iv_atm
        iv_oc = float(otm_c["impliedVolatility"].mean()) if not otm_c.empty else iv_atm
        result["iv_atm"] = iv_atm
        result["put_call_ratio"] = min(pc_ratio, 5.0)
        result["iv_skew"] = iv_op - iv_oc
    except Exception:
        pass
    return result

def apply_feature_lag(df: pd.DataFrame, keywords: list[str], lag_days: int = 5) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        col_l = col.lower()
        if any(k in col_l for k in keywords):
            out[col] = out[col].shift(lag_days)
    return out


def build_sentiment_features(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return build_sentiment_feature_dataframe(ticker, dates, finnhub_client=None, engine_level="finbert", sleep_rss=0.1, logger=None)

def build_research_feature_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_price_data(ticker, start, end)
    if df.empty:
        return df
    df = add_technical_features(df)
    dates = df.index
    frames = [
        build_multi_timeframe(df["Close"], dates),
        build_vix_features(dates, start, end),
        build_multi_market(ticker, dates, start, end),
        build_options_features_context(ticker, dates, live=False),  # FIX: avoid historical options leakage
        build_sentiment_features(ticker, dates),
        build_calendar_features(dates),
    ]
    if SOCIAL_SENTIMENT_ENABLED:
        try:
            frames.append(build_social_sentiment_features(ticker, dates))
        except Exception:
            pass
    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = apply_feature_lag(out, ["short_interest", "analyst", "recommend"], lag_days=5)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.dropna(subset=["Close", "target"], inplace=True)
    return out

def build_live_features_with_latest_news(ticker: str, feature_cols: list[str]) -> pd.DataFrame | None:
    end = datetime.utcnow().date()
    start = end - timedelta(days=270)
    df = fetch_price_data(ticker, start.isoformat(), (end + timedelta(days=1)).isoformat())
    if df.empty:
        return None
    df = add_technical_features(df)
    dates = df.index
    frames = [
        build_multi_timeframe(df["Close"], dates),
        build_vix_features(dates, start.isoformat(), (end + timedelta(days=1)).isoformat()),
        build_multi_market(ticker, dates, start.isoformat(), (end + timedelta(days=1)).isoformat()),
        build_options_features_context(ticker, dates, live=True),
        build_calendar_features(dates),
    ]
    try:
        sent = build_sentiment_feature_dataframe(ticker, dates, finnhub_client=None, engine_level="finbert", sleep_rss=0.1, logger=None)
        # overwrite latest row with freshest news signal
        try:
            engine = SentimentEngine(level="finbert")
            live_news = score_todays_news(ticker, engine)
            if not sent.empty:
                latest = sent.index[-1]
                sent.loc[latest, "news_sentiment"] = float(live_news.get("composite_score", 0.0) or 0.0)
                sent.loc[latest, "headline_volume"] = float(live_news.get("headline_count", 0) or 0.0)
        except Exception:
            pass
        frames.append(sent)
    except Exception:
        pass
    if SOCIAL_SENTIMENT_ENABLED:
        try:
            social_df = build_social_sentiment_features(ticker, dates)
            social_live = get_live_social_signal(ticker)
            if not social_df.empty and social_live.get("social_available"):
                latest = social_df.index[-1]
                social_df.loc[latest, "social_combined"] = float(social_live.get("combined", 0.0))
                social_df.loc[latest, "social_message_volume"] = float(social_live.get("message_volume", 0.0))
            frames.append(social_df)
        except Exception:
            pass
    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    sentiment_cols = [c for c in out.columns if ("sent_" in c or "sentiment" in c)]
    if sentiment_cols:
        recent = out[sentiment_cols].tail(min(5, len(out))).fillna(0.0)
        nonzero_ratio = float((recent.abs() > 1e-9).any(axis=1).mean()) if not recent.empty else 0.0
        if nonzero_ratio < 0.20:
            print(
                f"WARNING: [{ticker}] live sentiment features appear degraded at source "
                f"(recent non-zero ratio={nonzero_ratio:.2f}, cols={len(sentiment_cols)})"
            )
    return out
