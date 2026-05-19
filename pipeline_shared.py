"""
pipeline_shared.py — Single source of truth for research/live feature engineering.
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone

import sys
import numpy as np
import pandas as pd

# Add project root to path so we can import data_provider
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_provider import download_single as _dp_download_single, download_prices as _dp_download_prices

from settings import (
    DATA_DIR, MULTI_MARKET, SECTOR_MAP,
    USE_MULTI_TIMEFRAME, USE_VIX_TERM, USE_OPTIONS_DATA,
    RETURN_HORIZON_DAYS, SOCIAL_SENTIMENT_ALPHA_ENABLED, SOCIAL_SENTIMENT_SAFETY_ENABLED,
    SOCIAL_SENTIMENT_ENABLED, USE_EARNINGS_DATA, USE_NEWS_SENTIMENT,
    SENTIMENT_ENGINE_LEVEL, LIVE_SENTIMENT_ENGINE_LEVEL,
)
from sentiment_engine import build_sentiment_feature_dataframe, get_sentiment_engine, score_todays_news
from social_sentiment import build_social_sentiment_features, get_live_social_signal
from fundamental_features import (
    build_pead_features,
    build_iv_rank_features,
    build_sector_strength_features,
    build_market_breadth_features,
    build_market_concentration_features,
    build_gap_features,
    build_point_in_time_valuation_features,
    build_volume_features,
)

def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download price data with automatic fallback (yfinance → yahooquery → Stooq)."""
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    local_path = os.path.join(DATA_DIR, f"{ticker.upper()}.parquet")
    if os.path.exists(local_path):
        try:
            local = pd.read_parquet(local_path)
            local.index = pd.DatetimeIndex(local.index)
            if all(c in local.columns for c in price_cols):
                out = local.loc[:, price_cols].sort_index()
                if start:
                    out = out[out.index >= pd.Timestamp(start)]
                if end:
                    out = out[out.index < pd.Timestamp(end)]
                out = out.dropna().copy()
                if not out.empty:
                    return out
        except Exception:
            pass

    try:
        df = _dp_download_single(ticker, start=start, end=end)
    except RuntimeError:
        return pd.DataFrame()
    df = flatten_yf(df)
    if df.empty:
        return pd.DataFrame()
    if not all(c in df.columns for c in price_cols):
        return pd.DataFrame()
    return df[price_cols].dropna().copy()

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for lag in [1,3,5,10,20,60,120]:
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
        vix = flatten_yf(_dp_download_single("^VIX", start=start, end=end))["Close"]
        # Forward-fill only. Backfilling early rows would leak future VIX values
        # into dates before VIX data existed for this ticker's frame.
        vix = vix.reindex(dates, method="ffill")
        result["vix_level"] = vix.values
        result["vix_high"] = (vix > 25).astype(int).values
        result["vix_extreme"] = (vix > 35).astype(int).values
        result["vix_spike"] = (vix.pct_change(1, fill_method=None) > 0.15).astype(int).values
        result["vix_percentile"] = vix.rolling(252, min_periods=20).rank(pct=True).values
        try:
            vix3m = flatten_yf(_dp_download_single("^VIX3M", start=start, end=end))["Close"]
            vix3m = vix3m.reindex(dates, method="ffill").fillna(vix)
            result["vix3m_level"] = vix3m.values
            ratio_series = vix / (vix3m + 1e-9)
            result["vix_ratio"] = ratio_series.values
            result["vix_inverted"] = (ratio_series > 1.0).astype(int).values

            # ── Richer term-structure features (audit #9) ────────────────
            # The plain binary `vix_inverted` only fires on the day-of.
            # Markets digest term-structure changes over days, not ticks,
            # so the model benefits from features that capture:
            #   - vix_ratio_5d_chg: how fast the curve is steepening/flattening
            #   - vix_ratio_max_5d: was the curve inverted recently
            #     (even if it has normalised today, the stress signal lingers)
            #   - vix_inverted_persist: consecutive days of inversion
            #     (1 day = noise, 3+ days = real regime)
            #   - vix3m_chg_5d: outright back-end vol move
            result["vix_ratio_5d_chg"] = ratio_series.diff(5).fillna(0.0).values
            result["vix_ratio_max_5d"] = ratio_series.rolling(5, min_periods=1).max().values
            inverted_series = (ratio_series > 1.0).astype(int)
            # Count consecutive 1s — cumulative grouping trick: each time
            # the value drops to 0, a new "group" starts; cumsum within
            # each group gives the run length.
            groups = (inverted_series != inverted_series.shift()).cumsum()
            result["vix_inverted_persist"] = inverted_series.groupby(groups).cumsum().fillna(0).astype(int).values
            result["vix3m_chg_5d"] = vix3m.pct_change(5, fill_method=None).fillna(0.0).values
        except Exception:
            result["vix3m_level"] = result["vix_level"]
            result["vix_ratio"] = 1.0
            result["vix_inverted"] = 0
            result["vix_ratio_5d_chg"] = 0.0
            result["vix_ratio_max_5d"] = 1.0
            result["vix_inverted_persist"] = 0
            result["vix3m_chg_5d"] = 0.0
    except Exception:
        for col in [
            "vix_level", "vix_high", "vix_extreme", "vix_spike",
            "vix_percentile", "vix3m_level", "vix_ratio", "vix_inverted",
            "vix_ratio_5d_chg", "vix_ratio_max_5d", "vix_inverted_persist",
            "vix3m_chg_5d",
        ]:
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
    closes: dict[str, pd.Series] = {}

    for name, sym in symbols.items():
        try:
            raw = flatten_yf(_dp_download_single(sym, start=start, end=end))
            # Forward-fill only. Do not bfill early rows with future index/ETF
            # prices; missing early returns are handled as neutral below.
            close = raw["Close"].reindex(dates, method="ffill")
            closes[name] = close
            result[f"{name}_ret1d"] = close.pct_change(1, fill_method=None).fillna(0)
            result[f"{name}_ret5d"] = close.pct_change(5, fill_method=None).fillna(0)
            result[f"{name}_ret20d"] = close.pct_change(20, fill_method=None).fillna(0)
            result[f"{name}_ret60d"] = close.pct_change(60, fill_method=None).fillna(0)
            result[f"{name}_ret120d"] = close.pct_change(120, fill_method=None).fillna(0)
            # Horizon-matched return for the target function — must equal RETURN_HORIZON_DAYS
            # so make_direction_target can shift it forward to get the correct benchmark window.
            if name in ("spy", "sector") and RETURN_HORIZON_DAYS not in (5, 20):
                result[f"{name}_ret{RETURN_HORIZON_DAYS}d"] = close.pct_change(RETURN_HORIZON_DAYS, fill_method=None).fillna(0)
            if name == "sector":
                # Save raw sector ETF close so build_sector_strength_features()
                # can compute price-ratio z-scores (how unusual is today's
                # outperformance vs the past N days?).
                # This column is intermediate — it is dropped by
                # keep_conservative_feature_set() after the ratio features are built.
                result["sector_close"] = close.values
            if name in ("spy", "qqq", "dia"):
                ma20 = close.rolling(20, min_periods=20).mean()
                result[f"{name}_ma20_ready"] = ma20.notna().astype(float)
                result[f"{name}_above_ma20"] = ((close > ma20) & ma20.notna()).astype(int)
                result[f"{name}_vol10"] = close.pct_change(1, fill_method=None).rolling(10, min_periods=10).std().fillna(0)
            if name == "spy":
                spy_close = close
                ma200 = close.rolling(200, min_periods=200).mean()
                ma200_proxy = ma200.fillna(close.expanding(min_periods=1).mean())
                result["spy_ma200_ready"] = ma200.notna().astype(float)
                result["spy_dist_ma200"] = ((close - ma200_proxy) / (ma200_proxy + 1e-9)).clip(-0.3, 0.3).fillna(0)
                ma50 = close.rolling(50, min_periods=50).mean()
                ma50_proxy = ma50.fillna(close.expanding(min_periods=1).mean())
                result["spy_ma50_ready"] = ma50.notna().astype(float)
                result["spy_dist_ma50"] = ((close - ma50_proxy) / (ma50_proxy + 1e-9)).clip(-0.2, 0.2).fillna(0)
            if name == "vix":
                vix_close = close
            if name == "gld":
                result["gld_risk_off"] = (close.pct_change(5, fill_method=None) > 0.02).astype(int).fillna(0)
            if name == "hyg":
                result["credit_stress"] = (close.pct_change(5, fill_method=None) < -0.02).astype(int).fillna(0)
        except Exception:
            continue

    # Derived macro features use relative spreads/z-scores rather than raw ETF
    # returns. The raw tnx/hyg/uup/etc. columns stay excluded by the model
    # selector; only these macro_* summaries are allowed downstream.
    if {"tnx", "irx"}.issubset(closes):
        slope = (closes["tnx"] - closes["irx"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["macro_yield_curve_10y_3m"] = slope.clip(-10.0, 10.0).values
        mean = slope.rolling(252, min_periods=40).mean()
        std = slope.rolling(252, min_periods=40).std()
        result["macro_yield_curve_10y_3m_z"] = ((slope - mean) / (std + 1e-9)).clip(-4.0, 4.0).fillna(0.0).values

    if "uup" in closes:
        uup_ret = closes["uup"].pct_change(20, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = uup_ret.rolling(252, min_periods=40).mean()
        std = uup_ret.rolling(252, min_periods=40).std()
        result["macro_dxy_proxy_20d_z"] = ((uup_ret - mean) / (std + 1e-9)).clip(-4.0, 4.0).fillna(0.0).values

    if {"hyg", "ief"}.issubset(closes):
        credit = (
            closes["hyg"].pct_change(20, fill_method=None) -
            closes["ief"].pct_change(20, fill_method=None)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["macro_credit_spread_20d"] = credit.clip(-0.3, 0.3).values
        mean = credit.rolling(252, min_periods=40).mean()
        std = credit.rolling(252, min_periods=40).std()
        result["macro_credit_spread_20d_z"] = ((credit - mean) / (std + 1e-9)).clip(-4.0, 4.0).fillna(0.0).values

    if {"copper", "gld"}.issubset(closes):
        ratio = (closes["copper"] / (closes["gld"] + 1e-9)).replace([np.inf, -np.inf], np.nan)
        ratio_ret = ratio.pct_change(20, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["macro_copper_gold_20d"] = ratio_ret.clip(-0.3, 0.3).values
        mean = ratio_ret.rolling(252, min_periods=40).mean()
        std = ratio_ret.rolling(252, min_periods=40).std()
        result["macro_copper_gold_20d_z"] = ((ratio_ret - mean) / (std + 1e-9)).clip(-4.0, 4.0).fillna(0.0).values

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

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_literature_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classic equity anomaly factors, computed causally from trailing data."""
    result = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df.get("Close"), errors="coerce")
    volume = pd.to_numeric(df.get("Volume"), errors="coerce").fillna(0.0)
    ret_1d = close.pct_change(1, fill_method=None)

    # 12-1 momentum: prior 12-month return skipping the most recent month.
    mom_12_1 = close.shift(21) / (close.shift(252) + 1e-9) - 1.0
    result["factor_mom_12_1"] = mom_12_1.clip(-1.0, 5.0)

    if "sector_close" in df.columns:
        sector_close = pd.to_numeric(df["sector_close"], errors="coerce")
        sector_mom_12_1 = sector_close.shift(21) / (sector_close.shift(252) + 1e-9) - 1.0
        result["factor_resid_mom_sector_12_1"] = (mom_12_1 - sector_mom_12_1).clip(-2.0, 2.0)
    else:
        result["factor_resid_mom_sector_12_1"] = 0.0

    if "spy_ret1d" in df.columns:
        spy_ret = pd.to_numeric(df["spy_ret1d"], errors="coerce")
        cov = ret_1d.rolling(252, min_periods=80).cov(spy_ret)
        var = spy_ret.rolling(252, min_periods=80).var()
        beta = (cov / (var + 1e-12)).replace([np.inf, -np.inf], np.nan)
        residual = ret_1d - beta * spy_ret
        result["factor_beta_252_spy"] = beta.clip(-3.0, 5.0)
        result["factor_idio_vol_252_spy"] = (
            residual.rolling(252, min_periods=80).std() * np.sqrt(252)
        ).clip(0.0, 3.0)
    else:
        result["factor_beta_252_spy"] = 1.0
        result["factor_idio_vol_252_spy"] = 0.0

    dollar_vol = (close.abs() * volume).replace([np.inf, -np.inf], np.nan)
    avg_dollar_vol_20d = dollar_vol.rolling(20, min_periods=5).mean()
    result["factor_liquidity_dollar_vol_20d"] = np.log1p(avg_dollar_vol_20d).clip(0.0, 30.0)
    amihud = (ret_1d.abs() / (dollar_vol + 1e-9)).rolling(20, min_periods=5).mean()
    result["factor_illiquidity_amihud_20d"] = np.log1p(amihud * 1e9).clip(0.0, 30.0)

    # 52-week high proximity — George & Hwang (2004), Journal of Finance.
    # PLAIN ENGLISH: How close is today's price to its highest price in the
    # past year?  A value of 1.0 means the stock IS at its 52-week high;
    # 0.8 means it's 20% below.  Stocks near their 52-week high tend to
    # OUTPERFORM because investors anchor to the high as a mental ceiling,
    # creating underreaction when the stock breaks through.  This is distinct
    # from standard momentum (12-1) — it captures anchoring bias rather than
    # trend-following.  Cross-sectional t-stat > 5 in academic literature.
    high_252 = close.rolling(252, min_periods=60).max()
    result["factor_52w_high_proximity"] = (close / (high_252 + 1e-9)).clip(0.0, 1.2)

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar-based features for seasonality and market structure events.

    PLAIN ENGLISH: The stock market has predictable patterns tied to the
    calendar — stocks behave differently on Mondays vs Fridays, at month-end
    (when funds rebalance), and during options expiration week (when dealers
    unwind hedges causing extra volatility).  This function flags those days
    so the model can learn these patterns.

    Features produced:
      - dow_monday/dow_friday: day-of-week flags
      - month_sin/month_cos: cyclical month encoding (captures Jan effect, etc.)
      - month_end: 1 if this is one of the last 3 trading days of the month
      - opex_week: 1 if this trading day falls in the same Mon-Fri week as
        the third Friday of the month (monthly options expiration)
    """
    result = pd.DataFrame(index=dates)
    result["dow_monday"] = (dates.dayofweek == 0).astype(int)
    result["dow_friday"] = (dates.dayofweek == 4).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dates.month / 12)

    # ── Vectorized month_end: last 3 trading days of each month ──────────
    # PLAIN ENGLISH: Instead of looping through every date (slow O(n²)),
    # group dates by (year, month) and rank from the end.  The last 3 dates
    # in each group get flagged.  This is 100x faster for large date ranges.
    s = pd.Series(np.arange(len(dates)), index=dates)
    group_rank = s.groupby([dates.year, dates.month]).rank(
        method="first", ascending=False
    )
    result["month_end"] = (group_rank <= 3).astype(int)

    # ── OpEx week: the week containing the third Friday of each month ────
    # PLAIN ENGLISH: Monthly options expiration (OpEx) is the third Friday
    # of every month.  Massive open interest expires that day, causing
    # dealers to unwind delta hedges and creating abnormal volume/volatility.
    # The effect begins Monday of that week as dealers start adjusting.
    # We flag the entire Mon-Fri week so the model can learn this pattern.
    first_month = dates.min().replace(day=1)
    last_month = dates.max().replace(day=1) + pd.offsets.MonthBegin(1)
    month_starts = pd.date_range(first_month, last_month, freq="MS")

    opex_weeks = set()
    for ms in month_starts:
        # Third Friday = first Friday of month + 14 calendar days.
        # First Friday: offset from day 1 based on its weekday.
        first_day_dow = ms.weekday()  # 0=Mon, 4=Fri
        days_to_friday = (4 - first_day_dow) % 7
        first_friday = ms + pd.Timedelta(days=days_to_friday)
        third_friday = first_friday + pd.Timedelta(days=14)
        # Flag Mon-Fri of that week (third_friday is Friday, so Mon = Fri - 4)
        week_monday = third_friday - pd.Timedelta(days=4)
        for d in range(5):
            opex_weeks.add(week_monday + pd.Timedelta(days=d))

    result["opex_week"] = dates.normalize().isin(opex_weeks).astype(int)

    # ── FOMC proximity flags ────────────────────────────────────────────
    # PLAIN ENGLISH: The Federal Reserve's FOMC meeting decides interest
    # rates 8 times a year.  Markets often see abnormal moves on the
    # announcement day (~14:00 ET) and during the press conference.  The
    # day BEFORE FOMC is also distinctive — traders unwind positions
    # ahead of the binary risk, leading to lower volume + drift.  The
    # day AFTER is the "post-FOMC bounce/sell" regime.
    #
    # We hard-code the published FOMC calendar through 2026 (the Fed
    # publishes 1-2 years ahead).  Add new years here when the FOMC
    # schedule is announced; default to NO flag for unknown years rather
    # than guess (silently mis-flagging would corrupt training).
    fomc_dates = _fomc_meeting_dates()
    if fomc_dates:
        fomc_index = pd.DatetimeIndex(fomc_dates).normalize()
        # Day-of-meeting flag (the announcement happens that afternoon)
        result["fomc_day"] = dates.normalize().isin(fomc_index).astype(int)
        # Day-before (pre-FOMC drift regime): map each FOMC date back one
        # trading day.  Using -1 calendar day is an approximation; close
        # enough for a binary feature.
        pre_fomc = pd.DatetimeIndex([d - pd.Timedelta(days=1) for d in fomc_dates]).normalize()
        result["fomc_day_minus_1"] = dates.normalize().isin(pre_fomc).astype(int)
        # Day-after (post-FOMC trade)
        post_fomc = pd.DatetimeIndex([d + pd.Timedelta(days=1) for d in fomc_dates]).normalize()
        result["fomc_day_plus_1"] = dates.normalize().isin(post_fomc).astype(int)
    else:
        result["fomc_day"] = 0
        result["fomc_day_minus_1"] = 0
        result["fomc_day_plus_1"] = 0

    return result


# Hard-coded FOMC meeting dates (the day the policy decision is announced).
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm.  Update this
# list when the Fed publishes the next year's calendar — usually around
# October.  Leaving years unmapped is safer than guessing wrong dates.
_FOMC_DATES_BY_YEAR: dict[int, list[str]] = {
    2018: ["2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
           "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19"],
    2019: ["2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
           "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11"],
    2020: ["2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
           "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16"],
    2021: ["2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
           "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15"],
    2022: ["2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
           "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14"],
    2023: ["2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
           "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13"],
    2024: ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
           "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"],
    2025: ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
           "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"],
    2026: ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
           "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"],
}


def _fomc_meeting_dates() -> list[pd.Timestamp]:
    """Return all known FOMC announcement dates as a flat Timestamp list."""
    flat: list[pd.Timestamp] = []
    for year_dates in _FOMC_DATES_BY_YEAR.values():
        flat.extend(pd.Timestamp(d) for d in year_dates)
    return flat


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
    "vol_zscore_",
    "vol_pct_vs_",
    "vol_trend_",
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
    "macro_",
    "regime",
    "ret_vs_spy",
    "ret_vs_qqq",
    "breadth_",
    "pct_above_",
    "concentration_",
    "fund_",
    "factor_",
    "xs_rank_",
    # Sector-relative momentum features (added: ratio z-scores, RS slope/MA)
    "sector_vs_",      # sector_vs_spy_5d — sector leadership vs broad market
    "sector_ratio_",   # sector_ratio_z20, sector_ratio_z60 — z-score of price/sector_ETF
    "sector_rs_",      # sector_rs_slope_5d, sector_rs_above_ma20 — RS momentum
    "sector_rel_",     # sector_rel_return_60d, sector_rel_mom_20d — stock vs sector ETF
    "alt_",            # alternative data: EPS revisions, analyst recs, short interest, institutional ownership
)
CONSERVATIVE_FEATURE_EXACT = {
    "obv_slope",
    "vwap_dist",
    "uptick_ratio",
    "variance_ratio",
    # PEAD earnings features — point-in-time safe (guarded by USE_EARNINGS_DATA)
    "eps_surprise_pct",
    "days_since_earnings",
    "days_to_next_earnings",
    # Options IV features — real data from Tradier (neutral defaults for backtest)
    "iv_atm",
    "iv_rank_proxy",
    "iv_hv_spread",
    "iv_skew",
    "put_call_ratio",
    # Intraday features — VWAP and volume profile from Alpaca minute bars
    "vwap_dist_5d",
    "volume_am_ratio",
    "volume_last_hour",
    "intraday_range_pct",
    "close_position",
}
RISKY_FEATURE_KEYWORDS = (
    # Social sentiment is live safety-only by default. It is intentionally
    # filtered from model feature frames unless SOCIAL_SENTIMENT_ALPHA_ENABLED
    # is explicitly enabled after separate validation.
    "social",
    "iv_",
    "put_call",
    "option",
    "analyst",
    "recommend",
    "short_interest",
    "dark_pool",
)
# Earnings features that are explicitly point-in-time safe (PEAD).
# Previously blocked by "earn"/"eps_" risky keywords, now whitelisted
# because USE_EARNINGS_DATA guards them at the source.
EARNINGS_SAFE_COLUMNS = {
    "eps_surprise_pct",
    "days_since_earnings",
    "days_to_next_earnings",
}

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


# Trailing-window length for rolling sentiment z-scores.  ~1 trading year —
# long enough to be statistically stable, short enough to discount
# multi-year regime shifts (e.g. the meme-stock era's sentiment baseline
# vs the 2022 bear's).  Env-overridable for experiments.
SENTIMENT_ZSCORE_ROLLING_WINDOW = int(os.environ.get("SENTIMENT_ZSCORE_ROLLING_WINDOW", "252"))
SENTIMENT_ZSCORE_MIN_OBS = int(os.environ.get("SENTIMENT_ZSCORE_MIN_OBS", "60"))


def _rolling_zscore(
    series: pd.Series,
    *,
    window: int,
    min_obs: int,
) -> pd.Series:
    """Return a point-in-time rolling z-score of `series`.

    PLAIN ENGLISH: For each row, computes (value - mean_of_last_252_days) /
    std_of_last_252_days using ONLY data up to and including that row.  No
    forward leakage — safe to use in training AND live prediction.  Early
    rows where the rolling window has fewer than min_obs observations
    return 0 (sentiment is treated as neutral when we have no baseline).
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rmean = s.rolling(window=window, min_periods=min_obs).mean()
    rstd = s.rolling(window=window, min_periods=min_obs).std(ddof=0)
    z = (s - rmean) / rstd.where(rstd > 1e-6, 1.0)
    return z.clip(-5.0, 5.0).fillna(0.0)


def apply_sentiment_distribution_matching(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, float]] | None = None,
    *,
    use_rolling: bool = True,
    rolling_window: int = SENTIMENT_ZSCORE_ROLLING_WINDOW,
    min_obs: int = SENTIMENT_ZSCORE_MIN_OBS,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Z-score sentiment columns and write them back as ``sent_z_<col>``.

    Two modes:

      use_rolling=True (default, addresses audit #11) — per-ticker
        trailing-window z-score.  Each row uses ONLY the last
        `rolling_window` observations of the same ticker.  Adapts to
        sentiment regime shifts (e.g. post-meme-stock era recalibration)
        without forward leakage.  The returned ``stats`` dict still
        contains the global fallback (used when a ticker has fewer than
        `min_obs` history rows).

      use_rolling=False — legacy behaviour.  Computes a single global
        mean/std on the entire frame (or uses pre-fit stats) and applies
        it uniformly.  Kept for backwards compatibility with saved model
        artefacts that depend on those stats.

    Both modes write ``sent_z_<col>`` features, so downstream code is
    unchanged.
    """
    out = frame.copy()
    fitted = stats or fit_sentiment_zscore_stats(out)

    if not use_rolling or "ticker" not in out.columns:
        # Legacy mode — single global mean/std for the whole frame.
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

    # Rolling mode — per-ticker trailing z-score (point-in-time).
    # We also write the global fallback as a fillna source for new
    # tickers / early rows where the trailing window is too thin.
    out = out.sort_values(["ticker", "date"]) if "date" in out.columns else out.sort_values(["ticker"])
    for col, st in fitted.items():
        if col not in out.columns:
            out[col] = 0.0
        # Global fallback z-score (uses fitted stats — same math as legacy).
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = float(st.get("mean", 0.0))
        std = float(st.get("std", 1.0)) or 1.0
        if not np.isfinite(std) or abs(std) < 1e-6:
            std = 1.0
        global_z = ((s - mean) / std).clip(-5.0, 5.0)
        # Per-ticker rolling z-score (groupby preserves index alignment).
        rolling_z = out.groupby("ticker", sort=False, group_keys=False)[col].apply(
            lambda g: _rolling_zscore(g, window=rolling_window, min_obs=min_obs)
        )
        # Use rolling where defined, fall back to global where rolling is
        # NaN (early rows with insufficient history).
        out[f"sent_z_{col}"] = rolling_z.where(rolling_z.notna(), global_z)

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


def apply_live_fundamental_sector_zscores(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply latest cached sector valuation stats for live prediction frames."""
    out = frame.copy()
    for col in ["fund_pe_sector_z", "fund_fcf_yield_sector_z", "fund_value_combo_z"]:
        if col not in out.columns:
            out[col] = 0.0
    if not {"fund_pe_ttm", "fund_fcf_yield_ttm"}.issubset(out.columns):
        return out
    try:
        import json
        stats_path = os.path.join(DATA_DIR, "fundamental_sector_latest_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        sector = SECTOR_MAP.get(ticker.upper(), "OTHER")
        st = stats.get(sector) or stats.get("OTHER") or {}
        pe_std = float(st.get("pe_std", 1.0)) or 1.0
        fcf_std = float(st.get("fcf_yield_std", 1.0)) or 1.0
        pe_z = (pd.to_numeric(out["fund_pe_ttm"], errors="coerce") - float(st.get("pe_mean", 0.0))) / pe_std
        fcf_z = (pd.to_numeric(out["fund_fcf_yield_ttm"], errors="coerce") - float(st.get("fcf_yield_mean", 0.0))) / fcf_std
        out["fund_pe_sector_z"] = pe_z.clip(-5.0, 5.0).fillna(0.0)
        out["fund_fcf_yield_sector_z"] = fcf_z.clip(-5.0, 5.0).fillna(0.0)
        out["fund_value_combo_z"] = (out["fund_fcf_yield_sector_z"] - out["fund_pe_sector_z"]).clip(-8.0, 8.0)
    except Exception:
        pass
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
        # alt_ prefixed features are curated alternative data (EPS revisions,
        # analyst recs, short interest) — skip the risky-keyword blocker for them.
        if not col_l.startswith("alt_") and any(k in col_l for k in RISKY_FEATURE_KEYWORDS):
            continue
        if col.startswith("sent_z_"):
            keep.append(col)
            continue
        if col in CONSERVATIVE_FEATURE_EXACT or any(col.startswith(prefix) for prefix in CONSERVATIVE_FEATURE_PREFIXES):
            keep.append(col)
    return out.loc[:, keep]

def build_options_features_context(ticker: str, dates: pd.DatetimeIndex, live: bool = False) -> pd.DataFrame:
    """Build options features -- real data from Tradier if live, else neutral defaults.

    PLAIN ENGLISH: For backtesting (live=False), we don't have historical
    options data so we return neutral defaults.  For live predictions
    (live=True), we call Tradier's API to get real implied volatility,
    put/call ratio, and IV skew.  The iv_rank_proxy column from
    build_iv_rank_features() is ALSO computed from historical vol, but
    these real IV features are much more informative when available.
    """
    result = pd.DataFrame(index=dates)
    result["iv_atm"] = 0.25
    result["put_call_ratio"] = 1.0
    result["iv_skew"] = 0.0

    if live:
        try:
            from options_iv_provider import fetch_live_iv_features
            iv_data = fetch_live_iv_features(ticker)
            if iv_data.get("iv_has_real_data", 0) > 0:
                result["iv_atm"] = float(iv_data["iv_atm"])
                result["put_call_ratio"] = float(iv_data["put_call_ratio"])
                result["iv_skew"] = float(iv_data["iv_skew"])
        except ImportError:
            pass  # Tradier not installed/configured -- use defaults
        except Exception:
            pass  # API error -- use defaults

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
        build_point_in_time_valuation_features(ticker, df),
        build_market_breadth_features(dates, start, end),
        build_market_concentration_features(dates, start, end),
        # Earnings-surprise features (PEAD): eps_surprise_pct, days_since/to
        # earnings.  Returns neutral values when USE_EARNINGS_DATA=False.
        build_earnings_features_context(ticker, dates),
    ]
    if USE_NEWS_SENTIMENT:
        try:
            frames.append(build_sentiment_features(ticker, dates))
        except Exception:
            pass
    if SOCIAL_SENTIMENT_ALPHA_ENABLED:
        try:
            frames.append(build_social_sentiment_features(ticker, dates))
        except Exception:
            pass

    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    factors = build_literature_factor_features(out)
    out = pd.concat([out, factors], axis=1)
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
    verbose: bool = False,
) -> pd.DataFrame | None:
    end = datetime.now(timezone.utc).date()
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
        build_point_in_time_valuation_features(ticker, df),
        build_market_breadth_features(dates, start_str, end_str),
        build_market_concentration_features(dates, start_str, end_str),
        # Earnings-surprise features (PEAD): eps_surprise_pct, days_since/to
        # earnings.  Returns neutral values when USE_EARNINGS_DATA=False.
        build_earnings_features_context(ticker, dates),
    ]

    diagnostic_frames: list[pd.DataFrame] = []
    if USE_NEWS_SENTIMENT:
        try:
            sent = build_sentiment_feature_dataframe(
                ticker,
                dates,
                finnhub_client=None,
                engine_level=LIVE_SENTIMENT_ENGINE_LEVEL,
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
                "sentiment_primary_available",
                "sentiment_fallback_available",
                "sentiment_fresh_headline_count",
            ]:
                if col not in sent.columns:
                    sent[col] = 0.0
            if "sentiment_provider_status_json" not in sent.columns:
                sent["sentiment_provider_status_json"] = ""

            try:
                engine = get_sentiment_engine(LIVE_SENTIMENT_ENGINE_LEVEL)
                live_news = score_todays_news(ticker, engine, verbose=verbose) or {}
                latest = dates[-1]
                composite_score = float(live_news.get("composite_score", 0.0) or 0.0)
                headline_count = float(live_news.get("headline_count", 0) or 0.0)
                breaking_count = float(live_news.get("breaking_count", 0) or 0.0)
                score_abs = float(live_news.get("avg_abs_score", 0.0) or 0.0)
                primary_available = 1.0 if live_news.get("primary_available") else 0.0
                fallback_available = 1.0 if live_news.get("fallback_available") else 0.0
                fresh_count = float(live_news.get("fresh_headline_count", headline_count) or 0.0)
                has_signal = 1.0 if (
                    fresh_count > 0 or headline_count > 0 or abs(composite_score) > 1e-9 or score_abs > 1e-9
                ) else 0.0

                sent.loc[latest, "news_sentiment"] = composite_score
                sent.loc[latest, "headline_volume"] = headline_count
                sent.loc[latest, "live_news_headline_count"] = headline_count
                sent.loc[latest, "live_news_breaking"] = breaking_count
                sent.loc[latest, "live_news_score_abs"] = score_abs
                sent.loc[latest, "live_news_has_signal"] = has_signal
                sent.loc[latest, "sentiment_primary_available"] = primary_available
                sent.loc[latest, "sentiment_fallback_available"] = fallback_available
                sent.loc[latest, "sentiment_fresh_headline_count"] = fresh_count
                sent.loc[latest, "sentiment_provider_status_json"] = json.dumps(
                    live_news.get("provider_status", []),
                    sort_keys=True,
                )
            except Exception as e:
                print(f"[{ticker}] live news refresh failed: {e}")

            diagnostic_frames.append(sent)
        except Exception as e:
            print(f"[{ticker}] sentiment feature builder failed: {e}")

    if SOCIAL_SENTIMENT_ALPHA_ENABLED:
        try:
            social_df = build_social_sentiment_features(ticker, dates)
            social_live = get_live_social_signal(ticker)
            if not social_df.empty and social_live.get("social_available"):
                latest = social_df.index[-1]
                social_fresh_count = max(
                    float(social_live.get("message_volume", 0.0) or 0.0),
                    float(social_live.get("x_tweets", 0.0) or 0.0),
                )
                social_df.loc[latest, "social_combined"] = float(social_live.get("combined", 0.0))
                social_df.loc[latest, "social_message_volume"] = float(social_live.get("message_volume", 0.0))
                social_df.loc[latest, "sentiment_fallback_available"] = 1.0
                social_df.loc[latest, "sentiment_fresh_headline_count"] = max(
                    float(social_df.get("sentiment_fresh_headline_count", pd.Series(0.0, index=social_df.index)).loc[latest]),
                    social_fresh_count,
                )
                if diagnostic_frames:
                    base = diagnostic_frames[0]
                    base.loc[latest, "sentiment_fallback_available"] = max(
                        float(base.get("sentiment_fallback_available", pd.Series(0.0, index=base.index)).loc[latest]),
                        1.0,
                    )
                    base.loc[latest, "sentiment_fresh_headline_count"] = max(
                        float(base.get("sentiment_fresh_headline_count", pd.Series(0.0, index=base.index)).loc[latest]),
                        social_fresh_count,
                    )
            diagnostic_frames.append(social_df)
        except Exception:
            pass
    elif SOCIAL_SENTIMENT_SAFETY_ENABLED:
        try:
            social_live = get_live_social_signal(ticker)
            if social_live.get("social_available"):
                latest = dates[-1]
                social_fresh_count = max(
                    float(social_live.get("message_volume", 0.0) or 0.0),
                    float(social_live.get("x_tweets", 0.0) or 0.0),
                )
                if diagnostic_frames:
                    base = diagnostic_frames[0]
                else:
                    base = pd.DataFrame(index=dates, data={
                        "sentiment_primary_available": 0.0,
                        "sentiment_fallback_available": 0.0,
                        "sentiment_fresh_headline_count": 0.0,
                        "sentiment_provider_status_json": "",
                    })
                    diagnostic_frames.append(base)
                base.loc[latest, "sentiment_fallback_available"] = max(
                    float(base.get("sentiment_fallback_available", pd.Series(0.0, index=base.index)).loc[latest]),
                    1.0,
                )
                base.loc[latest, "sentiment_fresh_headline_count"] = max(
                    float(base.get("sentiment_fresh_headline_count", pd.Series(0.0, index=base.index)).loc[latest]),
                    social_fresh_count,
                )
        except Exception:
            pass

    # ── Options IV features (real data from Tradier if available) ───────────
    # For live predictions, fetch real implied volatility, put/call ratio,
    # and IV skew from the options market.  For backtesting, these stay at
    # neutral defaults (iv_atm=0.25, put_call_ratio=1.0, iv_skew=0.0).
    frames.append(build_options_features_context(ticker, dates, live=True))

    # ── Intraday features (VWAP, volume profile from Alpaca) ─────────────
    # These capture institutional flow patterns that daily OHLCV misses:
    # where the stock closed relative to VWAP, whether volume was
    # front-loaded (smart money) or back-loaded (retail/MOC), and how
    # volatile the intraday price action was.
    try:
        from intraday_features import fetch_intraday_features
        intraday = fetch_intraday_features(ticker)
        if intraday.get("has_intraday_data", 0) > 0:
            intraday_df = pd.DataFrame(index=dates)
            for col, val in intraday.items():
                if col != "has_intraday_data":
                    intraday_df[col] = 0.0
                    intraday_df.iloc[-1, intraday_df.columns.get_loc(col)] = float(val)
            frames.append(intraday_df)
    except ImportError:
        pass  # Alpaca not configured -- skip
    except Exception:
        pass  # API error -- skip

    out = pd.concat([df] + frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    factors = build_literature_factor_features(out)
    out = pd.concat([out, factors], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = apply_live_fundamental_sector_zscores(ticker, out)
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
        for col in [
            "sentiment_provider_status_json",
            "sentiment_primary_available",
            "sentiment_fallback_available",
            "sentiment_fresh_headline_count",
        ]:
            if col in diagnostics.columns:
                keep_diagnostics.append(col)
        keep_diagnostics = list(dict.fromkeys(keep_diagnostics))
        if keep_diagnostics:
            out = pd.concat([out, diagnostics[keep_diagnostics]], axis=1)
    missing_cols = [c for c in feature_cols if c not in out.columns]
    if missing_cols:
        missing_frame = pd.DataFrame(0.0, index=out.index, columns=missing_cols)
        # Live prediction is built one ticker at a time, so true same-date
        # cross-sectional ranks are unavailable here. Use neutral percentile
        # ranks instead of zeros; 0 would mean "worst in universe".
        for col in missing_cols:
            if col.startswith("xs_rank_"):
                missing_frame[col] = 0.5
        out = pd.concat([out, missing_frame], axis=1)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
