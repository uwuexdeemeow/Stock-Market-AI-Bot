"""
fundamental_features.py — Free alpha features: PEAD, IV rank proxy,
sector relative strength, and market breadth.

What this file does
-------------------
Each function here builds a small table of new "features" — numbers the
model can learn from — and returns it as a DataFrame whose rows line up
with the dates in your price data.

The four feature groups:
  1. PEAD (Post-Earnings Announcement Drift)
       Stocks often keep drifting after a big earnings surprise.
       Features: eps_surprise_pct, days_since_earnings, days_to_next_earnings

  2. IV Rank proxy (Implied Volatility Rank)
       Expensive options often signal coming moves.
       We can't get historical options prices for free, so we use
       "historical volatility percentile" as a proxy — how high is
       today's realized vol compared to its past year?
       Features: iv_rank_proxy, iv_hv_spread

  3. Sector relative strength
       If AAPL is beating the XLK tech ETF, that's a bullish sign.
       Computed from already-downloaded sector ETF data.
       Features: ret_vs_sector_5d, ret_vs_sector_20d, sector_vs_spy_5d

  4. Market breadth
       "Is the whole market healthy, or just a few stocks?"
       Downloads the S&P 500 % above 50-day MA (^MMFI) from Yahoo Finance.
       Features: pct_above_50ma, pct_above_200ma, breadth_slope_5d
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from settings import USE_EARNINGS_DATA
except Exception:
    USE_EARNINGS_DATA = False


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Remove multi-level column headers that yfinance sometimes produces."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. PEAD — Post-Earnings Announcement Drift
# ─────────────────────────────────────────────────────────────────────────────

def build_pead_features(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build earnings-surprise features aligned to the price series dates.

    What "PEAD" means:
        After a company beats earnings expectations, the stock tends to keep
        drifting upward for days/weeks (and vice-versa for misses).
        These features let the model capture that pattern.

    Leakage guard:
        Earnings figures are only known AFTER the release date, so we use a
        strict "less-than" comparison — the surprise from earnings on day T is
        only visible to the model starting day T+1.

    Returns columns:
        eps_surprise_pct       — % beat/miss vs analyst estimates (clipped ±100%)
        days_since_earnings    — trading-calendar days since last report (max 120)
        days_to_next_earnings  — calendar days until next scheduled report (max 120)
    """
    # Default: 0% surprise, 60 days since/to earnings (neutral midpoint)
    result = pd.DataFrame(index=dates, data={
        "eps_surprise_pct": 0.0,
        "days_since_earnings": 60.0,
        "days_to_next_earnings": 60.0,
    })
    if not USE_EARNINGS_DATA:
        return result

    try:
        tk = yf.Ticker(ticker)
        # get_earnings_dates returns ~40 past + future scheduled dates
        earnings = tk.get_earnings_dates(limit=40)
        if earnings is None or earnings.empty:
            return result

        # Strip timezone so we can compare with tz-naive price dates.
        # tz_localize(None) fails on already-tz-aware indexes; use tz_convert(None).
        idx = pd.to_datetime(earnings.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        earnings.index = idx.normalize()
        earnings = earnings.sort_index()

        # Detect column names (they vary by yfinance version)
        surprise_col = next((c for c in earnings.columns if "surprise" in c.lower()), None)
        reported_col = next((c for c in earnings.columns if "reported" in c.lower()), None)

        # "Past earnings" = rows where the company actually reported (not future estimates)
        if reported_col is not None:
            past_earnings = earnings[earnings[reported_col].notna()].copy()
        else:
            # Fallback: assume all rows without NaN surprise are past earnings
            past_earnings = (
                earnings[earnings[surprise_col].notna()].copy()
                if surprise_col else earnings.copy()
            )

        if past_earnings.empty:
            return result

        # Convert to numpy datetime64[D] for fast vectorized date math
        earn_arr = past_earnings.index.values.astype("datetime64[D]")
        all_earn_arr = earnings.index.values.astype("datetime64[D]")
        dates_arr = pd.DatetimeIndex(dates).normalize().values.astype("datetime64[D]")

        # --- days_since_earnings ---
        # For each price date, find the most-recent PAST earnings date
        # (strictly before the price date — the "less-than" is the leakage guard)
        past_pos = np.searchsorted(earn_arr, dates_arr, side="left") - 1
        has_past = past_pos >= 0
        safe_past = np.clip(past_pos, 0, len(earn_arr) - 1)

        # Subtraction of datetime64[D] arrays gives integer days
        days_since = np.where(
            has_past,
            (dates_arr.astype("int64") - earn_arr[safe_past].astype("int64")),
            120,
        )
        result["days_since_earnings"] = np.minimum(days_since, 120).astype(float)

        # --- eps_surprise_pct ---
        if surprise_col is not None:
            # Pull the surprise percentage from the most recent past earnings row
            surprise_vals = past_earnings[surprise_col].to_numpy(dtype=float, na_value=0.0)
            result["eps_surprise_pct"] = np.where(
                has_past,
                np.clip(surprise_vals[safe_past], -100.0, 100.0),
                0.0,
            )

        # --- days_to_next_earnings ---
        # Find the NEXT scheduled earnings date (including future estimates)
        future_pos = np.searchsorted(all_earn_arr, dates_arr, side="left")
        has_future = future_pos < len(all_earn_arr)
        safe_future = np.clip(future_pos, 0, len(all_earn_arr) - 1)

        days_to = np.where(
            has_future,
            (all_earn_arr[safe_future].astype("int64") - dates_arr.astype("int64")),
            120,
        )
        result["days_to_next_earnings"] = np.minimum(days_to, 120).astype(float)

    except Exception:
        # If anything fails (rate-limit, missing data, etc.), return neutral defaults
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. IV Rank proxy (Historical Volatility Percentile)
# ─────────────────────────────────────────────────────────────────────────────

def build_iv_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Implied Volatility Rank using Historical Volatility as a proxy.

    Why we need a proxy:
        True IV (options-implied volatility) history costs money to download.
        Historical Volatility (HV) — computed from price returns — is a decent
        free substitute because IV and HV track each other over time.

    IV Rank formula:
        iv_rank = (current_HV - 52w_low_HV) / (52w_high_HV - 52w_low_HV)
        A value near 1.0 means volatility is at a yearly high (expensive options).
        A value near 0.0 means volatility is compressed (cheap options).

    iv_hv_spread:
        The gap between ATM implied vol and realized HV.
        Positive = options are priced rich vs realized vol (market scared).
        For historical research we leave this at 0.0 (no historical IV data).
        For live prediction, pipeline_shared.py fills it from the options chain.

    Returns columns:
        iv_rank_proxy    — 0.0 to 1.0 percentile of hvol_20d over past 252 days
        iv_hv_spread     — placeholder 0.0 (overwritten live)
    """
    result = pd.DataFrame(index=df.index, data={"iv_rank_proxy": 0.5, "iv_hv_spread": 0.0})

    # Use hvol_20d if the technical features were already computed, else compute it
    if "hvol_20d" in df.columns:
        hv = df["hvol_20d"]
    else:
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        hv = log_ret.rolling(20).std() * np.sqrt(252)

    hv_min = hv.rolling(252, min_periods=20).min()
    hv_max = hv.rolling(252, min_periods=20).max()

    # Clip to [0, 1] so the model always sees a valid range
    rank = ((hv - hv_min) / (hv_max - hv_min + 1e-9)).clip(0.0, 1.0)
    result["iv_rank_proxy"] = rank.fillna(0.5).values

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3a. Gap Behaviour
# ─────────────────────────────────────────────────────────────────────────────

def build_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overnight gap features derived from daily OHLCV.

    A "gap" is when today's open is meaningfully different from yesterday's close
    — the stock jumps overnight.  Gaps that hold open (price doesn't reverse) are
    continuation signals.  Gaps that immediately fill are mean-reversion signals.

    Returns columns:
        gap_pct          — (open - prev_close) / prev_close.  Positive = gap up.
        gap_abs          — absolute gap size (unsigned magnitude)
        gap_z20          — gap_pct standardised over its own 20-day distribution
        gap_filled       — 1 if the gap was filled within the same day (open→close reversal)
        gap_direction    — sign of the gap: +1 up, -1 down, 0 flat
        gap_5d_avg       — 5-day rolling mean of gap_pct (trend in gap direction)
    """
    result = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)

    gap = (df["Open"] - prev_close) / (prev_close.abs() + 1e-9)
    result["gap_pct"] = gap.clip(-0.15, 0.15)
    result["gap_abs"] = result["gap_pct"].abs()
    result["gap_direction"] = np.sign(result["gap_pct"])

    roll_mean = result["gap_pct"].rolling(20, min_periods=5).mean()
    roll_std  = result["gap_pct"].rolling(20, min_periods=5).std()
    result["gap_z20"] = ((result["gap_pct"] - roll_mean) / (roll_std + 1e-9)).clip(-4, 4)

    # Gap filled: gap-up day where close falls back below open, or vice versa
    result["gap_filled"] = (
        ((gap > 0) & (df["Close"] < df["Open"])) |
        ((gap < 0) & (df["Close"] > df["Open"]))
    ).astype(float)

    result["gap_5d_avg"] = result["gap_pct"].rolling(5, min_periods=1).mean()

    return result.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Volume Z-Score
# ─────────────────────────────────────────────────────────────────────────────

def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume anomaly features.

    vol_ratio (already a technical feature) only compares 5-day vs 20-day
    averages.  These features look at the full trailing year so the model can
    detect truly unusual volume — e.g. a 3-sigma volume day that often precedes
    a sustained move.

    Returns columns:
        vol_zscore_252   — today's volume standardised over its trailing 252-day
                           distribution.  +2 = abnormally high, -2 = unusually quiet.
        vol_pct_vs_20d   — percent above/below the 20-day average volume (short-term)
        vol_trend_20_60  — ratio of 20-day avg to 60-day avg minus 1.
                           Positive = volume trending up (expanding interest).
    """
    result = pd.DataFrame(index=df.index)
    vol = df["Volume"].replace(0, np.nan).astype(float)

    v_mean = vol.rolling(252, min_periods=20).mean()
    v_std  = vol.rolling(252, min_periods=20).std()
    result["vol_zscore_252"] = ((vol - v_mean) / (v_std + 1e-9)).clip(-4, 4)

    v20 = vol.rolling(20, min_periods=5).mean()
    result["vol_pct_vs_20d"] = ((vol - v20) / (v20 + 1e-9)).clip(-3, 3)

    v60 = vol.rolling(60, min_periods=10).mean()
    result["vol_trend_20_60"] = ((v20 / (v60 + 1e-9)) - 1.0).clip(-1, 1)

    return result.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sector Relative Strength
# ─────────────────────────────────────────────────────────────────────────────

def build_sector_strength_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how the stock is performing RELATIVE to its sector ETF and SPY.

    Why this matters:
        A stock rising 3% is great if its sector is flat.
        The same 3% is less impressive if the whole sector rose 5%.
        Relative strength tells the model whether there is true buying interest
        specifically in this stock, vs just a sector-wide tide lifting all boats.

    Requires these columns to already be in df (added by build_multi_market):
        ret_5d, ret_20d       — the ticker's 5-day and 20-day returns
        sector_ret5d          — the sector ETF's 5-day return
        sector_ret20d         — the sector ETF's 20-day return (added below)
        spy_ret5d             — SPY's 5-day return

    Returns columns:
        ret_vs_sector_5d   — ticker 5d return minus sector ETF 5d return
        ret_vs_sector_20d  — ticker 20d return minus sector ETF 20d return
        sector_vs_spy_5d   — sector ETF 5d return minus SPY 5d return
                             (positive = sector is a leader, negative = laggard)
    """
    result = pd.DataFrame(index=df.index, data={
        "ret_vs_sector_5d": 0.0,
        "ret_vs_sector_20d": 0.0,
        "sector_vs_spy_5d": 0.0,
    })

    if "sector_ret5d" in df.columns and "ret_5d" in df.columns:
        result["ret_vs_sector_5d"] = (df["ret_5d"] - df["sector_ret5d"]).fillna(0.0).values

    if "sector_ret20d" in df.columns and "ret_20d" in df.columns:
        result["ret_vs_sector_20d"] = (df["ret_20d"] - df["sector_ret20d"]).fillna(0.0).values

    if "sector_ret5d" in df.columns and "spy_ret5d" in df.columns:
        result["sector_vs_spy_5d"] = (df["sector_ret5d"] - df["spy_ret5d"]).fillna(0.0).values

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Market Breadth
# ─────────────────────────────────────────────────────────────────────────────

def build_market_breadth_features(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """
    Build market breadth features from free Yahoo Finance ETFs.

    What "market breadth" is:
        When most stocks are participating in a rally, it's healthy and likely
        to continue. When only a handful of mega-caps are rising while most
        stocks drift lower, the rally is fragile. Breadth features let the
        model sense this difference.

    How we measure it (all free on Yahoo Finance):
        RSP  — Invesco S&P 500 Equal Weight ETF
               Equal-weight means small S&P names count as much as Apple/Microsoft.
               When RSP beats SPY, smaller names are participating → broad market.
               When RSP lags SPY, only mega-caps are moving → narrow market.

        IWM  — iShares Russell 2000 (small caps)
               Small caps are economically sensitive — when they lead, risk
               appetite is broad, which is a healthy market sign.

    Returns columns:
        breadth_rsp_vs_spy_5d   — RSP 5-day return minus SPY 5-day return
                                  (positive = broad participation, negative = narrow)
        breadth_iwm_vs_spy_5d   — IWM 5-day return minus SPY 5-day return
                                  (positive = small caps leading, risk-on breadth)
        breadth_rsp_ma20_dist   — RSP distance from its own 20-day MA (normalised)
                                  (positive = equal-weight market in uptrend)
        breadth_slope_5d        — 5-day momentum of RSP/SPY ratio
    """
    result = pd.DataFrame(index=dates, data={
        "breadth_rsp_vs_spy_5d": 0.0,
        "breadth_iwm_vs_spy_5d": 0.0,
        "breadth_rsp_ma20_dist": 0.0,
        "breadth_slope_5d": 0.0,
    })

    try:
        import io, sys as _sys
        # Suppress yfinance's "Failed downloads" noise when network is unavailable.
        _devnull = io.StringIO()
        _old_stderr = _sys.stderr
        try:
            _sys.stderr = _devnull
            raw = yf.download(["RSP", "SPY", "IWM"], start=start, end=end,
                              progress=False, auto_adjust=True)
        finally:
            _sys.stderr = _old_stderr

        # Do NOT use _flatten_yf here — multi-ticker downloads use a two-level
        # MultiIndex like ("Close", "RSP"). We access raw["Close"] to get a
        # DataFrame whose columns are the ticker symbols.
        if raw.empty:
            return result

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]  # columns = ["IWM", "RSP", "SPY"]
        else:
            close = raw[["Close"]].rename(columns={"Close": "SPY"})

        def _get(sym: str) -> pd.Series | None:
            if sym in close.columns:
                return close[sym].reindex(dates, method="ffill").bfill()
            return None

        rsp = _get("RSP")
        spy = _get("SPY")
        iwm = _get("IWM")

        if rsp is not None and spy is not None:
            # How much RSP over/underperformed SPY in last 5 days
            result["breadth_rsp_vs_spy_5d"] = (
                (rsp.pct_change(5) - spy.pct_change(5)).fillna(0.0).values
            )

            # RSP's position relative to its own 20-day MA — are small/mid names trending up?
            ma20 = rsp.rolling(20).mean()
            result["breadth_rsp_ma20_dist"] = (
                ((rsp - ma20) / (ma20 + 1e-9)).fillna(0.0).clip(-0.2, 0.2).values
            )

            # 5-day momentum of RSP/SPY ratio (is breadth expanding or contracting?)
            ratio = rsp / (spy + 1e-9)
            result["breadth_slope_5d"] = ratio.pct_change(5).fillna(0.0).values

        if iwm is not None and spy is not None:
            result["breadth_iwm_vs_spy_5d"] = (
                (iwm.pct_change(5) - spy.pct_change(5)).fillna(0.0).values
            )

    except Exception:
        pass

    return result
