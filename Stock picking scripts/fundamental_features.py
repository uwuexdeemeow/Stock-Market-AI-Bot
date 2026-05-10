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

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf  # still needed for yf.Ticker() fundamentals

# Multi-source data provider for price downloads
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from data_provider import download_prices as _dp_download_prices

try:
    from settings import (
        DATA_DIR,
        FUNDAMENTAL_REPORT_LAG_DAYS,
        SECTOR_MAP,
        USE_EARNINGS_DATA,
        USE_FUNDAMENTAL_ZSCORES,
    )
except Exception:
    DATA_DIR = "data"
    FUNDAMENTAL_REPORT_LAG_DAYS = 45
    SECTOR_MAP = {}
    USE_EARNINGS_DATA = False
    USE_FUNDAMENTAL_ZSCORES = False


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
# 3c. Point-in-time valuation fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def _statement_row(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    lookup = {str(idx).strip().lower(): idx for idx in frame.index}
    for alias in aliases:
        idx = lookup.get(alias.strip().lower())
        if idx is not None:
            return pd.to_numeric(frame.loc[idx], errors="coerce")
    return pd.Series(dtype=float)


def _normalise_statement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, out.columns.notna()]
    return out.sort_index(axis=1)


def _daily_known_series(values: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return pd.Series(np.nan, index=dates, dtype=float)
    known = values.copy()
    known.index = pd.DatetimeIndex(known.index) + pd.Timedelta(days=int(lag_days))
    known = known[~known.index.duplicated(keep="last")].sort_index()
    return known.reindex(dates, method="ffill")


def build_point_in_time_valuation_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Build valuation features using dated quarterly statements only.

    Yahoo does not provide full historical filing timestamps for free, so this
    uses fiscal quarter end + FUNDAMENTAL_REPORT_LAG_DAYS as a conservative
    availability date. That avoids using today's yfinance ``info`` snapshot in
    old backtest rows.
    """
    dates = pd.DatetimeIndex(df.index)
    neutral = pd.DataFrame(index=dates, data={
        "fund_pe_ttm": 0.0,
        "fund_fcf_yield_ttm": 0.0,
        "fund_has_valuation": 0.0,
        "fund_pe_sector_z": 0.0,
        "fund_fcf_yield_sector_z": 0.0,
        "fund_value_combo_z": 0.0,
    })
    if not USE_FUNDAMENTAL_ZSCORES or ticker.upper().endswith("-USD"):
        return neutral

    try:
        tk = yf.Ticker(ticker)
        inc = _normalise_statement_columns(tk.quarterly_financials)
        cf = _normalise_statement_columns(tk.quarterly_cashflow)
        bs = _normalise_statement_columns(tk.quarterly_balance_sheet)
        if inc.empty and cf.empty:
            return neutral

        eps_q = _statement_row(inc, ("Diluted EPS", "Basic EPS"))
        if eps_q.empty:
            net_income = _statement_row(inc, ("Net Income", "Net Income Common Stockholders"))
            avg_shares = _statement_row(inc, ("Diluted Average Shares", "Basic Average Shares"))
            eps_q = net_income / avg_shares.replace(0, np.nan)

        fcf_q = _statement_row(cf, ("Free Cash Flow",))
        if fcf_q.empty:
            operating_cf = _statement_row(cf, (
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ))
            capex = _statement_row(cf, ("Capital Expenditure", "Capital Expenditures"))
            fcf_q = operating_cf + capex

        shares_q = _statement_row(inc, ("Diluted Average Shares", "Basic Average Shares"))
        if shares_q.empty:
            shares_q = _statement_row(bs, ("Ordinary Shares Number", "Share Issued"))

        all_periods = sorted(set(eps_q.index.tolist()) | set(fcf_q.index.tolist()) | set(shares_q.index.tolist()))
        if not all_periods:
            return neutral
        idx = pd.DatetimeIndex(all_periods)
        eps_q = eps_q.reindex(idx).sort_index()
        fcf_q = fcf_q.reindex(idx).sort_index()
        shares_q = shares_q.reindex(idx).sort_index()

        ttm_eps = eps_q.rolling(4, min_periods=4).sum()
        ttm_fcf = fcf_q.rolling(4, min_periods=4).sum()

        eps_daily = _daily_known_series(ttm_eps, dates, FUNDAMENTAL_REPORT_LAG_DAYS)
        fcf_daily = _daily_known_series(ttm_fcf, dates, FUNDAMENTAL_REPORT_LAG_DAYS)
        shares_daily = _daily_known_series(shares_q, dates, FUNDAMENTAL_REPORT_LAG_DAYS)

        close = pd.to_numeric(df["Close"], errors="coerce").reindex(dates)
        market_cap = close * shares_daily
        pe = (close / eps_daily.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        fcf_yield = (fcf_daily / market_cap.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        has = pe.notna() | fcf_yield.notna()
        neutral["fund_pe_ttm"] = pe.clip(lower=0.0, upper=200.0).fillna(0.0)
        neutral["fund_fcf_yield_ttm"] = fcf_yield.clip(lower=-0.50, upper=0.50).fillna(0.0)
        neutral["fund_has_valuation"] = has.astype(float)
    except Exception:
        return neutral

    return neutral.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def apply_sector_fundamental_zscores(
    tickers: list[str],
    data_dir: str = DATA_DIR,
    sector_map: dict[str, str] | None = None,
) -> dict:
    """Update built parquet files with within-sector valuation z-scores."""
    sector_map = sector_map or SECTOR_MAP
    frames = []
    paths: dict[str, str] = {}
    for ticker in tickers:
        ticker = ticker.upper()
        path = f"{data_dir}/{ticker}.parquet"
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        needed = {"fund_pe_ttm", "fund_fcf_yield_ttm", "fund_has_valuation"}
        if not needed.issubset(df.columns):
            continue
        tmp = df[["fund_pe_ttm", "fund_fcf_yield_ttm", "fund_has_valuation"]].copy()
        tmp["ticker"] = ticker
        tmp["sector"] = sector_map.get(ticker, "OTHER")
        tmp["date"] = pd.to_datetime(df.index)
        frames.append(tmp.reset_index(drop=True))
        paths[ticker] = path

    if not frames:
        return {"updated": 0, "reason": "no_fundamental_columns"}

    panel = pd.concat(frames, ignore_index=True)
    valid = panel["fund_has_valuation"].astype(float) > 0
    panel["fund_pe_sector_z"] = 0.0
    panel["fund_fcf_yield_sector_z"] = 0.0

    def _zscore(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.count() < 2:
            return pd.Series(np.nan, index=s.index)
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or std < 1e-9:
            return pd.Series(0.0, index=s.index)
        return (s - float(s.mean())) / std

    for col, zcol in [
        ("fund_pe_ttm", "fund_pe_sector_z"),
        ("fund_fcf_yield_ttm", "fund_fcf_yield_sector_z"),
    ]:
        sector_z = panel.loc[valid].groupby(["date", "sector"], group_keys=False)[col].apply(_zscore)
        market_z = panel.loc[valid].groupby("date", group_keys=False)[col].apply(_zscore)
        z = sector_z.combine_first(market_z).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        panel.loc[z.index, zcol] = z.clip(-5.0, 5.0)

    panel["fund_value_combo_z"] = (
        panel["fund_fcf_yield_sector_z"] - panel["fund_pe_sector_z"]
    ).clip(-8.0, 8.0)

    latest_stats: dict[str, dict[str, float]] = {}
    latest_valid = panel[valid].sort_values("date").groupby("ticker").tail(1)
    for sector, grp in latest_valid.groupby("sector"):
        pe_std = float(grp["fund_pe_ttm"].std(ddof=0))
        fcf_std = float(grp["fund_fcf_yield_ttm"].std(ddof=0))
        if not np.isfinite(pe_std) or pe_std < 1e-9:
            pe_std = 1.0
        if not np.isfinite(fcf_std) or fcf_std < 1e-9:
            fcf_std = 1.0
        latest_stats[str(sector)] = {
            "pe_mean": float(grp["fund_pe_ttm"].mean()),
            "pe_std": pe_std,
            "fcf_yield_mean": float(grp["fund_fcf_yield_ttm"].mean()),
            "fcf_yield_std": fcf_std,
        }

    updated = 0
    zcols = ["fund_pe_sector_z", "fund_fcf_yield_sector_z", "fund_value_combo_z"]
    for ticker, path in paths.items():
        df = pd.read_parquet(path)
        sub = panel[panel["ticker"] == ticker].set_index("date")
        for col in zcols:
            df[col] = sub[col].reindex(pd.DatetimeIndex(df.index)).fillna(0.0).values
        df.to_parquet(path, index=True)
        updated += 1

    try:
        import json, os
        os.makedirs(data_dir, exist_ok=True)
        with open(f"{data_dir}/fundamental_sector_latest_stats.json", "w") as f:
            json.dump(latest_stats, f, indent=2)
    except Exception:
        pass

    return {"updated": updated, "latest_sector_stats": latest_stats}


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
        sector_ret5d/20d/1d   — the sector ETF's returns at those horizons
        sector_close          — raw sector ETF close prices (for ratio features)
        spy_ret5d             — SPY's 5-day return

    Returns columns:
        ret_vs_sector_5d    — ticker 5d return minus sector ETF 5d return
        ret_vs_sector_20d   — ticker 20d return minus sector ETF 20d return
        ret_vs_sector_1d    — ticker 1d return minus sector ETF 1d return
        sector_vs_spy_5d    — sector ETF 5d return minus SPY 5d return
                              (positive = sector leading, negative = laggard)
        sector_ratio_z20    — z-score of (stock price / sector ETF price) over
                              last 20 days.  +2 means the stock is unusually
                              strong vs its sector right now.
        sector_ratio_z60    — same but using a 60-day window (slower-moving)
        sector_rs_slope_5d  — 5-day % change in the stock/sector ratio.
                              Positive = relative strength is IMPROVING.
        sector_rs_above_ma20 — 1 if the stock/sector ratio is above its own
                              20-day moving average (trending outperformance).
    """
    result = pd.DataFrame(index=df.index, data={
        "ret_vs_sector_5d":    0.0,
        "ret_vs_sector_20d":   0.0,
        "ret_vs_sector_1d":    0.0,
        "sector_vs_spy_5d":    0.0,
        "sector_ratio_z20":    0.0,
        "sector_ratio_z60":    0.0,
        "sector_rs_slope_5d":  0.0,
        "sector_rs_above_ma20": 0.0,
    })

    # ── Simple return-difference features ──────────────────────────────────────
    if "sector_ret5d" in df.columns and "ret_5d" in df.columns:
        result["ret_vs_sector_5d"] = (df["ret_5d"] - df["sector_ret5d"]).fillna(0.0).values

    if "sector_ret20d" in df.columns and "ret_20d" in df.columns:
        result["ret_vs_sector_20d"] = (df["ret_20d"] - df["sector_ret20d"]).fillna(0.0).values

    # 1-day relative strength: is the stock beating its sector TODAY?
    if "ret_1d" in df.columns and "sector_ret1d" in df.columns:
        result["ret_vs_sector_1d"] = (df["ret_1d"] - df["sector_ret1d"]).fillna(0.0).values

    if "sector_ret5d" in df.columns and "spy_ret5d" in df.columns:
        result["sector_vs_spy_5d"] = (df["sector_ret5d"] - df["spy_ret5d"]).fillna(0.0).values

    # ── Ratio z-score features ─────────────────────────────────────────────────
    # These are the HIGH-LEVERAGE addition: instead of raw return differences,
    # we z-score the stock/sector price ratio against its own recent history.
    # This answers: "Is today's outperformance UNUSUAL compared to the last
    # N days?" — much cleaner signal than an absolute return gap.
    if "Close" in df.columns and "sector_close" in df.columns:
        sector_close = pd.Series(df["sector_close"].values, index=df.index)
        ticker_close = pd.Series(df["Close"].values, index=df.index)

        # Price ratio: stock / sector ETF.  Rising ratio = stock outperforming.
        ratio = (ticker_close / (sector_close + 1e-9)).replace([np.inf, -np.inf], np.nan)

        # Z-score at two horizons — 20-day (tactical) and 60-day (strategic)
        for window, col in [(20, "sector_ratio_z20"), (60, "sector_ratio_z60")]:
            min_p = max(window // 2, 5)
            roll_mean = ratio.rolling(window, min_periods=min_p).mean()
            roll_std  = ratio.rolling(window, min_periods=min_p).std()
            # Clip to ±4 sigma so outliers don't distort the model
            result[col] = (
                ((ratio - roll_mean) / (roll_std + 1e-9))
                .clip(-4.0, 4.0)
                .fillna(0.0)
                .values
            )

        # 5-day momentum of the ratio: is relative strength improving or fading?
        # Positive = stock strengthening vs sector (bullish for the stock)
        result["sector_rs_slope_5d"] = (
            ratio.pct_change(5).clip(-0.2, 0.2).fillna(0.0).values
        )

        # Binary: is the current stock/sector ratio above its own 20-day MA?
        # Acts like a moving-average crossover signal, but for relative strength.
        rs_ma20 = ratio.rolling(20, min_periods=5).mean()
        result["sector_rs_above_ma20"] = (
            (ratio > rs_ma20).astype(float).fillna(0.0).values
        )

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
        # Use multi-source data provider with automatic fallback
        raw = _dp_download_prices(["RSP", "SPY", "IWM"], start=start, end=end)

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
