"""
data_provider.py — Multi-source price data with automatic fallback.

PLAIN ENGLISH: When you need stock/ETF prices, you call download_prices()
and it tries multiple data sources in order.  If yfinance is down (like now),
it automatically falls back to yahooquery, then pandas-datareader (Stooq),
so your pipeline doesn't break just because one provider has an outage.

Usage:
    from data_provider import download_prices, download_single

    # Download one ticker
    df = download_single("SPY", period="max")

    # Download multiple tickers at once
    df = download_prices(["SPY", "QQQ", "TQQQ"], start="2020-01-01", end="2026-05-01")

    # Check which provider worked
    from data_provider import last_provider
    print(f"Data came from: {last_provider}")

How it works:
    1. Tries yfinance first (most complete, most familiar)
    2. If that fails, tries yahooquery (different Yahoo API path, often works
       when yfinance is broken)
    3. If that fails, tries pandas-datareader with Stooq backend (completely
       different data source, free, no API key needed)
    4. If ALL fail, raises an error with details about what went wrong

The returned DataFrame always has the same format regardless of source:
    - Index: DatetimeIndex (dates)
    - Columns: "Open", "High", "Low", "Close", "Volume" (for single ticker)
    - Or MultiIndex columns like ("Close", "SPY"), ("Close", "QQQ") for multi

Note: Each source has slightly different data coverage and quirks:
    - yfinance: best coverage, supports "period" parameter, has adjusted prices
    - yahooquery: same Yahoo data, different API, no "period" param (uses dates)
    - Stooq (via pandas-datareader): free Polish data provider, good US stock
      coverage, no API key needed, but less reliable for obscure tickers
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

# ── Track which provider was last used (for diagnostics) ──────────────
last_provider: str = "none"

# ── Period-to-date conversion ─────────────────────────────────────────
# PLAIN ENGLISH: yfinance supports "period" like "5d", "6mo", "max".
# Other providers need actual dates.  This converts period strings to
# (start_date, end_date) tuples so all providers can use them.
_PERIOD_MAP = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 10000,
}


def _period_to_dates(period: str) -> tuple[str, str]:
    """Convert a yfinance-style period string to (start, end) date strings."""
    days = _PERIOD_MAP.get(period, 365)
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════
#  PROVIDER 1: yfinance (primary)
# ══════════════════════════════════════════════════════════════════════
def _try_yfinance(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame | None:
    """
    Try downloading from yfinance.

    PLAIN ENGLISH: This is the same library you've been using.  It talks
    to Yahoo Finance's API.  Sometimes Yahoo changes their API or has
    outages, which is why we need fallbacks.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        # Suppress yfinance's noisy ERROR logging when Yahoo is down.
        # PLAIN ENGLISH: yfinance prints scary "ERROR" and "Failed download"
        # messages to stderr even when we're about to fall back to another
        # provider.  This temporarily silences those so the user doesn't
        # think something is broken when the fallback is working fine.
        import logging
        yf_logger = logging.getLogger("yfinance")
        old_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            if period and not start:
                if len(tickers) == 1:
                    raw = yf.download(
                        tickers[0], period=period,
                        auto_adjust=auto_adjust, progress=progress, threads=False,
                    )
                else:
                    raw = yf.download(
                        tickers, period=period,
                        auto_adjust=auto_adjust, progress=progress, threads=False,
                    )
            else:
                if len(tickers) == 1:
                    raw = yf.download(
                        tickers[0], start=start, end=end,
                        auto_adjust=auto_adjust, progress=progress, threads=False,
                    )
                else:
                    raw = yf.download(
                        tickers, start=start, end=end,
                        auto_adjust=auto_adjust, progress=progress, threads=False,
                    )
        finally:
            yf_logger.setLevel(old_level)

        if raw is None or raw.empty:
            return None

        # Flatten MultiIndex columns if needed
        if isinstance(raw.columns, pd.MultiIndex):
            # For single ticker, yfinance sometimes wraps in MultiIndex
            if len(tickers) == 1:
                raw.columns = raw.columns.get_level_values(0)

        return raw

    except Exception as e:
        warnings.warn(f"yfinance failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  PROVIDER 2: yahooquery (same Yahoo data, different API path)
# ══════════════════════════════════════════════════════════════════════
def _try_yahooquery(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame | None:
    """
    Try downloading from yahooquery.

    PLAIN ENGLISH: yahooquery talks to Yahoo Finance through a different
    API endpoint than yfinance.  When yfinance breaks because Yahoo changed
    their main API, yahooquery often still works because it uses a
    different path to the same data.
    """
    try:
        from yahooquery import Ticker
    except ImportError:
        return None

    try:
        # yahooquery needs actual dates, not periods
        if period and not start:
            start, end = _period_to_dates(period)

        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        t = Ticker(tickers)
        raw = t.history(start=start, end=end, adj_ohlc=auto_adjust)

        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            return None

        # yahooquery returns data in a different format — normalize it
        if isinstance(raw, dict):
            # Error case — yahooquery returns dict when it fails
            return None

        if isinstance(raw, pd.DataFrame):
            # Reset multi-index (symbol, date) → regular DataFrame
            if isinstance(raw.index, pd.MultiIndex):
                if len(tickers) == 1:
                    # Single ticker: drop the symbol level, keep date
                    raw = raw.reset_index(level="symbol", drop=True)
                    # Rename columns to match yfinance format
                    col_map = {
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume",
                        "adjclose": "Adj Close",
                    }
                    raw.columns = [col_map.get(c, c) for c in raw.columns]
                else:
                    # Multiple tickers: reshape to match yfinance multi-ticker format
                    raw = raw.reset_index()
                    raw = raw.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume", "date": "Date",
                    })
                    # Pivot to get MultiIndex columns like yfinance
                    raw["Date"] = pd.to_datetime(raw["Date"])
                    result = pd.DataFrame(index=sorted(raw["Date"].unique()))
                    result.index.name = "Date"
                    for sym in tickers:
                        sym_data = raw[raw["symbol"] == sym].set_index("Date")
                        for col in ["Open", "High", "Low", "Close", "Volume"]:
                            if col in sym_data.columns:
                                result[(col, sym)] = sym_data[col]
                    result.columns = pd.MultiIndex.from_tuples(result.columns)
                    raw = result

            raw.index = pd.to_datetime(raw.index)
            if raw.empty:
                return None
            return raw

    except Exception as e:
        warnings.warn(f"yahooquery failed: {e}")
        return None

    return None


# ══════════════════════════════════════════════════════════════════════
#  PROVIDER 3: pandas-datareader (Stooq — free, no API key)
# ══════════════════════════════════════════════════════════════════════
def _try_stooq(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
) -> pd.DataFrame | None:
    """
    Try downloading from Stooq via pandas-datareader.

    PLAIN ENGLISH: Stooq is a Polish financial data website that provides
    free historical price data for US stocks and ETFs.  No API key needed.
    It's a completely different data source from Yahoo, so if Yahoo is
    entirely down, Stooq might still work.

    Limitations:
    - Doesn't support all tickers (mainly major US stocks/ETFs)
    - Data might be slightly delayed
    - No "Adj Close" column (only raw OHLCV)
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        return None

    try:
        if period and not start:
            start, end = _period_to_dates(period)

        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        if len(tickers) == 1:
            raw = pdr.DataReader(tickers[0], "stooq", start=start, end=end)
            if raw is not None and not raw.empty:
                # Stooq returns data in reverse chronological order — fix that
                raw = raw.sort_index()
                return raw
        else:
            # Download each ticker separately and combine
            frames = {}
            for sym in tickers:
                try:
                    df = pdr.DataReader(sym, "stooq", start=start, end=end)
                    if df is not None and not df.empty:
                        df = df.sort_index()
                        frames[sym] = df
                except Exception:
                    continue

            if not frames:
                return None

            # Build MultiIndex DataFrame matching yfinance format
            all_dates = sorted(set().union(*(f.index for f in frames.values())))
            result = pd.DataFrame(index=all_dates)
            result.index.name = "Date"
            for sym, df in frames.items():
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        result[(col, sym)] = df[col].reindex(all_dates)
            result.columns = pd.MultiIndex.from_tuples(result.columns)
            return result

    except Exception as e:
        warnings.warn(f"Stooq/pandas-datareader failed: {e}")
        return None

    return None


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API — use these functions in your scripts
# ══════════════════════════════════════════════════════════════════════

def download_prices(
    tickers: list[str] | str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download price data with automatic fallback across multiple providers.

    PLAIN ENGLISH: Give it a ticker (or list of tickers) and it will try
    yfinance first, then yahooquery, then Stooq.  If all three fail, it
    raises an error.  The returned data always looks the same regardless
    of which source provided it.

    Args:
        tickers: One ticker string or list of tickers (e.g. "SPY" or ["SPY", "QQQ"])
        start: Start date string like "2020-01-01" (optional if period given)
        end: End date string like "2026-05-01" (optional, defaults to today)
        period: yfinance-style period like "5d", "6mo", "max" (optional)
        auto_adjust: Whether to use adjusted prices (default: True)
        progress: Show download progress bar (default: False)

    Returns:
        DataFrame with OHLCV columns (or MultiIndex for multiple tickers)

    Raises:
        RuntimeError: If all providers fail
    """
    global last_provider

    if isinstance(tickers, str):
        tickers = [tickers]

    errors: list[str] = []

    # ── Try Provider 1: yfinance ──────────────────────────────────────
    result = _try_yfinance(
        tickers, start=start, end=end, period=period,
        auto_adjust=auto_adjust, progress=progress,
    )
    if result is not None and not result.empty:
        last_provider = "yfinance"
        return result
    errors.append("yfinance: returned empty or failed")

    # ── Try Provider 2: yahooquery ────────────────────────────────────
    result = _try_yahooquery(
        tickers, start=start, end=end, period=period,
        auto_adjust=auto_adjust,
    )
    if result is not None and not result.empty:
        # Only print the fallback notice once per session, not for every ticker
        if last_provider != "yahooquery":
            print(f"  ℹ Using yahooquery fallback (yfinance unavailable)")
        last_provider = "yahooquery"
        return result
    errors.append("yahooquery: returned empty or failed")

    # ── Try Provider 3: Stooq ─────────────────────────────────────────
    result = _try_stooq(
        tickers, start=start, end=end, period=period,
    )
    if result is not None and not result.empty:
        if last_provider != "stooq":
            print(f"  ℹ Using Stooq fallback (yfinance and yahooquery unavailable)")
        last_provider = "stooq"
        return result
    errors.append("stooq: returned empty or failed")

    # ── All providers failed ──────────────────────────────────────────
    raise RuntimeError(
        f"All data providers failed for {tickers}:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def download_single(
    ticker: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download price data for a single ticker with automatic fallback.

    PLAIN ENGLISH: Same as download_prices() but for one ticker.
    Returns a simple DataFrame (not MultiIndex) with columns:
    Open, High, Low, Close, Volume.

    This is a convenience wrapper — use it when you want clean single-ticker
    data without dealing with MultiIndex columns.
    """
    return download_prices(
        [ticker], start=start, end=end, period=period,
        auto_adjust=auto_adjust, progress=progress,
    )


def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a yfinance-style MultiIndex DataFrame to simple columns.

    PLAIN ENGLISH: yfinance sometimes returns columns like ("Close", "AAPL")
    instead of just "Close".  This flattens them back to simple names.
    Handles both single-ticker MultiIndex and regular DataFrames.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If all second-level values are the same ticker, just use first level
        level1 = df.columns.get_level_values(1).unique()
        if len(level1) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            # Multiple tickers — join levels with underscore
            df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    return df


# ══════════════════════════════════════════════════════════════════════
#  SELF-TEST — run this file directly to test all providers
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing data providers...\n")

    # Test single ticker
    print("1. Single ticker (SPY, last 5 days):")
    try:
        df = download_single("SPY", period="5d")
        print(f"   Provider: {last_provider}")
        print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
        print(f"   Latest: {df.index[-1].date()} Close={df['Close'].iloc[-1]:.2f}")
        print("   ✓ OK\n")
    except Exception as e:
        print(f"   ✗ FAILED: {e}\n")

    # Test multiple tickers
    print("2. Multiple tickers (SPY, QQQ, TQQQ, last 5 days):")
    try:
        df = download_prices(["SPY", "QQQ", "TQQQ"], period="5d")
        print(f"   Provider: {last_provider}")
        print(f"   Rows: {len(df)}")
        print(f"   ✓ OK\n")
    except Exception as e:
        print(f"   ✗ FAILED: {e}\n")

    # Test with date range
    print("3. Date range (SPY, 2024-01-01 to 2024-02-01):")
    try:
        df = download_single("SPY", start="2024-01-01", end="2024-02-01")
        print(f"   Provider: {last_provider}")
        print(f"   Rows: {len(df)}")
        print(f"   ✓ OK\n")
    except Exception as e:
        print(f"   ✗ FAILED: {e}\n")

    print(f"Last provider used: {last_provider}")
