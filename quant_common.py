"""
quant_common.py — small helpers the quant scripts share.

PLAIN ENGLISH: these few pieces used to live inside backtest.py, the old
machine-learning backtester.  Quant scripts (core_satellite_alpha.py,
alpha_factor_backtest.py, factor_decay_monitor.py, ...) only needed these
helpers, but importing backtest.py also loaded the whole legacy ML layer
(train.py, xgboost, scikit-learn, matplotlib).  Moving them here lets the
quant path run without any of that.  The code is copied unchanged, so the
numbers it gives are the same as before.

What is here:
- INITIAL_CAPITAL: the starting cash every simulated portfolio begins with.
- _newey_west_tstat: a t-statistic that allows for returns being linked
  from one day to the next (an "honest" test that the average is not zero).
- _load_etf_price_frame: loads ETF closing prices from data/*.parquet
  (or downloads them), lines them up on the requested dates, and scales
  each to start at 1.0.  It never fills a missing price from the future.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from settings import DATA_DIR

# Starting cash for every simulated portfolio (dollars).
INITIAL_CAPITAL = 10_000.0
# Memory of prices already loaded, so repeat calls in one run are fast.
# The key includes each file's size and change time, so a data refresh
# automatically makes old entries unused.
_ETF_PRICE_FRAME_CACHE: dict[tuple, dict[str, pd.Series]] = {}


def _download_yfinance(*args, **kwargs) -> pd.DataFrame:
    """Import yfinance only when a download fallback is actually needed."""
    from data_provider import _quiet_yfinance_logging
    import yfinance as yf

    with _quiet_yfinance_logging():
        return yf.download(*args, **kwargs)



def _newey_west_tstat(excess: pd.Series, lag: int = 5) -> float:
    """
    Computes a Newey-West corrected t-statistic for the mean of `excess` returns.

    A plain t-stat assumes each day's return is independent, but stock returns
    have autocorrelation (today's big move can influence tomorrow's).  Newey-West
    adjusts the variance estimate to account for that, giving a more honest test.

    Returns 0.0 when there is not enough data (< 30 observations).
    """
    n = len(excess)
    if n < 30:
        return 0.0
    centered = excess.to_numpy(dtype=float) - float(excess.mean())
    # gamma0 is the basic variance of the centered series
    gamma0 = float(np.mean(centered * centered))
    lrv = gamma0
    # add weighted autocovariance terms up to `lag` lags
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)          # Bartlett (triangular) weight
        # Standard Newey-West divides the lag-k sum by n (not n-k), which keeps
        # the long-run variance estimate positive semi-definite.
        gk = float(np.sum(centered[k:] * centered[:-k]) / n)
        lrv += 2 * w * gk
    se = float(np.sqrt(max(lrv, 0.0) / n))
    if se == 0:
        return 0.0
    return float(excess.mean() / se)



def _load_etf_price_frame(
    index: pd.DatetimeIndex,
    symbols: tuple[str, ...] | list[str],
    *,
    warmup_calendar_days: int = 0,
) -> pd.DataFrame:
    """Cache prices and optionally include causal history for indicators."""
    index = pd.DatetimeIndex(index)
    if index.empty:
        return pd.DataFrame(index=index)
    clean_symbols = tuple(sorted({str(sym).upper() for sym in symbols}))
    # File versions invalidate cached prices after a local data refresh.
    versions = []
    for symbol in clean_symbols:
        path = os.path.join(DATA_DIR, f"{symbol}.parquet")
        stat = os.stat(path) if os.path.exists(path) else None
        versions.append((symbol, stat.st_mtime_ns if stat else None, stat.st_size if stat else None))
    # Preserve the existing cache lifetime for prices downloaded in long runs.
    ttl = max(1, int(os.environ.get("ETF_PRICE_CACHE_TTL_SEC", "1800")))
    warmup_calendar_days = max(0, int(warmup_calendar_days))
    cache_key = (
        clean_symbols,
        str((index.min() - pd.Timedelta(days=warmup_calendar_days)).date()),
        str(index.max().date()),
        tuple(versions),
        int(time.time() // ttl),
    )
    source = _ETF_PRICE_FRAME_CACHE.get(cache_key)
    if source is None:
        series = {}
        for symbol in clean_symbols:
            path = os.path.join(DATA_DIR, f"{symbol}.parquet")
            close = None
            if os.path.exists(path):
                local = pd.read_parquet(path)
                if "Close" in local:
                    close = local["Close"]
            if close is None:
                # Include enough earlier sessions to seed long moving averages.
                raw = _download_yfinance(
                    [symbol], start=(index.min() - pd.Timedelta(days=warmup_calendar_days + 10)).strftime("%Y-%m-%d"),
                    end=(index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True, group_by="ticker", threads=False, timeout=20,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    if symbol in raw.columns.get_level_values(0):
                        close = raw[(symbol, "Close")]
                    elif "Close" in raw.columns.get_level_values(0):
                        close = raw["Close"][symbol]
                elif "Close" in raw:
                    close = raw["Close"]
            if close is None or close.empty:
                raise ValueError(f"ETF price data unavailable for {symbol}; retry after restoring data")
            close = pd.to_numeric(close, errors="coerce").copy()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            close = close.loc[~close.index.duplicated(keep="last")].sort_index()
            # An invalid observed bar must remain visible instead of being filled.
            series[symbol] = close.loc[close.index <= index.max()]
        source = series

    alignment_index = index
    if warmup_calendar_days:
        warmup_start = index.min() - pd.Timedelta(days=warmup_calendar_days)
        historical_dates = pd.DatetimeIndex([])
        for symbol in clean_symbols:
            available = source[symbol].index
            historical_dates = historical_dates.union(
                available[(available >= warmup_start) & (available <= index.max())]
            )
        # Keep requested dates too, because a signal date can occasionally be a
        # holiday and must causally inherit the last known close.
        alignment_index = historical_dates.union(index).sort_values()

    frame = pd.DataFrame(index=alignment_index)
    for symbol in clean_symbols:
        # Reindex the full source, so interior observations are not discarded.
        close = source[symbol].reindex(alignment_index, method="ffill")
        invalid = ~np.isfinite(close) | close.le(0)
        if invalid.any():
            bad_date = close.index[invalid][0]
            raise ValueError(f"ETF price unavailable or invalid for {symbol} at {bad_date}; no future-price filling allowed")
        # Anchor to the earliest requested date even when request order is reversed.
        base_value = close.loc[alignment_index.min()]
        base = float(base_value.iloc[0]) if isinstance(base_value, pd.Series) else float(base_value)
        frame[symbol] = close / base
    # Failed downloads/alignment never enter the cache and can be retried.
    if len(_ETF_PRICE_FRAME_CACHE) >= 32 and cache_key not in _ETF_PRICE_FRAME_CACHE:
        _ETF_PRICE_FRAME_CACHE.pop(next(iter(_ETF_PRICE_FRAME_CACHE)))
    _ETF_PRICE_FRAME_CACHE[cache_key] = {symbol: close.copy() for symbol, close in source.items()}
    return frame.copy()

