"""
labels.py — Alternative target labels for the model.

PLAIN ENGLISH:
A "label" is the answer the model tries to predict. Today the project uses
a fixed 5-day forward return. That's noisy — a 1% move in calm markets is
meaningful; the same 1% in a 40-VIX regime is rounding error. This module
offers two better choices:

  1. vol_normalized_return: divides raw return by realized volatility so
     "big" means big relative to recent noise.
  2. triple_barrier: the López de Prado label. Walks forward from bar t
     until price hits an UP barrier (profit target), a DOWN barrier
     (stop), or a TIME barrier (max hold days). Label = +1 / -1 / 0.

Train with each label family; whichever yields the best out-of-sample
Sharpe wins. Backtest/train code can swap labels by importing from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from settings import (
    DIRECTION_LABEL_THRESHOLD,
    EXCESS_RETURN_MIN_PCT,
    PREDICTION_TARGET,
    RETURN_BINS,
    RETURN_HORIZON_DAYS,
    VOL_ADJUSTED_SHARPE_THRESHOLD,
)


def make_forward_return_target(close: pd.Series, horizon: int = RETURN_HORIZON_DAYS) -> pd.Series:
    return close.pct_change(horizon).shift(-horizon)


def make_spy_forward_return(
    df: pd.DataFrame,
    horizon: int = RETURN_HORIZON_DAYS,
    index: pd.Index | None = None,
) -> pd.Series:
    idx = df.index if index is None else index
    bench_col = f"spy_ret{horizon}d"
    if bench_col not in df.columns:
        return pd.Series(0.0, index=idx)
    return df[bench_col].shift(-horizon).reindex(idx)


def make_valid_label_mask(df: pd.DataFrame, horizon: int = RETURN_HORIZON_DAYS) -> pd.Series:
    stock_fwd = make_forward_return_target(df["Close"], horizon=horizon)
    spy_fwd = make_spy_forward_return(df, horizon=horizon)
    return stock_fwd.notna() & spy_fwd.notna()


def make_direction_target(
    df: pd.DataFrame | pd.Series,
    *,
    prediction_target: str = PREDICTION_TARGET,
    horizon: int = RETURN_HORIZON_DAYS,
    stock_fwd: pd.Series | None = None,
    spy_fwd: pd.Series | None = None,
    direction_threshold: float = DIRECTION_LABEL_THRESHOLD,
    excess_return_min_pct: float = EXCESS_RETURN_MIN_PCT,
    vol_adjusted_sharpe_threshold: float = VOL_ADJUSTED_SHARPE_THRESHOLD,
) -> np.ndarray:
    """
    Build the shared binary direction label.

    Forward returns are intentionally not filled. Callers that train/evaluate
    models should filter with make_valid_label_mask() so unknown future returns
    are excluded instead of being converted to fake flat-return examples.
    """
    if isinstance(df, pd.Series):
        close = df
        if spy_fwd is None:
            spy_fwd = pd.Series(0.0, index=close.index)
        hvol = pd.Series(0.20, index=close.index)
    else:
        close = df["Close"]
        if spy_fwd is None:
            spy_fwd = make_spy_forward_return(df, horizon=horizon)
        hvol = df["hvol_20d"] if "hvol_20d" in df.columns else pd.Series(0.20, index=close.index)

    if stock_fwd is None:
        stock_fwd = make_forward_return_target(close, horizon=horizon)

    if prediction_target == "triple_barrier" and not isinstance(df, pd.Series):
        labels = triple_barrier(
            close,
            high=df["High"] if "High" in df.columns else None,
            low=df["Low"] if "Low" in df.columns else None,
            max_hold=horizon,
        )
        return (labels.values > 0).astype(np.int64)

    if prediction_target == "excess_return":
        excess = stock_fwd - spy_fwd
        return (excess.values > excess_return_min_pct).astype(np.int64)

    if prediction_target == "vol_adjusted":
        excess = stock_fwd - spy_fwd
        vol_scaled = (hvol * np.sqrt(horizon)).clip(0.01, 1.0)
        return ((excess / vol_scaled).values > vol_adjusted_sharpe_threshold).astype(np.int64)

    return (stock_fwd.values > direction_threshold).astype(np.int64)


def make_return_bucket_target(
    close: pd.Series,
    *,
    return_bins: list[float] = RETURN_BINS,
    n_return_bins: int | None = None,
    horizon: int = RETURN_HORIZON_DAYS,
    stock_fwd: pd.Series | None = None,
) -> np.ndarray:
    fwd = stock_fwd if stock_fwd is not None else make_forward_return_target(close, horizon=horizon)
    n_bins = len(return_bins) + 1 if n_return_bins is None else n_return_bins
    buckets = np.digitize(fwd.values, return_bins)
    return np.clip(buckets, 0, n_bins - 1).astype(np.int64)


def forward_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Simple H-day forward log return (baseline label)."""
    return np.log(close.shift(-horizon) / close)


def vol_normalized_return(close: pd.Series, horizon: int = 5, vol_window: int = 20) -> pd.Series:
    """Forward return divided by trailing realized vol (annualized-agnostic)."""
    ret = forward_return(close, horizon)
    vol = close.pct_change().rolling(vol_window).std()
    return ret / (vol * np.sqrt(horizon)).replace(0, np.nan)


def triple_barrier(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_hold: int = 10,
    atr_window: int = 14,
) -> pd.Series:
    """Triple-barrier labels per López de Prado, Ch. 3.

    Up barrier = entry + pt_mult * ATR ; Down barrier = entry - sl_mult * ATR.
    Returns +1 if up barrier hit first, -1 if down, 0 on time-out.
    """
    if high is None or low is None:
        high = close
        low = close
    tr = pd.concat(
        [(high - low),
         (high - close.shift()).abs(),
         (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_window).mean()

    labels = pd.Series(index=close.index, dtype="float64")
    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    a = atr.to_numpy()
    n = len(close)

    for i in range(n - 1):
        if np.isnan(a[i]):
            continue
        up = c[i] + pt_mult * a[i]
        dn = c[i] - sl_mult * a[i]
        end = min(i + max_hold, n - 1)
        outcome = 0
        for j in range(i + 1, end + 1):
            if h[j] >= up:
                outcome = 1
                break
            if l[j] <= dn:
                outcome = -1
                break
        labels.iloc[i] = outcome
    return labels
