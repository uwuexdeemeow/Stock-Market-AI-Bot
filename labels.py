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
