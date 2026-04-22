"""
risk_sizing.py — Position sizing primitives.

PLAIN ENGLISH:
A great signal is worthless if you bet too much (blow up on one bad day)
or too little (great forecasts, tiny P&L). This file has three tools:

  1. vol_target_size — target a constant portfolio volatility. If a stock
     is twice as volatile as another, you buy half as much.
  2. fractional_kelly — the Kelly criterion is the math-optimal bet size
     for a repeated edge, but full Kelly is too aggressive because our
     edge estimate is noisy. We use 25–50% Kelly in practice.
  3. atr_stop_loss — place a stop at `k * ATR` below entry. ATR = Average
     True Range, a volatility measure. k=2 is common; tighter = more whipsaws.

These are pure functions; backtest.py / predict.py call them to turn a
"go long AAPL with 68% confidence" signal into a dollar (or share) size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_size(
    equity: float,
    asset_vol_annual: float,
    target_vol_annual: float = 0.15,
    max_weight: float = 0.20,
) -> float:
    """Return dollar notional so that position contributes target_vol to portfolio."""
    if asset_vol_annual <= 0 or np.isnan(asset_vol_annual):
        return 0.0
    weight = target_vol_annual / asset_vol_annual
    weight = min(weight, max_weight)
    return float(equity * weight)


def fractional_kelly(edge: float, odds: float = 1.0, fraction: float = 0.25) -> float:
    """Kelly fraction * user-chosen safety multiplier.

    `edge`  — estimated win probability minus 0.5 for a symmetric bet.
    `odds`  — payoff ratio; 1.0 means equal up/down magnitude.
    Returns a bet fraction of bankroll in [0, 1].
    """
    if edge <= 0:
        return 0.0
    kelly = edge / odds if odds > 0 else 0.0
    return float(max(0.0, min(1.0, kelly * fraction)))


def atr_stop(entry_price: float, atr: float, k: float = 2.0, side: str = "long") -> float:
    """Return the stop-loss price."""
    if side == "long":
        return float(entry_price - k * atr)
    return float(entry_price + k * atr)


def position_size_with_stop(
    equity: float,
    entry_price: float,
    atr: float,
    risk_per_trade: float = 0.01,
    k_atr: float = 2.0,
) -> int:
    """Shares so that risk (entry - stop) * shares = risk_per_trade * equity."""
    stop_dist = k_atr * atr
    if stop_dist <= 0 or entry_price <= 0:
        return 0
    dollars_at_risk = equity * risk_per_trade
    return int(dollars_at_risk // stop_dist)
