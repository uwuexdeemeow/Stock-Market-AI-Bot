"""
execution_model.py — Realistic cost/slippage model used by backtest & paper trade.

PLAIN ENGLISH:
Your backtest P&L is a lie unless you subtract the cost of trading:
  * Commission — broker fee per share.
  * Spread — the gap between bid and ask; half of it is paid every fill.
  * Slippage — price moves between your decision and the fill. Worse when
    your order is big relative to volume.
  * Capacity — once your order exceeds ~5–10% of daily volume, it starts
    to move the market against you. Back-of-envelope: sqrt-law impact.

Call `realistic_fill_price(...)` from the backtest loop and compare to the
naive close-price fill to see how much of the paper edge is execution fog.
"""

from __future__ import annotations

import math

import pandas as pd

from settings import COMMISSION_PER_SHARE, SLIPPAGE_BASE_PCT


def _finite_float(value: float) -> float | None:
    """Convert a value to a normal finite float, or None when it is unusable."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def bid_ask_spread_cost(
    mid: float,
    spread_bps: float = 5.0,
    side: str = "buy",
    asset_vol: float | None = None,
) -> float:
    """Half-spread paid on every fill.

    When asset_vol is provided the spread is scaled by realized volatility:
    a stock with 40% annual vol pays ~2x the spread of a 20%-vol stock.
    Capped at 3x (very volatile/illiquid) and floored at 0.5x (blue-chip).
    """
    if asset_vol is not None and asset_vol > 0:
        vol_mult = min(max(asset_vol / 0.20, 0.5), 3.0)
        spread_bps = spread_bps * vol_mult
    half = mid * (spread_bps / 10_000) / 2
    return mid + half if side == "buy" else mid - half


def sqrt_impact_bps(order_shares: float, adv_shares: float, k: float = 10.0) -> float:
    """Almgren-style square-root market impact in basis points.
    adv_shares = average daily volume. k=10 is a conservative retail default."""
    order_val = _finite_float(order_shares)
    adv_val = _finite_float(adv_shares)
    k_val = _finite_float(k)
    if order_val is None or adv_val is None or k_val is None:
        return 0.0
    if adv_val <= 0 or order_val <= 0 or k_val < 0:
        return 0.0
    return float(k_val * math.sqrt(order_val / adv_val) * 100)


def realistic_fill_price(
    mid: float,
    order_shares: float,
    adv_shares: float,
    side: str = "buy",
    spread_bps: float = 5.0,
    base_slippage_pct: float = SLIPPAGE_BASE_PCT,
    asset_vol: float | None = None,
) -> float:
    """Fill price with vol-scaled spread + sqrt-impact + baseline slippage."""
    px = bid_ask_spread_cost(mid, spread_bps, side, asset_vol=asset_vol)
    impact_bps = sqrt_impact_bps(order_shares, adv_shares)
    slip = (base_slippage_pct + impact_bps / 10_000)
    return px * (1 + slip) if side == "buy" else px * (1 - slip)


def commission(shares: int) -> float:
    return abs(shares) * COMMISSION_PER_SHARE


def capacity_warning(order_shares: float, adv_shares: float, threshold: float = 0.05) -> bool:
    """True if order exceeds `threshold` of ADV — size is likely unfillable cleanly."""
    return adv_shares > 0 and (order_shares / adv_shares) > threshold
