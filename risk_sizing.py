"""
risk_sizing.py — Position sizing primitives.

PLAIN ENGLISH:
A great signal is worthless if you bet too much (blow up on one bad day)
or too little (great forecasts, tiny P&L). This file has three tools:

  1. vol_target_size — target a constant portfolio volatility. If a stock
     is twice as volatile as another, you buy half as much. This is the
     "base" size before the model says anything.

  2. fractional_kelly — the Kelly criterion is the math-optimal bet size
     for a repeated edge. Full Kelly is too aggressive (edge estimates are
     noisy), so we use 25% of Kelly (quarter-Kelly) as a confidence
     MULTIPLIER on top of the vol-target base.

  3. compute_position_size — the main function called by backtest.py.
     Combines vol-target (for risk equalisation across tickers) and
     fractional Kelly (to size up when the model is more confident).

These are pure functions; backtest.py calls them to turn a
"go long AAPL with 68% confidence" signal into a portfolio weight.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Realized-vol estimator — used for position sizing
# ─────────────────────────────────────────────────────────────────────────
# A plain 60-day rolling std is the textbook choice but it has a real bug:
# during a vol shock (Feb 2018 vol-mageddon, March 2020 COVID, Aug 2024
# JPY-carry unwind), the rolling window mixes the calm "before" days with
# the violent "after" days for ~60 days.  The vol estimate stays low for
# weeks while the market is actually moving 4% a day — and position sizes
# stay oversized through the entire shock.
#
# Fix: take max(EWMA, rolling) so the estimate snaps to the higher reading
# when stress hits.  EWMA with halflife=10 reacts within ~10 days; rolling
# is the conservative "long memory" floor.  In a calm regime the two are
# roughly equal; in a shock, EWMA leaps ahead and rules.
VOL_EWMA_HALFLIFE = float(os.environ.get("VOL_EWMA_HALFLIFE", "10"))
VOL_ROLLING_WINDOW = int(os.environ.get("VOL_ROLLING_WINDOW", "60"))
VOL_MIN_OBS = int(os.environ.get("VOL_MIN_OBS", "20"))


def annualized_realized_vol(
    close: "pd.Series",
    *,
    halflife: float = VOL_EWMA_HALFLIFE,
    window: int = VOL_ROLLING_WINDOW,
    min_obs: int = VOL_MIN_OBS,
    default: float = 0.20,
) -> float:
    """Stress-aware annualised realised vol from a Close-price series.

    Returns the MAX of (EWMA vol, rolling-window vol) annualised by
    sqrt(252).  In a vol shock the EWMA leaps within ~halflife days while
    the rolling window lags ~60 days; taking the max makes position sizing
    react quickly without overshooting in calm regimes.

    Args:
        close: Close-price series (chronological order).
        halflife: EWMA halflife in trading days (default 10).
        window: Rolling-window length in trading days (default 60).
        min_obs: Minimum observations required; returns `default` below this.
        default: Vol assumed when data is too thin.

    Returns:
        Annualised vol in decimal form (e.g. 0.30 = 30%).
    """
    if close is None:
        return float(default)
    try:
        prices = pd.Series(close).dropna()
    except (TypeError, ValueError):
        return float(default)
    if len(prices) < min_obs:
        return float(default)

    returns = prices.pct_change().dropna()
    if len(returns) < min_obs:
        return float(default)

    # Annualisation factor for daily returns.  Trading days in a year.
    ANN = 252 ** 0.5

    rolling_vol = float(returns.tail(window).std() * ANN)
    # EWMA std — recent days weighted exponentially more.  min_periods
    # guards against an unstable estimate on the first few rows.
    ewma_vol_series = returns.ewm(halflife=halflife, min_periods=min_obs).std()
    if ewma_vol_series.empty or pd.isna(ewma_vol_series.iloc[-1]):
        return max(0.05, min(1.5, rolling_vol)) if rolling_vol > 0 else float(default)
    ewma_vol = float(ewma_vol_series.iloc[-1] * ANN)

    # Take the more conservative (higher) reading.  Clamped to a sane
    # band — 5% floor (silly-low for any tradeable name), 150% ceiling
    # (probably bad data if vol exceeds that).
    vol = max(rolling_vol, ewma_vol)
    if not np.isfinite(vol) or vol <= 0:
        return float(default)
    return max(0.05, min(1.5, float(vol)))


def vol_target_size(
    equity: float,
    asset_vol_annual: float,
    target_vol_annual: float = 0.15,
    max_weight: float = 0.20,
) -> float:
    """Return position weight so that position contributes target_vol to portfolio.

    PLAIN ENGLISH: We want every position to add roughly the same amount of
    daily swings to the portfolio regardless of how volatile that stock is.
    If we target 15% annual vol contribution:
      - NVDA (40% vol) → weight = 15%/40% = 37.5% → capped at max_weight (20%)
      - MSFT (20% vol) → weight = 15%/20% = 75%  → capped at max_weight (20%)
      - UNH  (15% vol) → weight = 15%/15% = 100% → capped at max_weight (20%)
    So every position contributes the same risk, capped at the max allowed.

    Args:
        equity: Current portfolio value in dollars (used to compute dollar size).
        asset_vol_annual: Stock's annualised realised volatility (e.g. 0.30 = 30%).
        target_vol_annual: Desired vol contribution per position (default 15%).
        max_weight: Hard cap as a fraction of portfolio (default 20%).

    Returns:
        Dollar notional to allocate (equity × weight).
    """
    if asset_vol_annual <= 0 or np.isnan(asset_vol_annual):
        # No vol data — return 0; caller will fall back to base_pct
        return 0.0
    # weight = target_vol / asset_vol  →  higher vol → smaller weight
    weight = target_vol_annual / asset_vol_annual
    weight = min(weight, max_weight)   # never exceed the hard per-name cap
    return float(equity * weight)


def fractional_kelly(
    p_win: float,
    win_loss_ratio: float = 1.0,
    fraction: float = 0.25,
) -> float:
    """Full Kelly criterion scaled by a safety multiplier (quarter-Kelly default).

    PLAIN ENGLISH: Kelly formula tells you the mathematically optimal fraction
    of your bankroll to bet. Full Kelly is too aggressive because our model's
    win-rate estimate always has noise, so we use only 25% of it (quarter-Kelly).

    The Kelly formula for a binary bet:
        full_kelly = (p_win × b − p_lose) / b
    where b = win_loss_ratio (how much you win vs how much you lose on average).

    For our symmetric triple-barrier exits (equal profit and stop distance, b=1):
        full_kelly = p_win − p_lose = 2×p_win − 1
        At 60% win rate: full_kelly=0.20  → quarter_kelly = 0.05  (5% bonus)
        At 65% win rate: full_kelly=0.30  → quarter_kelly = 0.075 (7.5% bonus)
        At 50% win rate: full_kelly=0.0   → no extra sizing (no edge)

    Args:
        p_win: Estimated win probability in [0, 1]. Pass confidence/100 from model.
        win_loss_ratio: Avg win magnitude / avg loss magnitude. 1.0 for symmetric.
        fraction: Safety multiplier. 0.25 = quarter-Kelly (industry standard).

    Returns:
        Kelly fraction in [0, 1] — multiply by max_pct to get a position bump.
    """
    # Below 50% win rate the Kelly formula gives a negative bet → don't size up
    if p_win <= 0.5 or win_loss_ratio <= 0:
        return 0.0
    p_lose = 1.0 - p_win
    # Standard Kelly formula: (p×b − q) / b
    full_kelly = (p_win * win_loss_ratio - p_lose) / win_loss_ratio
    # Apply safety fraction and clamp to valid range
    return float(max(0.0, min(1.0, full_kelly * fraction)))


def atr_stop(entry_price: float, atr: float, k: float = 2.0, side: str = "long") -> float:
    """Return the stop-loss price given entry, ATR, and direction.

    PLAIN ENGLISH: ATR (Average True Range) measures how much a stock moves
    on a typical day. Placing a stop k × ATR away means the stop is k
    "typical daily moves" from entry, reducing false triggers from noise.
    """
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
    """Number of shares so that (entry - stop) × shares = risk_per_trade × equity.

    PLAIN ENGLISH: We decide upfront to risk at most 1% of the portfolio on
    any single trade. Given how far the stop is from entry, we calculate
    exactly how many shares we can buy so a stop-out costs us exactly 1%.
    """
    if equity <= 0 or risk_per_trade <= 0:
        return 0
    stop_dist = k_atr * atr
    if stop_dist <= 0 or entry_price <= 0:
        return 0
    dollars_at_risk = equity * risk_per_trade
    return int(dollars_at_risk // stop_dist)


def compute_position_size(
    confidence: float,
    expected_return: float,
    signal_quality: str,
    equity: float,
    asset_vol_annual: float,
    base_pct: float = 0.15,
    max_pct: float = 0.20,
    vol_target: float = 0.15,
    kelly_fraction: float = 0.25,
    use_confidence_kelly: bool = False,
    use_quality_boost: bool = False,
) -> float:
    """Vol-target base position, scaled up/down by fractional Kelly confidence.

    PLAIN ENGLISH — Three steps:

    Step 1 — Vol-target base:
        Size the position so it contributes `vol_target` annual vol to the
        portfolio. High-vol stocks (NVDA) get smaller allocations than calm
        stocks (MSFT) so every name adds the same risk regardless of ticker.
        Result is clamped to [base_pct/2, max_pct].

    Step 2 — Kelly confidence multiplier:
        Disabled by default. Use the model's confidence as a win-rate estimate
        only after bucket diagnostics show that confidence is calibrated.
        Formula: modifier = 1 + fractional_kelly(p_win)
        At confidence=50% (no edge):  modifier = 1.00 → no change
        At confidence=60% (small edge): modifier ≈ 1.05 → +5% size
        At confidence=70% (good edge):  modifier ≈ 1.10 → +10% size
        This means confident signals get modestly larger allocations.

    Step 3 — Signal quality boost:
        HIGH-quality signals (top rank in cross-sectional sort) get +5% extra
        on top of the Kelly modifier. This is a small additional multiplier.

    Step 4 — Final clamp to [base_pct/2, max_pct].

    Args:
        confidence: Model confidence in percent (50–99). 50 = coin flip, 99 = certain.
        expected_return: Predicted return (used for minimum size floor). Not heavily used.
        signal_quality: "HIGH", "MEDIUM", or "LOW" from cross-sectional ranking.
        equity: Current portfolio value in dollars.
        asset_vol_annual: Ticker's annualised realised vol (e.g. 0.30 = 30%).
        base_pct: Minimum meaningful position size (default 15%).
        max_pct: Hard cap per name (default 20%, matches MAX_SINGLE_NAME_EXPOSURE).
        vol_target: Target annual vol contribution per position (default 15%).
        kelly_fraction: Safety fraction for Kelly (0.25 = quarter-Kelly).

    Returns:
        Position size as a fraction of total equity (e.g. 0.18 = 18% of portfolio).
    """
    # ── Step 1: vol-target base ───────────────────────────────────────────────
    # Compute raw vol-target weight = vol_target / asset_vol
    # IMPORTANT: only apply a LOWER bound here.  The upper bound (max_pct) is
    # applied AFTER the Kelly multiplier, so confident signals have room to
    # push high-vol positions slightly above what the vol-target alone suggests.
    # If we clamped to max_pct now, Kelly could never increase size (already at cap).
    if equity > 0 and asset_vol_annual > 0 and not np.isnan(asset_vol_annual):
        vt = vol_target / asset_vol_annual            # e.g. 8%/40% = 20%, 8%/55% = 14.5%
    else:
        vt = base_pct                                 # fallback: use base size

    # Only apply lower floor here — upper cap comes after Kelly scaling
    vt = float(max(base_pct * 0.5, vt))

    # ── Step 2: Kelly confidence multiplier ───────────────────────────────────
    # Convert percent confidence to win probability in [0.5, 0.99]
    # confidence = 50 → no edge → modifier = 1.0 (no change)
    # confidence = 60 → 10pp edge → fractional_kelly ≈ 0.05 → modifier = 1.05
    p_win = float(np.clip(confidence / 100.0, 0.50, 0.99))
    k = (
        fractional_kelly(p_win, win_loss_ratio=1.0, fraction=kelly_fraction)
        if use_confidence_kelly
        else 0.0
    )
    modifier = 1.0 + k           # ranges 1.00 (no edge) → ~1.12 at 75% confidence

    # ── Step 3: signal quality boost ─────────────────────────────────────────
    # HIGH is a reporting label until the rank-orientation diagnostics prove
    # monotonic value, so do not size it up by default.
    quality_mult = 1.05 if use_quality_boost and signal_quality == "HIGH" else 1.0

    # ── Step 4: combine and clamp ─────────────────────────────────────────────
    size = vt * modifier * quality_mult
    return float(np.clip(size, base_pct * 0.5, max_pct))
