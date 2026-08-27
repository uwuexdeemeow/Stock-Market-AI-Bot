"""
core_satellite_tqqq.py — TQQQ-enhanced core-satellite research module.

Runs the SAME strategy logic as core_satellite_alpha.py (factor overlay,
regime switching, sticky scoring, etc.) but replaces part of the QQQ core
allocation with TQQQ during risk_on periods.

Standalone live signal generation from this file is retired. TQQQ remains a
research/backtest component and a tunable `tqqq_weight` inside the unified
core-alpha live configuration.

Usage:
    python3 core_satellite_tqqq.py --backtest --tqqq-weight 0.30  # research backtest
    python3 core_satellite_tqqq.py --grid                        # research grid search
    python3 core_satellite_alpha.py                              # unified live signal

The strategy:
- During risk_on: replace some QQQ in the core with TQQQ (3x leveraged QQQ)
- During neutral/risk_off: NO TQQQ — use defensive allocation only
- Factor overlay (top-3 stocks by walkforward score) stays the same
- Regime switching (SPY 200MA + QQQ 100MA + vol) stays the same
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import the existing core-satellite building blocks ──────────────────────
# We reuse everything from the current strategy, only changing the core
# ETF allocation during risk_on periods.
from alpha_factor_backtest import (
    HORIZON_DAYS,
    MAX_GROSS_EXPOSURE,
    benchmark_equity,
    compare_to_benchmarks,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
    attach_scores,
    portfolio_stats,
    subperiod_metrics,
)
from backtest import INITIAL_CAPITAL
from core_satellite_alpha import (
    REGIME_PRESETS,
    SCORE_SOURCES,
    SHAPES,
    WEIGHTING_MODES,
    MAX_PER_SECTOR_OPTIONS,
    EARNINGS_BLACKOUT_DAY_OPTIONS,
    HOLDING_DAY_OPTIONS,
    OVERLAY_GROSS_OPTIONS,
    _cached_etf_prices,
    _ensure_robust_score_columns,
    _load_regime_indicators,
    _panel_day_map,
    _resolve_allocation,
    _score_col_for_regime,
    _exit_floor_for_regime,
    _select_sticky_holdings,
    _sticky_overlay_weights,
    _compute_regime_strength,
    _blended_score_col,
    _regime_preset_with_overlay_gross,
    _apply_concentration_overlay_target,
    check_factor_freshness,
    MAX_SINGLE_NAME_WEIGHT,
    COST_STRESS_MULTIPLIERS,
    DRAWDOWN_CIRCUIT_BREAKER_OPTIONS,
    VOL_TARGET_OPTIONS,
    VOL_TARGET_LOOKBACK,
    VOL_TARGET_MAX_SCALE,
    VOL_TARGET_MIN_SCALE,
    SENTIMENT_VETO_ENABLED,
    _apply_sentiment_veto,
    _load_feature_quality_filter,
)
from robustness_scoring import add_cost_stress_approval_columns, robustness_score_components
from settings import DATA_DIR, SIGNAL_DIR, SLIPPAGE_BASE_PCT

TQQQ_LIVE_DISABLED_MESSAGE = (
    "Standalone core_satellite_tqqq.py live signal generation is retired. "
    "Use `python3 core_satellite_alpha.py` for live signals; TQQQ is controlled "
    "by `tqqq_weight` inside the unified core-alpha nested configuration."
)

# Paper-trading safety: cap gross exposure at 1.0x for broker submissions
# (no leverage on paper accounts unless explicitly allowed)
PAPER_MAX_GROSS_EXPOSURE = 1.00

# ── TQQQ fast circuit breaker ───────────────────────────────────────────────
# PLAIN ENGLISH: TQQQ is 3x leveraged, so it can fall 70% in a crash before the
# 100-day-MA regime detector fires (roughly 20 trading days of lag).  This fast
# circuit breaker checks whether TQQQ has already dropped more than the threshold
# from its recent high.  If so, the signal immediately sets TQQQ weight to 0 —
# no waiting for the regime to flip.
#
# TQQQ_FAST_DD_THRESHOLD: if TQQQ is down more than this from its N-day high,
#   force TQQQ weight to 0 in the signal.  Default -15% (e.g. -0.15).
# TQQQ_FAST_DD_LOOKBACK: how many recent trading days to look back for the high.
import os as _os
TQQQ_FAST_DD_THRESHOLD = float(_os.environ.get("TQQQ_FAST_DD_THRESHOLD", "-0.15"))
TQQQ_FAST_DD_LOOKBACK = int(_os.environ.get("TQQQ_FAST_DD_LOOKBACK", "5"))


def _tqqq_drawdown_ok(prices: "pd.Series", threshold: float = TQQQ_FAST_DD_THRESHOLD, lookback: int = TQQQ_FAST_DD_LOOKBACK) -> tuple[bool, float]:
    """
    Return (is_ok, drawdown_from_high) for TQQQ's recent price action.

    PLAIN ENGLISH: Looks at the last `lookback` closing prices for TQQQ.
    Finds the highest price in that window.  If today's price is more than
    `threshold` below that high (e.g. -15%), returns is_ok=False — meaning
    the fast circuit breaker should fire and TQQQ weight should be set to 0.

    Returns (True, drawdown) if within threshold (safe to hold TQQQ).
    Returns (False, drawdown) if circuit breaker should fire.
    Drawdown is a negative number, e.g. -0.18 means 18% below recent high.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return True, 0.0
    window = prices.iloc[-lookback:] if len(prices) >= lookback else prices
    recent_high = float(window.max())
    current = float(prices.iloc[-1])
    if recent_high <= 0:
        return True, 0.0
    drawdown = (current - recent_high) / recent_high
    return drawdown > threshold, drawdown


# ── TQQQ-specific presets ───────────────────────────────────────────────────
# These mirror the winning config (qqq_trend_switch_overlay70_core55) but
# substitute TQQQ for part of the QQQ allocation during risk_on.

def build_tqqq_presets(tqqq_weight: float = 0.20) -> dict:
    """
    Build regime presets with TQQQ in the risk_on core allocation.

    PLAIN ENGLISH: During confirmed uptrends (both SPY and QQQ above their
    moving averages, volatility low), we hold some TQQQ (3x leveraged QQQ)
    instead of all QQQ.  During uncertain or bearish periods, we hold ZERO
    TQQQ — it's too volatile for risk_off markets.

    The tqqq_weight controls what fraction of the risk_on core is TQQQ.
    The rest stays QQQ.  Example: tqqq_weight=0.20 means:
      risk_on core = 80% QQQ + 20% TQQQ (by weight within core allocation)
    """
    # How much QQQ remains in risk_on after TQQQ takes its share
    qqq_risk_on = 1.0 - tqqq_weight

    return {
        "tqqq_enhanced": {
            "ma_window": 100,
            "high_vol": 0.30,
            "risk_on": {
                # Core is split between QQQ and TQQQ.  SPY stays at 0.
                "core_weights": {"SPY": 0.00, "QQQ": qqq_risk_on, "TQQQ": tqqq_weight},
                "core_gross": 0.55,
                "overlay_gross": 0.70,
            },
            "neutral": {
                # No TQQQ during neutral — too risky when trend is uncertain
                "core_weights": {"SPY": 0.25, "QQQ": 0.75, "TQQQ": 0.00},
                "core_gross": 0.55,
                "overlay_gross": 0.70,
            },
            "risk_off": {
                # Definitely no TQQQ during risk_off
                "core_weights": {"SPY": 0.60, "QQQ": 0.40, "TQQQ": 0.00},
                "core_gross": 0.65,
                "overlay_gross": 0.35,
            },
        },
        # --- CASH BUFFER VARIANT ---
        # PLAIN ENGLISH: Same TQQQ allocation in risk_on, but holds 20% cash
        # in risk_off and ~10% cash in neutral.  Reduces total market exposure
        # during stressed periods instead of just reshuffling ETFs.
        "tqqq_enhanced_cashbuffer": {
            "ma_window": 100,
            "high_vol": 0.30,
            "risk_on": {
                "core_weights": {"SPY": 0.00, "QQQ": qqq_risk_on, "TQQQ": tqqq_weight},
                "core_gross": 0.55,
                "overlay_gross": 0.70,
            },
            "neutral": {
                "core_weights": {"SPY": 0.25, "QQQ": 0.75, "TQQQ": 0.00},
                "core_gross": 0.50,
                "overlay_gross": 0.60,
            },
            "risk_off": {
                "core_weights": {"SPY": 0.60, "QQQ": 0.40, "TQQQ": 0.00},
                "core_gross": 0.55,
                "overlay_gross": 0.25,
            },
        },
        # --- ADAPTIVE VOL + CASH BUFFER ---
        # PLAIN ENGLISH: Combines cash buffer with percentile-based vol detection,
        # blended scores (no hard flip), and early rebalance on regime change.
        "tqqq_enhanced_adaptive": {
            "ma_window": 100,
            "high_vol": 0.30,
            "high_vol_mode": "percentile",
            "score_blend": True,
            "early_rebalance_on_regime_change": True,
            "risk_on": {
                "core_weights": {"SPY": 0.00, "QQQ": qqq_risk_on, "TQQQ": tqqq_weight},
                "core_gross": 0.55,
                "overlay_gross": 0.70,
            },
            "neutral": {
                "core_weights": {"SPY": 0.25, "QQQ": 0.75, "TQQQ": 0.00},
                "core_gross": 0.50,
                "overlay_gross": 0.60,
            },
            "risk_off": {
                "core_weights": {"SPY": 0.60, "QQQ": 0.40, "TQQQ": 0.00},
                "core_gross": 0.55,
                "overlay_gross": 0.25,
            },
        },
    }


# Module-level cache so we don't re-download ETF prices for every grid config
_TQQQ_ETF_PRICE_CACHE: dict[tuple, pd.DataFrame] = {}


def _load_etf_prices_with_tqqq(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load ETF prices for SPY, QQQ, and TQQQ with automatic fallback.

    PLAIN ENGLISH: Uses the multi-source data_provider to download prices.
    If yfinance is down, it automatically tries yahooquery, then Stooq.
    Falls back gracefully if TQQQ data is unavailable for early dates.
    """
    from data_provider import download_single, flatten_yf

    # Cache key: just the date range (start, end) — all grid configs share the same dates
    cache_key = (price_index.min(), price_index.max())
    if cache_key in _TQQQ_ETF_PRICE_CACHE:
        # Reindex cached prices to this specific price_index
        cached = _TQQQ_ETF_PRICE_CACHE[cache_key]
        return cached.reindex(price_index, method="ffill").ffill().bfill()

    tickers = ["SPY", "QQQ", "TQQQ"]
    start = pd.Timestamp(price_index.min()) - pd.tseries.offsets.BDay(5)
    end = pd.Timestamp(price_index.max()) + pd.tseries.offsets.BDay(5)

    # Prefer local parquet files produced by the research refresh. Nested
    # walk-forward should not depend on live provider availability when the
    # required ETF history is already on disk.
    all_dates = pd.bdate_range(start, end)
    prices = pd.DataFrame(index=all_dates)
    for sym in tickers:
        local_path = Path(DATA_DIR) / f"{sym}.parquet"
        if local_path.exists():
            try:
                local = pd.read_parquet(local_path)
                local.index = pd.DatetimeIndex(local.index)
                if "Close" in local.columns and not local["Close"].dropna().empty:
                    prices[sym] = local["Close"].reindex(all_dates).ffill().bfill()
                    continue
            except Exception as e:
                print(f"  WARNING: Could not load local {sym} data from {local_path}: {e}")

        try:
            raw = download_single(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            raw = flatten_yf(raw)
            close = raw["Close"]
            prices[sym] = close.reindex(all_dates, method="ffill")
        except Exception as e:
            print(f"  WARNING: Could not download {sym}: {e}")
            prices[sym] = np.nan

    # Forward-fill any gaps and cache
    prices = prices.ffill().bfill()
    _TQQQ_ETF_PRICE_CACHE[cache_key] = prices

    # Return subset for this specific price_index
    return prices.reindex(price_index, method="ffill").ffill().bfill()


def run_tqqq_backtest(
    panel: pd.DataFrame,
    tqqq_weight: float = 0.20,
    cost_stress: float = 2.0,
    holding_days: int = 10,
    preset_name: str = "tqqq_enhanced",
    score_source: str = "regime_adaptive",
    shape: str = "top5",
    weighting: str = "sticky_score",
    overlay_gross: float | None = None,
    regime_ma_window: int | None = None,
    regime_high_vol: float | None = None,
    high_vol_mode: str | None = None,
    max_per_sector: int = 2,
    earnings_blackout_days: int = 0,
    drawdown_circuit_breaker: float = 0.0,
    concentration_overlay_mode: str = "off",
    concentration_overlay_low_gross: float = 0.30,
    concentration_overlay_high_gross: float = 0.70,
    concentration_overlay_threshold: float = 0.05,
    concentration_overlay_span: float = 0.05,
    quiet: bool = False,
) -> dict:
    """
    Run the TQQQ-enhanced core-satellite backtest.

    PLAIN ENGLISH: This is the exact same strategy as the winning
    core-satellite config, but during risk_on periods we hold TQQQ
    (3x leveraged QQQ) for part of the core allocation.  Everything
    else — regime detection, factor overlay, sticky scoring — is identical.

    Parameters now match the alpha grid search dimensions so TQQQ can
    be tested across the same combinations (score sources, shapes,
    sector caps, holding days, blackout periods).
    """
    # ── Build config matching the winning core-satellite config ──────────
    # PLAIN ENGLISH: preset_name selects which variant to use:
    # "tqqq_enhanced" = original, "tqqq_enhanced_cashbuffer" = with cash buffer,
    # "tqqq_enhanced_adaptive" = cash buffer + percentile vol detection.
    presets = build_tqqq_presets(tqqq_weight)
    preset = presets[preset_name]
    if overlay_gross is not None:
        preset = _regime_preset_with_overlay_gross(preset, float(overlay_gross))
    if regime_ma_window is not None:
        preset["ma_window"] = int(regime_ma_window)
    if regime_high_vol is not None:
        preset["high_vol"] = float(regime_high_vol)
    if high_vol_mode is not None:
        preset["high_vol_mode"] = str(high_vol_mode)

    config = {
        "core_preset": preset_name,
        "regime_mode": preset_name,
        "regime_ma_window": int(preset["ma_window"]),
        "regime_high_vol": float(preset["high_vol"]),
        "high_vol_mode": str(preset.get("high_vol_mode", "fixed")),
        "core_weights": dict(preset["risk_on"]["core_weights"]),
        # These were hardcoded before — now they're grid-searchable parameters
        "score_source": score_source,
        "shape": shape,
        "weighting": weighting,
        "exit_rank_floor": 0.80,
        "adaptive_exit_mode": "fixed",
        "max_per_sector": max_per_sector,
        "earnings_blackout_days": earnings_blackout_days,
        "core_gross": float(preset["risk_on"]["core_gross"]),
        "overlay_gross": float(preset["risk_on"]["overlay_gross"]),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "max_single_name_weight": MAX_SINGLE_NAME_WEIGHT,
        # Score blending and early rebalance come from the preset (adaptive variant)
        "score_blend": bool(preset.get("score_blend", False)),
        "early_rebalance_on_regime_change": bool(preset.get("early_rebalance_on_regime_change", False)),
        # Drawdown circuit breaker: 0.0 = disabled, e.g. 0.20 = go to cash if down 20%
        "drawdown_circuit_breaker": drawdown_circuit_breaker,
        "cost_stress": cost_stress,
        "holding_days": holding_days,
        "concentration_overlay_mode": concentration_overlay_mode,
        "concentration_overlay_low_gross": concentration_overlay_low_gross,
        "concentration_overlay_high_gross": concentration_overlay_high_gross,
        "concentration_overlay_threshold": concentration_overlay_threshold,
        "concentration_overlay_span": concentration_overlay_span,
    }

    # ── Resolve dates and load prices ───────────────────────────────────
    return_col = "forward_return" if holding_days == HORIZON_DAYS else f"forward_return_{holding_days}d"
    if return_col not in panel.columns:
        if not quiet:
            print(f"  WARNING: {return_col} not in panel, falling back to forward_return")
        return_col = "forward_return"

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rebalance_dates = list(dates[::holding_days])
    exit_dates = pd.DatetimeIndex([
        pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days)
        for dt in rebalance_dates
    ])
    # Build price index from both regular and potential early-rebalance dates
    price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(exit_dates) | set(dates)))

    # Load ETF prices including TQQQ
    if not quiet:
        print("  Loading ETF prices (SPY, QQQ, TQQQ)...")
    etf_prices = _load_etf_prices_with_tqqq(price_index)

    # Also load regime indicators using the standard core-satellite logic
    # We temporarily register our preset in REGIME_PRESETS for _resolve_allocation
    REGIME_PRESETS[preset_name] = preset
    regime_indicators = _load_regime_indicators(
        pd.DatetimeIndex(dates), pd.DatetimeIndex(dates), config
    )

    # ── Early rebalance on regime change ────────────────────────────────
    # PLAIN ENGLISH: If the regime changes between scheduled rebalances,
    # we insert an extra rebalance on the day the regime flipped.  This
    # prevents being stuck in the wrong positioning for days.
    if config.get("early_rebalance_on_regime_change", False) and regime_indicators is not None:
        sched = set(rebalance_dates)
        for i, dt in enumerate(dates):
            if dt in sched or i == 0:
                continue
            prev_regime, *_ = _resolve_allocation(pd.Timestamp(dates[i - 1]), config, regime_indicators)
            curr_regime, *_ = _resolve_allocation(pd.Timestamp(dt), config, regime_indicators)
            if prev_regime != curr_regime:
                sched.add(dt)
        rebalance_dates = sorted(sched)

    day_map = _panel_day_map(panel)

    # ── Simulate the strategy ───────────────────────────────────────────
    equity = INITIAL_CAPITAL
    held: set[str] = set()
    prev_overlay = pd.Series(dtype=float)
    rows = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0
    ticker_contrib: dict[str, float] = {}
    concentration_overlay_adjustment_sum = 0.0
    concentration_overlay_active_count = 0

    # ── Drawdown circuit breaker state ─────────────────────────────────
    # PLAIN ENGLISH: Same as alpha strategy — track peak equity, go to cash
    # if we drop too far.  Re-enter when regime turns risk_on (not equity
    # recovery, because cash doesn't grow so we'd be stuck forever).
    dd_threshold = float(config.get("drawdown_circuit_breaker", 0.0))
    peak_equity = equity
    circuit_breaker_active = False

    # ── Volatility targeting state ─────────────────────────────────────
    vol_target = float(config.get("vol_target", 0.0))
    recent_returns: list[float] = []

    for dt in rebalance_dates:
        if pd.Timestamp(dt) not in day_map:
            continue
        day = day_map[pd.Timestamp(dt)]

        # Resolve regime and allocation (uses our TQQQ-enhanced preset)
        regime, core_weights, core_gross, overlay_gross = _resolve_allocation(
            pd.Timestamp(dt), config, regime_indicators
        )

        # ── Check circuit breaker ──────────────────────────────────────
        if dd_threshold > 0.0:
            drawdown_from_peak = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
            if circuit_breaker_active:
                # Re-enter when regime turns risk_on (market showing strength)
                if regime == "risk_on":
                    circuit_breaker_active = False
                    peak_equity = equity  # reset peak so we don't re-trigger
            else:
                if drawdown_from_peak >= dd_threshold:
                    circuit_breaker_active = True

        # If circuit breaker is active, override to zero exposure (100% cash)
        if circuit_breaker_active:
            core_gross = 0.0
            overlay_gross = 0.0

        # ── Volatility targeting ──────────────────────────────────────────
        if vol_target > 0.0 and len(recent_returns) >= VOL_TARGET_LOOKBACK:
            recent_arr = np.array(recent_returns[-VOL_TARGET_LOOKBACK:])
            realized_vol = float(np.std(recent_arr, ddof=1)) * np.sqrt(252.0 / holding_days)
            if realized_vol > 1e-6:
                vol_scale = float(np.clip(
                    vol_target / realized_vol,
                    VOL_TARGET_MIN_SCALE,
                    VOL_TARGET_MAX_SCALE,
                ))
                core_gross *= vol_scale
                overlay_gross *= vol_scale

        # Factor overlay — exact same logic as core-satellite
        overlay_before_concentration = overlay_gross
        overlay_gross, concentration_overlay_target, concentration_gap = _apply_concentration_overlay_target(
            pd.Timestamp(dt),
            core_gross,
            overlay_gross,
            regime_indicators,
            config,
        )
        concentration_overlay_adjustment = overlay_gross - overlay_before_concentration
        concentration_overlay_adjustment_sum += concentration_overlay_adjustment
        if abs(concentration_overlay_adjustment) > 1e-12:
            concentration_overlay_active_count += 1

        # PLAIN ENGLISH: If score_blend is enabled, we blend risk_on and risk_off
        # scores proportionally based on regime "strength" (0-1).  Otherwise we
        # hard-switch to the right score column for the current regime.
        if config.get("score_blend", False):
            strength = _compute_regime_strength(pd.Timestamp(dt), regime_indicators)
            score_col = _blended_score_col(day, str(config["score_source"]), regime, strength)
        else:
            score_col = _score_col_for_regime(str(config["score_source"]), regime)
        exit_floor = _exit_floor_for_regime(config, regime)
        selected = _select_sticky_holdings(
            day, held,
            score_col=score_col,
            return_col=return_col,
            shape=config["shape"],
            exit_rank_floor=exit_floor,
            max_per_sector=int(config["max_per_sector"]),
            earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
        )
        overlay = _sticky_overlay_weights(
            selected, overlay_gross, config["weighting"], prev_overlay,
            max_single_name_weight=float(config["max_single_name_weight"]),
            sticky_blend=float(config.get("sticky_blend", 0.65)),
        )
        held = set(overlay.index.astype(str))

        # Turnover and cost calculation
        aligned = pd.concat([prev_overlay.rename("prev"), overlay.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        cost = turnover * SLIPPAGE_BASE_PCT * float(config["cost_stress"])
        total_turnover += turnover
        total_cost += cost

        # ── Compute exit date and factor_scale FIRST ─────────────────────
        # PLAIN ENGLISH: Use actual next rebalance date as exit (not full holding_days)
        # to avoid double-counting returns when early rebalance shortens the period.
        dt_idx = rebalance_dates.index(dt) if dt in rebalance_dates else -1
        if dt_idx >= 0 and dt_idx + 1 < len(rebalance_dates):
            exit_dt = pd.Timestamp(rebalance_dates[dt_idx + 1])
        else:
            exit_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days)
        # Scale factor returns by actual/expected holding period
        actual_bdays = max(1, len(pd.bdate_range(pd.Timestamp(dt), exit_dt)) - 1)
        factor_scale = actual_bdays / max(1, holding_days)

        # Factor overlay return (stock picks)
        # PLAIN ENGLISH: Scale returns by actual/expected holding period when
        # early rebalance shortens the hold.
        factor_ret = 0.0
        if not overlay.empty and return_col in selected.columns:
            selected_by_ticker = selected.set_index("ticker")
            valid_tickers = [t for t in overlay.index if t in selected_by_ticker.index]
            if valid_tickers:
                ticker_returns = selected_by_ticker.loc[valid_tickers, return_col]
                ticker_period_contrib = ticker_returns * overlay.loc[valid_tickers] * factor_scale
                factor_ret = float(ticker_period_contrib.sum())
                for ticker, value in ticker_period_contrib.items():
                    ticker_contrib[str(ticker)] = ticker_contrib.get(str(ticker), 0.0) + float(value)
        core_ret = 0.0
        for sym, weight in core_weights.items():
            w = float(weight)
            if w == 0.0 or sym not in etf_prices.columns:
                continue
            if exit_dt in etf_prices.index and dt in etf_prices.index:
                entry_px = float(etf_prices.loc[pd.Timestamp(dt), sym])
                exit_px = float(etf_prices.loc[exit_dt, sym])
                if entry_px > 0 and not np.isnan(exit_px):
                    sym_ret = exit_px / entry_px - 1.0
                    core_ret += core_gross * w * sym_ret

        # Total strategy return for this period
        strategy_ret = core_ret + factor_ret - cost
        equity *= 1.0 + strategy_ret
        recent_returns.append(strategy_ret)
        # Update peak equity for circuit breaker tracking
        if equity > peak_equity:
            peak_equity = equity

        rows.append({"date": exit_dt, "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "exit_date": exit_dt,
            "regime": regime,
            "circuit_breaker_active": circuit_breaker_active,
            "core_ret": core_ret,
            "factor_ret": factor_ret,
            "cost": cost,
            "turnover": turnover,
            "strategy_ret": strategy_ret,
            "n_overlay": len(overlay),
            # Store the target positions so the independent daily audit can
            # rebuild each day's profit/loss from raw prices instead of
            # trusting only this backtest's holding-period return.
            "entry_delay_days": 0,
            "core_gross": core_gross,
            "overlay_gross": overlay_gross,
            "core_weights_json": json.dumps(
                {str(k): round(float(v), 6) for k, v in core_weights.items()},
                sort_keys=True,
            ),
            "overlay_weights_json": json.dumps(
                {str(k): round(float(v), 6) for k, v in overlay.items()},
                sort_keys=True,
            ),
            "tqqq_weight_used": float(core_weights.get("TQQQ", 0.0)),
            "concentration_overlay_target": concentration_overlay_target,
            "concentration_overlay_adjustment": concentration_overlay_adjustment,
            "concentration_qqq_spy_120d": concentration_gap,
        })
        prev_overlay = overlay

    # ── Compute metrics ─────────────────────────────────────────────────
    equity_series = pd.DataFrame(rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    trades = pd.DataFrame(trade_rows)
    periods_per_year = 252.0 / holding_days
    stats = portfolio_stats(equity_series, periods_per_year)

    bench = benchmark_equity(pd.DatetimeIndex(equity_series.index))
    bench_stats = {symbol: portfolio_stats(bench[symbol], periods_per_year) for symbol in bench.columns}
    comps = compare_to_benchmarks(equity_series, bench)
    subs = subperiod_metrics(equity_series, bench)

    # Holdout period check
    eq_holdout = equity_series.loc[(equity_series.index >= "2023-01-01") & (equity_series.index <= "2026-12-31")]
    holdout = {"data_available": False}
    if len(eq_holdout) >= 3:
        holdout_ret = float(eq_holdout.iloc[-1] / eq_holdout.iloc[0] - 1.0)
        holdout = {"data_available": True, "strategy_return_pct": round(holdout_ret * 100, 2)}
        for symbol in ("SPY", "QQQ", "BLEND"):
            b = bench[symbol].reindex(eq_holdout.index).ffill().bfill()
            bm_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
            holdout[f"{symbol.lower()}_return_pct"] = round(bm_ret * 100, 2)
            holdout[f"alpha_vs_{symbol.lower()}_pct"] = round((holdout_ret - bm_ret) * 100, 2)

    # Yearly alpha
    strat_rets = equity_series.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity_series.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity_series.index.year).sum() * 100.0

    # Regime distribution
    regime_counts = {}
    if not trades.empty:
        regime_counts = {str(k): int(v) for k, v in trades["regime"].value_counts().items()}

    metrics = {
        "strategy": "core_satellite_tqqq",
        "tqqq_weight": tqqq_weight,
        "cost_stress": cost_stress,
        "holding_days": holding_days,
        **stats,
        "turnover_pct": round(total_turnover * 100.0, 2),
        "estimated_cost_pct": round(total_cost * 100.0, 4),
        "n_rebalances": len(trades),
        "regime_counts": regime_counts,
        "concentration_overlay_active_rebalances": int(concentration_overlay_active_count),
        "avg_concentration_overlay_adjustment": round(
            float(concentration_overlay_adjustment_sum / max(len(trades), 1)),
            4,
        ),
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "holdout_2023_2026": holdout,
        "yearly_alpha_pct": {str(k): round(float(v), 2) for k, v in yearly_alpha.items()},
    }

    # Clean up — remove the temp preset we registered
    if preset_name in REGIME_PRESETS:
        del REGIME_PRESETS[preset_name]

    return metrics, equity_series, trades


def _scale_paper_targets(
    *,
    etf_targets: dict[str, float],
    overlay: pd.Series,
    max_gross: float = PAPER_MAX_GROSS_EXPOSURE,
) -> tuple[dict[str, float], pd.Series, float, float, bool]:
    """
    Scale all target weights down if gross exposure exceeds broker limit.

    PLAIN ENGLISH: If the strategy says invest 125% of your money, but the
    paper account only allows 100%, this shrinks everything proportionally
    so no orders get rejected by the broker.

    Returns: (scaled_etf_targets, scaled_overlay, raw_gross, scale_factor, was_scaled)
    """
    raw_gross = sum(abs(v) for v in etf_targets.values()) + float(overlay.abs().sum())
    if raw_gross <= 0 or raw_gross <= float(max_gross) + 1e-9:
        return etf_targets.copy(), overlay.copy(), raw_gross, 1.0, False
    scale = float(max_gross) / raw_gross
    scaled_etf = {k: v * scale for k, v in etf_targets.items()}
    return scaled_etf, overlay * scale, raw_gross, scale, True


LIVE_CONFIG_PATH = Path(SIGNAL_DIR) / "core_satellite_live_configs.json"


def _load_approved_tqqq_config() -> dict:
    """
    Retired compatibility shim.

    TQQQ exposure is now controlled by the unified core-alpha live config.
    """
    raise SystemExit(TQQQ_LIVE_DISABLED_MESSAGE)


def write_tqqq_signal(panel: pd.DataFrame, tqqq_weight: float = 0.20, factor_freshness: dict | None = None) -> Path:
    """
    Retired compatibility shim.

    Live signals must be generated by core_satellite_alpha.py. TQQQ exposure is
    controlled by the unified core-alpha config's `tqqq_weight` field.
    """
    raise SystemExit(TQQQ_LIVE_DISABLED_MESSAGE)
    saved = _load_approved_tqqq_config()
    tqqq_weight = float(saved.get("tqqq_weight", tqqq_weight))
    score_source = str(saved.get("score_source", "regime_adaptive"))
    shape = str(saved.get("shape", "top3"))
    weighting = str(saved.get("weighting", "sticky_score"))
    max_per_sector = int(saved.get("max_per_sector", 2))
    earnings_blackout_days = int(saved.get("earnings_blackout_days", 0))
    holding_days = int(saved.get("holding_days", HORIZON_DAYS))
    overlay_gross = float(saved.get("overlay_gross", 0.50))
    preset_name = str(saved.get("core_preset", "tqqq_enhanced"))
    regime_ma_window = int(saved.get("regime_ma_window", 100))
    regime_high_vol = float(saved.get("regime_high_vol", 0.30))
    high_vol_mode = str(saved.get("high_vol_mode", "fixed"))
    print(f"  ✓ Loaded nested-approved config: preset={preset_name}, TQQQ={tqqq_weight*100:.0f}%, "
          f"overlay={overlay_gross:.0%}, score={score_source}, shape={shape}, hd={holding_days}")

    # Build TQQQ preset and register it for regime resolution
    presets = build_tqqq_presets(tqqq_weight)
    preset = presets[preset_name] if preset_name in presets else presets["tqqq_enhanced"]
    preset = _regime_preset_with_overlay_gross(preset, overlay_gross)
    preset["ma_window"] = regime_ma_window
    preset["high_vol"] = regime_high_vol
    preset["high_vol_mode"] = high_vol_mode
    REGIME_PRESETS[preset_name] = preset

    # Config matching the winning backtest settings
    config = {
        "core_preset": preset_name,
        "regime_mode": preset_name,
        "regime_ma_window": int(preset["ma_window"]),
        "regime_high_vol": float(preset["high_vol"]),
        "high_vol_mode": str(preset.get("high_vol_mode", "fixed")),
        "core_weights": dict(preset["risk_on"]["core_weights"]),
        "score_source": score_source,
        "shape": shape,
        "weighting": weighting,
        "exit_rank_floor": 0.80,
        "adaptive_exit_mode": "fixed",
        "max_per_sector": max_per_sector,
        "earnings_blackout_days": earnings_blackout_days,
        "core_gross": float(preset["risk_on"]["core_gross"]),
        "overlay_gross": float(preset["risk_on"]["overlay_gross"]),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "max_single_name_weight": MAX_SINGLE_NAME_WEIGHT,
        "holding_days": holding_days,
        "live_config_source": "nested_walkforward",
        "approved_config_family": saved.get("approved_config_family"),
        "walkforward_source_json": saved.get("source_json"),
        "walkforward_source_metrics": saved.get("source_metrics", {}),
    }

    # Get latest date and resolve regime
    latest_date = pd.Timestamp(panel["date"].max())
    regime_indicators = _load_regime_indicators(
        pd.DatetimeIndex([latest_date]),
        pd.DatetimeIndex([latest_date]),
        config,
    )
    current_regime, core_weights, core_gross, overlay_gross = _resolve_allocation(
        latest_date, config, regime_indicators,
    )

    # Select overlay stocks using the right score column for this regime
    score_col = _score_col_for_regime(str(config["score_source"]), current_regime)
    day = panel[panel["date"] == latest_date]
    selected = _select_sticky_holdings(
        day,
        set(),  # no prior holdings for signal generation (fresh each day)
        score_col=score_col,
        return_col=None,
        shape=str(config["shape"]),
        exit_rank_floor=float(config["exit_rank_floor"]),
        max_per_sector=int(config["max_per_sector"]),
        earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
    )

    # ── Sentiment veto: check for strongly negative news ──────────────────
    # PLAIN ENGLISH: Same as alpha strategy — if a selected stock has terrible
    # recent news, drop it and pick the next-best candidate instead.
    sentiment_scores = {}
    if SENTIMENT_VETO_ENABLED and not selected.empty:
        try:
            selected, sentiment_scores = _apply_sentiment_veto(
                selected, day,
                score_col=score_col,
                shape=str(config["shape"]),
                exit_rank_floor=float(config["exit_rank_floor"]),
                max_per_sector=int(config["max_per_sector"]),
                earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
            )
        except Exception as exc:
            print(f"  ⚠ Sentiment veto failed (proceeding without): {exc}")

    overlay = _sticky_overlay_weights(
        selected,
        overlay_gross,
        str(config["weighting"]),
        pd.Series(dtype=float),
        max_single_name_weight=float(config["max_single_name_weight"]),
    )

    # Compute raw ETF target weights (core_gross * each weight)
    raw_etf_targets = {}
    for sym, w in core_weights.items():
        raw_etf_targets[sym] = core_gross * float(w)

    # Scale down to broker-safe gross exposure (<=1.0x)
    scaled_etf, paper_overlay, raw_gross, paper_scale, paper_scaled = _scale_paper_targets(
        etf_targets=raw_etf_targets,
        overlay=overlay,
        max_gross=PAPER_MAX_GROSS_EXPOSURE,
    )

    # Compute final gross exposure after scaling
    gross = sum(abs(v) for v in scaled_etf.values()) + float(paper_overlay.abs().sum())

    # ── TQQQ fast circuit breaker ─────────────────────────────────────────
    # PLAIN ENGLISH: Before writing the signal, check whether TQQQ has already
    # dropped more than TQQQ_FAST_DD_THRESHOLD from its recent high.  If so,
    # override the TQQQ weight to 0 right now — don't wait 20 days for the
    # regime detector to catch up.  The regime-switching flag will still say
    # "risk_on", but TQQQ specifically gets zeroed out.
    tqqq_cb_active = False
    tqqq_cb_drawdown = 0.0
    if scaled_etf.get("TQQQ", 0.0) > 0:
        try:
            lookback_dates = pd.bdate_range(
                end=latest_date, periods=TQQQ_FAST_DD_LOOKBACK + 1
            )
            tqqq_prices_df = _load_etf_prices_with_tqqq(lookback_dates)
            if "TQQQ" in tqqq_prices_df.columns:
                tqqq_ok, tqqq_cb_drawdown = _tqqq_drawdown_ok(tqqq_prices_df["TQQQ"])
                if not tqqq_ok:
                    tqqq_cb_active = True
                    scaled_etf["TQQQ"] = 0.0
                    # Redistribute the freed TQQQ weight to QQQ (keeping gross the same)
                    scaled_etf["QQQ"] = scaled_etf.get("QQQ", 0.0) + (tqqq_weight * core_gross * paper_scale)
                    gross = sum(abs(v) for v in scaled_etf.values()) + float(paper_overlay.abs().sum())
                    print(f"  🛑 TQQQ fast circuit breaker FIRED: "
                          f"{tqqq_cb_drawdown*100:.1f}% drawdown from {TQQQ_FAST_DD_LOOKBACK}-day high "
                          f"(threshold {TQQQ_FAST_DD_THRESHOLD*100:.0f}%). TQQQ weight → 0.")
        except Exception as exc:
            print(f"  ⚠ TQQQ fast circuit breaker check failed (proceeding with signal): {exc}")

    # Build signal row — same format as core_satellite_alpha_signal.csv
    # but with target_tqqq_weight added
    row = {
        "paper_signal_type": "core_satellite_tqqq",
        "paper_ready": True if (factor_freshness is None or factor_freshness.get("fresh", True)) else False,
        "core_preset": config["core_preset"],
        "regime_mode": config["regime_mode"],
        "current_regime": current_regime,
        "score_source": config["score_source"],
        "tqqq_weight_config": tqqq_weight,
        "tqqq_circuit_breaker_active": tqqq_cb_active,
        "tqqq_circuit_breaker_drawdown": round(tqqq_cb_drawdown, 4),
        "target_spy_weight": round(scaled_etf.get("SPY", 0.0), 4),
        "target_qqq_weight": round(scaled_etf.get("QQQ", 0.0), 4),
        "target_tqqq_weight": round(scaled_etf.get("TQQQ", 0.0), 4),
        "target_cash_weight": round(max(0.0, 1.0 - gross), 4),
        "gross_exposure": round(gross, 4),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "paper_max_gross_exposure": PAPER_MAX_GROSS_EXPOSURE,
        "raw_research_gross_exposure": round(raw_gross, 4),
        "paper_weight_scale": round(paper_scale, 8),
        "paper_weights_scaled": bool(paper_scaled),
        "overlay_gross": round(float(paper_overlay.abs().sum()), 4),
        "raw_overlay_gross": round(float(overlay.abs().sum()), 4),
        "core_gross": round(sum(abs(v) for v in scaled_etf.values()), 4),
        "raw_core_gross": round(core_gross, 4),
        "max_single_name_weight": round(float(config["max_single_name_weight"]), 4),
        "robust_cost_stress_pass": bool(
            saved.get("source_metrics", {}).get("cost_stress_approval_pass", saved.get("approval", {}).get("approved", False))
        ),
        "adaptive_exit_mode": str(config["adaptive_exit_mode"]),
        "earnings_blackout_days": int(config["earnings_blackout_days"]),
        "holding_days": int(config["holding_days"]),
        "overlay_tickers": ",".join(paper_overlay.index.astype(str).tolist()),
        "overlay_weights_json": json.dumps(
            {str(k): round(float(v), 6) for k, v in paper_overlay.items()}, sort_keys=True
        ),
        "raw_overlay_weights_json": json.dumps(
            {str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True
        ),
        "single_name_stock_picker_enabled": False,
        "ml_overlay_enabled": False,
        "factor_overlay_enabled": True,
        "latest_factor_date": str(latest_date.date()),
        "cost_stress": 2.0,
        "gates_all_pass": True if (factor_freshness is None or factor_freshness.get("fresh", True)) else False,
        "factor_data_stale": False if (factor_freshness is None or factor_freshness.get("fresh", True)) else True,
        "factor_data_age_trading_days": int(factor_freshness["age_trading_days"]) if factor_freshness else 0,
        "reason": "TQQQ-enhanced core-satellite signal",
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Write to dedicated TQQQ signal file
    out = Path(SIGNAL_DIR) / "core_satellite_tqqq_signal.csv"
    pd.DataFrame([row]).to_csv(out, index=False)

    # Clean up temp preset
    if "tqqq_enhanced" in REGIME_PRESETS:
        del REGIME_PRESETS["tqqq_enhanced"]

    return out


def main():
    parser = argparse.ArgumentParser(
        description="TQQQ-enhanced core-satellite research backtest. Standalone live signal generation is retired."
    )
    parser.add_argument("--tqqq-weight", type=float, default=0.20,
                        help="Research fraction of risk_on core allocated to TQQQ (default: 0.20)")
    parser.add_argument("--grid", action="store_true",
                        help="Run research grid search over multiple TQQQ weights")
    parser.add_argument("--backtest", action="store_true",
                        help="Run full research backtest")
    parser.add_argument("--cost-stress", type=float, default=2.0,
                        help="Cost stress multiplier (default: 2.0)")
    parser.add_argument("--holding-days", type=int, default=10,
                        help="Rebalance frequency in trading days (default: 10)")
    parser.add_argument("--ignore-stale", action="store_true",
                        help="Override stale data block for research backtests/grids")
    args = parser.parse_args()

    if not args.grid and not args.backtest:
        raise SystemExit(TQQQ_LIVE_DISABLED_MESSAGE)

    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)

    # Load factor panel (same as core-satellite)
    print("Loading factor panel and computing scores...")

    # ── FEATURE QUALITY FILTER ────────────────────────────────────────────────
    # PLAIN ENGLISH: Same filter as core-satellite alpha — if the feature quality
    # diagnostic has been run, drop grade D/F features to reduce noise.
    quality_filter = _load_feature_quality_filter()
    specs = load_feature_specs()
    if quality_filter is not None:
        original_count = len(specs)
        specs = [s for s in specs if s["feature"] in quality_filter]
        if len(specs) < original_count:
            print(f"  Feature filter applied: {original_count} → {len(specs)} specs")
    scores = load_prediction_scores()

    # Backtest/training panel: requires forward returns. It naturally lags the
    # newest raw data by the forward-return horizon, so it is correct for grid /
    # backtest but wrong for live freshness checks.
    panel = _ensure_robust_score_columns(
        attach_scores(load_factor_panel(specs), specs, scores)
    )
    print(f"  Backtest panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates, latest={pd.Timestamp(panel['date'].max()).date()}")

    # Live signal panel: does not require future returns, so it includes the
    # freshest feature rows and should be used for signal freshness + generation.
    signal_panel = _ensure_robust_score_columns(
        attach_scores(load_factor_panel(specs, require_forward_returns=False), specs, scores)
    )
    print(f"  Live signal panel: {len(signal_panel)} rows, {signal_panel['ticker'].nunique()} tickers, "
          f"{signal_panel['date'].nunique()} dates, latest={pd.Timestamp(signal_panel['date'].max()).date()}")

    # ── Data freshness gate ────────────────────────────────────────────
    # PLAIN ENGLISH: Check live feature freshness, not the backtest-label panel.
    freshness = check_factor_freshness(signal_panel, ignore_stale=args.ignore_stale)
    print(f"\n  {freshness['message']}")
    if freshness["blocked"]:
        raise SystemExit(f"Aborting: {freshness['message']}")

    if args.grid:
        # ── REDUCED Grid search ─────────────────────────────────────────
        # Keep only coarse, economically distinct choices to limit selection bias.
        tqqq_weights = [0.00, 0.10, 0.20, 0.30]
        cost_stresses = list(COST_STRESS_MULTIPLIERS)
        overlay_grosses = list(OVERLAY_GROSS_OPTIONS)
        preset_variants = ["tqqq_enhanced", "tqqq_enhanced_cashbuffer", "tqqq_enhanced_adaptive"]
        # DD breaker disabled — redundant with regime switching (same finding as alpha)
        dd_breaker_options = [0.0]

        # Count total configs to estimate runtime
        n_configs = (
            len(preset_variants) * len(tqqq_weights) * len(overlay_grosses) * len(SCORE_SOURCES)
            * len(SHAPES) * len(WEIGHTING_MODES) * len(MAX_PER_SECTOR_OPTIONS)
            * len(EARNINGS_BLACKOUT_DAY_OPTIONS) * len(HOLDING_DAY_OPTIONS)
            * len(cost_stresses) * len(dd_breaker_options)
        )
        print(f"\nReduced grid search: {n_configs} configs")
        print(f"  {len(preset_variants)} presets × {len(tqqq_weights)} TQQQ weights"
              f" × {len(overlay_grosses)} overlay levels"
              f" × {len(SCORE_SOURCES)} score sources × {len(SHAPES)} shapes"
              f" × {len(WEIGHTING_MODES)} weighting modes"
              f" × {len(MAX_PER_SECTOR_OPTIONS)} sector caps"
              f" × {len(EARNINGS_BLACKOUT_DAY_OPTIONS)} blackout options"
              f" × {len(HOLDING_DAY_OPTIONS)} holding days"
              f" × {len(cost_stresses)} cost validation levels"
              f" × {len(dd_breaker_options)} DD breaker options")
        print(f"\n{'#':>5} {'Preset':<30} {'TQQQ%':>5} {'Ov%':>4} {'Score':>12} {'Shp':>5} {'Wgt':>15} {'Sec':>3} "
              f"{'BO':>2} {'HD':>3} {'Cost':>5} | {'Return%':>10} {'Sharpe':>8} "
              f"{'MaxDD%':>8} {'Holdout%':>9}")
        print("-" * 130)

        grid_rows = []
        config_num = 0
        for pv in preset_variants:
            for tw in tqqq_weights:
                for og in overlay_grosses:
                    for ss in SCORE_SOURCES:
                        for shp in SHAPES:
                            for weighting in WEIGHTING_MODES:
                                for mps in MAX_PER_SECTOR_OPTIONS:
                                    for ebd in EARNINGS_BLACKOUT_DAY_OPTIONS:
                                        for hd in HOLDING_DAY_OPTIONS:
                                            for ddb in dd_breaker_options:
                                                for cs in cost_stresses:
                                                    config_num += 1
                                                    try:
                                                        metrics, eq, trades = run_tqqq_backtest(
                                                            panel,
                                                            tqqq_weight=tw,
                                                            cost_stress=cs,
                                                            holding_days=hd,
                                                            preset_name=pv,
                                                            score_source=ss,
                                                            shape=shp,
                                                            weighting=weighting,
                                                            overlay_gross=og,
                                                            max_per_sector=mps,
                                                            earnings_blackout_days=ebd,
                                                            drawdown_circuit_breaker=ddb,
                                                            quiet=True,
                                                        )
                                                    except Exception as e:
                                                        if not args.ignore_stale:
                                                            print(f"  SKIP {config_num}: {e}")
                                                        continue
                                                    holdout_ret = metrics.get("holdout_2023_2026", {}).get("strategy_return_pct", 0)
                                                    holdout = metrics.get("holdout_2023_2026", {})
                                                    comps = metrics.get("benchmark_comparisons", {})

                                                    # Print progress every 100 configs, or always for cost_stress=2.0 and dd=0
                                                    if (cs == 2.0 and ddb == 0.0) or config_num % 200 == 0:
                                                        print(
                                                            f"{config_num:>5} {pv:<30} {tw*100:>5.0f}% {og*100:>4.0f} "
                                                            f"{ss:>12} {shp:>5} {weighting:>15} "
                                                            f"{mps:>3} {ebd:>2} {hd:>3} dd={ddb:.0%} {cs:>5.1f} | "
                                                            f"{metrics['total_return_pct']:>10.1f} "
                                                            f"{metrics['sharpe']:>8.3f} "
                                                            f"{metrics['max_drawdown_pct']:>8.1f} "
                                                            f"{holdout_ret:>9.1f}"
                                                        )

                                                    grid_rows.append({
                                                        "core_preset": pv,
                                                        "tqqq_weight": tw,
                                                        "overlay_gross": og,
                                                        "score_source": ss,
                                                        "shape": shp,
                                                        "weighting": weighting,
                                                        "max_per_sector": mps,
                                                        "earnings_blackout_days": ebd,
                                                        "holding_days": hd,
                                                        "drawdown_circuit_breaker": ddb,
                                                        "cost_stress": cs,
                                                        "total_return_pct": metrics["total_return_pct"],
                                                        "cagr_pct": metrics["cagr_pct"],
                                                        "sharpe": metrics["sharpe"],
                                                        "max_drawdown_pct": metrics["max_drawdown_pct"],
                                                        "holdout_return_pct": holdout_ret,
                                                        "holdout_alpha_vs_qqq_pct": holdout.get("alpha_vs_qqq_pct", 0),
                                                        "holdout_alpha_vs_blend_pct": holdout.get("alpha_vs_blend_pct", 0),
                                                        "turnover_pct": metrics.get("turnover_pct", 0),
                                                        "alpha_vs_spy_pct": comps.get("SPY", {}).get("alpha_pct", 0),
                                                        "alpha_vs_qqq_pct": comps.get("QQQ", {}).get("alpha_pct", 0),
                                                        "alpha_vs_blend_pct": comps.get("BLEND", {}).get("alpha_pct", 0),
                                                        "tqqq_stress_row_gate_pass": bool(metrics["max_drawdown_pct"] > -50.0),
                                                    })

        grid_df = pd.DataFrame(grid_rows)
        grid_path = Path(SIGNAL_DIR) / "core_satellite_tqqq_grid.csv"
        robust_key_cols = [
            "core_preset",
            "tqqq_weight",
            "overlay_gross",
            "score_source",
            "shape",
            "weighting",
            "max_per_sector",
            "earnings_blackout_days",
            "holding_days",
            "drawdown_circuit_breaker",
        ]
        grid_df = add_cost_stress_approval_columns(
            grid_df,
            key_cols=robust_key_cols,
            required_costs=COST_STRESS_MULTIPLIERS,
            row_gate_col="tqqq_stress_row_gate_pass",
        )

        # ── Auto-select winning config using ROBUSTNESS SCORE ─────────────
        # PLAIN ENGLISH: Select in Sharpe units and subtract explicit penalties.
        # No CAGR, total return, or holdout return bonus is part of this score.
        robustness_cols = grid_df.apply(
            lambda row: pd.Series(robustness_score_components(row)),
            axis=1,
        )
        grid_df = pd.concat([grid_df, robustness_cols], axis=1)
        grid_df["paper_ready"] = (
            grid_df["tqqq_stress_row_gate_pass"].astype(bool)
            & grid_df["robust_cost_stress_pass"].astype(bool)
        )
        grid_df.to_csv(grid_path, index=False)
        print(f"\nGrid saved → {grid_path}  ({len(grid_df)} rows)")
        print(
            "Cost-stress gate: "
            f"{int(grid_df['robust_cost_stress_pass'].sum())} rows pass grouped checks "
            f"at levels {COST_STRESS_MULTIPLIERS}"
        )

        base_grid = grid_df[grid_df["cost_stress"].astype(float) == float(COST_STRESS_MULTIPLIERS[0])].copy()
        if base_grid.empty:
            raise SystemExit(f"No TQQQ base cost-stress rows found at {COST_STRESS_MULTIPLIERS[0]}x.")
        base_grid = base_grid.sort_values(
            ["paper_ready", "robust_cost_stress_pass", "robustness_score"],
            ascending=[False, False, False],
        )
        hardened = base_grid[base_grid["paper_ready"]]
        if hardened.empty:
            hardened = base_grid[base_grid["robust_cost_stress_pass"]]
        if hardened.empty:
            hardened = base_grid

        if not hardened.empty:
            winner = hardened.iloc[0]
            winning_config = {
                "core_preset": str(winner["core_preset"]),
                "tqqq_weight": float(winner["tqqq_weight"]),
                "overlay_gross": float(winner["overlay_gross"]),
                "score_source": str(winner["score_source"]),
                "shape": str(winner["shape"]),
                "weighting": str(winner["weighting"]),
                "max_per_sector": int(winner["max_per_sector"]),
                "earnings_blackout_days": int(winner.get("earnings_blackout_days", 0)),
                "holding_days": int(winner["holding_days"]),
                "drawdown_circuit_breaker": float(winner.get("drawdown_circuit_breaker", 0.0)),
                # Record the metrics so we know what we're targeting
                "grid_sharpe": float(winner["sharpe"]),
                "grid_robustness_score": float(winner.get("robustness_score", 0)),
                "grid_drawdown_penalty": float(winner.get("drawdown_penalty", 0)),
                "grid_turnover_penalty": float(winner.get("turnover_penalty", 0)),
                "grid_instability_penalty": float(winner.get("instability_penalty", 0)),
                "grid_return_pct": float(winner["total_return_pct"]),
                "grid_max_dd_pct": float(winner["max_drawdown_pct"]),
                "grid_holdout_pct": float(winner.get("holdout_return_pct", 0)),
                "base_cost_stress": float(winner.get("cost_stress", COST_STRESS_MULTIPLIERS[0])),
                "robust_cost_stress_pass": bool(winner.get("robust_cost_stress_pass", False)),
                "robust_cost_stress_summary": {
                    "cost_levels": str(winner.get("stress_cost_levels", "")),
                    "has_required_costs": bool(winner.get("stress_has_required_costs", False)),
                    "all_gates_pass": bool(winner.get("stress_all_gates_pass", False)),
                    "min_alpha_vs_spy_pct": float(winner.get("stress_min_alpha_vs_spy_pct", 0)),
                    "min_alpha_vs_qqq_pct": float(winner.get("stress_min_alpha_vs_qqq_pct", 0)),
                    "min_alpha_vs_blend_pct": float(winner.get("stress_min_alpha_vs_blend_pct", 0)),
                    "min_holdout_alpha_vs_qqq_pct": float(winner.get("stress_min_holdout_alpha_vs_qqq_pct", 0)),
                    "min_holdout_alpha_vs_blend_pct": float(winner.get("stress_min_holdout_alpha_vs_blend_pct", 0)),
                },
                "selection_method": "best_base_cost_robustness_score_cost_stress_gate",
                "selected_at": datetime.now().isoformat(),
            }
            config_path = Path(SIGNAL_DIR) / "core_satellite_tqqq_winning_config.json"
            with open(config_path, "w") as f:
                json.dump(winning_config, f, indent=2)
            print(f"\n✓ Winning config saved → {config_path}")
            print(f"  Preset={winning_config['core_preset']}, TQQQ={winning_config['tqqq_weight']*100:.0f}%, "
                  f"Overlay={winning_config['overlay_gross']*100:.0f}%, "
                  f"Score={winning_config['score_source']}, Shape={winning_config['shape']}, "
                  f"Weighting={winning_config['weighting']}, "
                  f"HD={winning_config['holding_days']}")
            print(f"  Sharpe={winning_config['grid_sharpe']:.3f}, "
                  f"Robust={winning_config['grid_robustness_score']:.3f}, "
                  f"Return={winning_config['grid_return_pct']:.0f}%, "
                  f"MaxDD={winning_config['grid_max_dd_pct']:.1f}%")
            print(f"  Penalties: DD={winning_config['grid_drawdown_penalty']:.3f}, "
                  f"turnover={winning_config['grid_turnover_penalty']:.3f}, "
                  f"instability={winning_config['grid_instability_penalty']:.3f}")
            print(
                "  Cost stress approval: "
                f"{'PASS' if winning_config['robust_cost_stress_pass'] else 'FAIL'} "
                f"levels={winning_config['robust_cost_stress_summary']['cost_levels']}"
            )

        # Show best by Sharpe at base cost stress
        print("\nTop 10 by Sharpe (at cost_stress=2.0):")
        top = grid_df[grid_df["cost_stress"] == 2.0].nlargest(10, "sharpe")
        for _, r in top.iterrows():
            print(f"  {r['core_preset']:<30} TQQQ={r['tqqq_weight']*100:.0f}%  "
                  f"Ov={r['overlay_gross']*100:.0f}%  "
                  f"score={r['score_source']:<25} shape={r['shape']}  "
                  f"weighting={r['weighting']}  "
                  f"sec={r['max_per_sector']:.0f}  hd={r['holding_days']:.0f}  "
                  f"Return={r['total_return_pct']:.0f}%  "
                  f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_drawdown_pct']:.1f}%")

        # Show best by total return
        print("\nTop 10 by Total Return (at cost_stress=2.0):")
        top_ret = grid_df[grid_df["cost_stress"] == 2.0].nlargest(10, "total_return_pct")
        for _, r in top_ret.iterrows():
            print(f"  {r['core_preset']:<30} TQQQ={r['tqqq_weight']*100:.0f}%  "
                  f"Ov={r['overlay_gross']*100:.0f}%  "
                  f"score={r['score_source']:<25} shape={r['shape']}  "
                  f"weighting={r['weighting']}  "
                  f"sec={r['max_per_sector']:.0f}  hd={r['holding_days']:.0f}  "
                  f"Return={r['total_return_pct']:.0f}%  "
                  f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_drawdown_pct']:.1f}%")

    elif args.backtest:
        # ── Single backtest run ────────────────────────────────────────
        print(f"\nRunning TQQQ-enhanced backtest (TQQQ weight={args.tqqq_weight*100:.0f}%, "
              f"cost_stress={args.cost_stress}x, holding={args.holding_days}d)...")

        metrics, equity_series, trades = run_tqqq_backtest(
            panel,
            tqqq_weight=args.tqqq_weight,
            cost_stress=args.cost_stress,
            holding_days=args.holding_days,
        )

        # Save outputs
        eq_path = Path(SIGNAL_DIR) / "core_satellite_tqqq_equity.csv"
        pd.DataFrame({"equity": equity_series}).to_csv(eq_path)

        trades_path = Path(SIGNAL_DIR) / "core_satellite_tqqq_trades.csv"
        trades.to_csv(trades_path, index=False)

        metrics_path = Path(SIGNAL_DIR) / "core_satellite_tqqq_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Print results
        print(f"\n{'='*60}")
        print(f"TQQQ-ENHANCED CORE-SATELLITE BACKTEST")
        print(f"{'='*60}")
        print(f"TQQQ weight (risk_on): {args.tqqq_weight*100:.0f}%")
        print(f"Cost stress:           {args.cost_stress}x")
        print(f"Holding days:          {args.holding_days}")
        print(f"{'─'*60}")
        print(f"Total return:          {metrics['total_return_pct']:.1f}%")
        print(f"CAGR:                  {metrics['cagr_pct']:.2f}%")
        print(f"Sharpe:                {metrics['sharpe']:.3f}")
        print(f"Sortino:               {metrics.get('sortino', 0):.3f}")
        print(f"Max drawdown:          {metrics['max_drawdown_pct']:.1f}%")
        print(f"Turnover:              {metrics['turnover_pct']:.1f}%")
        print(f"Est. cost:             {metrics['estimated_cost_pct']:.2f}%")
        print(f"Rebalances:            {metrics['n_rebalances']}")
        print(f"{'─'*60}")

        comps = metrics.get("benchmark_comparisons", {})
        for bm in ["SPY", "QQQ", "BLEND"]:
            if bm in comps:
                print(f"Alpha vs {bm:5s}:         {comps[bm].get('alpha_pct', 0):.1f}%")

        holdout = metrics.get("holdout_2023_2026", {})
        if holdout.get("data_available"):
            print(f"{'─'*60}")
            print(f"Holdout 2023-2026:     {holdout['strategy_return_pct']:.1f}%")
            print(f"  vs QQQ:              +{holdout.get('alpha_vs_qqq_pct', 0):.1f}%")
            print(f"  vs BLEND:            +{holdout.get('alpha_vs_blend_pct', 0):.1f}%")

        print(f"{'─'*60}")
        print(f"Regime distribution:   {metrics.get('regime_counts', {})}")

        print(f"\nYearly alpha vs BLEND:")
        for yr, a in sorted(metrics.get("yearly_alpha_pct", {}).items()):
            bar = "+" * max(0, int(a/5)) if a > 0 else "-" * max(0, int(-a/5))
            print(f"  {yr}: {a:+7.1f}%  {bar}")

        print(f"\nSaved → {eq_path}")
        print(f"Saved → {trades_path}")
        print(f"Saved → {metrics_path}")

        # Compare to standard core-satellite
        try:
            with open(Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json") as f:
                std = json.load(f)
            print(f"\n{'='*60}")
            print(f"COMPARISON vs Standard Core-Satellite")
            print(f"{'='*60}")
            print(f"{'Metric':<25s} {'Standard':>12s} {'TQQQ-Enhanced':>14s} {'Diff':>10s}")
            print(f"{'─'*60}")
            for key, label in [
                ("total_return_pct", "Total Return %"),
                ("cagr_pct", "CAGR %"),
                ("sharpe", "Sharpe"),
                ("max_drawdown_pct", "Max Drawdown %"),
            ]:
                sv = float(std.get(key, 0))
                tv = float(metrics.get(key, 0))
                diff = tv - sv
                print(f"{label:<25s} {sv:>12.1f} {tv:>14.1f} {diff:>+10.1f}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
