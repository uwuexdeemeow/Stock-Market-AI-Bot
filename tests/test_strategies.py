"""
test_strategies.py — Tests for core strategy logic: regime switching,
allocation resolution, overlay weights, drawdown throttle, freshness gate,
order building, and sell-wait-buy phasing.

PLAIN ENGLISH:
These tests check the decision-making brain of the trading system.
If any of these break, your strategy might allocate wrong weights,
pick the wrong regime, or build orders incorrectly.
Every test should take < 1 second.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ── Strategy imports ──────────────────────────────────────────────────────────
from core_satellite_alpha import (
    REGIME_PRESETS,
    _resolve_allocation,
    _overlay_weights,
    _cap_and_rescale,
    _exit_floor_for_regime,
    _score_col,
    _score_col_for_regime,
    _top_count,
    check_factor_freshness,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — tiny builders so tests stay short
# ═══════════════════════════════════════════════════════════════════════════════

def _regime_row(qqq_trend_ok: bool, spy_trend_ok: bool, high_vol: bool) -> pd.DataFrame:
    """
    Build a one-row DataFrame that looks like the regime_indicators table.
    PLAIN ENGLISH: fakes the market conditions for a single day so we can
    test which regime the strategy would pick.
    """
    dt = pd.Timestamp("2025-01-15")
    return pd.DataFrame(
        {"qqq_trend_ok": [qqq_trend_ok], "spy_trend_ok": [spy_trend_ok], "high_vol": [high_vol]},
        index=[dt],
    )


def _qqq_trend_switch_config(regime_mode: str = "qqq_trend_switch") -> dict:
    """Minimal config dict using a regime preset."""
    return {"regime_mode": regime_mode}


def _selected_df(tickers: list[str], scores: list[float]) -> pd.DataFrame:
    """
    Build a mini DataFrame that looks like the ranked stock picks.
    PLAIN ENGLISH: fakes the factor-scored stock list so we can test how
    weights get assigned to each stock.
    """
    return pd.DataFrame({
        "ticker": tickers,
        "_rank_score": scores,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REGIME SWITCHING — which regime does the strategy pick?
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeSwitching:
    """
    PLAIN ENGLISH: The strategy looks at QQQ trend, SPY trend, and volatility
    to decide if the market is risk_on, neutral, or risk_off.  These tests
    verify every combination picks the right regime.
    """
    DT = pd.Timestamp("2025-01-15")
    MODE = "qqq_trend_switch"

    def test_risk_on_when_both_trends_up_low_vol(self):
        """QQQ above MA + SPY above MA + low vol → risk_on"""
        indicators = _regime_row(qqq_trend_ok=True, spy_trend_ok=True, high_vol=False)
        regime, weights, core_gross, overlay_gross = _resolve_allocation(
            self.DT, _qqq_trend_switch_config(), indicators
        )
        assert regime == "risk_on"
        # risk_on for qqq_trend_switch means 100% QQQ, 0% SPY
        assert weights == {"SPY": 0.00, "QQQ": 1.00}

    def test_neutral_when_spy_up_qqq_down_low_vol(self):
        """SPY above MA + QQQ below MA + low vol → neutral"""
        indicators = _regime_row(qqq_trend_ok=False, spy_trend_ok=True, high_vol=False)
        regime, weights, core_gross, overlay_gross = _resolve_allocation(
            self.DT, _qqq_trend_switch_config(), indicators
        )
        assert regime == "neutral"
        assert weights == {"SPY": 0.25, "QQQ": 0.75}

    def test_risk_off_when_spy_down(self):
        """SPY below MA → risk_off regardless of QQQ"""
        indicators = _regime_row(qqq_trend_ok=True, spy_trend_ok=False, high_vol=False)
        regime, *_ = _resolve_allocation(
            self.DT, _qqq_trend_switch_config(), indicators
        )
        assert regime == "risk_off"

    def test_risk_off_when_high_vol(self):
        """High volatility → risk_off even if both trends are up"""
        indicators = _regime_row(qqq_trend_ok=True, spy_trend_ok=True, high_vol=True)
        regime, *_ = _resolve_allocation(
            self.DT, _qqq_trend_switch_config(), indicators
        )
        assert regime == "risk_off"

    def test_risk_off_weights_favor_spy(self):
        """In risk_off, SPY weight should be higher than QQQ (defensive)"""
        indicators = _regime_row(qqq_trend_ok=False, spy_trend_ok=False, high_vol=True)
        regime, weights, *_ = _resolve_allocation(
            self.DT, _qqq_trend_switch_config(), indicators
        )
        assert regime == "risk_off"
        assert weights["SPY"] > weights["QQQ"]

    def test_risk_on_has_highest_core_gross(self):
        """risk_on should have higher core_gross than risk_off (more invested)"""
        ri_on = _regime_row(qqq_trend_ok=True, spy_trend_ok=True, high_vol=False)
        ri_off = _regime_row(qqq_trend_ok=False, spy_trend_ok=False, high_vol=True)
        _, _, gross_on, _ = _resolve_allocation(self.DT, _qqq_trend_switch_config(), ri_on)
        _, _, gross_off, _ = _resolve_allocation(self.DT, _qqq_trend_switch_config(), ri_off)
        assert gross_on >= gross_off

    def test_static_fallback_when_no_preset(self):
        """If regime_mode isn't a known preset, falls back to 'static'"""
        config = {
            "regime_mode": "unknown_mode",
            "core_weights": {"SPY": 0.50, "QQQ": 0.50},
            "core_gross": 0.80,
            "overlay_gross": 0.20,
        }
        regime, weights, core_gross, overlay_gross = _resolve_allocation(
            self.DT, config, None
        )
        assert regime == "static"
        assert weights == {"SPY": 0.50, "QQQ": 0.50}
        assert core_gross == 0.80

    def test_all_presets_have_three_regimes(self):
        """Every regime preset must define risk_on, neutral, and risk_off"""
        for name, preset in REGIME_PRESETS.items():
            for regime in ("risk_on", "neutral", "risk_off"):
                assert regime in preset, f"{name} missing {regime}"
                assert "core_weights" in preset[regime]
                assert "core_gross" in preset[regime]
                assert "overlay_gross" in preset[regime]

    def test_overlay70_core55_preset_values(self):
        """The winning preset (overlay70_core55) should have specific known values"""
        preset = REGIME_PRESETS["qqq_trend_switch_overlay70_core55"]
        assert preset["risk_on"]["core_gross"] == 0.55
        assert preset["risk_on"]["overlay_gross"] == 0.70
        assert preset["risk_off"]["core_gross"] == 0.65
        assert preset["risk_off"]["overlay_gross"] == 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OVERLAY WEIGHTS — how much to put in each stock pick
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlayWeights:
    """
    PLAIN ENGLISH: After picking stocks, the strategy assigns a weight
    (percentage of portfolio) to each.  These tests check that the weights
    add up correctly and no single stock gets too much.
    """

    def test_equal_weight_sums_to_gross(self):
        """Equal weighting: 3 stocks with overlay_gross=0.30 → each gets 0.10"""
        selected = _selected_df(["AAPL", "MSFT", "GOOG"], [0.8, 0.8, 0.8])
        weights = _overlay_weights(selected, overlay_gross=0.30, weighting="equal")
        assert abs(weights.sum() - 0.30) < 1e-9

    def test_score_weight_sums_to_gross(self):
        """Score weighting should still sum to the target overlay_gross"""
        selected = _selected_df(["AAPL", "MSFT", "GOOG"], [0.9, 0.5, 0.3])
        weights = _overlay_weights(selected, overlay_gross=0.25, weighting="score")
        assert abs(weights.sum() - 0.25) < 1e-6

    def test_score_weight_higher_score_gets_more(self):
        """Higher-scored stocks should get bigger weights"""
        # Use overlay_gross=0.40 so the single-name cap (0.25) doesn't bind
        # equally on both stocks — HIGH should get more than LOW.
        selected = _selected_df(["HIGH", "LOW"], [0.95, 0.10])
        weights = _overlay_weights(selected, overlay_gross=0.40, weighting="score")
        assert weights["HIGH"] > weights["LOW"]

    def test_single_name_cap_enforced(self):
        """No stock should exceed the max single-name weight cap"""
        selected = _selected_df(["AAPL", "TINY"], [0.99, 0.01])
        cap = 0.15
        weights = _overlay_weights(
            selected, overlay_gross=0.50, weighting="score",
            max_single_name_weight=cap,
        )
        assert weights.max() <= cap + 1e-9

    def test_empty_selection_returns_empty(self):
        """If no stocks are selected, weights should be empty"""
        selected = _selected_df([], [])
        weights = _overlay_weights(selected, overlay_gross=0.25, weighting="equal")
        assert weights.empty

    def test_cap_and_rescale_preserves_total(self):
        """After capping one position, total weight should still hit gross"""
        weights = pd.Series({"A": 0.30, "B": 0.15, "C": 0.05})
        gross = 0.50
        cap = 0.20
        result = _cap_and_rescale(weights, gross, cap)
        assert result.max() <= cap + 1e-9
        assert abs(result.sum() - gross) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXIT FLOOR — regime-adaptive exit rank threshold
# ═══════════════════════════════════════════════════════════════════════════════

class TestExitFloor:
    """
    PLAIN ENGLISH: When the strategy decides whether to keep or drop a stock,
    it uses an 'exit floor' — a threshold that changes with the regime.
    In risk_on, we're more lenient (lower floor). In risk_off, stricter.
    """

    def test_fixed_mode_ignores_regime(self):
        """Fixed exit mode returns the base floor regardless of regime"""
        config = {"exit_rank_floor": 0.80, "adaptive_exit_mode": "fixed"}
        assert _exit_floor_for_regime(config, "risk_on") == 0.80
        assert _exit_floor_for_regime(config, "risk_off") == 0.80

    def test_regime_mode_risk_on_lowers_floor(self):
        """risk_on → more lenient (lower floor, keep stocks longer)"""
        config = {"exit_rank_floor": 0.80, "adaptive_exit_mode": "regime"}
        floor = _exit_floor_for_regime(config, "risk_on")
        assert floor <= 0.70

    def test_regime_mode_risk_off_raises_floor(self):
        """risk_off → stricter (higher floor, dump stocks faster)"""
        config = {"exit_rank_floor": 0.80, "adaptive_exit_mode": "regime"}
        floor = _exit_floor_for_regime(config, "risk_off")
        assert floor >= 0.90

    def test_regime_mode_neutral_returns_base(self):
        """neutral → just uses the base floor"""
        config = {"exit_rank_floor": 0.80, "adaptive_exit_mode": "regime"}
        assert _exit_floor_for_regime(config, "neutral") == 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCORE COLUMN SELECTION — which factor score to use per regime
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreColumns:
    """
    PLAIN ENGLISH: Different regimes use different scoring columns.
    risk_on uses aggressive scores, risk_off uses defensive scores.
    """

    def test_regime_adaptive_risk_on(self):
        col = _score_col_for_regime("regime_adaptive", "risk_on")
        assert "risk_on" in col

    def test_regime_adaptive_risk_off(self):
        col = _score_col_for_regime("regime_adaptive", "risk_off")
        assert "defensive" in col

    def test_regime_adaptive_neutral(self):
        col = _score_col_for_regime("regime_adaptive", "neutral")
        assert "walkforward" in col

    def test_consensus_risk_on(self):
        col = _score_col_for_regime("regime_adaptive_consensus", "risk_on")
        assert "consensus" in col

    def test_non_adaptive_ignores_regime(self):
        """Non-adaptive sources return the same column regardless of regime"""
        col_on = _score_col_for_regime("factor_walkforward", "risk_on")
        col_off = _score_col_for_regime("factor_walkforward", "risk_off")
        assert col_on == col_off

    def test_top_count_shapes(self):
        assert _top_count(50, "top3") == 3
        assert _top_count(50, "top5") == 5
        assert _top_count(50, "top10") == 10
        assert _top_count(50, "top10pct") == 5  # 10% of 50


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FACTOR FRESHNESS GATE — is data too old to trade on?
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorFreshness:
    """
    PLAIN ENGLISH: Before generating a signal, we check how old the factor
    data is.  Too old → block signal.  Somewhat old → warn but proceed.
    """

    def _panel_with_date(self, latest_date: str) -> pd.DataFrame:
        """Make a minimal panel with just a 'date' column."""
        return pd.DataFrame({"date": [latest_date]})

    def test_fresh_data_passes(self):
        """Data from today should be fresh"""
        today = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        result = check_factor_freshness(self._panel_with_date(today))
        assert result["fresh"] is True
        assert result["blocked"] is False

    def test_warn_zone_not_blocked(self):
        """Data 7 trading days old: warn but don't block"""
        # Go back ~10 calendar days to get ~7 trading days
        old_date = (pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(7)).strftime("%Y-%m-%d")
        result = check_factor_freshness(self._panel_with_date(old_date), warn_days=5, block_days=10)
        assert result["fresh"] is False
        assert result["blocked"] is False
        assert "WARNING" in result["message"]

    def test_block_zone_blocks(self):
        """Data 15 trading days old: block signal generation"""
        old_date = (pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(15)).strftime("%Y-%m-%d")
        result = check_factor_freshness(self._panel_with_date(old_date), warn_days=5, block_days=10)
        assert result["blocked"] is True
        assert "BLOCKED" in result["message"]

    def test_ignore_stale_overrides_block(self):
        """--ignore-stale flag should override the block"""
        old_date = (pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(15)).strftime("%Y-%m-%d")
        result = check_factor_freshness(
            self._panel_with_date(old_date), warn_days=5, block_days=10, ignore_stale=True
        )
        assert result["blocked"] is False
        assert "OVERRIDE" in result["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORDER BUILDING — converting target weights into buy/sell orders
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Legacy broker order-builder tests removed from the Alpaca-only path")
class TestOrderBuilding:
    """
    PLAIN ENGLISH: Given target portfolio weights and current positions,
    the system figures out what shares to buy/sell. These tests check
    that math is correct and edge cases are handled.
    """

    def test_buy_from_zero(self):
        """Starting with no shares, should generate BUY orders"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"SPY": 0.50, "QQQ": 0.50},
            current_positions={},
            prices={"SPY": 500.0, "QQQ": 400.0},
            min_trade_value=100,
            limit_offset_bps=5,
        )
        buys = orders[orders["action"] == "BUY"]
        assert len(buys) == 2
        # SPY: 50000/500 = 100 shares, QQQ: 50000/400 = 125 shares
        spy = orders[orders["ticker"] == "SPY"].iloc[0]
        assert spy["target_shares"] == 100
        assert spy["delta_shares"] == 100

    def test_sell_to_exit(self):
        """Holding shares with zero target weight → SELL"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"SPY": 1.00},
            current_positions={"SPY": 100, "OLD_STOCK": 50},
            prices={"SPY": 500.0, "OLD_STOCK": 200.0},
            min_trade_value=100,
            limit_offset_bps=5,
        )
        old = orders[orders["ticker"] == "OLD_STOCK"].iloc[0]
        assert old["action"] == "SELL"
        assert old["delta_shares"] == -50

    def test_hold_when_at_target(self):
        """If current shares match target, action should be HOLD"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"SPY": 0.50},
            current_positions={"SPY": 100},
            prices={"SPY": 500.0},
            min_trade_value=100,
            limit_offset_bps=5,
        )
        spy = orders[orders["ticker"] == "SPY"].iloc[0]
        assert spy["action"] == "HOLD"

    def test_skip_below_min_trade_value(self):
        """Small rebalances below min_trade_value → SKIP"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"SPY": 0.50},
            current_positions={"SPY": 99},  # off by 1 share = $500
            prices={"SPY": 500.0},
            min_trade_value=1000,  # threshold higher than $500
            limit_offset_bps=5,
        )
        spy = orders[orders["ticker"] == "SPY"].iloc[0]
        assert spy["action"] == "SKIP"
        assert spy["reason"] == "below_min_trade_value"

    def test_skip_missing_price(self):
        """If we can't get a price, skip the ticker"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"MYSTERY": 0.10},
            current_positions={},
            prices={"MYSTERY": 0.0},
            min_trade_value=100,
            limit_offset_bps=5,
        )
        assert orders.iloc[0]["action"] == "SKIP"
        assert orders.iloc[0]["reason"] == "missing_price"

    def test_buy_limit_above_market(self):
        """BUY limit price should be slightly above market (willing to pay a bit more)"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={"SPY": 0.50},
            current_positions={},
            prices={"SPY": 500.0},
            min_trade_value=100,
            limit_offset_bps=10,  # 10 bps above
        )
        spy = orders[orders["ticker"] == "SPY"].iloc[0]
        assert spy["limit_price"] > 500.0

    def test_sell_limit_below_market(self):
        """SELL limit price should be slightly below market (willing to take a bit less)"""
        orders = build_core_satellite_orders(
            equity=100_000,
            target_weights={},
            current_positions={"SPY": 100},
            prices={"SPY": 500.0},
            min_trade_value=100,
            limit_offset_bps=10,
        )
        spy = orders[orders["ticker"] == "SPY"].iloc[0]
        assert spy["limit_price"] < 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LIMIT PRICE ROUNDING — US stock rules
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Legacy broker limit-price tests removed from the Alpaca-only path")
class TestLimitPriceRounding:
    """
    PLAIN ENGLISH: US stocks above $1 must be priced in pennies.
    BUY prices round up (willing to pay more), SELL prices round down
    (willing to accept less).
    """

    def test_buy_rounds_up(self):
        """BUY limit should round up to next penny"""
        price = _round_us_limit_price(100.123, "BUY")
        assert price == 100.13

    def test_sell_rounds_down(self):
        """SELL limit should round down to nearest penny"""
        price = _round_us_limit_price(100.127, "SELL")
        assert price == 100.12

    def test_exact_penny_unchanged(self):
        """Already-round prices stay the same"""
        assert _round_us_limit_price(50.00, "BUY") == 50.00
        assert _round_us_limit_price(50.00, "SELL") == 50.00


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TARGET WEIGHT EXTRACTION — parsing signal into weights dict
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Legacy broker signal-parser tests removed from the Alpaca-only path")
class TestTargetWeights:
    """
    PLAIN ENGLISH: The signal CSV has core ETF weights and overlay stock
    weights stored in specific columns. This test checks that parsing
    produces the right target portfolio.
    """

    def test_core_plus_overlay(self):
        """Signal with core SPY/QQQ + overlay stocks should merge correctly"""
        signal = pd.Series({
            "target_spy_weight": 0.1375,
            "target_qqq_weight": 0.4125,
            "overlay_weights_json": '{"AAPL": 0.15, "MSFT": 0.10}',
        })
        weights = core_satellite_target_weights(signal)
        # Should have SPY, QQQ, AAPL, MSFT
        assert "SPY" in weights
        assert "QQQ" in weights
        assert "AAPL" in weights
        assert "MSFT" in weights
        assert abs(weights["SPY"] - 0.1375) < 1e-6
        assert abs(weights["QQQ"] - 0.4125) < 1e-6

    def test_no_overlay_just_core(self):
        """Signal with no overlay stocks should only have core ETFs"""
        signal = pd.Series({
            "target_spy_weight": 0.50,
            "target_qqq_weight": 0.50,
            "overlay_weights_json": "{}",
        })
        weights = core_satellite_target_weights(signal)
        assert len(weights) == 2
        assert abs(weights["SPY"] - 0.50) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ORDER STATUS BUCKETING — classify broker order states
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Legacy broker order-status tests removed from the Alpaca-only path")
class TestOrderStatusBucket:
    """
    PLAIN ENGLISH: Legacy paper execution reported order status strings here.
    We bucket them into simple categories: filled, partial, cancelled,
    pending, etc.
    """

    def test_filled_variants(self):
        """All filled-like statuses map to 'filled'"""
        for status in ["FILLED_ALL", "FILLED_PART", "FILLED", "DEALT"]:
            bucket = _order_status_bucket(status, dealt_qty=100, qty=100)
            assert bucket in ("filled", "partial"), f"{status} → {bucket}"

    def test_cancelled(self):
        bucket = _order_status_bucket("CANCELLED_ALL", dealt_qty=0, qty=100)
        assert bucket == "cancelled"

    def test_pending(self):
        """Submitting/waiting statuses map to 'open'"""
        bucket = _order_status_bucket("SUBMITTING", dealt_qty=0, qty=100)
        assert bucket == "open"
