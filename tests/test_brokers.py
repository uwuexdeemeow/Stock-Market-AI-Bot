"""
test_brokers.py — Integration tests for broker scripts and signal parsing.

PLAIN ENGLISH:
Tests that Alpaca broker can connect, parse signals correctly, generate
valid orders, and that safety features (duplicate prevention, weight
scaling) work as expected.  Every test should take < 5 seconds.

Run:  python -m pytest tests/test_brokers.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _alpaca_package_or_skip() -> None:
    if importlib.util.find_spec("alpaca_trade_api") is None:
        pytest.skip("alpaca-trade-api not installed")


def test_halt_sentinel_uses_atomic_writer(monkeypatch, tmp_path):
    import alpaca_paper_trading as apt

    calls = []

    def fake_write_text(path, content, **_kwargs):
        calls.append((path.name, content))

    monkeypatch.setattr(apt, "atomic_write_text", fake_write_text)

    apt._write_halt_sentinel(
        tmp_path / "alpaca_halt_active.txt",
        now=datetime(2026, 6, 5, 14, tzinfo=timezone.utc),
    )

    assert calls
    assert calls[0][0] == "alpaca_halt_active.txt"
    assert "Emergency liquidation triggered at 2026-06-05T14:00:00+00:00" in calls[0][1]
    assert "Delete this file to re-enable trading" in calls[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PARSING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTargetWeights:
    """Test that parse_target_weights handles all signal formats correctly."""

    def test_basic_spy_qqq(self):
        """Standard core-satellite signal with SPY and QQQ only."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.30,
            "target_qqq_weight": 0.55,
            "overlay_weights_json": "{}",
        })
        weights = parse_target_weights(signal)
        assert "SPY" in weights
        assert "QQQ" in weights
        assert abs(weights["SPY"] - 0.30) < 1e-6
        assert abs(weights["QQQ"] - 0.55) < 1e-6
        assert len(weights) == 2

    def test_tqqq_signal(self):
        """TQQQ-enhanced signal includes target_tqqq_weight."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.0,
            "target_qqq_weight": 0.44,
            "target_tqqq_weight": 0.11,
            "overlay_weights_json": '{"AAPL": 0.15}',
        })
        weights = parse_target_weights(signal)
        assert "TQQQ" in weights
        assert abs(weights["TQQQ"] - 0.11) < 1e-6
        assert "QQQ" in weights
        assert "AAPL" in weights
        assert "SPY" not in weights  # weight is 0.0, should be excluded

    def test_zero_weights_excluded(self):
        """Weights of 0.0 should not appear in the result."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.0,
            "target_qqq_weight": 0.0,
            "target_tqqq_weight": 0.0,
            "overlay_weights_json": "{}",
        })
        weights = parse_target_weights(signal)
        assert len(weights) == 0

    def test_overlay_weights_parsed(self):
        """Overlay stock weights from JSON are correctly included."""
        from alpaca_paper_trading import parse_target_weights

        overlay = {"AAPL": 0.10, "MSFT": 0.08, "GOOGL": 0.07}
        signal = pd.Series({
            "target_spy_weight": 0.30,
            "target_qqq_weight": 0.40,
            "overlay_weights_json": json.dumps(overlay),
        })
        weights = parse_target_weights(signal)
        assert len(weights) == 5  # SPY, QQQ, AAPL, MSFT, GOOGL
        assert abs(weights["AAPL"] - 0.10) < 1e-6
        assert abs(weights["MSFT"] - 0.08) < 1e-6

    def test_missing_overlay_json(self):
        """Missing or NaN overlay_weights_json shouldn't crash."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.50,
            "target_qqq_weight": 0.50,
        })
        weights = parse_target_weights(signal)
        assert len(weights) == 2

    def test_malformed_overlay_json(self):
        """Malformed JSON should be handled gracefully (empty overlay)."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.50,
            "target_qqq_weight": 0.50,
            "overlay_weights_json": "not valid json {{{",
        })
        weights = parse_target_weights(signal)
        assert len(weights) == 2  # only SPY and QQQ

    def test_missing_tqqq_weight(self):
        """Old-format signals without target_tqqq_weight shouldn't crash."""
        from alpaca_paper_trading import parse_target_weights

        signal = pd.Series({
            "target_spy_weight": 0.40,
            "target_qqq_weight": 0.60,
            "overlay_weights_json": "{}",
        })
        # No target_tqqq_weight key at all
        weights = parse_target_weights(signal)
        assert "TQQQ" not in weights
        assert len(weights) == 2


class TestSignalSafetyParsing:
    def test_sanity_reads_current_signal_schema_without_blocking_core_etfs(self):
        """Core ETF weights can exceed stock caps, but overlay stocks cannot."""
        from signal_freshness import validate_signal_sanity

        signal = pd.Series({
            "target_spy_weight": 0.0,
            "target_qqq_weight": 0.55,
            "target_tqqq_weight": 0.0,
            "overlay_weights_json": json.dumps({"AAPL": 0.15, "MSFT": 0.10}),
        })

        ok, issues = validate_signal_sanity(signal, max_single_weight=0.25)

        assert ok is True
        assert issues == []

    def test_sanity_fails_closed_when_weights_are_missing(self):
        from signal_freshness import validate_signal_sanity

        ok, issues = validate_signal_sanity(pd.Series({}))

        assert ok is False
        assert "missing_target_weights" in issues

    def test_sanity_catches_overlay_concentration_in_current_schema(self):
        from signal_freshness import validate_signal_sanity

        signal = pd.Series({
            "target_qqq_weight": 0.40,
            "overlay_weights_json": json.dumps({"AAPL": 0.35}),
        })

        ok, issues = validate_signal_sanity(signal, max_single_weight=0.25)

        assert ok is False
        assert any(issue.startswith("AAPL_weight") for issue in issues)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SCALING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleWeights:
    """Test that scale_weights caps gross exposure correctly."""

    def test_no_scaling_needed(self):
        """Weights within limit should not be changed."""
        from alpaca_paper_trading import scale_weights

        weights = {"SPY": 0.30, "QQQ": 0.40, "AAPL": 0.10}
        scaled = scale_weights(weights, max_gross=1.0)
        for k in weights:
            assert abs(scaled[k] - weights[k]) < 1e-9

    def test_scales_down_when_over_limit(self):
        """Weights exceeding max_gross should be scaled proportionally."""
        from alpaca_paper_trading import scale_weights

        weights = {"SPY": 0.50, "QQQ": 0.50, "AAPL": 0.25}  # 1.25x gross
        scaled = scale_weights(weights, max_gross=1.0)
        gross = sum(abs(v) for v in scaled.values())
        assert gross <= 1.0 + 1e-9
        # Proportions should be preserved
        assert abs(scaled["SPY"] / scaled["QQQ"] - 1.0) < 1e-6

    def test_empty_weights(self):
        """Empty weights dict should return empty."""
        from alpaca_paper_trading import scale_weights

        scaled = scale_weights({}, max_gross=1.0)
        assert len(scaled) == 0

    def test_nonfinite_weights_are_dropped_before_scaling(self):
        from alpaca_paper_trading import scale_weights

        scaled = scale_weights({"SPY": 0.80, "BAD": float("inf"), "NAN": float("nan")}, max_gross=0.40)

        assert scaled == {"SPY": 0.40}

    def test_nonfinite_max_gross_returns_empty(self):
        from alpaca_paper_trading import scale_weights

        assert scale_weights({"SPY": 0.80}, max_gross=float("nan")) == {}


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA SIGNAL FRESHNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaSignalFreshness:
    def test_latest_completed_us_trading_day_skips_nyse_holiday(self):
        pytest.importorskip("exchange_calendars")
        from signal_freshness import latest_completed_us_trading_day

        now = datetime(2024, 6, 19, 22, 0, tzinfo=timezone.utc)

        assert latest_completed_us_trading_day(now=now) == pd.Timestamp("2024-06-18")

    def test_signal_freshness_allows_prior_session_on_nyse_holiday(self):
        pytest.importorskip("exchange_calendars")
        from signal_freshness import validate_signal_freshness

        now = datetime(2024, 6, 19, 22, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": now.isoformat(),
            "latest_factor_date": "2024-06-18",
        })

        ok, issues = validate_signal_freshness(
            signal,
            max_signal_age_hours=24,
            max_factor_age_trading_days=0,
            now=now,
        )

        assert ok is True
        assert issues == []

    def test_fresh_signal_passes(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": now.isoformat(),
            "latest_factor_date": "2026-05-08",
        })
        ok, issues = apt.check_signal_freshness(signal, now=now)
        assert ok is True
        assert issues == []

    def test_stale_predicted_at_fails_closed(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": "2026-05-07T20:00:00+00:00",
            "latest_factor_date": "2026-05-08",
        })
        with pytest.raises(RuntimeError, match="signal_age"):
            apt.check_signal_freshness(signal, now=now)

    def test_future_predicted_at_fails_closed(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": "2026-05-08T21:10:00+00:00",
            "latest_factor_date": "2026-05-08",
        })
        with pytest.raises(RuntimeError, match="signal_from_future"):
            apt.check_signal_freshness(signal, now=now)

    def test_stale_latest_factor_date_fails_closed(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": now.isoformat(),
            "latest_factor_date": "2026-04-24",
        })
        with pytest.raises(RuntimeError, match="factor_age"):
            apt.check_signal_freshness(signal, now=now)

    def test_missing_freshness_fields_fail_closed(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        with pytest.raises(RuntimeError, match="missing_predicted_at"):
            apt.check_signal_freshness(pd.Series({}), now=now)

    def test_allow_stale_signal_returns_issues_without_raising(self):
        import alpaca_paper_trading as apt

        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": "2026-05-07T20:00:00+00:00",
            "latest_factor_date": "2026-04-24",
        })
        ok, issues = apt.check_signal_freshness(
            signal,
            allow_stale_signal=True,
            now=now,
        )
        assert ok is False
        assert any(issue.startswith("signal_age") for issue in issues)
        assert any(issue.startswith("factor_age") for issue in issues)

    def test_live_config_hash_match_passes(self, tmp_path, monkeypatch):
        import alpaca_paper_trading as apt
        from signal_freshness import live_config_fingerprint

        payload = {
            "created_at": "2026-05-08T20:00:00+00:00",
            "source_json": "signals/core_satellite_nested_walkforward.json",
            "approved_live_configs": {
                "core-alpha": {
                    "approved_config_family": "family-a",
                    "approved_family_signature": "family-a",
                    "approved_exact_config": "h=20,ov=0.5",
                    "config": {"holding_days": 20, "risk_control_mode": "off"},
                }
            },
        }
        live_path = tmp_path / "core_satellite_live_configs.json"
        live_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(apt, "LIVE_CONFIG_FILE", live_path)
        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": now.isoformat(),
            "latest_factor_date": "2026-05-08",
            "live_config_hash": live_config_fingerprint(payload),
            "live_config_created_at": payload["created_at"],
        })

        ok, issues = apt.check_signal_freshness(signal, now=now, check_live_config_match=True)

        assert ok is True
        assert issues == []

    def test_live_config_hash_mismatch_fails_closed(self, tmp_path, monkeypatch):
        import alpaca_paper_trading as apt

        payload = {
            "created_at": "2026-05-08T20:00:00+00:00",
            "approved_live_configs": {
                "core-alpha": {
                    "approved_config_family": "family-a",
                    "approved_exact_config": "h=20,ov=0.5",
                    "config": {"holding_days": 20},
                }
            },
        }
        live_path = tmp_path / "core_satellite_live_configs.json"
        live_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(apt, "LIVE_CONFIG_FILE", live_path)
        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": now.isoformat(),
            "latest_factor_date": "2026-05-08",
            "live_config_hash": "oldbadbadbadbad",
            "live_config_created_at": "2026-05-07T20:00:00+00:00",
        })

        with pytest.raises(RuntimeError, match="live_config_hash_mismatch"):
            apt.check_signal_freshness(signal, now=now, check_live_config_match=True)

    def test_allow_stale_signal_does_not_bypass_live_config_mismatch(self, tmp_path, monkeypatch):
        import alpaca_paper_trading as apt

        payload = {
            "created_at": "2026-05-08T20:00:00+00:00",
            "approved_live_configs": {
                "core-alpha": {
                    "approved_exact_config": "h=20,ov=0.5",
                    "config": {"holding_days": 20},
                }
            },
        }
        live_path = tmp_path / "core_satellite_live_configs.json"
        live_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(apt, "LIVE_CONFIG_FILE", live_path)
        now = datetime(2026, 5, 8, 21, 0, tzinfo=timezone.utc)
        signal = pd.Series({
            "predicted_at": "2026-05-07T20:00:00+00:00",
            "latest_factor_date": "2026-04-24",
            "live_config_hash": "oldbadbadbadbad",
        })

        with pytest.raises(RuntimeError, match="Signal/live config mismatch"):
            apt.check_signal_freshness(
                signal,
                now=now,
                allow_stale_signal=True,
                check_live_config_match=True,
            )


def test_alpaca_load_signal_requires_medium_risk_review(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    signal_path = tmp_path / "core_satellite_alpha_signal.csv"
    pd.DataFrame([{
        "paper_ready": True,
        "gates_all_pass": True,
        "medium_risk_review_pass": False,
        "reason": "old signal",
    }]).to_csv(signal_path, index=False)
    monkeypatch.setattr(apt, "SIGNAL_FILE", signal_path)

    with pytest.raises(RuntimeError, match="Medium-risk review"):
        apt.load_signal()


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE ORDER PREVENTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class _OrderBroker:
    def __init__(self, *, equity=100_000.0, positions=None, prices=None, sellable=None):
        self._equity = equity
        self._positions = dict(positions or {})
        self._prices = dict(prices or {})
        self._sellable = dict(sellable or {})

    def get_equity(self):
        return self._equity

    def get_position_map(self):
        return dict(self._positions)

    def get_last_price(self, ticker):
        return float(self._prices.get(ticker, 100.0))

    def get_sellable_qty(self, ticker):
        return int(self._sellable.get(ticker, self._positions.get(ticker, 0)))


def test_generate_orders_rejects_nonpositive_broker_equity():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(equity=0.0, prices={"SPY": 500.0})

    with pytest.raises(RuntimeError, match="Invalid broker equity"):
        apt.generate_orders(broker, {"SPY": 0.5}, force=True)


def test_generate_orders_rejects_nonfinite_broker_equity():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(equity=float("nan"), prices={"SPY": 500.0})

    with pytest.raises(RuntimeError, match="Invalid broker equity"):
        apt.generate_orders(broker, {"SPY": 0.5}, force=True)


def test_generate_orders_skips_nonfinite_prices():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(equity=100_000.0, prices={"SPY": float("inf")})

    assert apt.generate_orders(broker, {"SPY": 0.5}, force=True) == []


def test_experimental_live_risk_cap_is_off_by_default(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "LIVE_RISK_CAP_ENABLED", False)
    monkeypatch.setattr(apt, "EXECUTION_RISK_ENABLED", False)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", False)
    broker = _OrderBroker(equity=100_000.0, prices={"MU": 100.0})

    orders = apt.generate_orders(broker, {"MU": 0.20}, force=True)

    assert orders[0]["quantity"] == 200
    assert orders[0]["live_risk_cap_enabled"] is False
    assert orders[0]["live_risk_signal_target_qty"] == 200
    assert orders[0]["live_risk_capped_target_qty"] == 200


def test_experimental_live_risk_cap_can_only_reduce_buy(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    close = pd.Series(np.linspace(90.0, 100.0, len(dates)), index=dates)
    pd.DataFrame({"High": close + 5.0, "Low": close - 5.0, "Close": close}).to_parquet(
        tmp_path / "MU.parquet"
    )
    monkeypatch.setattr(apt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(apt, "LIVE_RISK_CAP_ENABLED", True)
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(apt, "EXECUTION_RISK_ENABLED", False)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", False)
    broker = _OrderBroker(equity=100_000.0, positions={"MU": 20}, prices={"MU": 100.0})

    orders = apt.generate_orders(broker, {"MU": 0.20}, force=True)

    # ATR is $10, so risking 1% with a 2×ATR stop permits 50 total shares.
    assert orders[0]["side"] == "buy"
    assert orders[0]["quantity"] == 30
    assert orders[0]["live_risk_signal_target_qty"] == 200
    assert orders[0]["live_risk_capped_target_qty"] == 50
    assert orders[0]["target_qty"] == 50
    assert orders[0]["live_risk_reason"].endswith("atr_qty=50")


def test_experimental_live_risk_cap_never_turns_buy_into_sell(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "LIVE_RISK_CAP_ENABLED", True)
    monkeypatch.setattr(apt, "_overlay_risk_cap", lambda *args, **kwargs: (
        kwargs["current_qty"],
        {
            "live_risk_cap_enabled": True,
            "live_risk_signal_target_qty": kwargs["signal_target_qty"],
            "live_risk_capped_target_qty": kwargs["current_qty"],
            "live_risk_realized_vol": 0.5,
            "live_risk_atr_14": 10.0,
            "live_risk_reason": "cap_below_holding",
        },
    ))
    broker = _OrderBroker(equity=100_000.0, positions={"MU": 100}, prices={"MU": 100.0})

    assert apt.generate_orders(broker, {"MU": 0.20}, force=True) == []


def test_experimental_live_risk_cap_refuses_real_endpoint(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "LIVE_RISK_CAP_ENABLED", True)
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://api.alpaca.markets")
    broker = _OrderBroker(equity=100_000.0, prices={"MU": 100.0})

    with pytest.raises(RuntimeError, match="only on Alpaca paper"):
        apt.generate_orders(broker, {"MU": 0.20}, force=True)


def test_generate_orders_adds_marketable_limit_prices():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(equity=100_000.0, positions={"SPY": 100}, prices={"SPY": 500.0})

    orders = apt.generate_orders(broker, {"SPY": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["limit_price"] < orders[0]["price"]
    assert orders[0]["raw_limit_price"] < orders[0]["price"]


def test_generate_orders_clamps_sell_to_broker_sellable_quantity():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(
        equity=100_000.0,
        positions={"FCX": 302},
        prices={"FCX": 64.665},
        sellable={"FCX": 136},
    )

    orders = apt.generate_orders(broker, {"FCX": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["quantity"] == 136
    assert orders[0]["requested_quantity"] == 302
    assert orders[0]["broker_sellable_qty"] == 136
    assert orders[0]["quantity_clamped_to_sellable"] is True


def test_generate_orders_skips_sell_when_broker_reports_zero_sellable():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(
        equity=100_000.0,
        positions={"FCX": 302},
        prices={"FCX": 64.665},
        sellable={"FCX": 0},
    )

    assert apt.generate_orders(broker, {"FCX": 0.0}, force=True) == []


def test_generate_orders_counts_shares_reserved_by_bot_trailing_stop(monkeypatch):
    import alpaca_paper_trading as apt

    broker = _OrderBroker(
        equity=100_000.0,
        positions={"FCX": 302},
        prices={"FCX": 64.665},
        sellable={"FCX": 0},
    )
    protective_stop = SimpleNamespace(
        symbol="FCX",
        side="sell",
        type="trailing_stop",
        qty="302",
    )
    monkeypatch.setattr(apt, "list_open_orders", lambda _broker: [protective_stop])

    orders = apt.generate_orders(broker, {"FCX": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["quantity"] == 302
    assert orders[0]["broker_sellable_qty"] == 302
    assert orders[0]["broker_reserved_stop_qty"] == 302
    assert orders[0]["quantity_clamped_to_sellable"] is False


def test_execution_quality_summary_separates_minor_and_material_bad_slippage(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_BAD_SLIPPAGE_BPS", 2.0)

    summary = apt._execution_quality_summary([
        {"slippage_bps": -1.0, "worst_adverse_60m_bps": 0.0},
        {"slippage_bps": 0.5, "worst_adverse_60m_bps": 0.0},
        {"slippage_bps": 2.5, "worst_adverse_60m_bps": 0.0},
        {"slippage_bps": 6.0, "worst_adverse_60m_bps": 0.0},
    ])

    assert summary["slippage_bad_threshold_bps"] == 2.0
    assert summary["slippage_bad_count"] == 2
    assert summary["raw_slippage_bad_count"] == 3
    assert summary["minor_bad_slippage_count"] == 1


def test_generate_orders_throttles_high_risk_overlay_buys(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    report_path = tmp_path / "alpaca_slippage_reversal_report.json"
    report_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "by_symbol": [{
            "symbol": "MU",
            "orders": 5,
            "avg_slippage_bps": 12.0,
            "bad_slippage_rate": 1.0,
            "execution_risk_score": 65.0,
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "SLIPPAGE_REPORT_FILE", report_path)
    monkeypatch.setattr(apt, "EXECUTION_RISK_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_RISK_REPORT_MAX_AGE_HOURS", 168)
    monkeypatch.setattr(apt, "EXECUTION_RISK_WARN_SCORE", 40)
    monkeypatch.setattr(apt, "EXECUTION_RISK_HIGH_SCORE", 60)
    monkeypatch.setattr(apt, "EXECUTION_RISK_HIGH_BUY_SCALE", 0.50)

    broker = _OrderBroker(equity=100_000.0, positions={}, prices={"MU": 100.0})

    orders = apt.generate_orders(broker, {"MU": 0.20}, force=True)

    assert orders[0]["side"] == "buy"
    assert orders[0]["requested_quantity"] == 200
    assert orders[0]["quantity"] == 200
    assert orders[0]["execution_risk_score"] == 65.0
    assert orders[0]["execution_risk_band"] == "high"
    assert orders[0]["execution_risk_buy_scale"] == 1.0
    assert orders[0]["execution_risk_quantity_before_scale"] == 200
    assert orders[0]["execution_risk_quantity_after_scale"] == 200
    assert orders[0]["execution_risk_reason"] == ""


def test_generate_orders_keeps_high_risk_sells_exit_ready(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    report_path = tmp_path / "alpaca_slippage_reversal_report.json"
    report_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "by_symbol": [{"symbol": "MU", "orders": 5, "execution_risk_score": 65.0}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "SLIPPAGE_REPORT_FILE", report_path)
    monkeypatch.setattr(apt, "EXECUTION_RISK_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_RISK_HIGH_SCORE", 60)
    monkeypatch.setattr(apt, "EXECUTION_RISK_HIGH_BUY_SCALE", 0.50)

    broker = _OrderBroker(equity=100_000.0, positions={"MU": 200}, prices={"MU": 100.0})

    orders = apt.generate_orders(broker, {"MU": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["requested_quantity"] == 200
    assert orders[0]["quantity"] == 200
    assert orders[0]["execution_risk_band"] == "high"
    assert orders[0]["execution_risk_reason"] == ""


def test_generate_orders_ignores_stale_execution_risk_report(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    report_path = tmp_path / "alpaca_slippage_reversal_report.json"
    report_path.write_text(json.dumps({
        "generated_at": "2020-01-01T00:00:00+00:00",
        "by_symbol": [{"symbol": "MU", "execution_risk_score": 99.0}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "SLIPPAGE_REPORT_FILE", report_path)
    monkeypatch.setattr(apt, "EXECUTION_RISK_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_RISK_REPORT_MAX_AGE_HOURS", 1)
    # This test isolates the per-symbol report. A fresh project-wide
    # execution scorecard from a real paper run must not change its result.
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", False)

    broker = _OrderBroker(equity=100_000.0, positions={}, prices={"MU": 100.0})

    orders = apt.generate_orders(broker, {"MU": 0.20}, force=True)

    assert orders[0]["quantity"] == 200
    assert orders[0]["execution_risk_score"] == ""
    assert orders[0]["execution_risk_band"] == "none"


def test_generate_orders_throttles_buys_when_execution_scorecard_fails(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    scorecard_path = tmp_path / "alpaca_execution_scorecard.json"
    scorecard_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "score": 66.67,
        "decision_eligible": True,
        "checks": [{"name": "avg_slippage_bps", "status": "fail"}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_FILE", scorecard_path)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_MAX_AGE_HOURS", 72)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_FAIL_BUY_SCALE", 0.75)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_SEVERE_SCORE", 50)

    broker = _OrderBroker(equity=100_000.0, positions={}, prices={"QQQ": 100.0})

    orders = apt.generate_orders(broker, {"QQQ": 0.60}, force=True)

    assert orders[0]["side"] == "buy"
    assert orders[0]["requested_quantity"] == 600
    assert orders[0]["quantity"] == 600
    assert orders[0]["execution_risk_buy_scale"] == 1.0
    assert orders[0]["execution_scorecard_status"] == "fail"
    assert orders[0]["execution_scorecard_score"] == 66.67
    assert orders[0]["execution_scorecard_buy_scale"] == 1.0
    assert orders[0]["execution_scorecard_failed_checks"] == "avg_slippage_bps"
    assert orders[0]["execution_risk_reason"] == ""

    scorecard_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "score": 20.0,
        "decision_eligible": True,
        "checks": [{"name": "fill_rate", "status": "fail"}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_SEVERE_BUY_SCALE", 0.50)

    severe_orders = apt.generate_orders(broker, {"QQQ": 0.60}, force=True)

    assert severe_orders[0]["quantity"] == 600
    assert severe_orders[0]["execution_scorecard_buy_scale"] == 1.0
    assert severe_orders[0]["execution_risk_reason"] == ""


def test_generate_orders_keeps_scorecard_fail_sells_exit_ready(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    scorecard_path = tmp_path / "alpaca_execution_scorecard.json"
    scorecard_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "score": 20.0,
        "decision_eligible": True,
        "checks": [{"name": "fill_rate", "status": "fail"}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_FILE", scorecard_path)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_SEVERE_BUY_SCALE", 0.50)

    broker = _OrderBroker(equity=100_000.0, positions={"QQQ": 600}, prices={"QQQ": 100.0})

    orders = apt.generate_orders(broker, {"QQQ": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["requested_quantity"] == 600
    assert orders[0]["quantity"] == 600
    assert orders[0]["execution_scorecard_status"] == "fail"
    assert orders[0]["execution_risk_reason"] == ""


def test_generate_orders_ignores_stale_execution_scorecard(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    scorecard_path = tmp_path / "alpaca_execution_scorecard.json"
    scorecard_path.write_text(json.dumps({
        "generated_at": "2020-01-01T00:00:00+00:00",
        "status": "fail",
        "score": 0.0,
        "checks": [{"name": "fill_rate", "status": "fail"}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_FILE", scorecard_path)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", True)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_MAX_AGE_HOURS", 1)

    broker = _OrderBroker(equity=100_000.0, positions={}, prices={"QQQ": 100.0})

    orders = apt.generate_orders(broker, {"QQQ": 0.60}, force=True)

    assert orders[0]["quantity"] == 600
    assert orders[0]["execution_scorecard_status"] == "stale"
    assert orders[0]["execution_scorecard_buy_scale"] == 1.0
    assert orders[0]["execution_risk_reason"] == ""
    assert orders[0]["execution_scorecard_reason"] == "execution_scorecard_stale_sizing_unchanged"


def test_generate_orders_ignores_ineligible_execution_scorecard(tmp_path, monkeypatch):
    """Thin or immature execution evidence must never shrink a new buy."""
    import alpaca_paper_trading as apt

    scorecard_path = tmp_path / "alpaca_execution_scorecard.json"
    scorecard_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "score": 0.0,
        "decision_eligible": False,
        "decision_blockers": ["adverse_60m_sample_0_below_20"],
        "checks": [{"name": "adverse_60m_rate", "status": "fail"}],
    }), encoding="utf-8")
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_FILE", scorecard_path)
    monkeypatch.setattr(apt, "EXECUTION_SCORECARD_THROTTLE_ENABLED", True)

    broker = _OrderBroker(equity=100_000.0, positions={}, prices={"QQQ": 100.0})
    orders = apt.generate_orders(broker, {"QQQ": 0.60}, force=True)

    assert orders[0]["quantity"] == 600
    assert orders[0]["execution_scorecard_status"] == "collecting"
    assert orders[0]["execution_scorecard_buy_scale"] == 1.0


def test_build_submission_order_defaults_to_limit():
    import alpaca_paper_trading as apt

    planned = {
        "ticker": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 100.0,
        "limit_price": 100.12,
    }

    order = apt.build_submission_order(planned, use_market_order=False, client_id="test-1")

    assert order.type == "limit"
    assert order.limit_price == 100.12
    assert planned["submitted_order_type"] == "limit"
    assert planned["submitted_limit_price"] == 100.12


def test_build_submission_order_can_force_market():
    import alpaca_paper_trading as apt

    planned = {"ticker": "AAPL", "side": "buy", "quantity": 10, "price": 100.0}

    order = apt.build_submission_order(planned, use_market_order=True, client_id="test-2")

    assert order.type == "market"
    assert order.limit_price is None
    assert planned["submitted_order_type"] == "market"


def test_alpaca_broker_maps_fixed_stop_to_gtc_stop_order():
    import alpaca_paper_trading as apt
    from broker_interface import Order

    submitted = {}
    broker = apt.AlpacaBroker.__new__(apt.AlpacaBroker)
    broker._api = SimpleNamespace(
        submit_order=lambda **kwargs: submitted.update(kwargs) or SimpleNamespace(id="stop-1")
    )

    oid = broker.place_order(Order(
        ticker="MU",
        side="sell",
        quantity=10,
        type="stop",
        stop_price=91.234,
    ))

    assert oid == "stop-1"
    assert submitted["type"] == "stop"
    assert submitted["stop_price"] == 91.23
    assert submitted["time_in_force"] == "gtc"


def test_quote_based_limit_is_available_for_submission():
    import alpaca_paper_trading as apt

    planned = {
        "ticker": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 100.0,
        "limit_price": 100.05,
        "bid_price": 99.95,
        "ask_price": 100.20,
    }

    order = apt.build_submission_order(
        planned,
        use_market_order=False,
        use_quote_limit=True,
        client_id="test-3",
    )

    assert order.type == "limit"
    assert order.limit_price > 100.20
    assert order.limit_price != 100.05
    assert planned["submitted_limit_reference"] == "quote"


def test_quote_based_limit_falls_back_to_last_price():
    import alpaca_paper_trading as apt

    price = apt.quote_based_limit_price(
        bid_price=None,
        ask_price=None,
        fallback_price=100.0,
        side="sell",
        ticker="AAPL",
        limit_offset_bps=10,
    )

    assert price < 100.0


def test_quote_limits_are_default_but_can_be_disabled(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "LIMIT_REFERENCE", "quote")

    assert apt.should_use_quote_limits(SimpleNamespace(quote_limit=False, last_trade_limit=False)) is True
    assert apt.should_use_quote_limits(SimpleNamespace(quote_limit=False, last_trade_limit=True)) is False


def test_daily_action_uses_quote_limit_reference():
    workflow = Path(".github/workflows/daily_paper_trading.yml").read_text(encoding="utf-8")

    assert "ALPACA_ORDER_TYPE=limit" in workflow
    assert "ALPACA_LIMIT_REFERENCE=quote" in workflow
    assert "ALPACA_LIMIT_REFERENCE=last" not in workflow


def test_market_orders_require_explicit_env_unlock(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "DEFAULT_ORDER_TYPE", "limit")
    monkeypatch.setattr(apt, "ALLOW_MARKET_ORDER_OVERRIDE", False)

    assert apt.should_use_market_orders(SimpleNamespace(market_order=True, limit_order=False)) is False

    monkeypatch.setattr(apt, "ALLOW_MARKET_ORDER_OVERRIDE", True)
    assert apt.should_use_market_orders(SimpleNamespace(market_order=True, limit_order=False)) is False


def test_market_order_env_default_is_ignored_until_unlocked(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "DEFAULT_ORDER_TYPE", "market")
    monkeypatch.setattr(apt, "ALLOW_MARKET_ORDER_OVERRIDE", False)

    assert apt.should_use_market_orders(SimpleNamespace(market_order=False, limit_order=False)) is False


def test_execution_window_uses_new_york_time(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_WINDOW_START", "09:35")
    monkeypatch.setattr(apt, "EXECUTION_WINDOW_END", "10:30")

    assert apt.execution_window_allowed(datetime(2026, 6, 8, 14, 0, tzinfo=timezone.utc)) is True
    assert apt.execution_window_allowed(datetime(2026, 6, 8, 19, 0, tzinfo=timezone.utc)) is False


def test_two_stage_client_ids_are_deterministic_and_distinct():
    import alpaca_paper_trading as apt

    row = {"ticker": "MU", "side": "buy", "quantity": 10}
    day = datetime(2026, 6, 8, tzinfo=timezone.utc)

    assert apt.bot_client_order_id(row, today=day, attempt=1).endswith("-a1")
    assert apt.bot_client_order_id(row, today=day, attempt=2).endswith("-a2")
    assert apt.bot_client_order_id(row, today=day, attempt=1) != apt.bot_client_order_id(row, today=day, attempt=2)


def test_scorecard_changes_price_policy_not_quantity(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 15)
    row = {
        "ticker": "MU",
        "execution_scorecard_status": "fail",
        "execution_scorecard_failed_checks": "avg_slippage_bps,bad_slippage_rate",
    }

    wait_seconds, offset_bps, policy = apt._two_stage_price_policy(row)

    assert wait_seconds == 30
    assert offset_bps == 1
    assert policy == "price_failure_more_patient"


class _SubmitBroker:
    """Tiny broker fake for submit-phase guard tests."""

    def __init__(self, *, cash=1_000.0, equity=None, buying_power=None, fail_tickers=None, statuses=None):
        self.cash = cash
        self.equity = cash if equity is None else equity
        self.buying_power = cash if buying_power is None else buying_power
        self.fail_tickers = set(fail_tickers or [])
        self.statuses = dict(statuses or {})
        self.orders = []
        self._api = SimpleNamespace(get_order=self.get_order)

    def place_order(self, order):
        if order.ticker in self.fail_tickers:
            raise RuntimeError(f"{order.ticker} rejected")
        self.orders.append(order)
        return f"{order.side}-{order.ticker}-{len(self.orders)}"

    def get_order(self, oid):
        return SimpleNamespace(status=self.statuses.get(oid, "filled"))

    def get_cash(self):
        return float(self.cash)

    def get_equity(self):
        return float(self.equity)

    def get_buying_power(self):
        return float(self.buying_power)


class _TwoStageBroker:
    """Broker fake that exposes quote, partial-fill, and cancellation behavior."""

    def __init__(self, *, first_filled=0, cancel_ok=True, second_spread=0.001, fill_on_cancel=False):
        self.first_filled = int(first_filled)
        self.cancel_ok = bool(cancel_ok)
        self.second_spread = float(second_spread)
        self.fill_on_cancel = bool(fill_on_cancel)
        self.cancelled = False
        self.orders = []
        self.quote_calls = 0
        self._api = SimpleNamespace(get_order=self.get_order)

    def get_quote_snapshot(self, ticker):
        self.quote_calls += 1
        spread = 0.001 if self.quote_calls == 1 else self.second_spread
        return {
            "bid_price": 99.9,
            "ask_price": 100.1,
            "quote_mid_price": 100.0,
            "spread_pct": spread,
        }

    def place_order(self, order):
        self.orders.append(order)
        return f"oid-{len(self.orders)}"

    def get_order(self, oid):
        if oid == "oid-1":
            if self.first_filled >= self.orders[0].quantity:
                return SimpleNamespace(status="filled", filled_qty=self.first_filled, filled_avg_price="100.0")
            status = "canceled" if self.cancelled else ("partially_filled" if self.first_filled else "new")
            return SimpleNamespace(status=status, filled_qty=self.first_filled, filled_avg_price="100.0" if self.first_filled else None)
        return SimpleNamespace(status="new", filled_qty=0, filled_avg_price=None)

    def cancel_order(self, oid):
        if self.cancel_ok:
            if self.fill_on_cancel:
                self.first_filled = self.orders[0].quantity
            self.cancelled = True
        return self.cancel_ok


def _planned_order(ticker, side, quantity=1):
    return {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "price": 100.0,
        "limit_price": 100.10 if side == "buy" else 99.90,
        "trade_value": quantity * 100.0,
        "target_weight": 0.10,
    }


def test_two_stage_limit_finishes_at_passive_midpoint(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    broker = _TwoStageBroker(first_filled=10)
    row = _planned_order("MU", "buy", quantity=10)

    oid = apt._submit_two_stage_limit(broker, row)

    assert oid == "oid-1"
    assert len(broker.orders) == 1
    assert broker.orders[0].limit_price == 100.0
    assert broker.orders[0].client_id.endswith("-a1")
    assert row["execution_stage"] == "stage1"
    assert row["fill_status"] == "filled"


def test_two_stage_limit_reprices_only_unfilled_remainder(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "EXECUTION_CANCEL_WAIT_SECONDS", 0)
    broker = _TwoStageBroker(first_filled=4)
    row = _planned_order("MU", "buy", quantity=10)

    oid = apt._submit_two_stage_limit(broker, row)

    assert oid == "oid-2"
    assert [order.quantity for order in broker.orders] == [10, 6]
    assert broker.orders[1].client_id.endswith("-a2")
    assert broker.orders[1].type == "limit"
    assert row["stage1_cancel_status"] == "canceled"
    assert row["remaining_quantity"] == 6
    assert row["execution_stage"] == "stage2"


def test_two_stage_limit_does_not_replace_uncertain_cancel(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "EXECUTION_CANCEL_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    broker = _TwoStageBroker(cancel_ok=False)
    row = _planned_order("MU", "buy", quantity=10)

    oid = apt._submit_two_stage_limit(broker, row)

    assert oid == "oid-1"
    assert len(broker.orders) == 1
    assert row["execution_stage"] == "cancel_uncertain"


def test_two_stage_limit_handles_cancel_fill_race(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "EXECUTION_CANCEL_WAIT_SECONDS", 0)
    broker = _TwoStageBroker(fill_on_cancel=True)
    row = _planned_order("MU", "buy", quantity=10)

    oid = apt._submit_two_stage_limit(broker, row)

    assert oid == "oid-1"
    assert len(broker.orders) == 1
    assert row["execution_stage"] == "stage1_cancel_fill_race"
    assert row["fill_status"] == "filled"


def test_two_stage_limit_blocks_wide_refreshed_quote(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "EXECUTION_CANCEL_WAIT_SECONDS", 0)
    monkeypatch.setattr(apt, "MAX_SPREAD_PCT_OVERLAY", 0.005)
    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    broker = _TwoStageBroker(second_spread=0.02)
    row = _planned_order("MU", "buy", quantity=10)

    oid = apt._submit_two_stage_limit(broker, row)

    assert oid == "oid-1"
    assert len(broker.orders) == 1
    assert row["execution_stage"] == "stage2_blocked"


def test_broker_truth_gate_blocks_buys_on_fail_and_keeps_sells(monkeypatch):
    import alpaca_paper_trading as apt

    alerts = []
    monkeypatch.setattr(apt, "BROKER_TRUTH_GATE_ENABLED", True)
    monkeypatch.setattr(apt, "BROKER_TRUTH_BLOCK_BUYS_ON_FAIL", True)
    monkeypatch.setattr(
        apt,
        "_write_broker_truth_gate_report",
        lambda: {
            "status": "fail",
            "score": 65.0,
            "summary": {"fail_count": 1, "warning_count": 0},
            "global_issues": [{"severity": "fail", "issue": "broker_status_missing"}],
            "rows": [{"ticker": "MU", "issue_severity": "fail", "issues": "target_missing"}],
        },
    )
    monkeypatch.setattr(
        apt,
        "_send_submit_guard_alert",
        lambda title, message, priority="warning": alerts.append((title, message, priority)),
    )
    sell = _planned_order("FCX", "sell")
    buy = _planned_order("MU", "buy")

    remaining, skipped, ids = apt._apply_broker_truth_gate([sell, buy])

    assert remaining == [sell]
    assert skipped == [buy]
    assert ids == ["SKIPPED: broker_truth_fail"]
    assert buy["fill_status"] == "skipped"
    assert buy["submitted_limit_reference"] == "broker_truth_fail"
    assert alerts and alerts[0][0] == "Broker truth gate blocked buys"
    assert alerts[0][2] == "critical"


def test_broker_truth_gate_warning_allows_orders(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "BROKER_TRUTH_GATE_ENABLED", True)
    monkeypatch.setattr(apt, "BROKER_TRUTH_BLOCK_BUYS_ON_FAIL", True)
    monkeypatch.setattr(
        apt,
        "_write_broker_truth_gate_report",
        lambda: {
            "status": "warning",
            "score": 90.0,
            "summary": {"fail_count": 0, "warning_count": 2},
            "global_issues": [{"severity": "warning", "issue": "paper_log_stale_vs_broker_status"}],
            "rows": [],
        },
    )
    order = _planned_order("MU", "buy")

    remaining, skipped, ids = apt._apply_broker_truth_gate([order])

    assert remaining == [order]
    assert skipped == []
    assert ids == []
    assert "fill_status" not in order


class _QuoteBroker:
    """Tiny broker fake for quote/spread guard tests."""

    def __init__(self, quotes):
        self.quotes = quotes

    def get_quote_snapshot(self, ticker):
        return self.quotes.get(str(ticker).upper(), {
            "bid_price": None,
            "ask_price": None,
            "quote_mid_price": None,
            "spread_pct": None,
        })


def test_apply_spread_guard_logs_wide_spread_skip(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "MAX_SPREAD_PCT_OVERLAY", 0.01)
    monkeypatch.setattr(apt, "REQUIRE_QUOTE_FOR_SUBMIT", True)
    alerts = []
    monkeypatch.setattr(
        apt,
        "_send_deduped_submit_guard_alert",
        lambda title, message, **kwargs: alerts.append((title, message, kwargs)) or True,
    )
    broker = _QuoteBroker({
        "MU": {"bid_price": 100.0, "ask_price": 103.0, "quote_mid_price": 101.5, "spread_pct": 0.02956},
    })
    order = _planned_order("MU", "buy")

    remaining, skipped, ids = apt._apply_spread_guard(broker, [order])

    assert remaining == []
    assert skipped == [order]
    assert ids[0].startswith("SKIPPED: spread_guard")
    assert order["fill_status"] == "skipped"
    assert order["submitted_order_type"] == "skipped"
    assert alerts and alerts[0][0] == "Spread Guard"


def test_apply_spread_guard_blocks_missing_quote_when_required(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "REQUIRE_QUOTE_FOR_SUBMIT", True)
    monkeypatch.setattr(apt, "_send_deduped_submit_guard_alert", lambda *args, **kwargs: True)
    broker = _QuoteBroker({})
    order = _planned_order("MU", "buy")

    remaining, skipped, ids = apt._apply_spread_guard(broker, [order])

    assert remaining == []
    assert skipped == [order]
    assert ids == ["SKIPPED: quote_unavailable"]


def test_apply_spread_guard_blocks_stale_timestamped_quote(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "EXECUTION_QUOTE_MAX_AGE_SECONDS", 5)
    monkeypatch.setattr(apt, "_send_deduped_submit_guard_alert", lambda *args, **kwargs: True)
    broker = _QuoteBroker({
        "MU": {
            "bid_price": 99.9,
            "ask_price": 100.1,
            "quote_mid_price": 100.0,
            "spread_pct": 0.002,
            "quote_timestamp": "2020-01-01T00:00:00Z",
        }
    })
    order = _planned_order("MU", "buy")

    remaining, skipped, ids = apt._apply_spread_guard(broker, [order])

    assert remaining == []
    assert skipped == [order]
    assert ids[0].startswith("SKIPPED: quote_stale")


def test_spread_guard_alert_is_deduped(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    sent = []
    monkeypatch.setattr(apt, "ALERT_DEDUPE_FILE", tmp_path / "notification_dedupe.json")
    monkeypatch.setattr(
        apt,
        "_send_submit_guard_alert",
        lambda title, message, priority="warning": sent.append((title, message, priority)),
    )

    first = apt._send_deduped_submit_guard_alert(
        "Spread Guard",
        "Spread/quote guard blocked 1 orders: MU",
        alert_key="spread_guard:test",
        ttl_hours=20,
    )
    second = apt._send_deduped_submit_guard_alert(
        "Spread Guard",
        "Spread/quote guard blocked 1 orders: MU",
        alert_key="spread_guard:test",
        ttl_hours=20,
    )

    assert first is True
    assert second is False
    assert len(sent) == 1


def test_force_does_not_allow_closed_market_queue(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "ALLOW_CLOSED_MARKET_QUEUE", False)

    assert apt.closed_market_queue_allowed(
        SimpleNamespace(force=True, allow_closed_market_queue=False)
    ) is False
    assert apt.closed_market_queue_allowed(
        SimpleNamespace(force=False, allow_closed_market_queue=True)
    ) is True


def test_submit_rebalance_orders_skips_buys_when_sell_submission_fails(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    broker = _SubmitBroker(fail_tickers={"FCX"})
    orders = [_planned_order("FCX", "sell"), _planned_order("MU", "buy")]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["FCX", "MU"]
    assert order_ids[0].startswith("ERROR")
    assert order_ids[1].startswith("SKIPPED: sell_submission_failed")
    assert [o.ticker for o in broker.orders] == []


def test_submit_rebalance_orders_skips_buys_when_sell_not_filled(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SELL_FILL_WAIT_SECONDS", 0)
    broker = _SubmitBroker(statuses={"sell-FCX-1": "new"})
    orders = [_planned_order("FCX", "sell"), _planned_order("MU", "buy")]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["FCX", "MU"]
    assert order_ids[0] == "sell-FCX-1"
    assert order_ids[1].startswith("SKIPPED: sell_not_filled")
    assert [o.ticker for o in broker.orders] == ["FCX"]


def test_submit_rebalance_orders_skips_buys_when_cash_negative(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    broker = _SubmitBroker(cash=-5.0)
    orders = [_planned_order("MU", "buy")]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["MU"]
    assert order_ids == ["SKIPPED: cash_below_threshold:-5.00"]
    assert broker.orders == []


def test_submit_rebalance_orders_skips_buys_when_cash_check_errors(monkeypatch):
    """Unknown cash is unsafe, so broker read failure must never allow a buy."""
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    broker = _SubmitBroker(cash=1_000.0)

    def fail_cash_read():
        raise RuntimeError("account endpoint unavailable")

    broker.get_cash = fail_cash_read
    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        [_planned_order("MU", "buy")],
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [order["ticker"] for order in submitted] == ["MU"]
    assert order_ids == ["SKIPPED: cash_safety_check_failed:RuntimeError"]
    assert broker.orders == []


def test_drawdown_check_fails_closed_when_evidence_is_unreadable(tmp_path, monkeypatch):
    """Broken drawdown evidence must raise instead of pretending drawdown is zero."""
    import alpaca_paper_trading as apt

    equity_path = tmp_path / "equity.csv"
    equity_path.write_text("equity\n100000\n", encoding="utf-8")
    monkeypatch.setattr(apt, "EQUITY_FILE", equity_path)

    class BrokenBroker:
        def get_equity(self):
            raise RuntimeError("account unavailable")

    with pytest.raises(RuntimeError, match="Could not verify portfolio drawdown"):
        apt.check_portfolio_drawdown(BrokenBroker())


def test_abort_repair_attempts_both_core_and_overlay_stops(monkeypatch):
    """A partial pre-submit cancellation must immediately restore protection."""
    import alpaca_paper_trading as apt

    calls = []
    monkeypatch.setattr(apt, "CORE_PROTECTION_ENABLED", True)
    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(
        apt,
        "repair_core_etf_protective_stops",
        lambda broker, tickers, logger: calls.append(("core", set(tickers))),
    )
    monkeypatch.setattr(
        apt,
        "repair_overlay_trailing_stops",
        lambda broker, tickers: calls.append(("overlay", set(tickers))) or {"errors": []},
    )

    result = apt._repair_protection_after_aborted_rebalance(
        SimpleNamespace(),
        core_tickers={"QQQ"},
        overlay_tickers={"MU"},
    )

    assert calls == [("core", {"QQQ"}), ("overlay", {"MU"})]
    assert result["errors"] == []


def test_alpaca_broker_rejects_live_money_endpoint(monkeypatch):
    """This paper-only project must not connect when pointed at Alpaca live."""
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "ALPACA_API_KEY", "paper-key")
    monkeypatch.setattr(apt, "ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://api.alpaca.markets")

    with pytest.raises(ValueError, match="Refusing non-paper Alpaca endpoint"):
        apt.AlpacaBroker()


def test_submit_rebalance_orders_skips_buys_when_cash_cannot_afford_one_share(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    broker = _SubmitBroker(cash=50.0)
    orders = [_planned_order("MU", "buy", quantity=1)]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["MU"]
    assert order_ids[0].startswith("SKIPPED: insufficient_cash_for_one_share")
    assert broker.orders == []


def test_submit_rebalance_orders_cash_clamps_buy_quantity(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    broker = _SubmitBroker(cash=150.0)
    orders = [_planned_order("MU", "buy", quantity=2)]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["MU"]
    assert order_ids == ["buy-MU-1"]
    assert submitted[0]["quantity"] == 1
    assert submitted[0]["original_quantity_before_cash_clamp"] == 2
    assert submitted[0]["cash_clamped_to_available"] is True
    assert "cash_limited:2->1" in submitted[0]["cash_clamp_reason"]
    assert broker.orders[0].quantity == 1


def test_submit_rebalance_orders_reserves_cash_buffer(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    monkeypatch.setattr(apt, "BUY_CASH_BUFFER_PCT", 0.10)
    monkeypatch.setattr(apt, "BUY_CASH_BUFFER_DOLLARS", 0.0)
    broker = _SubmitBroker(cash=1_000.0, equity=1_000.0)
    orders = [_planned_order("MU", "buy", quantity=10)]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert order_ids == ["buy-MU-1"]
    assert submitted[0]["quantity"] == 8
    assert submitted[0]["cash_buffer_reserved"] == 100.0
    assert submitted[0]["cash_available_after_buffer"] == 900.0
    assert submitted[0]["cash_clamped_to_available"] is True


def test_submit_rebalance_orders_uses_lower_buying_power(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SKIP_BUYS_WHEN_CASH_BELOW", 0.0)
    monkeypatch.setattr(apt, "BUY_CASH_BUFFER_PCT", 0.0)
    monkeypatch.setattr(apt, "BUY_CASH_BUFFER_DOLLARS", 0.0)
    broker = _SubmitBroker(cash=1_000.0, equity=1_000.0, buying_power=250.0)
    orders = [_planned_order("MU", "buy", quantity=5)]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert order_ids == ["buy-MU-1"]
    assert submitted[0]["quantity"] == 2
    assert submitted[0]["buying_power_used_for_cash_check"] == 250.0
    assert broker.orders[0].quantity == 2


def test_buy_orders_for_stops_only_include_filled_overlay_buys(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "BUY_FILL_WAIT_SECONDS", 0)
    broker = _SubmitBroker(statuses={
        "buy-MU-1": "filled",
        "buy-FCX-2": "new",
        "buy-QQQ-3": "filled",
    })
    orders = [
        _planned_order("MU", "buy"),
        _planned_order("FCX", "buy"),
        _planned_order("QQQ", "buy"),
    ]

    filled = apt._buy_orders_filled_for_stop_submission(
        broker,
        orders,
        ["buy-MU-1", "buy-FCX-2", "buy-QQQ-3"],
    )

    assert [order["ticker"] for order in filled] == ["MU"]


def test_tqqq_pre_trade_check_blocks_when_data_missing_fail_closed(monkeypatch):
    import alpaca_paper_trading as apt

    fake_yf = SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr(apt, "TQQQ_FAST_DD_THRESHOLD", -0.15)
    monkeypatch.setattr(apt, "TQQQ_FAST_DD_FAIL_CLOSED", True)

    safe, drawdown = apt._tqqq_pre_trade_check(SimpleNamespace())

    assert safe is False
    assert np.isnan(drawdown)


def test_submit_rebalance_orders_allows_buys_after_filled_sells_and_cash(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(apt, "SELL_FILL_WAIT_SECONDS", 0)
    broker = _SubmitBroker(cash=1_000.0, statuses={"sell-FCX-1": "filled"})
    orders = [_planned_order("FCX", "sell"), _planned_order("MU", "buy")]

    submitted, order_ids = apt.submit_rebalance_orders(
        broker,
        orders,
        use_market_order=False,
        use_quote_limit=False,
    )

    assert [o["ticker"] for o in submitted] == ["FCX", "MU"]
    assert order_ids == ["sell-FCX-1", "buy-MU-2"]
    assert [o.ticker for o in broker.orders] == ["FCX", "MU"]


def test_reconcile_orders_rechecks_partially_filled_orders(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    log_path = tmp_path / "alpaca_paper_log.csv"
    pd.DataFrame(
        [
            {
                "submitted_at": "2026-06-01T10:00:00+00:00",
                "order_id": "oid-1",
                "ticker": "MU",
                "side": "buy",
                "quantity": 25,
                "price": 100.0,
                "trade_value": 2500.0,
                "target_weight": 0.2,
                "fill_status": "partially_filled",
                "filled_qty": 24,
                "filled_avg_price": 100.0,
            }
        ]
    ).to_csv(log_path, index=False)
    monkeypatch.setattr(apt, "PAPER_LOG_FILE", log_path)

    class Api:
        def __init__(self):
            self.calls = 0

        def get_order(self, order_id):
            self.calls += 1
            return SimpleNamespace(
                status="filled",
                filled_qty="25",
                filled_avg_price="101.25",
            )

    api = Api()
    broker = SimpleNamespace(_api=api)

    apt.reconcile_orders(broker)

    output = pd.read_csv(log_path)
    assert api.calls == 1
    assert output.loc[0, "fill_status"] == "filled"
    assert output.loc[0, "filled_qty"] == 25
    assert output.loc[0, "filled_avg_price"] == 101.25


def test_log_submission_marks_error_and_skipped_statuses(tmp_path):
    import alpaca_paper_trading as apt

    orig = apt.PAPER_LOG_FILE
    apt.PAPER_LOG_FILE = tmp_path / "paper_log.csv"
    try:
        orders = [_planned_order("FCX", "sell"), _planned_order("MU", "buy")]
        apt.log_submission(orders, ["ERROR: no shares", "SKIPPED: sell_not_filled"])
        df = pd.read_csv(apt.PAPER_LOG_FILE)
    finally:
        apt.PAPER_LOG_FILE = orig

    assert df["fill_status"].tolist() == ["submission_failed", "skipped"]
    assert "cash_clamped_to_available" in df.columns
    assert "cash_buffer_reserved" in df.columns
    assert "buying_power_used_for_cash_check" in df.columns
    assert "execution_risk_score" in df.columns
    assert "execution_risk_reason" in df.columns


def test_paper_health_submitted_orders_excludes_skipped_and_errors(tmp_path, monkeypatch):
    import paper_health

    trades_path = tmp_path / "alpaca_paper_log.csv"
    equity_path = tmp_path / "alpaca_paper_equity.csv"
    status_path = tmp_path / "alpaca_daily_status.json"
    order_plan_path = tmp_path / "core_satellite_alpha_orders.csv"
    pd.DataFrame([
        {"submitted_at": "2026-06-03T14:00:00Z", "order_id": "buy-MU-1", "ticker": "MU", "side": "buy", "fill_status": "pending"},
        {"submitted_at": "2026-06-03T14:00:00Z", "order_id": "SKIPPED: cash", "ticker": "INTC", "side": "buy", "fill_status": "skipped"},
        {"submitted_at": "2026-06-03T14:00:00Z", "order_id": "ERROR: rejected", "ticker": "FCX", "side": "sell", "fill_status": "submission_failed"},
    ]).to_csv(trades_path, index=False)
    pd.DataFrame([{"date": "2026-06-03", "equity": 100_000.0}]).to_csv(equity_path, index=False)
    status_path.write_text("{}", encoding="utf-8")
    pd.DataFrame().to_csv(order_plan_path, index=False)

    monkeypatch.setattr(paper_health, "PAPER_TRADES", trades_path)
    monkeypatch.setattr(paper_health, "PAPER_EQUITY", equity_path)
    monkeypatch.setattr(paper_health, "PAPER_STATUS", status_path)
    monkeypatch.setattr(paper_health, "CORE_ORDER_PLAN", order_plan_path)
    monkeypatch.setattr(
        paper_health.alpaca_paper_gauntlet,
        "evaluate_alpaca_paper",
        lambda: {
            "strategy": "test",
            "status": "collecting",
            "filled_orders": 0,
            "fill_rate": 0.0,
            "cancel_rate": 0.0,
            "approved_for_real_capital": False,
            "reason": "test",
        },
    )

    health = paper_health.build_health()

    assert health["paper_trades"] == 3
    assert health["submitted_orders"] == 1
    assert health["skipped_orders"] == 1
    assert health["submission_failed_orders"] == 1


class TestDuplicatePrevention:
    """Test that _already_submitted_today works correctly."""

    def test_no_log_file(self, tmp_path):
        """No log file → not submitted."""
        import alpaca_paper_trading as apt
        orig = apt.PAPER_LOG_FILE
        apt.PAPER_LOG_FILE = tmp_path / "nonexistent.csv"
        try:
            assert apt._already_submitted_today() is False
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_today_entry_blocks(self, tmp_path):
        """Entry from today → already submitted."""
        import alpaca_paper_trading as apt
        orig = apt.PAPER_LOG_FILE
        log_path = tmp_path / "test_log.csv"
        pd.DataFrame([{
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "order_id": "test-123",
            "ticker": "SPY", "side": "buy",
            "quantity": 10, "price": 500,
            "trade_value": 5000, "target_weight": 0.5,
            "fill_status": "filled",
        }]).to_csv(log_path, index=False)
        apt.PAPER_LOG_FILE = log_path
        try:
            assert apt._already_submitted_today() is True
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_today_skipped_or_error_only_does_not_block(self, tmp_path):
        """Skipped/error audit rows are not actual Alpaca submissions."""
        import alpaca_paper_trading as apt
        orig = apt.PAPER_LOG_FILE
        log_path = tmp_path / "test_log.csv"
        pd.DataFrame([
            {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "order_id": "SKIPPED: quote_unavailable",
                "ticker": "MU", "side": "buy",
                "quantity": 10, "price": 500,
                "trade_value": 5000, "target_weight": 0.5,
                "fill_status": "skipped",
            },
            {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "order_id": "ERROR: rejected",
                "ticker": "FCX", "side": "sell",
                "quantity": 1, "price": 50,
                "trade_value": 50, "target_weight": 0.0,
                "fill_status": "submission_failed",
            },
        ]).to_csv(log_path, index=False)
        apt.PAPER_LOG_FILE = log_path
        try:
            assert apt._already_submitted_today() is False
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_old_entry_allows(self, tmp_path):
        """Only old entries → not submitted today."""
        import alpaca_paper_trading as apt
        orig = apt.PAPER_LOG_FILE
        log_path = tmp_path / "test_log.csv"
        pd.DataFrame([{
            "submitted_at": "2025-01-01T00:00:00+00:00",
            "order_id": "old-123",
            "ticker": "SPY", "side": "buy",
            "quantity": 10, "price": 500,
            "trade_value": 5000, "target_weight": 0.5,
            "fill_status": "filled",
        }]).to_csv(log_path, index=False)
        apt.PAPER_LOG_FILE = log_path
        try:
            assert apt._already_submitted_today() is False
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_empty_log_allows(self, tmp_path):
        """Empty log file → not submitted."""
        import alpaca_paper_trading as apt
        orig = apt.PAPER_LOG_FILE
        log_path = tmp_path / "test_log.csv"
        pd.DataFrame().to_csv(log_path, index=False)
        apt.PAPER_LOG_FILE = log_path
        try:
            assert apt._already_submitted_today() is False
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_alpaca_today_bot_order_blocks_without_local_log(self, tmp_path):
        """A live Alpaca bot order from today should block even if CSV is gone."""
        import alpaca_paper_trading as apt

        now = datetime.now(timezone.utc)
        client_id = apt.bot_client_order_id(
            {"ticker": "MU", "side": "buy", "quantity": 25},
            today=now,
        )
        order = SimpleNamespace(
            symbol="MU",
            side="buy",
            type="limit",
            qty="25",
            id="alpaca-1",
            status="filled",
            submitted_at=now,
            client_order_id=client_id,
        )
        broker = SimpleNamespace(_api=SimpleNamespace(list_orders=lambda **_kwargs: [order]))

        orig = apt.PAPER_LOG_FILE
        apt.PAPER_LOG_FILE = tmp_path / "missing.csv"
        try:
            assert apt._already_submitted_today(broker) is True
        finally:
            apt.PAPER_LOG_FILE = orig

    def test_alpaca_manual_or_stop_order_does_not_block_without_local_log(self, tmp_path):
        """Manual orders and bot trailing stops are not duplicate rebalance submits."""
        import alpaca_paper_trading as apt

        now = datetime.now(timezone.utc)
        today_prefix = now.strftime("%Y%m%d") + "_"
        orders = [
            SimpleNamespace(
                symbol="MU",
                side="buy",
                type="limit",
                qty="25",
                id="manual-1",
                status="filled",
                submitted_at=now,
                client_order_id="manual-order",
            ),
            SimpleNamespace(
                symbol="FCX",
                side="sell",
                type="trailing_stop",
                qty="302",
                id="stop-1",
                status="new",
                submitted_at=now,
                client_order_id=today_prefix + "FCX_sell_302",
            ),
        ]
        broker = SimpleNamespace(_api=SimpleNamespace(list_orders=lambda **_kwargs: orders))

        orig = apt.PAPER_LOG_FILE
        apt.PAPER_LOG_FILE = tmp_path / "missing.csv"
        try:
            assert apt._already_submitted_today(broker) is False
        finally:
            apt.PAPER_LOG_FILE = orig


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY STOP REPAIR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _open_order(symbol, oid, order_type="trailing_stop", side="sell", qty=10):
    """Make a tiny fake Alpaca order object for stop-management tests."""
    return SimpleNamespace(
        symbol=symbol,
        id=oid,
        type=order_type,
        side=side,
        qty=str(qty),
    )


class _StopBroker:
    """Tiny broker fake that records cancelled and submitted stop orders."""

    def __init__(self, *, orders=None, positions=None):
        self._orders = list(orders or [])
        self._positions = dict(positions or {})
        self.cancelled = []
        self.placed = []
        self._api = SimpleNamespace(list_orders=self.list_orders)

    def list_orders(self, **_kwargs):
        return list(self._orders)

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        self._orders = [o for o in self._orders if getattr(o, "id", "") != oid]
        return True

    def get_positions(self):
        from broker_interface import Position

        return [
            Position(ticker=ticker, quantity=qty, avg_price=100.0)
            for ticker, qty in self._positions.items()
        ]

    def place_order(self, order):
        self.placed.append(order)
        return f"new-{len(self.placed)}"


def test_cancel_overlay_trailing_stops_for_sells_cancels_only_affected_overlay_stops(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    broker = _StopBroker(orders=[
        _open_order("FCX", "stop-fcx", order_type="trailing_stop", side="sell", qty=104),
        _open_order("QQQ", "stop-qqq", order_type="trailing_stop", side="sell", qty=50),
        _open_order("FCX", "sell-fcx", order_type="limit", side="sell", qty=20),
        _open_order("MU", "stop-mu", order_type="trailing_stop", side="sell", qty=2),
    ])

    actions = apt.cancel_overlay_trailing_stops_for_sells(
        broker,
        [{"ticker": "FCX", "side": "sell"}, {"ticker": "QQQ", "side": "sell"}],
    )

    assert broker.cancelled == ["stop-fcx"]
    assert actions == [{
        "ticker": "FCX",
        "qty": 104.0,
        "order_id": "stop-fcx",
        "status": "cancelled",
        "reason": "rebalance_sell",
    }]


def test_repair_overlay_trailing_stops_recreates_for_remaining_position(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(apt, "TRAILING_STOP_PCT", 0.08)
    broker = _StopBroker(positions={"FCX": 136})

    result = apt.repair_overlay_trailing_stops(broker, tickers={"FCX"})

    assert result["errors"] == []
    assert len(broker.placed) == 1
    order = broker.placed[0]
    assert order.ticker == "FCX"
    assert order.side == "sell"
    assert order.quantity == 136
    assert order.type == "trailing_stop"
    assert order.trail_percent == 0.08


def test_repair_overlay_trailing_stops_skips_when_normal_sell_is_open(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    broker = _StopBroker(
        orders=[_open_order("FCX", "sell-fcx", order_type="limit", side="sell", qty=20)],
        positions={"FCX": 136},
    )

    result = apt.repair_overlay_trailing_stops(broker, tickers={"FCX"})

    assert broker.placed == []
    assert result["skipped"] == [{
        "ticker": "FCX",
        "reason": "open_sell_order",
        "position_qty": 136,
    }]


def test_repair_all_overlay_trailing_stops_uses_live_positions(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    broker = _StopBroker(
        orders=[
            _open_order("FCX", "old-fcx-1", order_type="trailing_stop", side="sell", qty=104),
            _open_order("FCX", "old-fcx-2", order_type="trailing_stop", side="sell", qty=62),
            _open_order("QQQ", "core-qqq", order_type="trailing_stop", side="sell", qty=86),
        ],
        positions={"FCX": 302, "MU": 23, "QQQ": 86},
    )

    result = apt.repair_all_overlay_trailing_stops(broker)

    assert broker.cancelled == ["old-fcx-1", "old-fcx-2"]
    placed = {(order.ticker, order.quantity, order.type) for order in broker.placed}
    assert placed == {("FCX", 302, "trailing_stop"), ("MU", 23, "trailing_stop")}
    assert result["errors"] == []


def test_build_experimental_atr_stop_uses_local_true_range(tmp_path, monkeypatch):
    import alpaca_paper_trading as apt

    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    pd.DataFrame({
        "High": [105.0] * 20,
        "Low": [95.0] * 20,
        "Close": [100.0] * 20,
    }, index=dates).to_parquet(tmp_path / "MU.parquet")
    monkeypatch.setattr(apt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(apt, "OVERLAY_STOP_MODE", "atr")
    monkeypatch.setattr(apt, "EXPERIMENTAL_ATR_STOPS_ENABLED", True)
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(apt, "LIVE_RISK_ATR_MULT", 2.0)

    order = apt._build_overlay_stop_order("MU", 10, 100.0)

    assert order.type == "stop"
    assert order.stop_price == 80.0
    assert order.trail_percent is None


def test_experimental_atr_stop_needs_double_gate(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "OVERLAY_STOP_MODE", "atr")
    monkeypatch.setattr(apt, "EXPERIMENTAL_ATR_STOPS_ENABLED", False)

    with pytest.raises(RuntimeError, match="needs ALPACA_ENABLE_EXPERIMENTAL_ATR_STOPS"):
        apt._overlay_stop_type()


def test_experimental_atr_stop_refuses_real_endpoint(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "OVERLAY_STOP_MODE", "atr")
    monkeypatch.setattr(apt, "EXPERIMENTAL_ATR_STOPS_ENABLED", True)
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://api.alpaca.markets")

    with pytest.raises(RuntimeError, match="only on Alpaca paper"):
        apt._overlay_stop_type()


def test_cancel_overlay_stops_treats_fixed_atr_stop_as_protection(monkeypatch):
    import alpaca_paper_trading as apt

    monkeypatch.setattr(apt, "TRAILING_STOP_ENABLED", True)
    broker = _StopBroker(orders=[
        _open_order("MU", "fixed-mu", order_type="stop", side="sell", qty=20),
    ])

    actions = apt.cancel_overlay_trailing_stops_for_sells(
        broker,
        [{"ticker": "MU", "side": "sell"}],
    )

    assert broker.cancelled == ["fixed-mu"]
    assert actions[0]["qty"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA BROKER CONNECTION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaBroker:
    """Test live Alpaca paper trading connection (requires API keys)."""

    @pytest.fixture
    def broker(self):
        """Create an AlpacaBroker — skip if no API keys."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        if not api_key:
            pytest.skip("ALPACA_API_KEY not set")
        _alpaca_package_or_skip()
        from alpaca_paper_trading import AlpacaBroker
        return AlpacaBroker()

    def test_get_equity(self, broker):
        """Equity should be a positive number."""
        equity = broker.get_equity()
        assert equity > 0
        assert isinstance(equity, float)

    def test_get_cash(self, broker):
        """Cash should be a float (can be negative if using margin)."""
        cash = broker.get_cash()
        assert isinstance(cash, float)
        assert np.isfinite(cash)

    def test_get_positions(self, broker):
        """Positions should return a list."""
        positions = broker.get_positions()
        assert isinstance(positions, list)

    def test_get_position_map(self, broker):
        """Position map should be a dict of ticker → quantity."""
        pos_map = broker.get_position_map()
        assert isinstance(pos_map, dict)
        for ticker, qty in pos_map.items():
            assert isinstance(ticker, str)
            assert isinstance(qty, int)

    def test_get_last_price_valid_ticker(self, broker):
        """Price for SPY should be positive."""
        price = broker.get_last_price("SPY")
        assert price > 0

    def test_get_last_price_invalid_ticker(self, broker):
        """Invalid ticker should return 0.0 after retries."""
        price = broker.get_last_price("ZZZZNOTREAL")
        assert price == 0.0

    def test_is_market_open(self, broker):
        """Should return a boolean."""
        result = broker.is_market_open()
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY SNAPSHOT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEquitySnapshot:
    """Test equity snapshot deduplication logic."""

    def test_snapshot_creates_file(self, tmp_path):
        """Snapshot should create the equity CSV if it doesn't exist."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        if not api_key:
            pytest.skip("ALPACA_API_KEY not set")
        _alpaca_package_or_skip()

        import alpaca_paper_trading as apt
        orig = apt.EQUITY_FILE
        test_path = tmp_path / "test_equity.csv"
        apt.EQUITY_FILE = test_path
        try:
            broker = apt.AlpacaBroker()
            apt.snapshot_equity(broker)
            assert test_path.exists()
            df = pd.read_csv(test_path)
            assert len(df) == 1
            assert "equity" in df.columns
            assert "date" in df.columns
        finally:
            apt.EQUITY_FILE = orig


def test_execution_quality_segments_split_market_and_limit_orders():
    import alpaca_paper_trading as apt

    rows = [
        {
            "order_type": "limit",
            "slippage_bps": -1.0,
            "adverse_15m_bps": 2.0,
            "worst_adverse_60m_bps": 10.0,
        },
        {
            "order_type": "market",
            "slippage_bps": 12.0,
            "adverse_15m_bps": -3.0,
            "worst_adverse_60m_bps": 20.0,
        },
    ]

    segments = apt._execution_quality_segments(rows)

    assert segments["all_orders"]["orders_analyzed"] == 2
    assert segments["limit_orders"]["orders_analyzed"] == 1
    assert segments["market_orders"]["orders_analyzed"] == 1
    assert segments["trailing_stops"]["orders_analyzed"] == 0
    assert segments["limit_orders"]["avg_slippage_bps"] == -1.0
    assert segments["market_orders"]["avg_slippage_bps"] == 12.0

    def test_snapshot_deduplicates_same_day(self, tmp_path):
        """Running snapshot twice on same day should keep 1 row, not 2."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        if not api_key:
            pytest.skip("ALPACA_API_KEY not set")
        _alpaca_package_or_skip()

        import alpaca_paper_trading as apt
        orig = apt.EQUITY_FILE
        test_path = tmp_path / "test_equity.csv"
        apt.EQUITY_FILE = test_path
        try:
            broker = apt.AlpacaBroker()
            apt.snapshot_equity(broker)
            apt.snapshot_equity(broker)
            df = pd.read_csv(test_path)
            assert len(df) == 1  # Not 2 — same day deduplicated
        finally:
            apt.EQUITY_FILE = orig
