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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _alpaca_package_or_skip() -> None:
    if importlib.util.find_spec("alpaca_trade_api") is None:
        pytest.skip("alpaca-trade-api not installed")


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
    def __init__(self, *, equity=100_000.0, positions=None, prices=None):
        self._equity = equity
        self._positions = dict(positions or {})
        self._prices = dict(prices or {})

    def get_equity(self):
        return self._equity

    def get_position_map(self):
        return dict(self._positions)

    def get_last_price(self, ticker):
        return float(self._prices.get(ticker, 100.0))


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


def test_generate_orders_adds_marketable_limit_prices():
    import alpaca_paper_trading as apt

    broker = _OrderBroker(equity=100_000.0, positions={"SPY": 100}, prices={"SPY": 500.0})

    orders = apt.generate_orders(broker, {"SPY": 0.0}, force=True)

    assert orders[0]["side"] == "sell"
    assert orders[0]["limit_price"] < orders[0]["price"]
    assert orders[0]["raw_limit_price"] < orders[0]["price"]


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


def test_quote_based_limit_is_opt_in():
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


class _SubmitBroker:
    """Tiny broker fake for submit-phase guard tests."""

    def __init__(self, *, cash=1_000.0, fail_tickers=None, statuses=None):
        self.cash = cash
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


def test_submit_rebalance_orders_skips_buys_when_buy_value_exceeds_cash(monkeypatch):
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
    assert order_ids == ["SKIPPED: buy_value_exceeds_cash:100.00>50.00"]
    assert broker.orders == []


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
