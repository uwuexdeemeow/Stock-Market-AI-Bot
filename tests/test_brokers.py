"""
test_brokers.py — Integration tests for broker scripts and signal parsing.

PLAIN ENGLISH:
Tests that Alpaca broker can connect, parse signals correctly, generate
valid orders, and that safety features (duplicate prevention, weight
scaling) work as expected.  Every test should take < 5 seconds.

Run:  python -m pytest tests/test_brokers.py -v
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE ORDER PREVENTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

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
