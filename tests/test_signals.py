"""
test_signals.py — Tests for signal generation and validation.

PLAIN ENGLISH:
The signal CSV is the contract between strategy code and the broker script.
If the signal is malformed (missing columns, weights > 1.0, stale dates),
real money could be lost.  These tests verify the signal is always valid.

Run:  python -m pytest tests/test_signals.py -v
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from settings import SIGNAL_DIR

SIGNALS = Path(SIGNAL_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED SIGNAL FILE TESTS
# ─────────────────────────────────────────────────────────────────────────────

# Required columns in the unified core-satellite signal (includes TQQQ when
# the nested walkforward grid search determines it helps).  Both Moomoo and
# Alpaca read this same signal file.
UNIFIED_REQUIRED_COLUMNS = [
    "paper_signal_type",
    "paper_ready",
    "current_regime",
    "target_spy_weight",
    "target_qqq_weight",
    "target_tqqq_weight",
    "target_cash_weight",
    "gross_exposure",
    "overlay_tickers",
    "overlay_weights_json",
    "predicted_at",
    "gates_all_pass",
]

# Legacy alias for backward compat in any external tests
TQQQ_REQUIRED_COLUMNS = UNIFIED_REQUIRED_COLUMNS
CORE_SAT_REQUIRED_COLUMNS = UNIFIED_REQUIRED_COLUMNS


class TestUnifiedSignalFile:
    """Validate the unified signal CSV that both Moomoo and Alpaca read."""

    @pytest.fixture
    def signal_path(self):
        path = SIGNALS / "core_satellite_alpha_signal.csv"
        if not path.exists():
            pytest.skip("Unified signal file not generated yet")
        return path

    @pytest.fixture
    def signal(self, signal_path):
        df = pd.read_csv(signal_path)
        assert not df.empty, "Signal file is empty"
        return df.iloc[0]

    def test_required_columns_present(self, signal_path):
        """All required columns must exist in the signal CSV."""
        df = pd.read_csv(signal_path)
        missing = [c for c in UNIFIED_REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_signal_type_correct(self, signal):
        """Signal type should identify this as the unified core-satellite strategy."""
        assert signal["paper_signal_type"] == "core_satellite_alpha"

    def test_paper_ready_is_true(self, signal):
        """Signal should be paper-ready."""
        assert bool(signal["paper_ready"]) is True

    def test_gates_all_pass(self, signal):
        """Gates should all pass."""
        assert bool(signal["gates_all_pass"]) is True

    def test_tqqq_weight_non_negative(self, signal):
        """TQQQ weight should be 0 or positive (never negative)."""
        tqqq = float(signal.get("target_tqqq_weight", 0.0) or 0.0)
        assert tqqq >= 0.0, f"TQQQ weight is negative: {tqqq}"

    def test_regime_is_valid(self, signal):
        """Regime must be one of the three valid values."""
        valid_regimes = {"risk_on", "neutral", "risk_off"}
        assert str(signal["current_regime"]) in valid_regimes

    def test_gross_exposure_within_limit(self, signal):
        """Gross exposure should not exceed paper max (1.0x)."""
        gross = float(signal["gross_exposure"])
        assert 0.0 <= gross <= 1.0 + 1e-6, f"Gross {gross} exceeds 1.0x limit"

    def test_weights_are_finite(self, signal):
        """All weight values should be finite numbers."""
        for col in ("target_spy_weight", "target_qqq_weight", "target_tqqq_weight", "target_cash_weight"):
            val = float(signal[col])
            assert np.isfinite(val), f"{col} is not finite: {val}"

    def test_weights_non_negative(self, signal):
        """ETF weights should be non-negative (no shorting in paper)."""
        for col in ("target_spy_weight", "target_qqq_weight", "target_tqqq_weight"):
            val = float(signal[col])
            assert val >= -1e-9, f"{col} is negative: {val}"

    def test_tqqq_zero_during_risk_off(self, signal):
        """TQQQ weight should be 0 during risk_off regime."""
        regime = str(signal["current_regime"])
        tqqq_w = float(signal["target_tqqq_weight"])
        if regime == "risk_off":
            assert abs(tqqq_w) < 1e-9, \
                f"TQQQ weight is {tqqq_w} during risk_off — should be 0"

    def test_overlay_weights_json_valid(self, signal):
        """Overlay weights JSON should be valid and parseable."""
        raw = str(signal["overlay_weights_json"])
        overlay = json.loads(raw)
        assert isinstance(overlay, dict)
        for ticker, weight in overlay.items():
            assert isinstance(ticker, str)
            assert np.isfinite(float(weight))

    def test_overlay_tickers_match_json(self, signal):
        """Overlay tickers list should match the JSON keys."""
        tickers_str = str(signal.get("overlay_tickers", ""))
        tickers_list = [t.strip() for t in tickers_str.split(",") if t.strip()]

        overlay = json.loads(str(signal["overlay_weights_json"]))
        json_tickers = sorted(overlay.keys())
        csv_tickers = sorted(tickers_list)

        assert csv_tickers == json_tickers, \
            f"Mismatch: CSV tickers={csv_tickers}, JSON keys={json_tickers}"

    def test_predicted_at_is_recent(self, signal):
        """Signal should have been generated recently (not stale)."""
        predicted_at = str(signal["predicted_at"])
        ts = pd.to_datetime(predicted_at, errors="coerce")
        assert not pd.isna(ts), f"Cannot parse predicted_at: {predicted_at}"
        age_hours = (datetime.now() - ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
        # Allow up to 7 days for testing (might not regenerate signal daily)
        assert age_hours < 168, f"Signal is {age_hours:.0f} hours old"

    def test_gross_equals_sum_of_weights(self, signal):
        """Gross exposure should equal the sum of all absolute weights."""
        spy = abs(float(signal["target_spy_weight"]))
        qqq = abs(float(signal["target_qqq_weight"]))
        tqqq = abs(float(signal["target_tqqq_weight"]))
        overlay = json.loads(str(signal["overlay_weights_json"]))
        overlay_sum = sum(abs(float(v)) for v in overlay.values())
        computed_gross = spy + qqq + tqqq + overlay_sum
        reported_gross = float(signal["gross_exposure"])
        assert abs(computed_gross - reported_gross) < 0.01, \
            f"Computed gross {computed_gross:.4f} != reported {reported_gross:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE-SATELLITE SIGNAL FILE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreSatelliteSignalFile:
    """Validate the core-satellite signal CSV that Moomoo reads."""

    @pytest.fixture
    def signal_path(self):
        path = SIGNALS / "core_satellite_alpha_signal.csv"
        if not path.exists():
            pytest.skip("Core-satellite signal file not generated yet")
        return path

    @pytest.fixture
    def signal(self, signal_path):
        df = pd.read_csv(signal_path)
        assert not df.empty, "Signal file is empty"
        return df.iloc[0]

    def test_required_columns_present(self, signal_path):
        """All required columns must exist."""
        df = pd.read_csv(signal_path)
        missing = [c for c in CORE_SAT_REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_signal_type_correct(self, signal):
        """Signal type should identify this as core-satellite."""
        assert signal["paper_signal_type"] == "core_satellite_alpha"

    def test_gross_exposure_within_limit(self, signal):
        """Gross should not exceed paper max."""
        gross = float(signal["gross_exposure"])
        assert gross <= 1.0 + 1e-6

    def test_no_tqqq_in_core_satellite(self, signal):
        """Core-satellite should NOT have a TQQQ weight column with value > 0."""
        if "target_tqqq_weight" in signal.index:
            tqqq = float(signal.get("target_tqqq_weight", 0.0) or 0.0)
            assert abs(tqqq) < 1e-9, \
                f"Core-satellite has TQQQ weight {tqqq} — should be 0"

    def test_overlay_weights_json_valid(self, signal):
        """Overlay JSON should be parseable."""
        raw = str(signal["overlay_weights_json"])
        overlay = json.loads(raw)
        assert isinstance(overlay, dict)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONSISTENCY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalConsistency:
    """Cross-check that both signals are internally consistent."""

    def test_tqqq_weight_consistent_with_regime(self):
        """During non-risk_on regimes, TQQQ weight should be 0."""
        path = SIGNALS / "core_satellite_alpha_signal.csv"
        if not path.exists():
            pytest.skip("Signal not generated")

        sig = pd.read_csv(path).iloc[0]
        regime = str(sig["current_regime"])
        tqqq_w = float(sig.get("target_tqqq_weight", 0.0) or 0.0)

        # TQQQ is only held during risk_on regime; should be 0 in neutral/risk_off
        if regime in ("neutral", "risk_off"):
            assert tqqq_w == 0.0, \
                f"TQQQ weight should be 0 in {regime} regime, got {tqqq_w}"

    def test_unified_signal_exists(self):
        """The unified signal file should exist."""
        path = SIGNALS / "core_satellite_alpha_signal.csv"
        assert path.exists(), "Signal missing — run: python3 core_satellite_alpha.py"
