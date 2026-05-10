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
# TQQQ SIGNAL FILE TESTS
# ─────────────────────────────────────────────────────────────────────────────

# Required columns that must exist in every TQQQ signal
TQQQ_REQUIRED_COLUMNS = [
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

# Required columns for core-satellite signal
CORE_SAT_REQUIRED_COLUMNS = [
    "paper_signal_type",
    "paper_ready",
    "current_regime",
    "target_spy_weight",
    "target_qqq_weight",
    "gross_exposure",
    "overlay_tickers",
    "overlay_weights_json",
    "predicted_at",
    "gates_all_pass",
]


class TestTQQQSignalFile:
    """Validate the TQQQ signal CSV that Alpaca reads."""

    @pytest.fixture
    def signal_path(self):
        path = SIGNALS / "core_satellite_tqqq_signal.csv"
        if not path.exists():
            pytest.skip("TQQQ signal file not generated yet")
        return path

    @pytest.fixture
    def signal(self, signal_path):
        df = pd.read_csv(signal_path)
        assert not df.empty, "Signal file is empty"
        return df.iloc[0]

    def test_required_columns_present(self, signal_path):
        """All required columns must exist in the signal CSV."""
        df = pd.read_csv(signal_path)
        missing = [c for c in TQQQ_REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_signal_type_correct(self, signal):
        """Signal type should identify this as TQQQ strategy."""
        assert signal["paper_signal_type"] == "core_satellite_tqqq"

    def test_paper_ready_is_true(self, signal):
        """Signal should be paper-ready."""
        assert bool(signal["paper_ready"]) is True

    def test_gates_all_pass(self, signal):
        """Gates should all pass."""
        assert bool(signal["gates_all_pass"]) is True

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

    def test_tqqq_weight_config_matches_allocation(self):
        """The tqqq_weight_config should match the actual TQQQ allocation ratio."""
        path = SIGNALS / "core_satellite_tqqq_signal.csv"
        if not path.exists():
            pytest.skip("TQQQ signal not generated")

        sig = pd.read_csv(path).iloc[0]
        config_weight = float(sig.get("tqqq_weight_config", 0.0))
        regime = str(sig["current_regime"])

        # During risk_on, TQQQ should be approximately config_weight of core
        if regime == "risk_on":
            tqqq_w = float(sig["target_tqqq_weight"])
            qqq_w = float(sig["target_qqq_weight"])
            core_total = tqqq_w + qqq_w
            if core_total > 1e-6:
                actual_ratio = tqqq_w / core_total
                assert abs(actual_ratio - config_weight) < 0.02, \
                    f"TQQQ ratio {actual_ratio:.3f} != config {config_weight:.3f}"

    def test_both_signals_exist(self):
        """Both signal files should exist (both strategies active)."""
        tqqq_exists = (SIGNALS / "core_satellite_tqqq_signal.csv").exists()
        core_exists = (SIGNALS / "core_satellite_alpha_signal.csv").exists()
        assert tqqq_exists, "TQQQ signal missing — run: python core_satellite_tqqq.py"
        assert core_exists, "Core-satellite signal missing — run: python core_satellite_alpha.py"
