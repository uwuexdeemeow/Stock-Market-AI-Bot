"""Calibrator artifact lifecycle safety tests."""

from __future__ import annotations

from confidence_calibration import load_direction_calibrator, save_direction_calibrator


def test_fallback_retrain_removes_stale_calibrator(tmp_path):
    """No calibrator is safer than one fitted to an older model."""
    path = tmp_path / "direction_calibrator.pkl"
    path.write_bytes(b"stale-old-model")

    save_direction_calibrator(None, str(path))

    assert path.exists() is False
    assert load_direction_calibrator(str(path)) is None
