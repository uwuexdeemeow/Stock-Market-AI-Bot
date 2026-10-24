from __future__ import annotations

import numpy as np
import pandas as pd

import drift_monitor


def test_same_distribution_stays_ok_and_shifted_distribution_drifts(tmp_path, monkeypatch):
    """A large artificial shift must score worse than unchanged input."""
    monkeypatch.setattr(drift_monitor, "DRIFT_PSI_ALERT", 0.25)
    monkeypatch.setattr(drift_monitor, "DRIFT_KS_ALERT", 0.20)
    reference = pd.DataFrame({"momentum": np.linspace(-1.0, 1.0, 500)})
    baseline = drift_monitor.snapshot_baseline(
        reference,
        output_path=tmp_path / "baseline.json",
    )

    stable = drift_monitor.check_drift(reference.copy(), baseline)
    shifted = drift_monitor.check_drift(
        pd.DataFrame({"momentum": np.linspace(3.0, 5.0, 500)}),
        baseline,
    )

    assert stable["status"] == "ok"
    assert shifted["status"] == "drift"
    assert shifted["features"]["momentum"]["psi"] > stable["features"]["momentum"]["psi"]


def test_baseline_skips_constant_and_too_sparse_columns(tmp_path):
    """Unmeasurable inputs are excluded instead of producing false drift."""
    frame = pd.DataFrame({
        "useful": np.arange(30, dtype=float),
        "constant": np.ones(30),
        "sparse": [1.0] * 5 + [np.nan] * 25,
    })

    baseline = drift_monitor.snapshot_baseline(frame, output_path=tmp_path / "baseline.json")

    assert list(baseline["features"]) == ["useful"]


def test_metadata_snapshot_honors_isolated_output_directory(tmp_path, monkeypatch):
    """A shadow training run must not overwrite the production baseline."""
    data_dir = tmp_path / "data"
    shadow_dir = tmp_path / "shadow"
    data_dir.mkdir()
    pd.DataFrame({"momentum": np.linspace(-1.0, 1.0, 30)}).to_parquet(data_dir / "AAA.parquet")
    monkeypatch.setattr(drift_monitor, "DATA_DIR", str(data_dir))

    path = drift_monitor.snapshot_from_metadata(
        "AAA",
        {"feature_cols_raw": ["momentum"], "model_version": "test"},
        tickers=["AAA"],
        output_dir=shadow_dir,
    )

    assert path == shadow_dir / "AAA_drift_baseline.json"
    assert path.exists()


def test_metadata_snapshot_excludes_rows_after_training_cutoff(tmp_path, monkeypatch):
    """Calibration, test, and newer rows must not shape the training baseline."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    # The final ten values are deliberately huge so their accidental inclusion
    # is obvious in both the row count and the saved maximum quantile.
    values = np.concatenate([np.arange(30, dtype=float), np.full(10, 10_000.0)])
    pd.DataFrame({"momentum": values}, index=dates).to_parquet(data_dir / "AAA.parquet")
    monkeypatch.setattr(drift_monitor, "DATA_DIR", str(data_dir))

    path = drift_monitor.snapshot_from_metadata(
        "AAA",
        {
            "feature_cols_raw": ["momentum"],
            "model_version": "test",
            "training_data_end": dates[29].isoformat(),
        },
        tickers=["AAA"],
        output_dir=tmp_path / "models",
    )

    baseline = __import__("json").loads(path.read_text(encoding="utf-8"))
    feature = baseline["features"]["momentum"]
    assert feature["sample_count"] == 30
    assert feature["reference_quantiles"][-1] == 29.0
    assert baseline["metadata"]["training_data_end"] == dates[29].isoformat()
