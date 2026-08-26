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
