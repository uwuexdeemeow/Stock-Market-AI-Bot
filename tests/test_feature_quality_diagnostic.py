from __future__ import annotations

import pandas as pd

import feature_quality_diagnostic as diagnostic


def test_feature_quality_outputs_use_atomic_writers(monkeypatch, tmp_path):
    calls = []

    def fake_write_json(data, path, **_kwargs):
        calls.append(("json", path.name, data["n_features"]))

    def fake_write_csv(df, path, **_kwargs):
        calls.append(("csv", path.name, len(df)))

    monkeypatch.setattr(diagnostic, "atomic_write_json", fake_write_json)
    monkeypatch.setattr(diagnostic, "atomic_write_csv", fake_write_csv)

    report_path, summary_path = diagnostic.write_feature_quality_outputs(
        {"n_features": 2},
        pd.DataFrame([{"feature": "signal_a"}, {"feature": "signal_b"}]),
        signal_dir=tmp_path,
    )

    assert report_path == tmp_path / "feature_quality_report.json"
    assert summary_path == tmp_path / "feature_quality_summary.csv"
    assert calls == [
        ("json", "feature_quality_report.json", 2),
        ("csv", "feature_quality_summary.csv", 2),
    ]
