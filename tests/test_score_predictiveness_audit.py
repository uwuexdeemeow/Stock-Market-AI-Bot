from __future__ import annotations

import pandas as pd

import score_predictiveness_audit as audit_mod


def test_score_predictiveness_audit_outputs_use_atomic_writers(monkeypatch, tmp_path):
    calls = []

    def fake_write_csv(df, path, **kwargs):
        calls.append(("csv", path.name, len(df), kwargs.get("index")))

    def fake_write_json(data, path, **_kwargs):
        calls.append(("json", path.name, data["objective"]))

    monkeypatch.setattr(audit_mod, "atomic_write_csv", fake_write_csv)
    monkeypatch.setattr(audit_mod, "atomic_write_json", fake_write_json)

    out_csv, out_json = audit_mod.write_score_predictiveness_audit(
        pd.DataFrame([{"metric": "score"}]),
        {"objective": "alpha_vs_qqq"},
        tmp_path / "score_predictiveness_audit.csv",
        tmp_path / "score_predictiveness_audit.json",
    )

    assert out_csv == tmp_path / "score_predictiveness_audit.csv"
    assert out_json == tmp_path / "score_predictiveness_audit.json"
    assert calls == [
        ("csv", "score_predictiveness_audit.csv", 1, False),
        ("json", "score_predictiveness_audit.json", "alpha_vs_qqq"),
    ]
