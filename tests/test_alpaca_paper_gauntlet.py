from __future__ import annotations

from pathlib import Path

import alpaca_paper_gauntlet as gauntlet


def test_gauntlet_report_uses_atomic_writer(monkeypatch, tmp_path):
    calls = []

    def fake_write_json(data, path, **_kwargs):
        calls.append((path.name, data["status"]))

    monkeypatch.setattr(gauntlet, "atomic_write_json", fake_write_json)

    out = gauntlet.write_gauntlet_report(
        {"status": "passed"},
        output_path=tmp_path / "alpaca_paper_gauntlet_20260605.json",
    )

    assert out == tmp_path / "alpaca_paper_gauntlet_20260605.json"
    assert calls == [("alpaca_paper_gauntlet_20260605.json", "passed")]


def test_gauntlet_equity_snapshot_has_no_direct_csv_write():
    source = Path("alpaca_paper_gauntlet.py").read_text(encoding="utf-8")

    assert "atomic_write_csv" in source
    assert ".to_csv(ALPACA_EQUITY" not in source
