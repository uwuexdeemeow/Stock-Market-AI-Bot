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


def _run_failed_gauntlet(monkeypatch, tmp_path, argv):
    """Run main() with a gauntlet that fails after the 30-day grace period."""
    monkeypatch.setattr(gauntlet, "LOGS", tmp_path)
    monkeypatch.setattr(gauntlet, "SIGNALS", tmp_path)
    monkeypatch.setattr(gauntlet, "evaluate_alpaca_paper", lambda verbose=False: {
        "status": "failed", "reason": "fill_rate 0.413 < 0.950", "trading_days": 37,
        "approved_for_real_capital": False,
    })
    monkeypatch.setattr(gauntlet, "write_gauntlet_report", lambda result: tmp_path / "report.json")
    monkeypatch.setattr(gauntlet, "print_gauntlet_report", lambda result: None)
    monkeypatch.setattr("notifications.send_alert", lambda *args, **kwargs: True)
    monkeypatch.setattr("sys.argv", ["alpaca_paper_gauntlet.py", *argv])
    gauntlet.main()


def test_failed_gauntlet_is_report_only_by_default(monkeypatch, tmp_path, capsys):
    # No SystemExit: the daily paper workflow must stay green.
    _run_failed_gauntlet(monkeypatch, tmp_path, [])
    assert "reported only" in capsys.readouterr().out


def test_failed_gauntlet_exits_one_when_strict(monkeypatch, tmp_path):
    import pytest

    with pytest.raises(SystemExit) as stop:
        _run_failed_gauntlet(monkeypatch, tmp_path, ["--strict"])
    assert stop.value.code == 1


def test_gauntlet_equity_snapshot_has_no_direct_csv_write():
    source = Path("alpaca_paper_gauntlet.py").read_text(encoding="utf-8")

    assert "atomic_write_csv" in source
    assert ".to_csv(ALPACA_EQUITY" not in source
