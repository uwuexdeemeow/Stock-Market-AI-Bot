from __future__ import annotations

import json

import fill_monitor


def test_fill_monitor_writes_fresh_output_when_no_trade_file(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals"
    report_path = signal_dir / "fill_monitor.json"

    monkeypatch.setattr(fill_monitor, "SIGNALS", signal_dir)
    monkeypatch.setattr(fill_monitor, "PAPER_TRADES_FILE", signal_dir / "missing_paper_trades.csv")
    monkeypatch.setattr(fill_monitor, "FILL_MONITOR_LOG", report_path)

    result = fill_monitor.check_recent_fills(lookback_days=2, quiet=True)
    fill_monitor.print_fill_report(result, quiet=True)

    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["status"] == "ok"
    assert payload["reason"] == "paper_trades_missing"
    assert payload["total_checked"] == 0
    assert payload["problems"] == []
