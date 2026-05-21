from __future__ import annotations

import json
from datetime import datetime

import daily_run
import fill_monitor
import monitor_heartbeat


def test_fill_monitor_writes_fresh_output_when_no_trade_file(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals"
    report_path = signal_dir / "fill_monitor.json"

    monkeypatch.setattr(fill_monitor, "SIGNALS", signal_dir)
    monkeypatch.setattr(fill_monitor, "PAPER_TRADES_FILE", signal_dir / "missing_alpaca_paper_log.csv")
    monkeypatch.setattr(fill_monitor, "FILL_MONITOR_LOG", report_path)

    result = fill_monitor.check_recent_fills(lookback_days=2, quiet=True)
    fill_monitor.print_fill_report(result, quiet=True)

    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["status"] == "ok"
    assert payload["reason"] == "alpaca_paper_log_missing"
    assert payload["total_checked"] == 0
    assert payload["problems"] == []


def test_monitor_heartbeat_finds_daily_run_fill_stub(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals_abs"
    log_dir = tmp_path / "logs_abs"
    monkeypatch.setattr(daily_run, "SIGNAL_DIR", str(signal_dir))
    monkeypatch.setattr(daily_run, "LOGS", log_dir)

    now = datetime(2026, 5, 19, 9, 35)
    daily_run._write_startup_stubs(now, [daily_run.FILL_MONITOR_STEP])

    fill_path = signal_dir / "fill_monitor.json"
    monkeypatch.setattr(monitor_heartbeat, "SIGNALS", signal_dir)
    monkeypatch.setattr(monitor_heartbeat, "MONITORED_FILES", {"fill_monitor": fill_path})

    summary = monitor_heartbeat.check_monitors(max_age_hours=36)

    assert summary["all_ok"] is True
    assert summary["missing_monitors"] == []
    assert summary["monitors"]["fill_monitor"]["path"] == str(fill_path)
    assert summary["monitors"]["fill_monitor"]["status"] == "fresh"
