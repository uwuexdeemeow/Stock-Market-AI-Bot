from __future__ import annotations

import json
from datetime import datetime, timezone

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


def test_fill_monitor_accepts_current_alpaca_log_schema(tmp_path, monkeypatch):
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir()
    trade_path = signal_dir / "alpaca_paper_log.csv"
    now = datetime.now(timezone.utc).isoformat()
    trade_path.write_text(
        "\n".join([
            "submitted_at,order_id,ticker,side,quantity,price,trade_value,target_weight,fill_status,filled_qty,filled_avg_price",
            f"{now},ERR-1,FCX,sell,302,64.665,19528.83,0.0,submission_failed,,",
            f"{now},abc-2,MU,buy,25,855.77,21394.25,0.2,partially_filled,24.0,857.84",
            f"{now},abc-3,CAT,buy,3,899.71,2699.13,0.0292,filled,3.0,900.24",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(fill_monitor, "SIGNALS", signal_dir)
    monkeypatch.setattr(fill_monitor, "PAPER_TRADES_FILE", trade_path)
    monkeypatch.setattr(fill_monitor, "FILL_MONITOR_LOG", signal_dir / "fill_monitor.json")

    result = fill_monitor.check_recent_fills(lookback_days=2, quiet=True)

    assert result["status"] == "warning"
    assert result["total_checked"] == 3
    assert result["filled"] == 1
    assert result["cancelled"] == 1
    assert result["partial"] == 1
    assert {p["action"] for p in result["problems"]} == {"SELL", "BUY"}
    assert {p["status"] for p in result["problems"]} == {"submission_failed", "partially_filled"}


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


def test_monitor_heartbeat_tracks_execution_scorecard():
    assert "execution_scorecard" in monitor_heartbeat.MONITORED_FILES
    assert monitor_heartbeat.MONITORED_FILES["execution_scorecard"].name == "alpaca_execution_scorecard.json"


def test_monitor_heartbeat_tracks_broker_truth():
    assert "broker_truth" in monitor_heartbeat.MONITORED_FILES
    assert monitor_heartbeat.MONITORED_FILES["broker_truth"].name == "broker_truth.json"
