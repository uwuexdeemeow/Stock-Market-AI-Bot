import csv
import json
from pathlib import Path

import status


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write simple test CSV files so status.py can read them like real signals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_signal_blocks_when_gates_fail(tmp_path, monkeypatch):
    signal_path = tmp_path / "signal.csv"
    _write_csv(
        signal_path,
        [
            {
                "paper_ready": "True",
                "gates_all_pass": "False",
                "medium_risk_review_pass": "True",
                "current_regime": "neutral",
                "predicted_at": "2026-06-04T00:00:00Z",
                "overlay_weights_json": '{"MSFT": 0.1}',
                "core_gross": "0.6",
                "overlay_gross": "0.1",
            }
        ],
    )
    monkeypatch.setattr(status, "USE_COLOR", False)
    monkeypatch.setattr(status, "SIGNAL_PATH", signal_path)

    out = status.render_signal()

    assert out["paper_ready"] is True
    assert out["gates_all_pass"] is False
    assert out["trade_ready"] is False
    assert "BLOCKED" in out["line"]
    assert "gates_all_pass=false" in out["line"]


def test_signal_blocks_when_medium_risk_review_fails(tmp_path, monkeypatch):
    signal_path = tmp_path / "signal.csv"
    _write_csv(
        signal_path,
        [
            {
                "paper_ready": "True",
                "gates_all_pass": "True",
                "medium_risk_review_pass": "False",
                "current_regime": "risk_on",
                "predicted_at": "2026-06-04T00:00:00+00:00",
                "overlay_weights_json": "{}",
                "core_gross": "0.8",
                "overlay_gross": "0.0",
            }
        ],
    )
    monkeypatch.setattr(status, "USE_COLOR", False)
    monkeypatch.setattr(status, "SIGNAL_PATH", signal_path)

    out = status.render_signal()

    assert out["medium_risk_review_pass"] is False
    assert out["trade_ready"] is False
    assert "medium_risk_review_pass=false" in out["line"]


def test_equity_uses_first_recorded_equity_as_start(tmp_path, monkeypatch):
    equity_path = tmp_path / "alpaca_paper_equity.csv"
    _write_csv(
        equity_path,
        [
            {"date": "2026-06-01", "equity": "50000", "cash": "50000", "invested": "0"},
            {"date": "2026-06-04", "equity": "55000", "cash": "5000", "invested": "50000"},
        ],
    )
    monkeypatch.setattr(status, "USE_COLOR", False)
    monkeypatch.setattr(status, "EQUITY_PATH", equity_path)

    out = status.render_equity()

    assert out["equity"] == 55000.0
    assert out["start_equity"] == 50000.0
    assert out["pnl_pct"] == 10.0
    assert "$55,000.00" in out["line"]
    assert "+10.00%" in out["line"]


def test_equity_falls_back_to_alpaca_status_when_csv_missing(tmp_path, monkeypatch):
    status_path = tmp_path / "alpaca_daily_status.json"
    status_path.write_text(json.dumps({"account_equity": 43210.5}), encoding="utf-8")
    monkeypatch.setattr(status, "USE_COLOR", False)
    monkeypatch.setattr(status, "EQUITY_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(status, "PAPER_STATUS_PATH", status_path)

    out = status.render_equity()

    assert out["equity"] == 43210.5
    assert "from Alpaca status" in out["line"]


def test_positions_compute_exposure_when_snapshot_lacks_field(tmp_path, monkeypatch):
    status_path = tmp_path / "alpaca_daily_status.json"
    status_path.write_text(
        json.dumps(
            {
                "account_equity": 100000,
                "positions": {"QQQ": 10, "MSFT": 5},
                "position_values": {"QQQ": 60000, "MSFT": 25000},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "USE_COLOR", False)
    monkeypatch.setattr(status, "PAPER_STATUS_PATH", status_path)

    out = status.render_positions()

    assert out["n_positions"] == 2
    assert out["exposure"] == 85.0
    assert "gross exposure=85.0%" in out["lines"][0]
