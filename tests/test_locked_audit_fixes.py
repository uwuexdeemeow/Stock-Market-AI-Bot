"""Regression tests for the audit fixes made in locked paper-trading files.

PLAIN ENGLISH: each test uses fake data only; no broker is contacted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import alpaca_paper_trading as apt
import core_satellite_alpha as csa
import paper_health


class _FakeBroker:
    def get_equity(self) -> float:
        return 1000.0

    def get_cash(self) -> float:
        return 250.0


def test_snapshot_equity_dates_rows_in_new_york_time(tmp_path, monkeypatch):
    """At 01:30 UTC it is still the previous evening in New York."""
    real_datetime = apt.datetime

    class _FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            utc = real_datetime(2026, 9, 26, 1, 30, tzinfo=timezone.utc)
            # A UTC runner's naive clock reads 01:30 on the 26th.
            return utc.astimezone(tz) if tz is not None else utc.replace(tzinfo=None)

    monkeypatch.setattr(apt, "datetime", _FrozenDatetime)
    monkeypatch.setattr(apt, "EQUITY_FILE", tmp_path / "equity.csv")

    apt.snapshot_equity(_FakeBroker())

    row = pd.read_csv(tmp_path / "equity.csv").iloc[0]
    assert str(row["date"]) == "2026-09-25"
    assert str(row["timestamp"]) == "2026-09-25 21:30:00"


def test_live_cagr_counts_return_days_not_snapshots(tmp_path, monkeypatch):
    """253 daily snapshots span exactly 252 return days = one year."""
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"mean_oos_sharpe": 1.0, "mean_oos_cagr_pct": 10.0}), encoding="utf-8")
    monkeypatch.setattr(paper_health, "WALKFORWARD_RESULTS", wf)
    dates = pd.bdate_range("2025-01-01", periods=253)
    equity = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        # Grow smoothly from 100 to 110: a 10% total return.
        "equity": [100.0 * (1.10 ** (i / 252)) for i in range(253)],
    })

    result = paper_health._backtest_vs_live_drift(equity)

    assert result["data_available"] is True
    assert abs(float(result["live_annualised_cagr_pct"]) - 10.0) < 0.005


def _approved_live() -> dict:
    return {
        "approval": {"approved": True, "reasons": []},
        "source_metrics": {"cost_stress_approval_pass": True},
        "medium_risk_review": {
            "pass": True,
            "reasons": [],
            "execution_stress_review": {"pass": True, "stamp": "published"},
        },
    }


def test_signal_shows_current_stress_review_without_changing_gate(monkeypatch):
    monkeypatch.setattr(csa, "_current_robustness_reviews", lambda: {
        "pass": False,
        "execution_stress_review": {"pass": False, "stamp": "today"},
    })
    metrics = {"core_satellite_gate_results": {"all_pass": True}, "feature_health_gate_pass": True}

    out = csa._apply_nested_live_approval_gates(metrics, _approved_live(), {"fresh": True})

    assert out["execution_stress_review"] == {"pass": False, "stamp": "today"}
    assert out["execution_stress_review_published"] == {"pass": True, "stamp": "published"}
    assert out["robustness_review_source"] == "current_reports"
    # The gate decision still comes from the published approval.
    assert out["medium_risk_review_pass"] is True
    assert out["paper_ready"] is True


def test_signal_keeps_published_review_when_reports_unreadable(monkeypatch):
    monkeypatch.setattr(csa, "_current_robustness_reviews", lambda: {})
    metrics = {"core_satellite_gate_results": {"all_pass": True}, "feature_health_gate_pass": True}

    out = csa._apply_nested_live_approval_gates(metrics, _approved_live(), {"fresh": True})

    assert out["execution_stress_review"] == {"pass": True, "stamp": "published"}
    assert "execution_stress_review_published" not in out
    assert out["robustness_review_source"] == "published_live_config"
