"""Audit fix H1: the paper account trades only on the backtest's 20-day calendar.

PLAIN ENGLISH: the backtest re-picks stocks and the market regime only every
`holding_days` trading sessions, on a calendar that starts on the first day
of the data.  The paper account used to re-trade every day, so its results
could not confirm the backtest.  These tests pin:

  * the calendar the signal publishes (same rule as the backtest);
  * the order step's hold / rebalance / catch-up decision;
  * that the drawdown halt still runs on days with no strategy orders.

Fake data and fake brokers only.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

import alpaca_paper_trading as apt
import core_satellite_alpha as csa


# ── The calendar in the signal ─────────────────────────────────────────────

def test_schedule_matches_the_backtest_calendar():
    first = pd.Timestamp("2026-01-02")
    sessions = csa._nyse_sessions(first, pd.Timestamp("2026-06-30"))
    scheduled = sessions[::20]     # the backtest's rule: every 20th session

    on_day = csa.paper_rebalance_schedule(first, scheduled[2], 20)
    assert on_day["anchor"] == sessions[0]
    assert on_day["last"] == scheduled[2]
    assert on_day["next"] == scheduled[3]
    assert on_day["scheduled_today"] is True

    between = csa.paper_rebalance_schedule(first, sessions[45], 20)
    assert between["last"] == scheduled[2]
    assert between["next"] == scheduled[3]
    assert between["scheduled_today"] is False


def test_schedule_rejects_bad_holding_days():
    try:
        csa.paper_rebalance_schedule("2026-01-02", "2026-02-02", 0)
    except ValueError:
        return
    raise AssertionError("holding_days=0 must be rejected")


def test_signal_publishes_the_calendar(tmp_path, monkeypatch):
    monkeypatch.setattr(csa, "SIGNAL_DIR", str(tmp_path))
    monkeypatch.setattr(csa, "SENTIMENT_VETO_ENABLED", False)
    monkeypatch.setattr(csa, "_paper_signal_timestamp", lambda: "2026-05-13T22:00+08:00")
    first, latest = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-13")
    panel = pd.DataFrame({
        "date": [first, first, latest, latest],
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "sector": ["XLK", "XLF", "XLK", "XLF"],
        "factor_walkforward_score": [1.0, 0.9, 1.0, 0.9],
    })
    metrics = {
        "core_preset": "test_static", "regime_mode": "static", "current_regime": "static",
        "core_weights": {"QQQ": 1.0}, "core_gross": 0.5, "overlay_gross": 0.5,
        "score_source": "factor_walkforward", "shape": "top3", "weighting": "sticky_score",
        "exit_rank_floor": 0.20, "max_per_sector": 10, "max_single_name_weight": 1.0,
        "holding_days": 5, "feature_health_gate_pass": True, "paper_ready": True,
        "robust_cost_stress_pass": True, "core_satellite_gate_results": {"all_pass": True},
    }

    row = pd.read_csv(csa.write_paper_signal(panel, metrics)).iloc[0]

    # Sessions from 2026-05-01: May 1, 4, 5, 6, 7 | 8, 11, 12, 13, 14 | ...
    assert row["rebalance_policy"] == "tested_calendar_v1"
    assert row["rebalance_anchor_date"] == "2026-05-01"
    assert row["last_scheduled_rebalance_date"] == "2026-05-08"
    assert row["next_scheduled_rebalance_date"] == "2026-05-15"
    assert bool(row["scheduled_rebalance_today"]) is False


# ── The order step's decision ──────────────────────────────────────────────

def _signal(period="2026-05-08", today=False):
    return pd.Series({
        "last_scheduled_rebalance_date": period,
        "next_scheduled_rebalance_date": "2026-06-05",
        "scheduled_rebalance_today": today,
        "rebalance_policy": "tested_calendar_v1",
    })


def test_first_run_rebalances(tmp_path):
    assert apt._paper_rebalance_decision(_signal(), path=tmp_path / "p.json") == (
        "rebalance", "no_rebalance_recorded_yet")


def test_period_already_rebalanced_holds(tmp_path):
    path = tmp_path / "p.json"
    apt._record_paper_rebalance(_signal("2026-05-08"), path=path)
    action, reason = apt._paper_rebalance_decision(_signal("2026-05-08"), path=path)
    assert action == "hold"
    assert "2026-05-08" in reason


def test_new_scheduled_day_rebalances(tmp_path):
    path = tmp_path / "p.json"
    apt._record_paper_rebalance(_signal("2026-04-09"), path=path)
    assert apt._paper_rebalance_decision(_signal("2026-05-08", today=True), path=path) == (
        "rebalance", "scheduled_rebalance_2026-05-08")


def test_missed_scheduled_day_is_caught_up(tmp_path):
    # The scheduled day's run failed (or a halt blocked it): the next run
    # still sees an older recorded period and rebalances once.
    path = tmp_path / "p.json"
    apt._record_paper_rebalance(_signal("2026-04-09"), path=path)
    assert apt._paper_rebalance_decision(_signal("2026-05-08", today=False), path=path) == (
        "rebalance", "catch_up_missed_rebalance_2026-05-08")


def test_damaged_period_file_or_legacy_signal_rebalances(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    assert apt._paper_rebalance_decision(_signal(), path=path)[0] == "rebalance"
    legacy = pd.Series({"current_regime": "risk_on"})
    assert apt._paper_rebalance_decision(legacy, path=tmp_path / "missing.json") == (
        "rebalance", "signal_has_no_rebalance_calendar")


def test_record_writes_the_period(tmp_path):
    path = tmp_path / "p.json"
    apt._record_paper_rebalance(_signal("2026-05-08"), path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["period"] == "2026-05-08"
    assert payload["next_scheduled_rebalance_date"] == "2026-06-05"
    assert payload["policy"] == "tested_calendar_v1"


# ── The drawdown halt on days with no strategy orders ──────────────────────

def _broker():
    return SimpleNamespace(is_market_open=lambda: True)


def test_no_order_day_still_fires_the_drawdown_halt(monkeypatch, tmp_path):
    fired = []
    monkeypatch.setattr(apt, "_HALT_SENTINEL_FILE", tmp_path / "halt.txt")
    monkeypatch.setattr(apt, "_maybe_auto_clear_halt", lambda broker: False)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda broker: (True, -0.13))
    monkeypatch.setattr(apt, "_emergency_liquidate", lambda broker: fired.append(True) or {"complete": True})
    monkeypatch.setattr(apt, "_set_submit_outcome", lambda *a, **k: {})

    assert apt._no_order_day_drawdown_check(_broker()) == 2
    assert fired == [True]


def test_stops_are_off_by_default():
    # Pre-registered H-edge / H-edge-core-stop: both trailing stops dropped.
    import importlib
    import os
    import alpaca_protection
    saved = {k: os.environ.pop(k, None) for k in ("ALPACA_TRAILING_STOP", "GUARD_CORE_STOP")}
    try:
        assert importlib.reload(alpaca_protection).CORE_PROTECTION_ENABLED is False
        assert importlib.reload(apt).TRAILING_STOP_ENABLED is False
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        importlib.reload(alpaca_protection)
        importlib.reload(apt)


def test_retired_stock_stops_are_cancelled_but_nothing_else(monkeypatch):
    orders = [
        SimpleNamespace(id="s1", symbol="NVDA", side="sell", type="trailing_stop", qty=5),
        SimpleNamespace(id="s2", symbol="AMD", side="sell", type="stop", qty=3),
        SimpleNamespace(id="e1", symbol="QQQ", side="sell", type="trailing_stop", qty=9),
        SimpleNamespace(id="b1", symbol="NVDA", side="buy", type="limit", qty=1),
        SimpleNamespace(id="l1", symbol="AMD", side="sell", type="limit", qty=1),
    ]
    cancelled = []
    broker = SimpleNamespace(cancel_order=lambda oid: cancelled.append(oid) or True)
    monkeypatch.setattr(apt, "list_open_orders", lambda _broker: orders)

    actions = apt._cancel_retired_overlay_stops(broker)

    assert cancelled == ["s1", "s2"]
    assert [a["status"] for a in actions] == ["cancelled", "cancelled"]


def test_no_order_day_without_drawdown_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(apt, "_HALT_SENTINEL_FILE", tmp_path / "halt.txt")
    monkeypatch.setattr(apt, "_maybe_auto_clear_halt", lambda broker: False)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda broker: (False, -0.02))
    monkeypatch.setattr(apt, "_emergency_liquidate", lambda broker: (_ for _ in ()).throw(AssertionError("no sell-off")))

    assert apt._no_order_day_drawdown_check(_broker()) is None
