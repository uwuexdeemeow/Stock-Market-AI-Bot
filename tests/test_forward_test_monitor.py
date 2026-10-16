"""Tests for forward_test_monitor.py (fake data only).

PLAIN ENGLISH: the monthly forward-test numbers decide, at the checkpoints,
whether the paper strategy is stopped.  These tests pin the calendar, the
blocked-rebalance count (and the T-block trigger), the fair-yardstick
return and the fill-cost sign.
"""
from __future__ import annotations

import pandas as pd
import pytest

import forward_test_monitor as ftm

SESSIONS = pd.bdate_range("2026-01-01", periods=200)


def test_scheduled_dates_walk_the_20_session_grid():
    grid = ftm.scheduled_dates(str(SESSIONS[45].date()), SESSIONS)
    assert SESSIONS[5] in grid and SESSIONS[25] in grid and SESSIONS[45] in grid and SESSIONS[65] in grid
    assert all(int(SESSIONS.get_loc(d)) % 20 == 5 for d in grid)


def test_periods_done_counts_only_finished_periods_after_epoch():
    grid = ftm.scheduled_dates(str(SESSIONS[45].date()), SESSIONS)
    # Epoch starts after day 25; day 45's period ends on day 65.
    assert ftm.periods_done(grid, SESSIONS, SESSIONS[30], SESSIONS[64]) == 0
    assert ftm.periods_done(grid, SESSIONS, SESSIONS[30], SESSIONS[65]) == 1
    assert ftm.periods_done(grid, SESSIONS, SESSIONS[30], SESSIONS[105]) == 3


def _outcomes(days_status):
    return pd.DataFrame({"generated_at": [pd.Timestamp(d).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=10)
                                          for d, _ in days_status],
                         "status": [s for _, s in days_status]})


def test_blocked_share_uses_first_run_on_the_trade_day():
    grid = ftm.scheduled_dates(str(SESSIONS[45].date()), SESSIONS)
    out = _outcomes([(SESSIONS[46], "blocked"), (SESSIONS[46], "executed"),   # first run blocked
                     (SESSIONS[66], "executed")])                               # day 86: no run
    result = ftm.blocked_share(grid, SESSIONS, out, SESSIONS[30], SESSIONS[100])
    assert result["scheduled"] == 2 and result["blocked"] == 1
    assert result["missing_runs"] == [str(SESSIONS[85].date())]
    assert result["t_block_triggered"] is False


def test_t_block_trigger_needs_4_of_6():
    grid = ftm.scheduled_dates(str(SESSIONS[5].date()), SESSIONS)
    days = [SESSIONS[int(SESSIONS.get_loc(d)) + 1] for d in grid if d < SESSIONS[150]]
    statuses = ["blocked", "executed", "blocked", "blocked", "blocked", "executed", "executed"]
    out = _outcomes(list(zip(days, statuses)))
    assert ftm.blocked_share(grid, SESSIONS, out, SESSIONS[0], SESSIONS[199])["t_block_triggered"] is True


def test_relative_return_matches_hand_calculation():
    days = pd.bdate_range("2026-03-02", periods=3)
    equity = pd.DataFrame({"date": days, "equity": [100.0, 110.0, 99.0], "invested": [50.0, 110.0, 99.0]})
    qqq = pd.Series([10.0, 12.0, 12.0], index=days)
    bil = pd.Series([1.0, 1.0, 1.01], index=days)
    out = ftm.relative_return(equity, qqq, bil, days[0])
    assert out["days"] == 2
    assert out["account_return"] == pytest.approx(-0.01)
    # day 1: 50% QQQ (+20%) + 50% BIL (0%) = +10%; day 2: 100% QQQ (0%) = 0%
    assert out["yardstick_return"] == pytest.approx(0.10)
    assert out["m1_annualised"] is None          # fewer than 20 days


def test_fill_costs_flip_sign_for_sells():
    day = pd.Timestamp("2026-03-02")
    fills = pd.DataFrame({"submitted_at": ["2026-03-02T15:00:00+00:00"] * 2, "ticker": ["AAA", "BBB"],
                          "side": ["buy", "sell"], "filled_avg_price": [101.0, 99.0]})
    out = ftm.fill_costs(fills, {("AAA", day): 100.0, ("BBB", day): 100.0})
    assert out["fills"] == 2 and out["mean_bps"] == pytest.approx(100.0)   # both 1% worse than the Open


def test_append_log_adds_row_after_table(tmp_path):
    doc = tmp_path / "rule.md"
    doc.write_text("# x\n\n## Log\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    ftm.append_log("| 3 | 4 |", doc)
    assert doc.read_text(encoding="utf-8").splitlines()[-1] == "| 3 | 4 |"
