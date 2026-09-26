"""Tests for the pre-registered H-edge check (fake data only).

PLAIN ENGLISH: the edge check decides whether the incumbent's backtest edge
is real.  These tests pin the pieces that could quietly go wrong: the
point-in-time stock list, the random "monkey" scores, the simulated trailing
stop, and the pass/fail rules.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("research_evidence/edge_check_20260926/edge_check.py")
spec = importlib.util.spec_from_file_location("edge_check", SCRIPT)
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


# ── Point-in-time stock list ───────────────────────────────────────────────

def test_point_in_time_mask_uses_latest_snapshot_and_old_tickers():
    snapshots = pd.DataFrame({
        "date": pd.to_datetime(["2012-01-01", "2014-01-01"]),
        "members": [{"AAPL", "FB"}, {"AAPL", "FB", "NVDA"}],
    })
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2011-06-01", "2013-06-01", "2013-06-01", "2015-06-01", "2013-06-01"]),
        "ticker": ["AAPL", "NVDA", "META", "NVDA", "AAPL"],
    })
    mask = edge.point_in_time_mask(panel, snapshots, {"META": ("FB",)})
    # 2011 is before the first snapshot; NVDA joins only in the 2014 snapshot;
    # META counts through its old ticker FB.
    assert mask.tolist() == [False, False, True, True, True]


# ── Random "monkey" scores ─────────────────────────────────────────────────

def test_randomized_scores_keep_eligibility_and_share_one_draw():
    panel = pd.DataFrame({
        "factor_risk_on_score": [0.9, np.nan, 0.1, 0.5],
        "factor_defensive_score": [0.2, np.nan, 0.8, 0.5],
        "factor_walkforward_score": [0.3, np.nan, 0.3, 0.5],
    })
    first = edge.randomized_scores(panel, seed=1)
    again = edge.randomized_scores(panel, seed=1)
    other = edge.randomized_scores(panel, seed=2)
    for col in edge.SCORE_COLS:
        assert first[col].isna().tolist() == panel[col].isna().tolist()
    assert first["factor_risk_on_score"].equals(first["factor_defensive_score"])
    assert first.equals(again)
    assert not first.equals(other)
    # The real scores are gone.
    assert not np.allclose(first["factor_risk_on_score"].dropna(), panel["factor_risk_on_score"].dropna())


# ── Simulated trailing stop ────────────────────────────────────────────────

def _bars(rows):
    dates = pd.bdate_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=dates)


def test_stop_not_hit_keeps_the_plain_return():
    bars = _bars([(100, 101, 99, 100), (100, 103, 98, 102), (102, 104, 100, 104)])
    out = edge.stop_exit(bars, bars.index[0], bars.index[-1])
    assert out["stopped"] is False
    assert abs(out["stop_return"] - 0.04) < 1e-12


def test_stop_trails_the_high_of_earlier_days():
    # High of 110 on day 2 lifts the stop to 101.2; day 3's low of 100 hits it.
    bars = _bars([(100, 101, 99, 100), (100, 110, 100, 108), (108, 108, 100, 105), (105, 106, 104, 106)])
    out = edge.stop_exit(bars, bars.index[0], bars.index[-1])
    assert out["stopped"] is True
    assert out["stop_date"] == bars.index[2]
    assert abs(out["stop_return"] - (110 * 0.92 / 100 - 1)) < 1e-12
    assert abs(out["plain_return"] - 0.06) < 1e-12


def test_gap_below_the_stop_sells_at_the_open():
    bars = _bars([(100, 100, 99, 100), (80, 81, 78, 79)])
    out = edge.stop_exit(bars, bars.index[0], bars.index[-1])
    assert out["stopped"] is True
    assert abs(out["stop_return"] - (-0.20)) < 1e-12


def test_stop_adjusted_equity_changes_only_stopped_names():
    good = _bars([(100, 101, 99, 100), (100, 102, 99, 101), (101, 103, 100, 102)])
    crash = _bars([(100, 100, 99, 100), (100, 100, 85, 86), (86, 90, 85, 90)])
    equity = pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    trades = pd.DataFrame([{
        "date": pd.Timestamp("2019-12-31"), "exit_date": pd.Timestamp("2020-01-03"),
        "label_end_date": pd.Timestamp("2020-01-03"), "entry_delay_days": 0,
        "overlay_weights_json": json.dumps({"GOOD": 0.25, "CRASH": 0.25}), "period_return": 0.10,
    }])
    rebuilt, info = edge.stop_adjusted_equity(
        equity, trades, {"GOOD": good, "CRASH": crash},
        session_offset=lambda day, n: pd.Timestamp("2020-01-01"), stop_cost_pct=0.001)
    # CRASH: plain -10%, stopped at 92 -> -8%; change = 0.25 * 0.02 - 0.25 * 0.001.
    expected = 100.0 * (1.0 + 0.10 + 0.25 * 0.02 - 0.25 * 0.001)
    assert abs(rebuilt.iloc[-1] - expected) < 1e-9
    assert info == {"stock_positions": 2, "stop_exits": 1}


# ── Pre-registered judging ─────────────────────────────────────────────────

def _rows(r0_late, s_late, s_on_time, monkeys, t_alpha, s_dd, t_dd):
    rows = []
    for v in r0_late:
        rows.append({"test": "R0", "delay": 1, "decision_alpha_vs_qqq_pct": v})
    for v in s_late:
        rows.append({"test": "S", "delay": 1, "decision_alpha_vs_qqq_pct": v})
    for v, dd in zip(s_on_time, s_dd):
        rows.append({"test": "S", "delay": 0, "decision_alpha_vs_qqq_pct": v, "decision_period_max_drawdown_pct": dd})
    for v in monkeys:
        rows.append({"test": "M", "delay": 0, "decision_alpha_vs_qqq_pct": v})
    for v, dd in zip(t_alpha, t_dd):
        rows.append({"test": "T", "delay": 0, "decision_alpha_vs_qqq_pct": v, "decision_period_max_drawdown_pct": dd})
    return rows


def test_judge_edge_shown_and_stop_kept():
    rows = _rows(r0_late=[100, 100], s_late=[60, 60], s_on_time=[80, 80], monkeys=list(range(0, 50)),
                 t_alpha=[75, 75], s_dd=[-20, -20], t_dd=[-18, -18])
    result = edge.judge(rows)
    assert result["edge_verdict"] == "edge shown"
    assert result["stop_verdict"] == "keep the 8% stop"


def test_judge_fails_survivorship_below_half_of_r0():
    rows = _rows(r0_late=[100], s_late=[40], s_on_time=[80], monkeys=[0, 1, 2],
                 t_alpha=[80], s_dd=[-20], t_dd=[-20])
    result = edge.judge(rows)
    assert result["gates"]["S1_survivorship"] is False
    assert result["edge_verdict"] == "edge not shown"


def test_judge_fails_luck_when_monkeys_do_as_well():
    rows = _rows(r0_late=[100], s_late=[60], s_on_time=[80], monkeys=list(range(0, 200)),
                 t_alpha=[80], s_dd=[-20], t_dd=[-20])
    result = edge.judge(rows)
    assert result["gates"]["L1_beats_random_picks"] is False
    assert result["edge_verdict"] == "edge not shown"


def test_judge_drops_stop_that_costs_too_much_alpha():
    rows = _rows(r0_late=[100], s_late=[60], s_on_time=[80], monkeys=[0],
                 t_alpha=[70], s_dd=[-20], t_dd=[-15])
    result = edge.judge(rows)
    assert result["gates"]["stop_alpha_ok"] is False
    assert result["stop_verdict"] == "drop the 8% stop"


# ── Add-on: core ETF stops (edge_core_stop.py) ─────────────────────────────

core_spec = importlib.util.spec_from_file_location(
    "edge_core_stop", Path("research_evidence/edge_check_20260926/edge_core_stop.py"))
core_stop = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core_stop)


def test_core_stop_starts_the_day_after_a_close_entry():
    # Bought at the Close of day 1 (100).  Day 1's own low can't trigger it.
    # Day 3's low of 94 is below 95% of the high so far (100 -> stop 95).
    bars = _bars([(101, 101, 90, 100), (100, 100, 99, 99), (99, 99, 94, 96)])
    out = core_stop.stop_exit_from_close(bars, bars.index[0], bars.index[-1], 0.05)
    assert out["stopped"] is True
    assert out["stop_date"] == bars.index[2]
    assert abs(out["stop_return"] - (-0.05)) < 1e-12
    assert abs(out["plain_return"] - (-0.04)) < 1e-12


def test_core_stop_adjusted_equity_uses_core_gross_times_weight():
    qqq = _bars([(100, 100, 100, 100), (100, 100, 94, 90), (90, 91, 89, 90)])
    spy = _bars([(100, 100, 100, 100), (100, 101, 99, 101), (101, 102, 100, 102)])
    equity = pd.Series([100.0, 105.0], index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    trades = pd.DataFrame([{
        "date": pd.Timestamp("2020-01-01"), "exit_date": pd.Timestamp("2020-01-03"),
        "label_end_date": pd.Timestamp("2020-01-03"), "entry_delay_days": 0,
        "core_gross": 0.5, "core_weights_json": json.dumps({"QQQ": 0.8, "SPY": 0.2, "TQQQ": 0.0}),
        "period_return": 0.05,
    }])
    rebuilt, info = core_stop.core_stop_adjusted_equity(
        equity, trades, {"QQQ": qqq, "SPY": spy},
        session_offset=lambda day, n: pd.Timestamp("2020-01-01"), stop_cost_pct=0.0002)
    # QQQ weight 0.4: plain -10%, stopped at 95 -> -5%: change 0.4*0.05 - 0.4*0.0002.
    expected = 100.0 * (1.0 + 0.05 + 0.4 * 0.05 - 0.4 * 0.0002)
    assert abs(rebuilt.iloc[-1] - expected) < 1e-9
    assert info == {"core_holdings": 2, "core_stop_exits": 1}
