"""Tests for the pre-registered H-index-risk check (fake prices only).

PLAIN ENGLISH: the check compares two simple risk rules on QQQ with just
holding QQQ.  These tests pin the month-end signal days, the two rules, the
trade timing (next Open, plus the stress delay), trading costs, the
metrics and the pass/fail gates.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path("research_evidence/index_risk_20261016/index_risk_check.py")
spec = importlib.util.spec_from_file_location("index_risk_check", SCRIPT)
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)


def _frame(closes, opens=None, start="2020-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"Open": opens, "Close": closes}, index=days)


def test_month_end_sessions():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    ends = ir.month_end_sessions(days)
    assert list(ends.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31"]


def test_trend_weight_above_and_below_average():
    up = pd.Series(np.arange(1, 301, dtype=float), index=pd.bdate_range("2020-01-01", periods=300))
    assert ir.trend_weight(up, up.index[-1]) == 1.0
    down = pd.Series(np.arange(300, 0, -1, dtype=float), index=up.index)
    assert ir.trend_weight(down, down.index[-1]) == 0.0


def test_vol_weight_caps_at_one_and_scales_down():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=100)
    calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.002, 100)), index=days)
    assert ir.vol_weight(calm, days[-1]) == 1.0
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.04, 100)), index=days)
    w = ir.vol_weight(wild, days[-1])
    vol = wild.pct_change().dropna().tail(63).std(ddof=1) * np.sqrt(252)
    assert w == pytest.approx(0.20 / vol)


def test_simulate_buy_and_hold_pays_cost_once_and_tracks_price():
    qqq = _frame([100, 110, 121], opens=[100, 105, 115])
    bil = _frame([50, 50, 50])
    targets = pd.Series([1.0], index=[qqq.index[0] - pd.Timedelta(days=3)])
    sim = ir.simulate(qqq, bil, targets, qqq.index[0], qqq.index[-1], cost_bps=10, hold_only=True)
    assert sim["equity"].iloc[-1] == pytest.approx((1 - 0.001) * 121 / 100)


def test_simulate_trades_at_next_open_and_delay_shifts_it():
    # QQQ jumps at day 3's Open.  A month-end signal on day 1 switching to
    # QQQ is traded at day 2's Open (before the jump) or, one session late,
    # at day 3's Open (after it).
    qqq = _frame([100, 100, 100, 200, 200], opens=[100, 100, 100, 200, 200])
    bil = _frame([10] * 5)
    targets = pd.Series([0.0, 1.0], index=[qqq.index[0] - pd.Timedelta(days=3), qqq.index[1]])
    on_time = ir.simulate(qqq, bil, targets, qqq.index[0], qqq.index[-1], cost_bps=0)
    late = ir.simulate(qqq, bil, targets, qqq.index[0], qqq.index[-1], delay=1, cost_bps=0)
    assert on_time["equity"].iloc[-1] == pytest.approx(2.0)
    assert late["equity"].iloc[-1] == pytest.approx(1.0)


def test_metrics_drawdown_and_cagr():
    days = pd.bdate_range("2020-01-01", periods=3)
    curve = pd.Series([1.0, 0.5, 1.0], index=days)
    m = ir.metrics(curve, pd.Series(1.0, index=days))
    assert m["max_drawdown_pct"] == -50.0 and m["total_return_pct"] == 0.0


def _m(cagr, sharpe, dd):
    return {"cagr_pct": cagr, "sharpe": sharpe, "max_drawdown_pct": dd}


def test_judge_gates():
    decision = {"B": _m(15, 0.70, -50), "T": _m(13, 0.75, -35), "V": _m(11, 0.80, -38)}
    stress = {"B": _m(15, 0.70, -50), "T": _m(12, 0.72, -38), "V": _m(10, 0.65, -45)}
    gates = ir.judge(decision, stress)
    assert gates["T"]["passes_decision"] is True
    assert gates["V"]["R3_affordable"] is False and gates["V"]["R4_robust"] is False
