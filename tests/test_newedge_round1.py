"""Tests for the pre-registered H-newedge-1 round (fake data only).

PLAIN ENGLISH: these pin the three idea switches (momentum keep rule,
volatility scaling, trend weights), the Sharpe/drawdown helper and the
pass/fail rules, so the real run measures exactly what was written down.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "newedge_round1", Path("research_evidence/newedge_20261002/newedge_round1.py"))
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)


def test_momentum_keep_set_only_keeps_held_top_momentum_names():
    day = pd.DataFrame({
        "ticker": list("ABCDEFGHIJ"),
        "score": [0.5] * 9 + [np.nan],
        "factor_mom_12_1": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 5.0],
    })
    # Among the 9 scored names, A (rank 1.0), B (0.89) and C (0.78) are >= 0.70.
    assert ne.momentum_keep_set(day, {"A", "C", "D", "J"}, "score") == {"A", "C"}
    assert ne.momentum_keep_set(day, set(), "score") == set()


def test_vol_scale_shrinks_only_when_markets_are_wild():
    days = pd.bdate_range("2020-01-01", periods=30)
    calm = pd.Series(100.0 * np.cumprod(1 + np.tile([0.001, -0.001], 15)), index=days)
    wild = pd.Series(100.0 * np.cumprod(1 + np.tile([0.04, -0.04], 15)), index=days)
    assert ne.vol_scale(calm, days[-1]) == 1.0
    scale = ne.vol_scale(wild, days[-1])
    assert ne.V_MIN_SCALE <= scale < 0.5
    # Not enough history -> no scaling.
    assert ne.vol_scale(wild, days[5]) == 1.0


def test_vol_scale_uses_no_future_prices():
    days = pd.bdate_range("2020-01-01", periods=40)
    prices = pd.Series(100.0, index=days)
    prices.iloc[30:] = np.linspace(100, 300, 10)   # a wild move AFTER the decision day
    assert ne.vol_scale(prices, days[25]) == 1.0


def test_trend_weights_split_fifths_between_trending_assets_and_cash():
    days = pd.bdate_range("2019-01-01", periods=300)
    up = pd.Series(np.linspace(100, 150, 300), index=days)
    down = pd.Series(np.linspace(100, 80, 300), index=days)
    cash = pd.Series(np.linspace(100, 101, 300), index=days)
    closes = {"SPY": up, "QQQ": up, "TLT": down, "IEF": down, "GLD": up, "BIL": cash}
    w = ne.trend_weights(closes, days[-1], lookback=252)
    assert w == {"SPY": 0.2, "QQQ": 0.2, "GLD": 0.2, "BIL": 0.4}
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Too little history: everything sits in cash.
    assert ne.trend_weights(closes, days[100], lookback=252) == {"BIL": 1.0}


def test_window_stats_sharpe_and_drawdown():
    idx = pd.to_datetime(["2013-01-31", "2013-03-01", "2013-03-29", "2013-04-26", "2023-06-01"])
    eq = pd.Series([100.0, 110.0, 99.0, 108.9, 500.0], index=idx)
    stats = ne.window_stats(eq)
    assert round(stats["max_dd_pct"], 2) == -10.0
    rets = pd.Series([0.10, -0.10, 0.10])
    assert np.isclose(stats["sharpe"], rets.mean() / rets.std(ddof=1) * np.sqrt(252 / 20))


def _rows():
    rows = []
    for o in range(20):
        rows += [
            {"test": "S", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": 460.0, "sharpe": 1.2, "max_dd_pct": -25.0},
            {"test": "S", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": 436.0, "sharpe": 1.2, "max_dd_pct": -25.0},
            {"test": "W", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": 410.0},
            {"test": "W", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": 400.0},
            {"test": "W_N3", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": 250.0},
            {"test": "V", "delay": 1, "offset": o, "sharpe": 1.3 if o < 16 else 1.0, "max_dd_pct": -20.0},
            {"test": "T_core", "delay": 1, "offset": o, "sharpe": 0.9 if o < 10 else 0.5,
             "max_dd_pct": -15.0, "qqq_max_dd_pct": -35.0},
            {"test": "E_core", "delay": 1, "offset": o, "sharpe": 0.7},
        ]
    return rows


def test_judge_applies_each_ideas_gates():
    result = ne.judge(_rows())
    # W: 400 >= 0.9*436; delay cost 10 <= 0.5*24; 250 >= 0.5*400.
    assert result["gates"]["W1_alpha_kept"] and result["gates"]["W2_delay_cost_halved"]
    assert result["gates"]["W3_not_few_names"] and result["decision_pass"]["W"]
    # V: better on both in 16 of 20 start days.
    assert result["numbers"]["V_start_days_better_sharpe_and_dd"] == 16 and result["decision_pass"]["V"]
    # T: better Sharpe than E on only 10 of 20 -> fails even with a shallow drawdown.
    assert result["gates"]["T2_dd_vs_QQQ"] and not result["decision_pass"]["T"]


def test_judge_w_fails_when_delay_cost_is_not_halved():
    rows = [r for r in _rows() if not (r["test"] == "W" and r["delay"] == 0)]
    rows += [{"test": "W", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": 430.0} for o in range(20)]
    assert not ne.judge(rows)["gates"]["W2_delay_cost_halved"]
