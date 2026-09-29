"""Tests for the pre-registered H-top-names and H-benchmark checks (fake data only).

PLAIN ENGLISH: these pin the arithmetic that decides the two verdicts:
which tickers count as the biggest earners, how the "best pick made
typical" equity is rebuilt, how the exposure-matched benchmark and the
daily regression are computed, and the pass/fail rules.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tn = _load("top_names_check", "research_evidence/top_names_20260929/top_names_check.py")
bm = _load("benchmark_check", "research_evidence/benchmark_20260929/benchmark_check.py")

D1, D2 = pd.Timestamp("2015-01-02"), pd.Timestamp("2015-02-02")


def _trades():
    return pd.DataFrame({
        "date": [D1, D2],
        "exit_date": [pd.Timestamp("2015-01-30"), pd.Timestamp("2015-03-02")],
        "label_end_date": [pd.Timestamp("2015-01-30"), pd.Timestamp("2015-03-02")],
        "period_return": [0.05, 0.02],
        "overlay_weights_json": [json.dumps({"AAA": 0.2, "BBB": 0.2}), json.dumps({"AAA": 0.2, "CCC": 0.2})],
    })


LABELS = {(D1, "AAA"): 0.30, (D1, "BBB"): 0.05, (D2, "AAA"): -0.10, (D2, "CCC"): 0.10}


# ── H-top-names ────────────────────────────────────────────────────────────

def test_label_column_matches_engine_naming():
    assert tn.label_column(20, 5) == "forward_return_20d"
    assert tn.label_column(5, 5) == "forward_return"


def test_ticker_contributions_add_weight_times_return_inside_window():
    contrib = tn.ticker_contributions(_trades(), LABELS, window=("2015-01-01", "2015-12-31"))
    assert contrib == {"AAA": 0.2 * 0.30 + 0.2 * -0.10, "BBB": 0.2 * 0.05, "CCC": 0.2 * 0.10}
    assert tn.ticker_contributions(_trades(), LABELS, window=("2016-01-01", "2016-12-31")) == {}


def test_top_contributors_average_across_runs_counting_missing_as_zero():
    runs = [{"AAA": 3.0, "BBB": 1.0}, {"BBB": 1.0, "CCC": 2.5}]
    # averages: AAA 1.5, BBB 1.0, CCC 1.25
    assert tn.top_contributors(runs, 2) == ["AAA", "CCC"]
    assert tn.top_contributors([{"B": 1.0, "A": 1.0}], 1) == ["A"]   # tie -> name order


def test_median_label_uses_only_scored_rows():
    panel = pd.DataFrame({
        "date": [D1, D1, D1, D1],
        "ticker": ["A", "B", "C", "D"],
        "forward_return_20d": [0.0, 0.1, 0.2, 9.9],
        "factor_walkforward_score": [0.5, 0.5, 0.5, np.nan],
    })
    assert tn.median_label_by_date(panel, "forward_return_20d") == {D1: 0.1}


def test_best_pick_swap_replaces_only_the_best_pick():
    equity = pd.Series([100.0], index=[pd.Timestamp("2014-12-31")])
    medians = {D1: 0.02, D2: 0.0}
    rebuilt, info = tn.best_pick_swapped_equity(equity, _trades(), LABELS, medians)
    # Period 1: best is AAA (0.30) -> change 0.2*(0.02-0.30) = -0.056.
    # Period 2: best is CCC (0.10) -> change 0.2*(0.0-0.10) = -0.02.
    assert rebuilt.iloc[1] == 100.0 * (1 + 0.05 - 0.056)
    assert rebuilt.iloc[2] == rebuilt.iloc[1] * (1 + 0.02 - 0.02)
    assert info["periods_swapped"] == 2


def _tn_rows(s_late, n3_late, p1, monkeys):
    rows = []
    for i in range(3):
        rows += [{"test": "S", "delay": 1, "offset": i, "decision_alpha_vs_qqq_pct": s_late},
                 {"test": "S", "delay": 0, "offset": i, "decision_alpha_vs_qqq_pct": s_late},
                 {"test": "N3", "delay": 1, "offset": i, "decision_alpha_vs_qqq_pct": n3_late},
                 {"test": "P1", "delay": 0, "offset": i, "decision_alpha_vs_qqq_pct": p1}]
    rows += [{"test": "MP", "delay": 0, "offset": i, "decision_alpha_vs_qqq_pct": m} for i, m in enumerate(monkeys)]
    return rows


def test_top_names_judge_broad_edge():
    result = tn.judge(_tn_rows(400.0, 250.0, 150.0, list(range(-200, 100))))
    assert result["gates"] == {"T1_names": True, "T2_big_winners": True}
    assert result["verdict"] == "edge is broad"


def test_top_names_judge_both_failures_are_named():
    result = tn.judge(_tn_rows(400.0, 100.0, 50.0, list(range(-200, 100))))
    assert result["gates"] == {"T1_names": False, "T2_big_winners": False}
    assert "few names" in result["verdict"] and "big winner" in result["verdict"]


# ── H-benchmark ────────────────────────────────────────────────────────────

def test_portfolio_weights_scale_core_by_core_gross():
    trade = {"core_gross": 0.6, "core_weights_json": json.dumps({"QQQ": 1.0, "SPY": 0.0}),
             "overlay_weights_json": json.dumps({"AAA": 0.2, "BBB": 0.2})}
    assert bm.portfolio_weights(trade) == {"QQQ": 0.6, "AAA": 0.2, "BBB": 0.2}


def _bars(opens, closes, days):
    return pd.DataFrame({"Open": opens, "Close": closes}, index=pd.DatetimeIndex(days))


def test_period_path_from_entry_open_with_cash_and_cost():
    days = ["2015-01-05", "2015-01-06", "2015-01-07"]
    stock = _bars([10.0, 11.0, 12.0], [11.0, 12.0, 12.0], days)
    cash = _bars([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], days)
    ret = bm.period_path({"AAA": stock}, {"AAA": 0.5}, cash, pd.Timestamp(days[0]), pd.Timestamp(days[2]), cost=0.001)
    # value: day1 0.5 + 0.5*1.1 = 1.05; day2 0.5 + 0.5*1.2 = 1.10; day3 same.
    assert np.isclose(ret.iloc[0], 0.05 - 0.001)
    assert np.isclose(ret.iloc[1], 1.10 / 1.05 - 1)
    assert np.isclose(ret.iloc[2], 0.0)


def test_exposure_matched_equity_mixes_qqq_and_cash_by_gross():
    days = pd.bdate_range("2015-01-02", "2015-03-06")
    qqq = _bars(np.full(len(days), 100.0), np.full(len(days), 110.0), days)
    cash = _bars(np.full(len(days), 1.0), np.full(len(days), 1.01), days)
    trades = _trades().assign(gross_exposure=[0.6, 1.2], entry_delay_days=0)
    step = lambda d, n: d + pd.offsets.BDay(n)
    eq = bm.exposure_matched_equity(trades, qqq, cash, 100.0, "2014-12-31", step)
    assert np.isclose(eq.iloc[1], 100.0 * (1 + 0.6 * 0.10 + 0.4 * 0.01))
    # g > 1: the extra 0.2 is borrowed at the cash return.
    assert np.isclose(eq.iloc[2], eq.iloc[1] * (1 + 1.2 * 0.10 - 0.2 * 0.01))


def test_window_return_needs_three_points():
    s = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2013-01-02", "2014-01-02", "2015-01-02"]))
    assert np.isclose(bm.window_return(s, ("2013-01-01", "2022-12-31")), 0.21)
    assert bm.window_return(s.iloc[:2], ("2013-01-01", "2022-12-31")) is None


def test_beta_adjusted_recovers_known_beta_and_alpha():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2013-01-01", periods=2000)
    x = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    cash = pd.Series(0.0, index=idx)
    y = 0.001 + 0.5 * x + pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    out = bm.beta_adjusted(y, x, cash, lambda s: float(s.mean() / (s.std() / np.sqrt(len(s)))))
    assert abs(out["beta"] - 0.5) < 0.01
    assert abs(out["alpha_annual_pct"] - 0.001 * 252 * 100) < 1.0
    assert out["nw_tstat"] > 10


def _bm_rows(bexp, t):
    return [{"test": "S", "delay": 1, "offset": i, "bexp_alpha_pct": bexp, "nw_tstat": t} for i in range(3)]


def test_benchmark_judge_verdicts():
    assert bm.judge(_bm_rows(50.0, 2.5))["verdict"] == "edge holds against a fair benchmark"
    assert bm.judge(_bm_rows(50.0, 1.5))["verdict"] == "beats a fair benchmark, but not reliably"
    assert bm.judge(_bm_rows(-5.0, 3.0))["verdict"].startswith("headline alpha is mostly market exposure")
