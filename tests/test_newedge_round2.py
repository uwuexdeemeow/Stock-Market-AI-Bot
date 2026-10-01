"""Tests for the pre-registered H-newedge-2 round (fake data only).

PLAIN ENGLISH: these pin the three score changes (low-volatility filter,
5-day smoothing, sector-neutral ranks) and the pass/fail rules.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "newedge_round2", Path("research_evidence/newedge_20261002/newedge_round2.py"))
ne2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne2)


def _panel():
    dates = pd.bdate_range("2015-01-02", periods=6)
    rows = []
    for d in dates:
        for i, (t, sector) in enumerate([("A", "XLK"), ("B", "XLK"), ("C", "XLF"),
                                         ("D", "XLF"), ("E", "XLE"), ("F", "XLE")]):
            rows.append({"date": d, "ticker": t, "sector": sector, "factor_idio_vol_252_spy": float(i),
                         "factor_risk_on_score": float(i + d.day), "factor_defensive_score": float(i),
                         "factor_walkforward_score": float(i)})
    return pd.DataFrame(rows)


def test_quality_filter_removes_the_most_volatile_third():
    out = ne2.quality_filter(_panel())
    day = out[out["date"] == out["date"].min()]
    # Vol ranks 1/6..6/6: E (5/6) and F (6/6) are above 2/3 and lose their scores.
    assert day.loc[day["ticker"].isin(["E", "F"]), "factor_risk_on_score"].isna().all()
    assert day.loc[~day["ticker"].isin(["E", "F"]), "factor_risk_on_score"].notna().all()


def test_smooth_scores_use_only_the_last_five_rows():
    out = ne2.smooth_scores(_panel())
    a = out[out["ticker"] == "A"].sort_values("date")
    raw = _panel()
    raw_a = raw[raw["ticker"] == "A"].sort_values("date")["factor_risk_on_score"].to_numpy()
    assert a["factor_risk_on_score"].iloc[:4].isna().all()          # fewer than 5 rows -> no score
    assert np.isclose(a["factor_risk_on_score"].iloc[4], raw_a[:5].mean())
    assert np.isclose(a["factor_risk_on_score"].iloc[5], raw_a[1:6].mean())


def test_sector_neutral_ranks_within_sector():
    out = ne2.sector_neutral(_panel())
    day = out[out["date"] == out["date"].min()].set_index("ticker")
    # In each two-stock sector the higher score gets rank 1.0, the lower 0.5.
    assert day.loc["B", "factor_walkforward_score"] == 1.0 and day.loc["A", "factor_walkforward_score"] == 0.5
    assert day.loc["F", "factor_walkforward_score"] == 1.0


def test_idea_config_sets_one_per_sector_only_for_n():
    assert ne2.idea_config("N", {"max_per_sector": 2})["max_per_sector"] == 1
    assert ne2.idea_config("Q", {"max_per_sector": 2})["max_per_sector"] == 2


def _rows(q_med=420.0, q_n3=230.0, m_on=None, m_late=420.0, n_med=300.0, n_n3=200.0):
    rows = []
    for o in range(20):
        s_on = 400.0 + (o % 5) * 50   # spread 200
        rows += [{"test": "S", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": s_on + 10},
                 {"test": "S", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": 436.0},
                 {"test": "Q", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": q_med},
                 {"test": "Q_N3", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": q_n3},
                 {"test": "M", "delay": 0, "offset": o,
                  "decision_alpha_vs_qqq_pct": (m_on if m_on is not None else 420.0 + (o % 5) * 10)},
                 {"test": "M", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": m_late},
                 {"test": "N", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": n_med},
                 {"test": "N_N3", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": n_n3}]
    return rows


def test_judge_applies_each_ideas_gates():
    result = ne2.judge(_rows())
    # Q: 420 >= 0.9*436 and 230 >= 0.5*420.
    assert result["decision_pass"]["Q"]
    # M: spread 40 <= 0.75*200; late 420 >= 392; delay cost of M is small.
    assert result["gates"]["M_G1"] and result["gates"]["M_G2"] and result["decision_pass"]["M"]
    # N: 300 < 392 -> fails G1.
    assert not result["gates"]["N_G1"] and not result["decision_pass"]["N"]


def test_judge_q_fails_when_alpha_rests_on_top3():
    assert not ne2.judge(_rows(q_n3=150.0))["gates"]["Q_G2"]
