"""Tests for the pre-registered H-newedge-3 round (fake data only).

PLAIN ENGLISH: these pin how SEC filing times become earnings numbers (which
day reacts, no peeking at the future), the three panel changes (earnings
score, real blackout, blended score), the coverage gate and the pass/fail
rules.  No network and no real prices are used.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "newedge_round3", Path("research_evidence/newedge_20261007/newedge_round3.py"))
ne3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne3)

DAYS = pd.bdate_range("2015-01-05", periods=60)   # Monday 5 Jan 2015 onwards, no holidays removed


def _panel():
    """Four stocks with flat prices, except A jumps +10% and B drops 10% on day 10."""
    rows = []
    for i, d in enumerate(DAYS):
        for j, t in enumerate(["A", "B", "C", "D"]):
            close = 100.0
            if i >= 10 and t == "A":
                close = 110.0
            if i >= 10 and t == "B":
                close = 90.0
            rows.append({"date": d, "ticker": t, "Close": close, "days_to_next_earnings": 60.0,
                         "factor_risk_on_score": float(j), "factor_defensive_score": float(j),
                         "factor_walkforward_score": float(j)})
    return pd.DataFrame(rows)


# A and B both report after the close of day 9, so day 10 is the reaction day.
AFTER_CLOSE = f"{DAYS[9].date()}T21:30:00.000Z"      # 16:30 New York (winter time)
EVENTS = {"A": [AFTER_CLOSE], "B": [AFTER_CLOSE]}


def test_filing_times_keep_only_8k_item_202():
    block = {"form": ["8-K", "8-K", "8-K/A", "10-Q"],
             "items": ["2.02,9.01", "5.02", "2.02", ""],
             "acceptanceDateTime": ["t1", "t2", "t3", "t4"]}
    assert ne3.filing_times(block) == ["t1"]


def test_reaction_index_before_and_after_the_close():
    monday = DAYS[0].date()
    assert ne3.reaction_index(f"{monday}T13:00:00.000Z", DAYS) == 0     # 08:00 New York -> same day
    assert ne3.reaction_index(f"{monday}T21:05:00.000Z", DAYS) == 1     # 16:05 New York -> next session
    saturday = (DAYS[4] + pd.Timedelta(days=1)).date()
    assert ne3.reaction_index(f"{saturday}T15:00:00.000Z", DAYS) == 5   # weekend -> next Monday
    assert ne3.reaction_index("2014-06-02T15:00:00.000Z", DAYS) is None  # before the data starts
    assert ne3.reaction_index(f"{DAYS[-1].date()}T21:05:00.000Z", DAYS) is None  # after the data ends


def test_earnings_frames_use_no_future_information():
    latest, days_to, table = ne3.earnings_frames(_panel(), EVENTS)
    # Reaction day is position 10; the score may first appear on position 11.
    assert latest["A"].iloc[:11].isna().all()
    assert np.isclose(latest["A"].iloc[11], 0.10)       # +10% vs a market median of 0
    assert np.isclose(latest["B"].iloc[11], -0.10)
    assert latest["A"].iloc[11:11 + ne3.DRIFT_WINDOW].notna().all()
    assert latest["A"].iloc[11 + ne3.DRIFT_WINDOW:].isna().all()
    assert latest["C"].isna().all()
    assert len(table) == 2
    # Calendar days to the reaction day; nothing known afterwards.
    assert days_to["A"].iloc[10] == 0.0
    assert days_to["A"].iloc[9] == 3.0                  # Friday to Monday: the weekend counts
    assert days_to["A"].iloc[5] == 7.0                  # Monday to the Monday after
    assert days_to["A"].iloc[11] == ne3.NO_EVENT_DAYS
    assert (days_to["C"] == ne3.NO_EVENT_DAYS).all()


def test_earnings_score_keeps_only_good_reactions():
    out = ne3.earnings_score(_panel(), EVENTS)
    day = out[out["date"] == DAYS[20]].set_index("ticker")
    assert day.loc["A", "factor_risk_on_score"] == 1.0                  # the only stock with good news
    assert day.loc[["B", "C", "D"], "factor_walkforward_score"].isna().all()
    before = out[out["date"] == DAYS[10]]
    assert before["factor_defensive_score"].isna().all()                # reaction day itself: nothing yet


def test_real_blackout_changes_only_the_earnings_column():
    panel = _panel()
    out = ne3.real_blackout(panel, EVENTS)
    assert out["factor_risk_on_score"].equals(panel["factor_risk_on_score"])
    row = out[(out["date"] == DAYS[8]) & (out["ticker"] == "A")].iloc[0]
    assert row["days_to_next_earnings"] == 4.0          # Thursday to the next Monday
    assert (out.loc[out["ticker"] == "D", "days_to_next_earnings"] == ne3.NO_EVENT_DAYS).all()


def test_blended_score_is_half_and_half():
    out = ne3.blended_score(_panel(), EVENTS)
    day = out[out["date"] == DAYS[20]].set_index("ticker")
    # Own ranks: A .25, B .5, C .75, D 1.  Earnings ranks: A 1, B .5, others neutral .5.
    assert np.isclose(day.loc["A", "factor_risk_on_score"], 0.5 * 0.25 + 0.5 * 1.0)
    assert np.isclose(day.loc["B", "factor_risk_on_score"], 0.5 * 0.50 + 0.5 * 0.5)
    assert np.isclose(day.loc["D", "factor_risk_on_score"], 0.5 * 1.00 + 0.5 * 0.5)
    early = out[out["date"] == DAYS[2]].set_index("ticker")
    assert np.isclose(early.loc["D", "factor_risk_on_score"], 0.75)     # nobody has reported yet


def test_coverage_counts_ticker_quarters_with_an_event():
    panel = _panel()
    _latest, _days_to, table = ne3.earnings_frames(panel, EVENTS)
    result = ne3.coverage(panel, table)
    # One quarter, four tickers with 60 rows each; only A and B have an event.
    assert result["ticker_quarters"] == 4 and result["covered"] == 2
    assert result["share"] == 0.5 and not result["C0_pass"]
    assert set(result["missing_quarters_by_ticker"]) == {"C", "D"}


def _rows(p_late=80.0, p_on=90.0, p_n3=50.0, x_on_step=10.0, x_late=430.0, ps_late=350.0, ps_n3=200.0):
    rows = []
    for o in range(20):
        s_on = 400.0 + (o % 5) * 50   # spread 200, mean delay cost 64
        rows += [{"test": "S", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": s_on},
                 {"test": "S", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": 436.0},
                 {"test": "P", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": p_on},
                 {"test": "P", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": p_late},
                 {"test": "P_N3", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": p_n3},
                 {"test": "X", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": 430.0 + (o % 5) * x_on_step},
                 {"test": "X", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": x_late},
                 {"test": "PS", "delay": 0, "offset": o, "decision_alpha_vs_qqq_pct": 360.0},
                 {"test": "PS", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": ps_late},
                 {"test": "PS_N3", "delay": 1, "offset": o, "decision_alpha_vs_qqq_pct": ps_n3}]
    return rows


def test_judge_passes_when_every_gate_holds():
    result = ne3.judge(_rows())
    assert all(result["gates"].values())
    assert result["decision_pass"] == {"P": True, "X": True, "PS": True}


def test_judge_p_needs_every_run_to_beat_qqq():
    rows = _rows()
    for row in rows:
        if row["test"] == "P" and row["delay"] == 0 and row["offset"] == 3:
            row["decision_alpha_vs_qqq_pct"] = -1.0
    result = ne3.judge(rows)
    assert not result["gates"]["P1"] and not result["decision_pass"]["P"]
    assert result["numbers"]["P_runs_beating_qqq"] == 39


def test_judge_p_fails_when_alpha_rests_on_top3():
    assert not ne3.judge(_rows(p_n3=30.0))["gates"]["P3"]     # 30 < half of 80


def test_judge_x_needs_a_smaller_spread():
    # Spread 4 * 40 = 160 > 0.75 * 200.
    assert not ne3.judge(_rows(x_on_step=40.0))["gates"]["X2"]
    # 380 < 0.9 * 436.
    assert not ne3.judge(_rows(x_late=380.0))["gates"]["X1"]


def test_judge_ps_uses_the_75_percent_bar():
    assert ne3.judge(_rows(ps_late=330.0))["gates"]["PS1"]        # 330 >= 327
    assert not ne3.judge(_rows(ps_late=320.0))["gates"]["PS1"]
    assert not ne3.judge(_rows(ps_n3=100.0))["gates"]["PS2"]      # 100 < half of 350
