"""Survivorship diagnostic for the stock list (research_evidence/survivorship_check_20260926).

PLAIN ENGLISH: fake index-membership snapshots check that the script counts
companies that later left the index, spots list names that joined later,
treats a ticker rename as "was already in the index", and counts how often
the strategy held a name before it joined.  No internet or real data needed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "watchlist_membership_check",
    ROOT / "research_evidence" / "survivorship_check_20260926" / "watchlist_membership_check.py")
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

CSV = """date,tickers
2009-06-01,"AAA,BBB,GONE,FB,BRK.B"
2012-06-01,"AAA,BBB,GONE,FB,BRK.B,NEWCO"
2020-01-02,"AAA,BBB,META,BRK.B,NEWCO"
"""


def test_counts_leavers_late_joiners_and_renames(monkeypatch):
    monkeypatch.setattr(check, "CHECK_DATES", ("2010-01-04",))
    snapshots = check.load_snapshots(CSV)
    result = check.analyse(snapshots, ["AAA", "META", "NEWCO", "BRK-B", "BTC-USD"])
    day = result["by_date"]["2010-01-04"]
    assert result["watchlist_size"] == 4            # BTC-USD is not a stock
    assert day["index_members"] == 5
    # GONE left; FB -> META is only a ticker change but still counts (documented caveat).
    assert day["members_no_longer_in_index_at_cutoff"] == 2
    assert day["watchlist_in_index_under_older_ticker"] == ["META"]
    assert day["watchlist_not_in_index_then"] == [{"ticker": "NEWCO", "first_in_index": "2012-06-01"}]
    assert day["watchlist_names_that_later_left_index"] == []


def test_holdings_before_joining_are_counted():
    trades = pd.DataFrame({
        "date": ["2011-01-03", "2013-01-02", "2013-02-01"],
        "overlay_weights_json": [json.dumps({"NEWCO": 0.2, "AAA": 0.2}), json.dumps({"NEWCO": 0.3}), None],
    })
    out = check.held_before_joining(trades, {"NEWCO": "2012-06-01"})
    assert out["overlay_holdings"] == 3
    assert out["holdings_of_names_not_in_index_at_start"] == 2
    assert out["holdings_before_name_joined_index"] == 1
    assert out["before_joining_by_ticker"] == {"NEWCO": 1}
