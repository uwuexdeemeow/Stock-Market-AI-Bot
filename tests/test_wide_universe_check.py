"""Tests for the H-wide-universe data step (fake data only).

PLAIN ENGLISH: the wide-universe check joins two free price sources (WIKI up
to March 2018, Tiingo after).  A bad join could quietly add fake returns or
glue two different companies together.  These tests pin the join rule, the
WIKI ticker spellings, the choice of WIKI file, and the WIKI loader.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("research_evidence/wide_universe_20261016/wide_universe_check.py")
spec = importlib.util.spec_from_file_location("wide_universe_check", SCRIPT)
wu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wu)


def _prices(start, periods, level=100.0, seed=0):
    days = pd.bdate_range(start, periods=periods)
    rets = np.random.default_rng(seed).normal(0, 0.02, periods)
    close = level * np.cumprod(1 + rets)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": 1000.0, "RawClose": close, "RawVolume": 1000.0}, index=days)


def test_wiki_candidates_tries_every_spelling():
    assert wu.wiki_candidates({"BRK-B"}) == ["BRK-B", "BRK_B", "BRK.B"]
    assert wu.wiki_candidates({"FB", "META"}) == ["FB", "META"]


def test_stitch_fills_only_before_tiingo_and_keeps_returns():
    wiki = _prices("2012-01-02", 600, level=50.0)
    tiingo = wiki.iloc[400:].copy()
    tiingo[["Open", "High", "Low", "Close"]] *= 2.0        # different adjustment base
    joined, info = wu.stitch(tiingo, wiki)
    assert info["source"] == "wiki+tiingo"
    assert info["wiki_days_added"] == 400
    assert abs(info["scale"] - 2.0) < 1e-9
    # Tiingo bars are untouched, and returns across the join are WIKI's own.
    pd.testing.assert_frame_equal(joined.loc[tiingo.index], tiingo)
    np.testing.assert_allclose(joined["Close"].pct_change().iloc[1:400],
                               wiki["Close"].pct_change().iloc[1:400])
    assert joined.index.is_monotonic_increasing and not joined.index.duplicated().any()


def test_stitch_drops_wiki_when_returns_disagree():
    wiki = _prices("2012-01-02", 600, seed=1)
    tiingo = _prices("2012-01-02", 600, seed=2).iloc[400:]   # a different security
    joined, info = wu.stitch(tiingo, wiki)
    assert info["wiki_dropped"] == "returns_disagree"
    pd.testing.assert_frame_equal(joined, tiingo)


def test_stitch_drops_wiki_when_overlap_too_short():
    wiki = _prices("2012-01-02", 300)
    tiingo = _prices("2013-02-01", 300, seed=3)               # shares < 20 days
    joined, info = wu.stitch(tiingo, wiki)
    assert info["wiki_dropped"] == "overlap_too_short"
    pd.testing.assert_frame_equal(joined, tiingo)


def test_stitch_uses_wiki_alone_when_tiingo_missing():
    wiki = _prices("2012-01-02", 100)
    joined, info = wu.stitch(pd.DataFrame(), wiki)
    assert info["source"] == "wiki"
    pd.testing.assert_frame_equal(joined, wiki)


def test_pick_wiki_name_takes_most_member_days():
    member_days = pd.bdate_range("2013-01-01", periods=100)
    frames = {"OLD": _prices("2013-01-01", 30), "NEW": _prices("2013-01-01", 90), "X": pd.DataFrame()}
    assert wu.pick_wiki_name(frames, member_days) == "NEW"
    assert wu.pick_wiki_name({"X": pd.DataFrame()}, member_days) is None


def test_wiki_frame_cuts_after_march_2018_and_bad_bars():
    days = pd.bdate_range("2018-03-20", periods=10)
    raw = pd.DataFrame({"ticker": "AAA", "date": days, "open": 10.0, "high": 10.0, "low": 10.0,
                        "close": 10.0, "volume": 5.0, "adj_open": 9.0, "adj_high": 9.0,
                        "adj_low": 9.0, "adj_close": 9.0, "adj_volume": 5.5})
    raw.loc[1, "adj_close"] = 0.0
    frame = wu.wiki_frame(raw)
    assert frame.index.max() <= pd.Timestamp(wu.WIKI_END)
    assert days[1] not in frame.index
    assert frame["RawClose"].iloc[0] == 10.0 and frame["Close"].iloc[0] == 9.0
