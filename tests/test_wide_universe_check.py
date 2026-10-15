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


# ── Runs step helpers ──────────────────────────────────────────────────────

def test_wikipedia_names_reads_both_cell_styles_and_markup():
    text = """{| class="wikitable sortable" id="changes"
|-
! Effective Date
|-
|October 6, 2026
|TWLO
|[[Twilio]]
|WBD
|[[Warner Bros. Discovery]]
|reason<ref name=":1">x</ref>
|-
|| July 26, 2017
|| AOS
|| [[A. O. Smith]]
|| BBBY
|| [[Bed Bath & Beyond Inc.|Bed Bath & Beyond]]
|| Market capitalization changes.
|}"""
    names = wu.wikipedia_names(text)
    assert names["BBBY"] == "Bed Bath & Beyond"
    assert names["TWLO"] == "Twilio" and names["AOS"] == "A. O. Smith"


def test_rerank_matches_live_cross_sectional_code(tmp_path):
    import cross_sectional_features as xs

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2015-01-01", periods=5)
    tickers = [f"T{i}" for i in range(10)]
    sector_map = {t: ("XLK" if i < 7 else "XLE") for i, t in enumerate(tickers)}
    cols = ["ret_5d", "hvol_20d"]
    for t in tickers:
        frame = pd.DataFrame(rng.normal(size=(len(days), 2)), index=days, columns=cols)
        if t == "T3":
            frame.iloc[1, 0] = np.nan
        frame.to_parquet(tmp_path / f"{t}.parquet")
    xs.apply_cross_sectional_rank_features(tickers, str(tmp_path), sector_map=sector_map,
                                           source_cols=cols, failure_dates={})
    source = pd.concat([pd.read_parquet(tmp_path / f"{t}.parquet")[cols].assign(ticker=t, date=days)
                        for t in tickers], ignore_index=True)
    mine = wu.rerank_cross_sectional(source, pd.Series(True, index=source.index), sector_map, source_cols=cols)
    for t in tickers:
        live = pd.read_parquet(tmp_path / f"{t}.parquet")
        ours = mine[mine["ticker"] == t].set_index("date")
        for col in ["xs_rank_sector_ret_5d", "xs_rank_market_ret_5d", "xs_rank_sector_hvol_20d"]:
            np.testing.assert_allclose(ours[col].to_numpy(), live[col].to_numpy())


def test_rerank_uses_only_eligible_rows():
    days = pd.bdate_range("2015-01-01", periods=1)
    source = pd.DataFrame({"date": list(days) * 3, "ticker": ["A", "B", "C"], "ret_5d": [1.0, 2.0, 3.0]})
    eligible = pd.Series([True, True, False])
    out = wu.rerank_cross_sectional(source, eligible, {}, source_cols=["ret_5d"])
    assert list(out["ticker"]) == ["A", "B"]
    assert list(out["xs_rank_market_ret_5d"]) == [0.5, 1.0]


def _rows(test, delay, values, start_seed=None):
    return [{"test": test, "delay": delay, "offset": i % 20, "decision_alpha_vs_qqq_pct": v,
             **({"seed": start_seed + i} if start_seed is not None else {})} for i, v in enumerate(values)]


def test_judge_stops_when_sources_disagree():
    rows = _rows("R0", 1, [100.0] * 3) + _rows("R0n", 1, [60.0] * 3)
    result = wu.judge(rows)
    assert result["gates"]["C2_source_agreement"] is False
    assert result["verdict"] == "source disagrees with project data"


def test_judge_applies_wide_universe_gates():
    rows = (_rows("R0", 1, [100.0] * 3) + _rows("R0n", 1, [90.0] * 3)
            + _rows("U", 1, [50.0] * 3) + _rows("U", 0, [60.0] * 3) + _rows("MU", 0, list(range(100)), 0)
            + _rows("W", 1, [80.0] * 3) + _rows("W", 0, [200.0] * 3) + _rows("MW", 0, list(range(100)), 0)
            + _rows("WN3", 1, [30.0] * 3))
    result = wu.judge(rows)
    gates = result["gates"]
    assert gates["C2_source_agreement"] and gates["U1_survivorship"] and gates["W1_width"] and gates["W2_skill"]
    assert gates["U2_beats_random_picks"] is False          # 60 < p95 of 0..99
    assert gates["W3_breadth"] is False                      # 30 < 50% of 80
    assert result["W_verdict"] == "real but too small or too narrow"
    assert result["U_verdict"] == "edge not shown"
