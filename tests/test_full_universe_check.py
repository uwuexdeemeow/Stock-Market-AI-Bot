"""Tests for the pre-registered H-full-universe check (fake data only).

PLAIN ENGLISH: the full-universe check decides whether the incumbent's edge
survives on a stock list that includes companies which later left the S&P
500.  These tests pin the pieces that could quietly go wrong: the member
pool and ticker changes, the coverage gate, the monthly "biggest 62" list,
the delisting sell price, the sector table, and the pass/fail rules.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("research_evidence/full_universe_20260929/full_universe_check.py")
spec = importlib.util.spec_from_file_location("full_universe_check", SCRIPT)
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)


def _snapshots():
    return pd.DataFrame({
        "date": pd.to_datetime(["2012-06-01", "2015-01-02", "2020-01-02", "2024-01-02"]),
        "members": [{"AAA", "OLD"}, {"AAA", "OLD", "FB"}, {"AAA", "META", "NEW"}, {"AAA", "META"}],
    })


# ── Pool and membership ────────────────────────────────────────────────────

def test_pool_includes_the_list_in_force_at_window_start_and_merges_renames():
    pool = fu.build_pool(_snapshots(), window=("2012-10-01", "2022-12-31"))
    # The 2012-06 snapshot was in force on 2012-10-01, so OLD counts.
    # FB and META are one company, stored under META.
    assert pool == {"AAA": {"AAA"}, "OLD": {"OLD"}, "META": {"FB", "META"}, "NEW": {"NEW"}}


def test_pool_ignores_snapshots_after_the_window():
    snaps = _snapshots()
    snaps.at[3, "members"] = {"AAA", "META", "LATE"}
    assert "LATE" not in fu.build_pool(snaps, window=("2012-10-01", "2022-12-31"))


def test_member_matrix_uses_latest_snapshot_and_aliases():
    sessions = pd.DatetimeIndex(["2012-01-03", "2013-01-02", "2016-01-04", "2021-01-04"])
    names = {"OLD": {"OLD"}, "META": {"FB", "META"}}
    members = fu.member_matrix(_snapshots(), sessions, names)
    assert members["OLD"].tolist() == [False, True, True, False]
    assert members["META"].tolist() == [False, False, True, True]


def test_coverage_gate_counts_removed_and_current_companies_separately():
    sessions = pd.bdate_range("2013-01-01", periods=10)
    members = pd.DataFrame({"A": True, "B": True, "C": True, "D": True}, index=sessions)
    price_dates = {
        "A": sessions,             # current, fully priced
        "B": sessions[:9],         # removed, 90% priced -> covered
        "C": sessions[:8],         # removed, 80% -> not covered
        # D: removed, no prices
    }
    table = fu.coverage_table(members, price_dates, current={"A"})
    result = fu.judge_coverage(table)
    assert result["current_covered_share"] == 1.0
    assert result["removed_covered_share"] == round(1 / 3, 4)
    assert result["C1_pass"] is False
    assert result["not_covered"] == ["C", "D"]


def test_coverage_gate_passes_at_the_thresholds():
    sessions = pd.bdate_range("2013-01-01", periods=10)
    names = [f"R{i}" for i in range(20)] + [f"C{i}" for i in range(20)]
    members = pd.DataFrame(True, index=sessions, columns=names)
    # 17 of 20 removed covered (85%), 19 of 20 current covered (95%).
    price_dates = {n: sessions for n in names[:17] + names[20:39]}
    table = fu.coverage_table(members, price_dates, current=set(names[20:]))
    assert fu.judge_coverage(table)["C1_pass"] is True


# ── Monthly universe ───────────────────────────────────────────────────────

def _dv_setup():
    sessions = pd.bdate_range("2012-06-01", "2012-12-31")
    dv = pd.DataFrame({"BIG": 300.0, "MID": 200.0, "SMALL": 100.0, "NONMEMBER": 999.0}, index=sessions)
    members = pd.DataFrame({"BIG": True, "MID": True, "SMALL": True, "NONMEMBER": False}, index=sessions)
    return dv, members


def test_monthly_universe_takes_biggest_members_only():
    dv, members = _dv_setup()
    lists = fu.monthly_universe(dv, members, size=2, lookback=20, start="2012-10-01")
    first = pd.Timestamp("2012-10-01")
    assert lists[first] == ["BIG", "MID"]
    assert sorted(lists) == [pd.Timestamp("2012-10-01"), pd.Timestamp("2012-11-01"), pd.Timestamp("2012-12-03")]


def test_monthly_universe_uses_only_data_before_the_month_starts():
    dv, members = _dv_setup()
    # SMALL becomes huge ON the first day of November: too late to count.
    dv.loc[pd.Timestamp("2012-11-01"):, "SMALL"] = 10_000.0
    lists = fu.monthly_universe(dv, members, size=2, lookback=20, start="2012-10-01")
    assert lists[pd.Timestamp("2012-11-01")] == ["BIG", "MID"]
    assert lists[pd.Timestamp("2012-12-03")][0] == "SMALL"


def test_monthly_universe_needs_history_and_recent_trading():
    dv, members = _dv_setup()
    dv.loc[:pd.Timestamp("2012-09-20"), "MID"] = np.nan     # too little history in Oct
    dv.loc[pd.Timestamp("2012-10-20"):, "BIG"] = np.nan     # stopped trading
    lists = fu.monthly_universe(dv, members, size=2, lookback=20, start="2012-10-01")
    assert lists[pd.Timestamp("2012-10-01")] == ["BIG", "SMALL"]
    assert "BIG" not in lists[pd.Timestamp("2012-11-01")]


def test_universe_mask_follows_the_month_in_force():
    lists = {pd.Timestamp("2013-01-02"): ["AAA"], pd.Timestamp("2013-02-01"): ["BBB"]}
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2012-12-31", "2013-01-15", "2013-01-15", "2013-02-15"]),
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
    })
    assert fu.universe_mask(panel, lists).tolist() == [False, True, False, True]


# ── Delisting ──────────────────────────────────────────────────────────────

def test_pad_after_delisting_holds_last_close_with_optional_haircut():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"Open": [10.0, 11.0], "High": [10, 11.0], "Low": [10, 11.0],
                          "Close": [10.0, 12.0], "Volume": [5.0, 5.0], "feat": [1.0, 2.0]}, index=idx)
    sessions = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    padded = fu.pad_after_delisting(frame, sessions)
    assert padded.index.tolist() == sessions.tolist()
    assert padded.loc["2020-01-07", "Open"] == 12.0 and padded.loc["2020-01-07", "Close"] == 12.0
    assert np.isnan(padded.loc["2020-01-06", "feat"])
    cut = fu.pad_after_delisting(frame, sessions, haircut=0.30)
    assert cut.loc["2020-01-06", "Close"] == 12.0 * 0.7
    assert cut.loc["2020-01-03", "Close"] == 12.0   # real bars are untouched


# ── Sectors ────────────────────────────────────────────────────────────────

def test_sic_table_narrow_codes_win_over_wide_ranges():
    assert fu.sic_to_sector(2834) == "XLV"     # drugs inside chemicals
    assert fu.sic_to_sector(2821) == "XLB"     # plastics
    assert fu.sic_to_sector(3674) == "XLK"     # semiconductors
    assert fu.sic_to_sector(6798) == "XLRE"    # REIT inside finance
    assert fu.sic_to_sector(6021) == "XLF"
    assert fu.sic_to_sector(6324) == "XLV"     # health insurer
    assert fu.sic_to_sector(4911) == "XLU"
    assert fu.sic_to_sector("5331") == "XLP"
    assert fu.sic_to_sector(None) == "OTHER"
    assert fu.sic_to_sector(9995) == "OTHER"


def test_every_sector_label_is_a_known_etf():
    assert {row[2] for row in fu.SIC_SECTOR_TABLE} <= set(fu.SECTOR_ETFS)


def test_renames_never_point_to_a_ticker_that_is_itself_renamed():
    assert not set(fu.RENAMES.values()) & set(fu.RENAMES)


# ── Pass/fail rules ────────────────────────────────────────────────────────

def _rows(r0_late, u_late, u_on, monkeys):
    rows = []
    for i in range(3):
        rows.append({"test": "R0", "delay": 1, "offset": i, "decision_alpha_vs_qqq_pct": r0_late})
        rows.append({"test": "U", "delay": 1, "offset": i, "decision_alpha_vs_qqq_pct": u_late})
        rows.append({"test": "U", "delay": 0, "offset": i, "decision_alpha_vs_qqq_pct": u_on})
    rows += [{"test": "MU", "delay": 0, "offset": i, "decision_alpha_vs_qqq_pct": m} for i, m in enumerate(monkeys)]
    return rows


def test_judge_edge_survives():
    result = fu.judge(_rows(400.0, 250.0, 260.0, list(range(-100, 100))))
    assert result["gates"] == {"U1_survivorship": True, "U2_beats_random_picks": True}
    assert result["verdict"] == "edge survives the full survivorship test"
    assert result["diagnostics"]["U_mean_delay_cost"] == 10.0


def test_judge_real_but_mostly_hindsight():
    result = fu.judge(_rows(400.0, 150.0, 160.0, list(range(-100, 100))))
    assert result["gates"]["U1_survivorship"] is False
    assert result["verdict"] == "edge real but mostly list hindsight"


def test_judge_edge_not_shown_when_random_picks_do_as_well():
    result = fu.judge(_rows(400.0, 250.0, 50.0, list(range(-100, 100))))
    assert result["gates"]["U2_beats_random_picks"] is False
    assert result["verdict"] == "edge not shown"


def test_judge_negative_late_alpha_fails_u1():
    assert fu.judge(_rows(-10.0, -8.0, 300.0, [0.0]))["gates"]["U1_survivorship"] is False


# ── Tiingo allowance messages ──────────────────────────────────────────────

def test_monthly_symbol_limit_reply_counts_as_quota_even_with_status_200():
    body = '{"detail": "You have run over your 500 symbol look up for this month. Please upgrade at https://api.tiingo.com/pricing"}'
    assert fu.is_quota_reply(200, body)
    assert fu.is_quota_reply(429, "")
    assert fu.is_quota_reply(200, '{"detail": "Error: You have run over your hourly request allocation."}')
    assert not fu.is_quota_reply(200, '[{"date": "2020-01-02T00:00:00.000Z", "close": 1.0}]')
    assert not fu.is_quota_reply(404, '{"detail": "Error: Ticker XYZ not found"}')
