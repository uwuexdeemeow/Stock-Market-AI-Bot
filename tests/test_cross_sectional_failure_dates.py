"""Failed companies must not affect other stocks' cross-sectional ranks.

PLAIN ENGLISH: First Republic and Bed Bath & Beyond kept trading at
fractions of a cent after they failed.  The ranking step should ignore those
rows, so healthy stocks are ranked only against other real stocks.
"""
from __future__ import annotations

import pandas as pd

import cross_sectional_features as xs


def _write(tmp_path, values_by_ticker, dates):
    for ticker, values in values_by_ticker.items():
        pd.DataFrame({"ret_5d": values}, index=dates).to_parquet(tmp_path / f"{ticker}.parquet")


def test_rows_after_failure_date_are_left_out_of_rankings(tmp_path):
    dates = pd.bdate_range("2023-04-27", periods=4)          # Thu, Fri, Mon, Tue
    # FAIL has the highest value every day; it fails on 2023-04-28.
    _write(tmp_path, {"A": [1.0] * 4, "B": [2.0] * 4, "FAIL": [9.0] * 4}, dates)
    xs.apply_cross_sectional_rank_features(
        ["A", "B", "FAIL"], data_dir=str(tmp_path), sector_map={}, source_cols=["ret_5d"],
        failure_dates={"FAIL": "2023-04-28"},
    )
    b = pd.read_parquet(tmp_path / "B.parquet")["xs_rank_market_ret_5d"]
    fail = pd.read_parquet(tmp_path / "FAIL.parquet")["xs_rank_market_ret_5d"]
    # Up to the failure date FAIL is ranked and pushes B into the middle...
    assert b.iloc[0] < b.iloc[-1]
    assert fail.iloc[1] == fail.iloc[0] > b.iloc[1]
    # ...after it, B is top of the two real stocks and FAIL gets neutral 0.5.
    assert b.iloc[2] == b.iloc[3] == 1.0
    assert fail.iloc[2] == fail.iloc[3] == 0.5


def test_no_failure_dates_keeps_every_row(tmp_path):
    dates = pd.bdate_range("2023-04-27", periods=2)
    _write(tmp_path, {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]}, dates)
    xs.apply_cross_sectional_rank_features(
        ["A", "B", "C"], data_dir=str(tmp_path), sector_map={}, source_cols=["ret_5d"], failure_dates={},
    )
    b = pd.read_parquet(tmp_path / "B.parquet")["xs_rank_market_ret_5d"]
    assert b.iloc[0] == b.iloc[1] < 1.0
