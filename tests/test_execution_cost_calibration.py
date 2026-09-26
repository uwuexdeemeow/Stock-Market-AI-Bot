"""Liquidity buckets in execution_cost_calibration.py.

PLAIN ENGLISH: the dollar-volume feature is stored as log(1 + dollars).  The
bucket must undo the log first, otherwise a $2B-a-day stock looks "low".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import execution_cost_calibration as calibration


def _write(tmp_path, ticker, column, value):
    pd.DataFrame({column: [value]}, index=pd.to_datetime(["2026-09-25"])).to_parquet(tmp_path / f"{ticker}.parquet")


def test_log_dollar_volume_feature_is_converted_back_to_dollars(tmp_path):
    _write(tmp_path, "BIG", "factor_liquidity_dollar_vol_20d", np.log1p(2e9))
    _write(tmp_path, "MID", "factor_liquidity_dollar_vol_20d", np.log1p(3e8))
    _write(tmp_path, "SML", "factor_liquidity_dollar_vol_20d", np.log1p(5e7))
    assert calibration._liquidity_bucket("BIG", tmp_path)[0] == "high"
    assert calibration._liquidity_bucket("MID", tmp_path)[0] == "medium"
    assert calibration._liquidity_bucket("SML", tmp_path)[0] == "low"
    assert abs(calibration._liquidity_bucket("BIG", tmp_path)[1] - 2e9) < 1.0


def test_raw_dollar_volume_column_is_used_as_is(tmp_path):
    _write(tmp_path, "RAW", "dollar_volume_20d", 1.5e9)
    assert calibration._liquidity_bucket("RAW", tmp_path) == ("high", 1.5e9)


def test_missing_file_is_unknown(tmp_path):
    assert calibration._liquidity_bucket("NONE", tmp_path) == ("unknown", None)
