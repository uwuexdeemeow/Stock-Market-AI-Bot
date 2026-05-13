from __future__ import annotations

import pandas as pd

import core_satellite_nested_walkforward as nested_wf


def test_nested_walkforward_hard_default_min_train_years_is_three():
    assert nested_wf.DEFAULT_MIN_TRAIN_YEARS == 3


def test_build_fold_splits_default_allows_three_training_years():
    panel = pd.DataFrame({
        "date": pd.date_range("2020-01-01", "2025-12-31", freq="B"),
    })
    panel["_date"] = pd.to_datetime(panel["date"])

    splits = nested_wf.build_fold_splits(panel)

    assert [split.outer_year for split in splits] == [2023, 2024, 2025]
