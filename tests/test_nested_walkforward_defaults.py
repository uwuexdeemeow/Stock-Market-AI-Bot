from __future__ import annotations

import pandas as pd
import numpy as np

import core_satellite_nested_walkforward as nested_wf
import nested_cv


def test_nested_walkforward_hard_default_min_train_years_is_three():
    assert nested_wf.DEFAULT_MIN_TRAIN_YEARS == 3


def test_build_fold_splits_default_allows_three_training_years():
    panel = pd.DataFrame({
        "date": pd.date_range("2020-01-01", "2025-12-31", freq="B"),
    })
    panel["_date"] = pd.to_datetime(panel["date"])

    splits = nested_wf.build_fold_splits(panel)

    assert [split.outer_year for split in splits] == [2023, 2024, 2025]


def test_pooled_splits_keep_dates_together_and_purge_session_horizon():
    sessions = pd.bdate_range("2024-01-02", periods=80)
    sampled = sessions[::5]
    row_dates = sampled.repeat(100)

    splits = nested_cv._date_group_splits(
        row_dates,
        n_splits=3,
        embargo=5,
        session_dates=sessions,
        min_train_size=500,
        min_test_size=200,
    )

    assert splits
    session_position = {date: pos for pos, date in enumerate(sessions)}
    for train_idx, test_idx in splits:
        train_dates = pd.DatetimeIndex(row_dates[train_idx])
        test_dates = pd.DatetimeIndex(row_dates[test_idx])
        assert not set(train_dates).intersection(test_dates)
        assert session_position[test_dates.min()] > session_position[train_dates.max()] + 5


def test_nested_search_exposes_only_inner_fold_winner(monkeypatch):
    sessions = pd.bdate_range("2023-01-02", periods=120)
    row_dates = sessions.repeat(20)
    y = pd.Series(np.tile([0, 1], len(row_dates) // 2))
    X = pd.DataFrame({"feature": y.astype(float)})

    class FakeClassifier:
        def __init__(self, quality=1, **_kwargs):
            self.quality = quality

        def fit(self, _x, _y, verbose=False):
            return self

        def predict_proba(self, x):
            base = np.asarray(x)[:, 0]
            rank = base if self.quality == 1 else -base
            probability = 1.0 / (1.0 + np.exp(-rank))
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(nested_cv, "XGBClassifier", FakeClassifier)
    results = nested_cv.nested_walk_forward_search(
        X,
        y,
        {"quality": [0, 1]},
        outer_splits=2,
        inner_splits=2,
        embargo=1,
        min_test_size=200,
        dates=row_dates,
        session_dates=sessions,
    )

    assert results
    assert len(results) <= 2
    assert all(result.params["quality"] == 1 for result in results)
