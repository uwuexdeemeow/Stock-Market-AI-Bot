"""Legacy nested_cv tests split out of tests/test_nested_walkforward_defaults.py (nested_cv.py is in legacy/)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import nested_cv


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
