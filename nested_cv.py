"""
nested_cv.py — Nested walk-forward cross-validation for hyperparameter search.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class FoldResult:
    params: dict
    score: float
    fold: int


def _time_splits(n: int, n_splits: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_size = n // (n_splits + 1)
    out = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if train_end < 50 or test_end <= test_start:
            break
        out.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return out


def nested_walk_forward_search(X: pd.DataFrame, y: pd.Series, param_grid: dict, outer_splits: int = 4, inner_splits: int = 3, embargo: int = 5) -> list[FoldResult]:
    n = len(X)
    outer = _time_splits(n, outer_splits, embargo)
    param_combos = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]
    results: list[FoldResult] = []

    for k, (tr, te) in enumerate(outer):
        inner = _time_splits(len(tr), inner_splits, embargo)
        best_params, best_score = None, -np.inf
        for params in param_combos:
            inner_scores = []
            for itr, ite in inner:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X.iloc[tr[itr]])
                X_test = scaler.transform(X.iloc[tr[ite]])
                model = XGBClassifier(**params, eval_metric="logloss")
                model.fit(X_train, y.iloc[tr[itr]])
                pred = model.predict(X_test)
                inner_scores.append(accuracy_score(y.iloc[tr[ite]], pred))
            mean = float(np.mean(inner_scores)) if inner_scores else -np.inf
            if mean > best_score:
                best_score, best_params = mean, params

        scaler = StandardScaler()
        X_train_outer = scaler.fit_transform(X.iloc[tr])
        X_test_outer = scaler.transform(X.iloc[te])
        model = XGBClassifier(**best_params, eval_metric="logloss")
        model.fit(X_train_outer, y.iloc[tr])
        outer_score = accuracy_score(y.iloc[te], model.predict(X_test_outer))
        results.append(FoldResult(params=best_params, score=float(outer_score), fold=k))
    return results
