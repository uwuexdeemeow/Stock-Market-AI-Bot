"""
nested_cv.py — Nested walk-forward cross-validation for hyperparameter search.

PLAIN ENGLISH:
Ordinary grid search can cheat by tuning on the same data you later test
on. The cure is NESTED cross-validation:
  * OUTER loop: split time into several consecutive test blocks.
  * INNER loop: inside each outer-training block, split again and search
    hyperparameters there. The outer test block is never touched until the
    end.
This gives an honest estimate of how well tuning *itself* will generalize.

Exposes: `nested_walk_forward_search(X, y, param_grid)`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


@dataclass
class FoldResult:
    params: dict
    score: float
    fold: int


def _time_splits(n: int, n_splits: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) for rolling walk-forward."""
    fold_size = n // (n_splits + 1)
    out = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        out.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return out


def nested_walk_forward_search(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    outer_splits: int = 4,
    inner_splits: int = 3,
    embargo: int = 5,
) -> list[FoldResult]:
    """Return best-params-per-outer-fold. Caller averages / diagnoses drift."""
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
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                model.fit(X.iloc[tr[itr]], y.iloc[tr[itr]])
                pred = model.predict(X.iloc[tr[ite]])
                inner_scores.append(accuracy_score(y.iloc[tr[ite]], pred))
            mean = float(np.mean(inner_scores)) if inner_scores else -np.inf
            if mean > best_score:
                best_score, best_params = mean, params

        # evaluate chosen params on held-out outer test
        model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
        model.fit(X.iloc[tr], y.iloc[tr])
        outer_score = accuracy_score(y.iloc[te], model.predict(X.iloc[te]))
        results.append(FoldResult(params=best_params, score=outer_score, fold=k))
    return results
