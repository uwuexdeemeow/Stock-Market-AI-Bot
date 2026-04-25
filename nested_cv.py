"""
nested_cv.py — Nested walk-forward cross-validation for hyperparameter search.

What this does
--------------
Finds the best XGBoost hyperparameters (e.g. max_depth, min_child_weight,
reg_lambda) by running a nested time-series cross-validation:

  Outer folds: sequential train/test splits across time (no lookahead).
  Inner folds: within each outer training window, try every param combo and
               pick the one with the best average inner-fold accuracy.

This is the standard way to avoid overfitting hyperparameters to the test set:
the test set is ONLY touched in the outer evaluation, never during inner tuning.

The function returns one FoldResult per outer fold, each recording which params
won on that fold.  Callers aggregate those results (e.g. pick best-scoring fold
or take the mode across folds) to choose final production parameters.

New in v2
---------
- `fixed_params` argument: pass XGBoost settings that should NOT be tuned
  (e.g. learning_rate, subsample, tree_method).  These are merged with each
  tunable param combo before fitting, so param_grid only needs to contain the
  parameters you actually want to search over.
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
    # Which hyperparameter values won on this outer fold.
    params: dict
    # Accuracy on the outer test fold (0–1).
    score: float
    # Which outer fold index (0-based).
    fold: int


def _time_splits(n: int, n_splits: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Produce sequential train/test index pairs.

    Each split uses all rows from 0..train_end for training and
    train_end+embargo..test_end for testing.  The embargo gap prevents
    forward leakage when return windows overlap.
    """
    fold_size = n // (n_splits + 1)
    out = []
    for k in range(n_splits):
        train_end  = fold_size * (k + 1)
        test_start = train_end + embargo
        test_end   = min(test_start + fold_size, n)
        if train_end < 50 or test_end <= test_start:
            break
        out.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return out


def nested_walk_forward_search(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    fixed_params: dict | None = None,
    outer_splits: int = 4,
    inner_splits: int = 3,
    embargo: int = 5,
) -> list[FoldResult]:
    """
    Nested walk-forward hyperparameter search.

    Args
    ----
    X            : Feature matrix (rows = time steps, columns = features).
                   Must be a pd.DataFrame so iloc works.
    y            : Binary label series (0 = DOWN, 1 = UP).
    param_grid   : Dict mapping param name → list of values to try.
                   Only the parameters you want to tune go here.
                   Example: {"max_depth": [3,4,5], "reg_lambda": [0.1,1,10]}
    fixed_params : Dict of XGBoost parameters that are fixed (not tuned).
                   These are merged with every param combo before fitting.
                   Example: {"n_estimators": 300, "learning_rate": 0.05,
                              "tree_method": "hist", "random_state": 42,
                              "objective": "binary:logistic",
                              "eval_metric": "logloss"}
    outer_splits : Number of outer train/test folds.
    inner_splits : Number of inner folds used to select best params.
    embargo      : Number of rows to skip between train and test to prevent
                   leakage (should equal the return horizon in days).

    Returns
    -------
    List of FoldResult, one per outer fold.  Each stores the best param combo
    found on that fold and the outer-fold test accuracy.
    """
    n = len(X)
    outer = _time_splits(n, outer_splits, embargo)

    # Generate every combination of the tunable params.
    param_combos = [
        dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())
    ]

    results: list[FoldResult] = []

    for k, (tr, te) in enumerate(outer):
        inner = _time_splits(len(tr), inner_splits, embargo)

        best_params, best_score = None, -np.inf

        for tunable in param_combos:
            # Merge fixed + tunable params; tunable values override fixed if
            # the same key appears in both (caller is responsible for no overlap).
            full_params = {**(fixed_params or {}), **tunable}

            inner_scores = []
            for itr, ite in inner:
                scaler  = StandardScaler()
                # fit_transform on inner training slice, transform on inner test slice
                X_train = scaler.fit_transform(X.iloc[tr[itr]])
                X_test  = scaler.transform(X.iloc[tr[ite]])

                model = XGBClassifier(**full_params)
                model.fit(X_train, y.iloc[tr[itr]], verbose=False)
                pred  = model.predict(X_test)
                inner_scores.append(accuracy_score(y.iloc[tr[ite]], pred))

            mean_score = float(np.mean(inner_scores)) if inner_scores else -np.inf
            if mean_score > best_score:
                best_score  = mean_score
                best_params = full_params   # store the FULL merged params

        # Evaluate the winner on the outer test fold.
        scaler       = StandardScaler()
        X_train_out  = scaler.fit_transform(X.iloc[tr])
        X_test_out   = scaler.transform(X.iloc[te])
        model = XGBClassifier(**(best_params or {}))
        model.fit(X_train_out, y.iloc[tr], verbose=False)
        outer_score = accuracy_score(y.iloc[te], model.predict(X_test_out))

        results.append(FoldResult(params=best_params, score=float(outer_score), fold=k))

    return results
