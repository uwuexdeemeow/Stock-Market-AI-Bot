"""
nested_cv.py — Nested walk-forward cross-validation for hyperparameter search.

What this does
--------------
Finds the best XGBoost hyperparameters (e.g. max_depth, min_child_weight,
reg_lambda) by running a nested time-series cross-validation:

  Outer folds: sequential train/test splits across time (no lookahead).
  Inner folds: within each outer training window, try every param combo and
               pick the one with the best average inner-fold ROC-AUC.

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class FoldResult:
    # Which hyperparameter values won on this outer fold.
    params: dict
    # AUC-ROC on the outer test fold (0–1, 0.5 = random baseline).
    score: float
    # Which outer fold index (0-based).
    fold: int


def _time_splits(
    n: int,
    n_splits: int,
    embargo: int,
    min_train_size: int = 50,
    min_test_size: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Produce sequential train/test index pairs.

    Each split uses all rows from 0..train_end for training and
    train_end+embargo..test_end for testing.  The embargo gap prevents
    forward leakage when return windows overlap.
    """
    if n_splits <= 0:
        return []

    available = n - min_train_size - embargo
    if available <= 0:
        return []

    # Adapt the test size downward when the data set is too short to support
    # the requested number of folds at the nominal floor. This avoids silently
    # evaluating only one or two outer folds and then over-trusting the result.
    effective_min_test = min(int(min_test_size), max(50, available // n_splits))
    latest_train_end = max(min_train_size, n - embargo - effective_min_test)
    if n_splits == 1:
        train_ends = [latest_train_end]
    else:
        train_ends = np.linspace(min_train_size, latest_train_end, n_splits).astype(int).tolist()

    out = []
    seen_train_ends: set[int] = set()
    for train_end in train_ends:
        if train_end in seen_train_ends:
            continue
        seen_train_ends.add(train_end)
        test_start = train_end + embargo
        test_end   = min(test_start + effective_min_test, n)
        if train_end < min_train_size:
            continue
        if test_end - test_start < max(50, effective_min_test):
            continue
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
    min_test_size: int = 500,
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
    List of FoldResult, one per parameter combo per outer fold.  This lets the
    caller average outer-fold ROC-AUC for every candidate instead of selecting
    the combo that happened to win one lucky fold.
    """
    n = len(X)
    outer = _time_splits(n, outer_splits, embargo, min_train_size=500, min_test_size=min_test_size)

    # Generate every combination of the tunable params.
    param_combos = [
        dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())
    ]

    results: list[FoldResult] = []

    for k, (tr, te) in enumerate(outer):
        inner = _time_splits(len(tr), inner_splits, embargo, min_train_size=500, min_test_size=min_test_size)
        if not inner:
            continue

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

                # Use AUC-ROC instead of accuracy.
                # Why: with ~30% UP labels (triple-barrier), accuracy is dominated
                # by the majority class — a model that always predicts DOWN gets
                # ~70% accuracy and beats a model with real signal.  AUC-ROC uses
                # predicted probabilities (not a hard threshold), so it measures
                # pure discriminative ability: can the model rank UP rows above
                # DOWN rows?  Random baseline = 0.5 regardless of class balance.
                y_te = y.iloc[tr[ite]]
                if len(np.unique(y_te)) < 2:
                    # Skip fold if only one class present (can't compute AUC)
                    continue
                proba = model.predict_proba(X_test)[:, 1]
                inner_scores.append(roc_auc_score(y_te, proba))

            if not inner_scores:
                continue

            # Evaluate every candidate on this outer fold. The caller can then
            # choose the parameter set with the best average outer AUC across
            # all folds, avoiding the old "pick the luckiest fold" behavior.
            scaler       = StandardScaler()
            X_train_out  = scaler.fit_transform(X.iloc[tr])
            X_test_out   = scaler.transform(X.iloc[te])
            model = XGBClassifier(**full_params)
            model.fit(X_train_out, y.iloc[tr], verbose=False)
            y_te_out    = y.iloc[te]
            outer_proba = model.predict_proba(X_test_out)[:, 1]
            outer_score = (
                roc_auc_score(y_te_out, outer_proba)
                if len(np.unique(y_te_out)) >= 2
                else 0.5
            )

            results.append(FoldResult(params=full_params, score=float(outer_score), fold=k))

    return results
