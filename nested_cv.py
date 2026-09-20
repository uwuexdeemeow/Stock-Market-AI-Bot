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

The function returns one FoldResult per outer fold, each recording the settings
selected by inner validation plus the untouched outer score. Callers may use
the mode of the inner-selected settings; outer scores are reporting evidence,
not another tuning surface.

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
    # Mean AUC from the inner folds that selected these parameters. This is
    # safe to use for choosing final production settings; outer AUC is not.
    inner_score: float = 0.5


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


def _date_group_splits(
    dates: pd.Index | pd.Series,
    n_splits: int,
    embargo: int,
    *,
    session_dates: pd.Index | pd.Series | None = None,
    min_train_size: int = 50,
    min_test_size: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split complete dates and purge the label horizon in market sessions.

    PLAIN ENGLISH: Pooled data has many stock rows for one date. Splitting by
    row can put Monday's AAPL in training and Monday's MSFT in testing. This
    helper keeps every date together and uses the full session calendar to
    leave an honest gap after training.
    """
    row_dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None)
    if len(row_dates) == 0 or n_splits <= 0:
        return []
    if not row_dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted chronologically")

    unique_dates = pd.DatetimeIndex(row_dates.unique()).sort_values()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(session_dates if session_dates is not None else unique_dates)
    ).tz_localize(None).unique().sort_values()
    missing = unique_dates.difference(calendar)
    if len(missing):
        raise ValueError("session_dates must contain every row date")
    calendar_position = {date: pos for pos, date in enumerate(calendar)}

    row_groups = {date: np.flatnonzero(row_dates == date) for date in unique_dates}
    available_rows = len(row_dates) - min_train_size
    if available_rows <= 0:
        return []
    effective_min_test = min(int(min_test_size), max(50, available_rows // n_splits))

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_end in range(1, len(unique_dates)):
        train_dates = unique_dates[:train_group_end]
        train_idx = np.concatenate([row_groups[date] for date in train_dates])
        if len(train_idx) < min_train_size:
            continue

        # A horizon-H label starting on the last training date may consume the
        # next H sessions. Testing begins strictly after that label can end.
        last_train_position = calendar_position[train_dates[-1]]
        eligible_test_dates = [
            date for date in unique_dates[train_group_end:]
            if calendar_position[date] > last_train_position + max(0, int(embargo))
        ]
        test_parts: list[np.ndarray] = []
        test_rows = 0
        for date in eligible_test_dates:
            part = row_groups[date]
            test_parts.append(part)
            test_rows += len(part)
            if test_rows >= effective_min_test:
                break
        if test_rows >= effective_min_test:
            candidates.append((train_idx, np.concatenate(test_parts)))

    if not candidates:
        return []
    chosen = np.linspace(0, len(candidates) - 1, min(n_splits, len(candidates))).astype(int)
    return [candidates[pos] for pos in dict.fromkeys(chosen.tolist())]


def nested_walk_forward_search(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    fixed_params: dict | None = None,
    outer_splits: int = 4,
    inner_splits: int = 3,
    embargo: int = 5,
    min_test_size: int = 500,
    dates: pd.Index | pd.Series | None = None,
    session_dates: pd.Index | pd.Series | None = None,
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
    embargo      : Number of market sessions to purge between train and test.
    dates        : Date for each feature row. Required for pooled panels where
                   many rows share a date.
    session_dates: Full ordered market-session calendar used to measure the
                   embargo even when ``dates`` was sampled at a wider stride.

    Returns
    -------
    List of FoldResult, one inner-selected winner per outer fold. Outer scores
    remain an honest estimate because losing candidates never see outer data.
    """
    n = len(X)
    if dates is not None and len(dates) != n:
        raise ValueError("dates must have one value per X row")
    split_fn = _date_group_splits if dates is not None else None
    outer = (
        split_fn(
            dates,
            outer_splits,
            embargo,
            session_dates=session_dates,
            min_train_size=500,
            min_test_size=min_test_size,
        )
        if split_fn
        else _time_splits(n, outer_splits, embargo, min_train_size=500, min_test_size=min_test_size)
    )

    # Generate every combination of the tunable params.
    param_combos = [
        dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())
    ]

    results: list[FoldResult] = []

    for k, (tr, te) in enumerate(outer):
        inner = (
            _date_group_splits(
                pd.DatetimeIndex(dates)[tr],
                inner_splits,
                embargo,
                session_dates=session_dates,
                min_train_size=500,
                min_test_size=min_test_size,
            )
            if dates is not None
            else _time_splits(len(tr), inner_splits, embargo, min_train_size=500, min_test_size=min_test_size)
        )
        if not inner:
            continue

        scored_candidates: list[tuple[float, dict]] = []
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
            scored_candidates.append((float(np.mean(inner_scores)), full_params))

        if not scored_candidates:
            continue
        # Only the inner-fold winner earns one look at the untouched outer set.
        best_inner_score, best_params = max(scored_candidates, key=lambda item: item[0])
        scaler = StandardScaler()
        X_train_out = scaler.fit_transform(X.iloc[tr])
        X_test_out = scaler.transform(X.iloc[te])
        model = XGBClassifier(**best_params)
        model.fit(X_train_out, y.iloc[tr], verbose=False)
        y_te_out = y.iloc[te]
        outer_proba = model.predict_proba(X_test_out)[:, 1]
        outer_score = (
            roc_auc_score(y_te_out, outer_proba)
            if len(np.unique(y_te_out)) >= 2
            else 0.5
        )
        results.append(FoldResult(
            params=best_params,
            score=float(outer_score),
            fold=k,
            inner_score=best_inner_score,
        ))

    return results
