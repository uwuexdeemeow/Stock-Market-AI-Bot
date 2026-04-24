"""
calibration_stability.py — Check that confidence calibration is stable over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def calibration_stability(raw_scores: pd.Series, labels: pd.Series, n_folds: int = 5, min_fold_size: int = 25) -> dict:
    """Return KS distance between consecutive valid isotonic calibrators.

    A fold is skipped if it is too small or only contains one class.
    This avoids false crashes on sparse calibration windows.
    """
    raw = pd.Series(raw_scores).reset_index(drop=True)
    y = pd.Series(labels).reset_index(drop=True).astype(int)
    n = min(len(raw), len(y))
    raw = raw.iloc[:n]
    y = y.iloc[:n]

    if n < max(n_folds, min_fold_size):
        return {
            "ks_between_folds": [],
            "max_ks": None,
            "stable": None,
            "valid_folds": 0,
            "skipped_folds": n_folds,
            "reason": "not_enough_samples",
        }

    idx = np.array_split(np.arange(n), n_folds)
    grid = np.linspace(float(raw.min()), float(raw.max()), 100)
    curves: list[np.ndarray] = []
    skipped = 0

    for fold in idx:
        fold_scores = raw.iloc[fold]
        fold_labels = y.iloc[fold]
        if len(fold_scores) < min_fold_size or fold_labels.nunique() < 2:
            skipped += 1
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(fold_scores, fold_labels)
        curves.append(ir.predict(grid))

    if len(curves) < 2:
        return {
            "ks_between_folds": [],
            "max_ks": None,
            "stable": None,
            "valid_folds": len(curves),
            "skipped_folds": skipped,
            "reason": "not_enough_valid_folds",
        }

    ks_vals = []
    for i in range(len(curves) - 1):
        stat, _ = ks_2samp(curves[i], curves[i + 1])
        ks_vals.append(float(stat))

    max_ks = float(max(ks_vals)) if ks_vals else None
    return {
        "ks_between_folds": ks_vals,
        "max_ks": max_ks,
        "stable": bool(max_ks is not None and max_ks < 0.10),
        "valid_folds": len(curves),
        "skipped_folds": skipped,
    }
