"""
calibration_stability.py — Check that calibrated scores are stable over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calibration_stability(raw_scores: pd.Series, labels: pd.Series, n_folds: int = 5, min_fold_size: int = 25) -> dict:
    """Return KS distance between consecutive calibrated score distributions.

    The input name is kept for compatibility, but callers should pass the
    probabilities actually used downstream.  Re-fitting isotonic curves inside
    this diagnostic made stable score distributions look unstable whenever fold
    base rates moved.  A direct KS test answers the operational question:
    "does a calibrated score mean roughly the same thing over time?"
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
    score_folds: list[pd.Series] = []
    fold_stats: list[dict] = []
    skipped = 0

    for fold in idx:
        fold_scores = raw.iloc[fold]
        fold_labels = y.iloc[fold]
        if len(fold_scores) < min_fold_size:
            skipped += 1
            continue
        score_folds.append(fold_scores)
        pred = (fold_scores >= 0.5).astype(int)
        accuracy = float((pred.to_numpy() == fold_labels.to_numpy()).mean())
        fold_stats.append({
            "n": int(len(fold_scores)),
            "mean_score": round(float(fold_scores.mean()), 6),
            "std_score": round(float(fold_scores.std(ddof=0)), 6),
            "base_rate": round(float(fold_labels.mean()), 6),
            "accuracy_at_0_5": round(accuracy, 6),
        })

    if len(score_folds) < 2:
        return {
            "ks_between_folds": [],
            "max_ks": None,
            "stable": None,
            "valid_folds": len(score_folds),
            "skipped_folds": skipped,
            "reason": "not_enough_valid_folds",
            "fold_stats": fold_stats,
        }

    ks_vals = []
    for i in range(len(score_folds) - 1):
        stat, _ = ks_2samp(score_folds[i], score_folds[i + 1])
        ks_vals.append(float(stat))

    max_ks = float(max(ks_vals)) if ks_vals else None
    return {
        "ks_between_folds": ks_vals,
        "max_ks": max_ks,
        "stable": bool(max_ks is not None and max_ks < 0.10),
        "valid_folds": len(score_folds),
        "skipped_folds": skipped,
        "fold_stats": fold_stats,
    }
