"""
calibration_stability.py — Check that the confidence calibration is stable over time.

PLAIN ENGLISH:
The model outputs a raw score (e.g., 0.72). We then use "isotonic
regression" to turn that into a probability ("72% chance of going up").
If the mapping between raw score and actual win-rate drifts a lot between
different time periods, our "confidence" numbers lie. This script:
  1. Splits the calibration set into K chronological folds.
  2. Fits isotonic regression on each fold.
  3. Compares the learned mappings — if Kolmogorov–Smirnov distance > 0.1
     between adjacent folds, calibration is unstable. Retraining more
     often (or using a Platt/sigmoid calibrator) may help.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def calibration_stability(raw_scores: pd.Series, labels: pd.Series, n_folds: int = 5) -> dict:
    """Return KS distance between consecutive isotonic calibrators."""
    idx = np.array_split(np.arange(len(raw_scores)), n_folds)
    curves = []
    grid = np.linspace(raw_scores.min(), raw_scores.max(), 100)
    for fold in idx:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_scores.iloc[fold], labels.iloc[fold])
        curves.append(ir.predict(grid))

    ks = []
    for i in range(len(curves) - 1):
        stat, _ = ks_2samp(curves[i], curves[i + 1])
        ks.append(float(stat))
    return {
        "ks_between_folds": ks,
        "max_ks": float(max(ks)) if ks else 0.0,
        "stable": bool(ks) and max(ks) < 0.10,
    }
