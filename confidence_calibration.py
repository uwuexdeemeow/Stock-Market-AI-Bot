"""
confidence_calibration.py — v4: Isotonic calibration + dynamic threshold.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_direction_calibrator(p_up_raw: np.ndarray,
                              y_true: np.ndarray) -> Optional[IsotonicRegression]:
    p_up_raw = np.asarray(p_up_raw, dtype=float).reshape(-1)
    y_true   = np.asarray(y_true,   dtype=int).reshape(-1)
    if len(p_up_raw) < 25:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(p_up_raw, y_true)
    return model


def save_direction_calibrator(model: Optional[IsotonicRegression], path: str) -> None:
    if model is None:
        return
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_direction_calibrator(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrate_p_up(model, p_up_raw: float) -> float:
    if model is None:
        return float(np.clip(p_up_raw, 0.0, 1.0))
    val = float(model.predict([float(np.clip(p_up_raw, 0.0, 1.0))])[0])
    return float(np.clip(val, 0.0, 1.0))


def compute_dynamic_threshold(calibrator,
                               target_precision: float = 0.58,
                               default: float = 57.5) -> float:
    """
    Find the calibrated confidence level at which the model's historical
    precision first exceeds `target_precision`.

    This replaces the hardcoded DEFAULT_FIXED_CONFIDENCE_THRESHOLD with a
    per-ticker data-driven value derived from the isotonic calibration curve.

    Args:
        calibrator:       Fitted IsotonicRegression from fit_direction_calibrator.
        target_precision: Minimum acceptable precision (default 58%).
        default:          Fallback value if calibrator is None or crossing not found.

    Returns:
        Confidence threshold as a percentage (e.g. 61.4).
    """
    if calibrator is None:
        return default

    try:
        # Sample a fine grid of raw p_up values [0.50, 0.99]
        raw_grid  = np.linspace(0.50, 0.99, 500)
        cal_grid  = calibrator.predict(raw_grid)          # calibrated p_up
        conf_grid = np.maximum(cal_grid, 1 - cal_grid) * 100  # confidence %

        # Find first index where calibrated p_up >= target_precision
        crossings = np.where(cal_grid >= target_precision)[0]
        if len(crossings) == 0:
            return default

        threshold = float(conf_grid[crossings[0]])
        # Clamp to a reasonable operating range
        threshold = float(np.clip(threshold, 52.0, 75.0))
        return threshold

    except Exception:
        return default
