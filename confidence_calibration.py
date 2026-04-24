"""
confidence_calibration.py — v5: Validated isotonic calibration + dynamic threshold.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and actual outcomes."""
    return float(np.mean((probs - y_true.astype(float)) ** 2))


def fit_direction_calibrator(p_up_raw: np.ndarray,
                              y_true: np.ndarray) -> Optional[IsotonicRegression]:
    """
    Fit an isotonic regression calibrator, but only keep it if it actually
    improves Brier score on a held-out validation slice.

    Isotonic regression can overfit small calibration sets and invert the
    probability ordering out-of-sample (higher confidence → worse accuracy).
    This function fits on the first 80% of the data and validates on the last
    20%. If calibration makes Brier score worse on that held-out slice, it
    returns None so raw XGBoost probabilities are used instead.
    """
    p_up_raw = np.asarray(p_up_raw, dtype=float).reshape(-1)
    y_true   = np.asarray(y_true,   dtype=int).reshape(-1)
    # Raised from 25 → 100: isotonic regression needs enough points to
    # fit a reliable monotone curve without noise spikes.
    if len(p_up_raw) < 100:
        return None
    if len(np.unique(y_true)) < 2:
        return None

    # Hold out the last 20% for out-of-sample validation.
    split = max(int(len(p_up_raw) * 0.8), 50)
    p_train, p_val = p_up_raw[:split], p_up_raw[split:]
    y_train, y_val = y_true[:split],   y_true[split:]

    if len(np.unique(y_train)) < 2 or len(p_val) < 10:
        return None

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(p_train, y_train)

    p_val_cal = np.clip(model.predict(p_val), 0.0, 1.0)

    # Reject if calibration raises Brier score on the held-out slice.
    if _brier_score(p_val_cal, y_val) >= _brier_score(p_val, y_val):
        return None

    # Reject if calibration inverts the probability ranking — the exact failure
    # mode seen in diagnostics where higher confidence → lower actual win rate.
    # Spearman rank correlation between calibrated probs and outcomes must be
    # at least weakly positive; if it's negative, calibration is harmful.
    ranks_raw = np.argsort(np.argsort(p_val))
    ranks_cal = np.argsort(np.argsort(p_val_cal))
    rank_corr_raw = float(np.corrcoef(ranks_raw, y_val)[0, 1])
    rank_corr_cal = float(np.corrcoef(ranks_cal, y_val)[0, 1])
    if rank_corr_cal < 0 or rank_corr_cal < rank_corr_raw - 0.05:
        return None

    # Validation passed — refit on all data.
    model_full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model_full.fit(p_up_raw, y_true)
    return model_full


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
