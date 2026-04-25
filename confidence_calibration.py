"""
confidence_calibration.py — v6: Isotonic → Platt → raw fallback ladder.

Calibration converts raw XGBoost probabilities (which tend to cluster near 0.5)
into more honest probability estimates.  Three methods are tried in order:

  1. Isotonic regression — flexible, fits any monotone curve.  Needs ~100+ samples
     and is validated on a held-out 20% slice.
  2. Platt scaling (logistic regression on raw probs) — simpler, more stable on
     small calibration sets.  Used when isotonic fails its quality test.
  3. Hard-threshold fallback — no calibration; raw XGB probs used directly with a
     data-driven threshold derived from scanning raw prob bins.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and actual outcomes."""
    return float(np.mean((probs - y_true.astype(float)) ** 2))


class _PlattCalibrator:
    """
    Thin wrapper around sklearn LogisticRegression that exposes a .predict()
    interface matching IsotonicRegression, so callers don't need to branch.

    Platt scaling fits a logistic curve P(y=1) = sigmoid(a * p_raw + b) where
    p_raw is the raw XGBoost probability.  It is more stable than isotonic on
    small calibration sets because it has only 2 parameters.
    """

    def __init__(self, model: LogisticRegression) -> None:
        self._model = model

    def predict(self, p_up_raw) -> np.ndarray:
        arr = np.asarray(p_up_raw, dtype=float).reshape(-1, 1)
        return self._model.predict_proba(arr)[:, 1]


def _fit_platt(p_up_raw: np.ndarray, y_true: np.ndarray,
               p_val: np.ndarray, y_val: np.ndarray):
    """
    Try Platt scaling.  Returns a _PlattCalibrator if it improves Brier score
    on the validation slice, else None.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
        lr.fit(p_up_raw.reshape(-1, 1), y_true)
        p_val_platt = np.clip(lr.predict_proba(p_val.reshape(-1, 1))[:, 1], 0.0, 1.0)
        if _brier_score(p_val_platt, y_val) < _brier_score(p_val, y_val):
            # Refit on all data
            lr_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
            lr_full.fit(np.concatenate([p_up_raw, p_val]).reshape(-1, 1),
                        np.concatenate([y_true, y_val]))
            return _PlattCalibrator(lr_full)
    except Exception:
        pass
    return None


def fit_direction_calibrator(p_up_raw: np.ndarray, y_true: np.ndarray):
    """
    Fit a calibrator using a three-rung fallback ladder:

      1. Isotonic regression  — validated on held-out 20%, kept only if Brier
                                 improves and rank correlation doesn't invert.
      2. Platt scaling        — logistic regression on raw probs, more stable
                                 on small calibration sets.
      3. None                 — caller uses raw XGB probs + data-driven threshold.

    Returns an object with a .predict(p_raw) → np.ndarray interface, or None.
    """
    p_up_raw = np.asarray(p_up_raw, dtype=float).reshape(-1)
    y_true   = np.asarray(y_true,   dtype=int).reshape(-1)

    if len(p_up_raw) < 50 or len(np.unique(y_true)) < 2:
        return None

    # Hold out the last 20% for out-of-sample validation.
    split = max(int(len(p_up_raw) * 0.8), 40)
    p_train, p_val = p_up_raw[:split], p_up_raw[split:]
    y_train, y_val = y_true[:split],   y_true[split:]

    if len(np.unique(y_train)) < 2 or len(p_val) < 10:
        return _fit_platt(p_up_raw, y_true, p_up_raw[-10:], y_true[-10:])

    # ── Rung 1: Isotonic ──────────────────────────────────────────────────────
    if len(p_up_raw) >= 100:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p_train, y_train)
        p_val_cal = np.clip(model.predict(p_val), 0.0, 1.0)

        brier_ok = _brier_score(p_val_cal, y_val) < _brier_score(p_val, y_val)
        ranks_raw = np.argsort(np.argsort(p_val))
        ranks_cal = np.argsort(np.argsort(p_val_cal))
        corr_raw  = float(np.corrcoef(ranks_raw, y_val)[0, 1]) if y_val.std() > 0 else 0.0
        corr_cal  = float(np.corrcoef(ranks_cal, y_val)[0, 1]) if y_val.std() > 0 else 0.0
        rank_ok   = corr_cal >= 0 and corr_cal >= corr_raw - 0.05

        if brier_ok and rank_ok:
            model_full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model_full.fit(p_up_raw, y_true)
            return model_full

    # ── Rung 2: Platt scaling ─────────────────────────────────────────────────
    platt = _fit_platt(p_train, y_train, p_val, y_val)
    if platt is not None:
        return platt

    # ── Rung 3: give up, caller handles raw probs ─────────────────────────────
    return None


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
