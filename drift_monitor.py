"""
drift_monitor.py — Population Stability Index (PSI) and KS drift detector.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from settings import LOG_DIR, MODEL_DIR

BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_distributions.npz")


def _safe_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population stability index between two 1-D distributions."""
    expected = _safe_array(expected)
    actual = _safe_array(actual)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    if np.allclose(expected, expected[0]) and np.allclose(actual, actual[0]):
        return 0.0

    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.percentile(expected, breakpoints)
    edges = np.unique(edges)
    if len(edges) < 2:
        lo = float(expected.min()) - 1e-9
        hi = float(expected.max()) + 1e-9
        edges = np.array([lo, hi])
    else:
        edges[0] -= 1e-9
        edges[-1] += 1e-9

    e_counts = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a_counts = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def snapshot_baseline(features: pd.DataFrame, scores: np.ndarray) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(
        BASELINE_PATH,
        features=features.to_numpy(),
        columns=np.array(features.columns.tolist(), dtype=object),
        scores=_safe_array(scores),
    )


def check_drift(features: pd.DataFrame, scores: np.ndarray) -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {"status": "no_baseline"}

    os.makedirs(LOG_DIR, exist_ok=True)
    z = np.load(BASELINE_PATH, allow_pickle=True)
    base_feat = z["features"]
    base_cols = z["columns"].tolist()
    base_scores = _safe_array(z["scores"])
    curr_scores = _safe_array(scores)

    results = {"run_at": datetime.utcnow().isoformat() + "Z", "features": {}, "score": {}}
    for i, col in enumerate(base_cols):
        if col not in features.columns or i >= base_feat.shape[1]:
            continue
        actual = _safe_array(features[col].to_numpy())
        expected = _safe_array(base_feat[:, i])
        if len(actual) == 0 or len(expected) == 0:
            continue
        results["features"][col] = {
            "psi": psi(expected, actual),
            "ks": float(ks_2samp(expected, actual).statistic),
        }

    if len(base_scores) and len(curr_scores):
        results["score"] = {
            "psi": psi(base_scores, curr_scores),
            "ks": float(ks_2samp(base_scores, curr_scores).statistic),
        }
    else:
        results["score"] = {"psi": 0.0, "ks": 0.0}

    worst_psi = max([v["psi"] for v in results["features"].values()] + [results["score"]["psi"]], default=0.0)
    worst_ks = max([v["ks"] for v in results["features"].values()] + [results["score"]["ks"]], default=0.0)
    if worst_psi > 0.25 or worst_ks > 0.10:
        results["status"] = "drift"
    elif worst_psi > 0.10 or worst_ks > 0.05:
        results["status"] = "caution"
    else:
        results["status"] = "stable"

    with open(os.path.join(LOG_DIR, "drift.jsonl"), "a") as f:
        f.write(json.dumps(results) + "\n")
    return results
