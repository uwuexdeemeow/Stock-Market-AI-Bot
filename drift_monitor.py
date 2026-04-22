"""
drift_monitor.py — Population Stability Index (PSI) and KS drift detector.

PLAIN ENGLISH:
The world changes. A model trained on 2015–2023 data can silently stop
working when 2026 market structure drifts (e.g., 0DTE options flipping
intraday volatility patterns). This module compares a recent window of
FEATURE values and raw MODEL SCORES against the training baseline.

Two metrics:
  * PSI (Population Stability Index): <0.1 stable, 0.1–0.25 caution, >0.25 drift.
  * KS (Kolmogorov–Smirnov): 0 identical, 1 disjoint. > 0.10 = investigate.

Run nightly against models/baseline_distributions.npz. Any red flag emits
a line in logs/drift.jsonl which the monitoring stack can alert on.
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


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population stability index between two 1-D distributions."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.percentile(expected, breakpoints)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    e_counts = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a_counts = np.histogram(actual, edges)[0] / max(len(actual), 1)
    # replace zeros to avoid log(0)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def snapshot_baseline(features: pd.DataFrame, scores: np.ndarray) -> None:
    np.savez(BASELINE_PATH, features=features.to_numpy(),
             columns=np.array(features.columns.tolist()), scores=scores)


def check_drift(features: pd.DataFrame, scores: np.ndarray) -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {"status": "no_baseline"}
    z = np.load(BASELINE_PATH, allow_pickle=True)
    base_feat = z["features"]
    base_cols = z["columns"].tolist()
    base_scores = z["scores"]

    results = {"run_at": datetime.utcnow().isoformat() + "Z", "features": {}, "score": {}}
    for i, col in enumerate(base_cols):
        if col not in features.columns:
            continue
        a = features[col].dropna().to_numpy()
        b = base_feat[:, i]
        b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        results["features"][col] = {
            "psi": psi(b, a),
            "ks": float(ks_2samp(b, a).statistic),
        }
    results["score"] = {
        "psi": psi(base_scores, scores),
        "ks": float(ks_2samp(base_scores, scores).statistic),
    }

    worst_psi = max([v["psi"] for v in results["features"].values()] + [results["score"]["psi"]], default=0.0)
    results["status"] = "drift" if worst_psi > 0.25 else ("caution" if worst_psi > 0.10 else "stable")

    with open(os.path.join(LOG_DIR, "drift.jsonl"), "a") as f:
        f.write(json.dumps(results) + "\n")
    return results
