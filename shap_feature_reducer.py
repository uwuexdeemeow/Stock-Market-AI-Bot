"""Rank model inputs by SHAP contribution and select a compact feature set.

PLAIN ENGLISH: gain importance says how often a tree used a feature; SHAP asks
how much that feature actually changed predictions across real training rows.
This module creates a research recommendation only.  It does not replace live
model features until a later out-of-sample test proves the smaller set holds up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import xgboost as xgb

from safe_io import atomic_write_json


DEFAULT_OUTPUT = Path("models/selected_features.json")


def _model_sha256(path: Path) -> str:
    """Fingerprint the model used to create the recommendation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduce_features(
    model,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int = 15,
    max_rows: int = 2000,
) -> dict:
    """Return top features by mean absolute SHAP value.

    The sample is evenly spaced instead of random, so repeating the command on
    identical inputs produces the same result and does not need a random seed.
    """
    matrix = np.asarray(X, dtype=float)
    names = [str(name) for name in feature_names]
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if matrix.shape[1] != len(names):
        raise ValueError("feature_names length must match X columns")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    keep_count = min(max(1, int(top_k)), matrix.shape[1])
    sample_count = min(max(1, int(max_rows)), matrix.shape[0])
    sample_indices = np.linspace(0, matrix.shape[0] - 1, sample_count, dtype=int)
    sample = matrix[sample_indices]

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    contributions = np.asarray(
        booster.predict(xgb.DMatrix(sample, feature_names=names), pred_contribs=True),
        dtype=float,
    )
    # XGBoost adds one final "bias" column. It is the model's base prediction,
    # not an input feature, so exclude it from feature ranking.
    if contributions.ndim != 2 or contributions.shape[1] != len(names) + 1:
        raise ValueError("unexpected SHAP contribution shape")
    importance = np.mean(np.abs(contributions[:, :-1]), axis=0)
    order = np.argsort(-importance, kind="stable")
    total = float(importance.sum())
    ranking = []
    cumulative = 0.0
    for rank, index in enumerate(order, start=1):
        share = float(importance[index] / total) if total > 0 else 0.0
        cumulative += share
        ranking.append({
            "rank": rank,
            "feature": names[index],
            "column_index": int(index),
            "mean_abs_shap": float(importance[index]),
            "importance_share": share,
            "cumulative_share": cumulative,
            "selected": rank <= keep_count,
        })
    selected = ranking[:keep_count]
    return {
        "method": "mean_absolute_xgboost_shap",
        "input_feature_count": len(names),
        "selected_feature_count": keep_count,
        "sample_rows": sample_count,
        "selected_features": [row["feature"] for row in selected],
        "selected_indices": [row["column_index"] for row in selected],
        "selected_importance_share": float(sum(row["importance_share"] for row in selected)),
        "ranking": ranking,
        "promotion_status": "research_only_requires_oos_validation",
    }


def main() -> int:
    """Create a recommendation from an XGBoost model and prepared NPZ matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved XGBoost JSON model.")
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="NPZ containing arrays named X and feature_names from the training split.",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    saved = np.load(args.matrix, allow_pickle=False)
    if "X" not in saved or "feature_names" not in saved:
        raise SystemExit("Matrix NPZ must contain X and feature_names arrays")
    booster = xgb.Booster()
    booster.load_model(args.model)
    result = reduce_features(
        booster,
        saved["X"],
        [str(value) for value in saved["feature_names"].tolist()],
        top_k=args.top_k,
        max_rows=args.max_rows,
    )
    result.update({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(args.model),
        "model_sha256": _model_sha256(args.model),
        "matrix_path": str(args.matrix),
    })
    atomic_write_json(result, args.output)
    print(f"Selected {result['selected_feature_count']}/{result['input_feature_count']} features")
    print(f"Research recommendation -> {args.output}")
    print("Promotion blocked until out-of-sample performance is verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
