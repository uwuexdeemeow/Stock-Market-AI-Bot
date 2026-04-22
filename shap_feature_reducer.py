"""
shap_feature_reducer.py — Keep only features that actually matter.

PLAIN ENGLISH:
"SHAP values" are a game-theory way to measure how much each feature
contributes to a model's predictions. Too many weak features = noise, slow
training, and fragile models. This script:
  1. Trains an XGBoost model on the FULL feature set.
  2. Computes SHAP importance for each feature.
  3. Keeps the top-N (default 15) most-important features.
  4. Writes the reduced list to models/selected_features.json so the
     training pipeline can reuse it.

Rule of thumb: start around 15 features. Too few starves the model; too
many lets it memorize noise. Re-run quarterly as markets evolve.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from settings import MODEL_DIR


def reduce_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_keep: int = 15,
    model_kwargs: dict | None = None,
) -> list[str]:
    """Return the top-`n_keep` feature names by mean |SHAP|."""
    model = XGBClassifier(**(model_kwargs or {"n_estimators": 300, "max_depth": 5}))
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # multiclass
        importance = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        importance = np.abs(sv).mean(axis=0)

    ranking = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    top = ranking.head(n_keep).index.tolist()

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "selected_features.json")
    with open(path, "w") as f:
        json.dump({"features": top, "importance": ranking.head(n_keep).to_dict()}, f, indent=2)
    return top
