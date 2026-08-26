from __future__ import annotations

import numpy as np
import xgboost as xgb

from shap_feature_reducer import reduce_features


def test_shap_reducer_selects_predictive_feature_deterministically():
    """A feature that drives labels should rank above an irrelevant constant."""
    x0 = np.linspace(-2.0, 2.0, 120)
    X = np.column_stack([x0, np.zeros_like(x0), np.sin(x0)])
    y = (x0 > 0).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=12,
        max_depth=2,
        learning_rate=0.2,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)

    first = reduce_features(model, X, ["signal", "constant", "wave"], top_k=1, max_rows=80)
    second = reduce_features(model, X, ["signal", "constant", "wave"], top_k=1, max_rows=80)

    assert first["selected_features"] == ["signal"]
    assert first["selected_indices"] == [0]
    assert first["promotion_status"] == "research_only_requires_oos_validation"
    assert first == second


def test_shap_reducer_rejects_misaligned_feature_names():
    """A mislabeled matrix must fail before it can produce a misleading report."""
    model = xgb.XGBClassifier(n_estimators=2, random_state=42)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    model.fit(X, np.array([0, 0, 1, 1]))

    try:
        reduce_features(model, X, ["only_one_name"])
    except ValueError as exc:
        assert "feature_names length" in str(exc)
    else:
        raise AssertionError("misaligned feature names were accepted")
