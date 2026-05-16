"""
ensemble_utils.py

Shared helpers for ensemble weighting and eligibility checks.
Use this in both train.py and post_train_optimizer.py to avoid drift.
"""

from __future__ import annotations


def compute_dynamic_weights(
    model_accs: dict,
    base_weights: dict,
    baseline_up_rate: float = 50.0,
    blend: float = 0.40,
) -> dict:
    """
    Same weighting rule as train.py:
    - accuracy above the ticker-specific baseline becomes score
    - dynamic weights are blended 40/60 with base weights
    """
    scores = {m: max(float(acc) - float(baseline_up_rate), 0.0) for m, acc in model_accs.items()}
    total = sum(scores.values())

    if total < 1e-6:
        return dict(base_weights)

    dynamic = {m: s / total for m, s in scores.items()}
    blended = {}
    for m in base_weights:
        b = float(base_weights.get(m, 0.0))
        d = float(dynamic.get(m, 0.0))
        blended[m] = blend * d + (1.0 - blend) * b

    total_b = sum(blended.values())
    if total_b <= 1e-12:
        return dict(base_weights)
    return {m: v / total_b for m, v in blended.items()}


def has_sufficient_ensemble(weights: dict, min_models: int = 2) -> bool:
    active = {m: w for m, w in weights.items() if float(w) > 0.01}

    if len(active) < min_models:
        return False

    has_xgb = float(active.get("xgboost", 0.0)) > 0.01
    neural_active = [m for m, w in active.items() if m != "xgboost" and float(w) > 0.01]
    has_neural = len(neural_active) > 0
    has_attn = float(active.get("lstm_attention", 0.0)) > 0.01
    has_transformer = float(active.get("transformer", 0.0)) > 0.01

    if has_xgb and has_neural:
        return True
    if has_attn and has_transformer:
        return True
    return False
