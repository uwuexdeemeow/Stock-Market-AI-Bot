"""
dynamic_weighting.py — Regime-aware ensemble weighting for the 3-model stack.

This module now assumes the active ensemble is:
- xgboost
- lstm_attention
- transformer

It lightly tilts weights by regime without reintroducing removed legacy models.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from settings import VIX_HIGH_THRESHOLD, VIX_EXTREME_THRESHOLD

# Conservative multipliers for the simplified 3-model ensemble.
XGB_STABLE_MULT = 1.15
XGB_DEFENSIVE_MULT = 0.90
XGB_CRISIS_MULT = 0.75
LSTM_ATTN_DEFENSIVE_MULT = 1.10
LSTM_ATTN_CRISIS_MULT = 1.20
TRANSFORMER_DEFENSIVE_MULT = 1.10
TRANSFORMER_CRISIS_MULT = 1.20
DYN_WEIGHT_MIN_FLOOR = 1e-4
ACTIVE_MODELS = ("lstm_attention", "transformer", "xgboost")


def infer_regime_from_row(row: pd.Series) -> Tuple[str, Dict[str, float]]:
    vix = float(row.get("vix_level", 0.0) or 0.0)
    spy_above = float(row.get("spy_above_ma20", 1.0) or 0.0)
    qqq_above = float(row.get("qqq_above_ma20", 1.0) or 0.0)
    credit_stress = float(row.get("credit_stress", 0.0) or 0.0)
    gld_risk_off = float(row.get("gld_risk_off", 0.0) or 0.0)
    weekly_ret = float(row.get("weekly_ret", 0.0) or 0.0)
    monthly_ret = float(row.get("monthly_ret", 0.0) or 0.0)

    high_vix = vix >= VIX_HIGH_THRESHOLD
    extreme_vix = vix >= VIX_EXTREME_THRESHOLD
    bear = (spy_above < 0.5 and qqq_above < 0.5 and (weekly_ret < 0 or monthly_ret < 0))
    risk_off = credit_stress > 0.5 or gld_risk_off > 0.5

    if extreme_vix or (bear and risk_off):
        return "crisis", {"vix": vix, "bear": float(bear), "risk_off": float(risk_off)}
    if bear or high_vix:
        return "defensive", {"vix": vix, "bear": float(bear), "risk_off": float(risk_off)}
    return "stable", {"vix": vix, "bear": float(bear), "risk_off": float(risk_off)}



def regime_adjust_weights(base_weights: Dict[str, float], regime_name: str) -> Dict[str, float]:
    # Keep only active models so old saved keys cannot creep back in.
    w = {m: float(base_weights.get(m, 0.0)) for m in ACTIVE_MODELS}

    if regime_name == "stable":
        w["xgboost"] *= XGB_STABLE_MULT
    elif regime_name == "defensive":
        w["xgboost"] *= XGB_DEFENSIVE_MULT
        w["lstm_attention"] *= LSTM_ATTN_DEFENSIVE_MULT
        w["transformer"] *= TRANSFORMER_DEFENSIVE_MULT
    elif regime_name == "crisis":
        w["xgboost"] *= XGB_CRISIS_MULT
        w["lstm_attention"] *= LSTM_ATTN_CRISIS_MULT
        w["transformer"] *= TRANSFORMER_CRISIS_MULT

    for k in ACTIVE_MODELS:
        w[k] = max(float(w.get(k, 0.0)), DYN_WEIGHT_MIN_FLOOR)

    total = sum(w.values())
    if total <= 0:
        return {m: float(base_weights.get(m, 0.0)) for m in ACTIVE_MODELS}
    return {k: float(v / total) for k, v in w.items()}



def get_live_regime_adjusted_weights(base_weights: Dict[str, float], feature_row: pd.Series):
    regime_name, regime_info = infer_regime_from_row(feature_row)
    adj = regime_adjust_weights(base_weights, regime_name)
    return adj, regime_name, regime_info
