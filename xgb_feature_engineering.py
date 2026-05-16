"""
xgb_feature_engineering.py — Build richer tabular features for XGBoost.

Why this exists
---------------
The neural models already see a 90-day sequence window.
XGBoost should not be limited to a single day's raw features.
This module builds compact rolling summaries from the same raw feature table.

Output
------
For each numeric source feature:
- raw value
- rolling mean over 3, 5, 10 days
- rolling std over 3, 5, 10 days
- delta vs rolling mean over 3, 5, 10 days

This is a strong tabular representation without exploding dimensionality too much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _add_interaction_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of bounded cross-terms for shallow XGBoost trees."""
    out = raw.copy()

    if {"rsi_14", "atr_norm"}.issubset(out.columns):
        rsi_centered = ((out["rsi_14"] - 50.0) / 50.0).clip(-2.0, 2.0)
        out["x_rsi14_atr_norm"] = rsi_centered * out["atr_norm"].clip(-1.0, 1.0)

    if {"regime", "hvol_20d"}.issubset(out.columns):
        out["x_regime_hvol20d"] = out["regime"].clip(-1.0, 1.0) * out["hvol_20d"].clip(0.0, 2.0)

    if {"vol_zscore_252", "ret_5d"}.issubset(out.columns):
        out["x_volz252_ret5d"] = out["vol_zscore_252"].clip(-4.0, 4.0) * out["ret_5d"].clip(-0.5, 0.5)

    if {"vol_ratio", "ret_5d"}.issubset(out.columns):
        vol_pressure = (out["vol_ratio"] - 1.0).clip(-3.0, 3.0)
        out["x_volratio_ret5d"] = vol_pressure * out["ret_5d"].clip(-0.5, 0.5)

    if {"hvol_20d", "ret_5d"}.issubset(out.columns):
        out["x_hvol20d_ret5d"] = out["hvol_20d"].clip(0.0, 2.0) * out["ret_5d"].clip(-0.5, 0.5)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_xgb_feature_frame(
    df: pd.DataFrame,
    feature_cols_raw: list[str],
    windows: tuple[int, ...] = (3, 5, 10),
    include_interactions: bool = True,
) -> pd.DataFrame:
    raw = df[feature_cols_raw].copy()
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if include_interactions:
        raw = _add_interaction_features(raw)

    out = raw.copy()
    for w in windows:
        roll_mean = raw.rolling(w, min_periods=1).mean()
        roll_std = raw.rolling(w, min_periods=2).std(ddof=0).fillna(0.0)
        # Departure from the recent past: use the trailing window ending at
        # t-1. Including today's value dampens the very move this feature is
        # meant to measure.
        past_mean = raw.shift(1).rolling(w, min_periods=1).mean()
        delta = raw - past_mean

        roll_mean.columns = [f"{c}__mean_{w}" for c in raw.columns]
        roll_std.columns = [f"{c}__std_{w}" for c in raw.columns]
        delta.columns = [f"{c}__delta_mean_{w}" for c in raw.columns]

        out = pd.concat([out, roll_mean, roll_std, delta], axis=1)

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def build_xgb_matrix(
    df: pd.DataFrame,
    feature_cols_raw: list[str],
    xgb_feature_cols: list[str] | None = None,
    include_interactions: bool = True,
) -> tuple[np.ndarray, list[str]]:
    frame = build_xgb_feature_frame(df, feature_cols_raw, include_interactions=include_interactions)
    cols = xgb_feature_cols if xgb_feature_cols is not None else list(frame.columns)
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0
    mat = frame[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0), cols
