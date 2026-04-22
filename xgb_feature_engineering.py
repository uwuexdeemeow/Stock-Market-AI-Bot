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


def build_xgb_feature_frame(df: pd.DataFrame, feature_cols_raw: list[str],
                            windows: tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    raw = df[feature_cols_raw].copy()
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = raw.copy()
    for w in windows:
        roll_mean = raw.rolling(w, min_periods=1).mean()
        roll_std = raw.rolling(w, min_periods=2).std(ddof=0).fillna(0.0)
        delta = raw - roll_mean

        roll_mean.columns = [f"{c}__mean_{w}" for c in raw.columns]
        roll_std.columns = [f"{c}__std_{w}" for c in raw.columns]
        delta.columns = [f"{c}__delta_mean_{w}" for c in raw.columns]

        out = pd.concat([out, roll_mean, roll_std, delta], axis=1)

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def build_xgb_matrix(df: pd.DataFrame, feature_cols_raw: list[str],
                     xgb_feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    frame = build_xgb_feature_frame(df, feature_cols_raw)
    cols = xgb_feature_cols if xgb_feature_cols is not None else list(frame.columns)
    for c in cols:
        if c not in frame.columns:
            frame[c] = 0.0
    mat = frame[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0), cols
