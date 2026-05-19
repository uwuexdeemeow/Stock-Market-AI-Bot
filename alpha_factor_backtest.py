"""
alpha_factor_backtest.py — benchmark-relative direct factor research harness.

This is intentionally separate from paper trading. It tests simple
cross-sectional factor portfolios directly before giving the ML model any
credit. The selection objective is robustness: Sharpe minus drawdown, turnover,
and instability penalties. Benchmark alpha is still reported as a gate/context
metric, but massive return or CAGR is not the optimizer target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import INITIAL_CAPITAL, _load_etf_price_frame, _newey_west_tstat
from feature_health import (
    MAX_CLUSTER_WEIGHT,
    MIN_ACTIVE_CLUSTERS,
    enrich_feature_specs,
)
from robustness_scoring import robustness_score_components
from settings import DATA_DIR, SECTOR_MAP, SIGNAL_DIR, SLIPPAGE_BASE_PCT, WATCHLIST


HORIZON_DAYS = 5
MAX_GROSS_EXPOSURE = 1.25
TOP_N_SHAPES = ("top5", "top10", "top_decile", "top_quintile", "top_bottom_decile", "top_bottom_quintile")
EXPOSURE_MODES = ("long_only", "long_short", "beta_neutral")
WEIGHTING_MODES = ("equal", "score")
SECTOR_MODES = ("unconstrained", "sector_neutral")
SCORE_SOURCES = ("factor_only", "factor_plus_model")
SUBPERIODS = {
    "2015_2019": ("2015-01-01", "2019-12-31"),
    "2020_2021": ("2020-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023_2026": ("2023-01-01", "2026-12-31"),
}
FACTOR_FAMILY_ALLOWLIST = (
    "factor_",
    "xs_rank_market_",
    "xs_rank_sector_",
    "dist_",
    "ret_",
    "sector_",
    # Phase 2: added families for newly allowed features
    "vwap_",
    "vol_",
    "uptick_",
    "macd_",
    "bb_",
    "rsi_",
    "hvol_",
)
FACTOR_NAME_ALLOWLIST = (
    "beta",
    "idio_vol",
    "liquidity",
    "illiquidity",
    "dist_ma",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_vs_sector",
    "sector_rel",
    "ret_vs_spy",
    "ret_vs_qqq",
    "sector_rs_slope",
    "factor_mom",
    "factor_resid_mom",
    "factor_52w_high",
    # --- Phase 2: added high-IC features that were previously excluded ---
    # These all have |t-stat| > 5 in the IC shortlist, meaning they
    # reliably predict forward sector-excess returns.
    "vwap_dist",        # VWAP distance — price vs volume-weighted average (t≈5.8)
    "vol_trend",        # Volume trend ratio 20d/60d (t≈5.8)
    "uptick_ratio",     # Uptick ratio — % of up-ticks (t≈5.8)
    "macd_hist",        # MACD histogram — momentum oscillator (t≈5.9)
    "bb_pos",           # Bollinger Band position — mean reversion (t≈5.5)
    "rsi_",             # RSI — relative strength index (t≈5.4)
    "hvol_",            # Historical volatility — vol features (useful for risk)
)


def load_feature_specs(max_specs: int = 48) -> list[dict]:
    path = Path("logs/feature_ic_shortlist.csv")
    if not path.exists():
        fallback = [
            {"feature": "xs_rank_sector_factor_illiquidity_amihud_20d", "preferred_direction": "high_is_good"},
            {"feature": "xs_rank_sector_factor_liquidity_dollar_vol_20d", "preferred_direction": "low_is_good"},
            {"feature": "xs_rank_sector_ret_5d", "preferred_direction": "low_is_good"},
            {"feature": "xs_rank_market_ret_5d", "preferred_direction": "low_is_good"},
            {"feature": "dist_ma5", "preferred_direction": "low_is_good"},
            {"feature": "ret_vs_sector_5d", "preferred_direction": "low_is_good"},
            {"feature": "factor_52w_high_proximity", "preferred_direction": "high_is_good"},
        ]
        enriched, _profile = enrich_feature_specs(fallback)
        return enriched
    df = pd.read_csv(path)
    df = df[(df.get("target") == "sector_excess") & (df.get("horizon") == HORIZON_DAYS)].copy()
    if df.empty:
        return []
    df = df[
        df["family"].astype(str).str.startswith(FACTOR_FAMILY_ALLOWLIST)
        & df["feature"].astype(str).apply(lambda x: any(key in x for key in FACTOR_NAME_ALLOWLIST))
    ].copy()
    df["abs_t_stat"] = pd.to_numeric(df["abs_t_stat"], errors="coerce").fillna(0.0)
    df = df.sort_values("abs_t_stat", ascending=False)

    specs: list[dict] = []
    for row in df.to_dict("records"):
        feature = str(row["feature"])
        specs.append({
            "feature": feature,
            "preferred_direction": str(row.get("preferred_direction", "high_is_good")),
            "t_stat": float(row.get("t_stat", 0.0) or 0.0),
            "mean_ic": float(row.get("mean_ic", 0.0) or 0.0),
        })
        if len(specs) >= max_specs:
            break
    enriched, _profile = enrich_feature_specs(specs)
    return enriched


def load_prediction_scores() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in WATCHLIST:
        path = Path(SIGNAL_DIR) / f"{ticker.upper()}_walkforward_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"date", "ticker", "p_up_raw", "confidence", "expected_return"})
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = ticker.upper()
        frames.append(df.dropna(subset=["date"]))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "ml_score"])
    panel = pd.concat(frames, ignore_index=True)
    score_parts = []
    for col in ("p_up_raw", "confidence", "expected_return"):
        if col in panel.columns:
            score_parts.append(pd.to_numeric(panel[col], errors="coerce").groupby(panel["date"]).rank(pct=True))
    panel["ml_score"] = pd.concat(score_parts, axis=1).mean(axis=1, skipna=True) if score_parts else np.nan
    return panel[["date", "ticker", "ml_score"]].dropna(subset=["ml_score"])


def load_factor_panel(specs: list[dict], *, require_forward_returns: bool = True) -> pd.DataFrame:
    requested = sorted(
        {s["feature"] for s in specs}
        | {
            "days_to_next_earnings",
            "days_since_earnings",
            "factor_beta_252_spy",
            "factor_idio_vol_252_spy",
            "hvol_20d",
            "xs_rank_sector_factor_idio_vol_252_spy",
            "xs_rank_market_factor_idio_vol_252_spy",
            "xs_rank_sector_hvol_20d",
            "xs_rank_market_hvol_20d",
        }
    )
    frames: list[pd.DataFrame] = []
    for ticker in WATCHLIST:
        path = Path(DATA_DIR) / f"{ticker.upper()}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, errors="coerce")
        keep = [c for c in requested if c in df.columns]
        required = ["Open", "Close"]
        if not all(c in df.columns for c in required) or not keep:
            continue
        sub = df[required + keep].copy()
        sub["date"] = df.index
        sub["ticker"] = ticker.upper()
        sub["sector"] = SECTOR_MAP.get(ticker.upper(), "OTHER")
        sub["entry_price"] = df["Open"].shift(-1)
        sub["exit_price"] = df["Close"].shift(-HORIZON_DAYS)
        sub["forward_return"] = sub["exit_price"] / sub["entry_price"] - 1.0
        for hold_days in (10, 20):
            sub[f"exit_price_{hold_days}d"] = df["Close"].shift(-hold_days)
            sub[f"forward_return_{hold_days}d"] = sub[f"exit_price_{hold_days}d"] / sub["entry_price"] - 1.0
        for hold_days in (HORIZON_DAYS, 10, 20):
            delayed_entry = df["Open"].shift(-2)
            delayed_exit = df["Close"].shift(-(hold_days + 1))
            sub[f"forward_return_delay1_{hold_days}d"] = delayed_exit / delayed_entry - 1.0
        frames.append(sub.reset_index(drop=True))
    if not frames:
        raise SystemExit("No factor panel data found. Run research.py to build data/*.parquet first.")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    if require_forward_returns:
        panel = panel.dropna(subset=["entry_price", "exit_price", "forward_return"])
        panel = panel[np.isfinite(panel["forward_return"])]
        panel = panel[(panel["entry_price"] > 0) & (panel["exit_price"] > 0)]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def attach_scores(panel: pd.DataFrame, specs: list[dict], ml_scores: pd.DataFrame) -> pd.DataFrame:
    specs, profile = enrich_feature_specs(specs, write_outputs=False)
    out = panel.copy()
    score_cols: list[str] = []
    score_by_feature: dict[str, str] = {}
    for spec in specs:
        col = spec["feature"]
        if col not in out.columns:
            continue
        raw = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rank = raw.groupby(out["date"]).rank(pct=True)
        raw_rank_col = f"_raw_rank_{col}"
        out[raw_rank_col] = rank
        oriented = 1.0 - rank if spec.get("preferred_direction") == "low_is_good" else rank
        score_col = f"_score_{col}"
        out[score_col] = oriented
        score_cols.append(score_col)
        score_by_feature[col] = score_col
    if not score_cols:
        raise SystemExit("None of the selected factor columns are available in data/*.parquet.")

    cluster_rows = {c["cluster_id"]: c for c in profile.get("clusters", [])}
    feature_rows = {r["feature"]: r for r in profile.get("features", [])}
    cluster_score_cols: list[str] = []
    cluster_weight_raw: dict[str, float] = {}
    cluster_features: dict[str, list[str]] = {}
    cluster_score_by_id: dict[str, str] = {}
    for cluster_id, cluster in cluster_rows.items():
        contributing = [
            feature
            for feature in cluster.get("contributing_features", [])
            if feature in score_by_feature
        ]
        if not contributing:
            continue
        cluster_col = f"_cluster_score_{cluster_id}"
        out[cluster_col] = out[[score_by_feature[f] for f in contributing]].mean(axis=1, skipna=True)
        cluster_score_cols.append(cluster_col)
        cluster_score_by_id[cluster_id] = cluster_col
        cluster_features[cluster_id] = contributing
        cluster_weight_raw[cluster_id] = float(cluster.get("raw_weight", 0.0) or 0.0)

    total_cluster_weight = sum(w for w in cluster_weight_raw.values() if w > 0)
    if cluster_score_cols and total_cluster_weight > 0:
        out["factor_score"] = 0.0
        for cluster_id, cluster_col in cluster_score_by_id.items():
            weight = cluster_weight_raw.get(cluster_id, 0.0) / total_cluster_weight
            out["factor_score"] = out["factor_score"] + pd.to_numeric(out[cluster_col], errors="coerce").fillna(0.5) * weight
    else:
        out["factor_score"] = 0.5

    out = out.dropna(subset=["factor_score"])

    normalized_cluster_weights = {
        cid: (weight / total_cluster_weight if total_cluster_weight > 0 else 0.0)
        for cid, weight in cluster_weight_raw.items()
    }
    feature_health_summary = dict(profile.get("summary", {}))
    feature_health_summary.update({
        "active_cluster_count": int(len(cluster_score_cols)),
        "effective_cluster_count": int(len(cluster_score_cols)),
        "max_cluster_weight": round(max(normalized_cluster_weights.values(), default=0.0), 6),
        "min_active_clusters": int(MIN_ACTIVE_CLUSTERS),
        "max_cluster_weight_allowed": float(MAX_CLUSTER_WEIGHT),
    })
    # Re-check the gate after the actual score columns are built.  This matters
    # because quarantined clusters are removed before capital is allowed.
    feature_health_summary["feature_health_gate_pass"] = bool(
        feature_health_summary["active_cluster_count"] >= MIN_ACTIVE_CLUSTERS
        and feature_health_summary["max_cluster_weight"] <= MAX_CLUSTER_WEIGHT + 1e-12
    )
    reasons: list[str] = []
    if feature_health_summary["active_cluster_count"] < MIN_ACTIVE_CLUSTERS:
        reasons.append(
            f"active_cluster_count {feature_health_summary['active_cluster_count']} < {MIN_ACTIVE_CLUSTERS}"
        )
    if feature_health_summary["max_cluster_weight"] > MAX_CLUSTER_WEIGHT + 1e-12:
        reasons.append(
            f"max_cluster_weight {feature_health_summary['max_cluster_weight']:.3f} > {MAX_CLUSTER_WEIGHT:.2f}"
        )
    feature_health_summary["feature_health_gate_reasons"] = reasons
    feature_health_summary["quarantined_features"] = [
        r["feature"] for r in feature_rows.values() if r.get("health_state") == "quarantined"
    ]
    feature_health_summary["watchlist_features"] = [
        r["feature"] for r in feature_rows.values() if r.get("health_state") == "watchlist"
    ]
    out["feature_health_overlay_allowed"] = bool(feature_health_summary["feature_health_gate_pass"])
    out["feature_health_active_cluster_count"] = int(feature_health_summary["active_cluster_count"])
    out["feature_health_max_cluster_weight"] = float(feature_health_summary["max_cluster_weight"])
    out.attrs["feature_health_summary"] = feature_health_summary
    out.attrs["feature_health_cluster_score_cols"] = cluster_score_cols
    out.attrs["feature_health_cluster_features"] = cluster_features
    out.attrs["feature_health_cluster_weights"] = normalized_cluster_weights

    if not ml_scores.empty:
        out = out.merge(ml_scores, on=["date", "ticker"], how="left")
    else:
        out["ml_score"] = np.nan
    out["factor_plus_model_score"] = np.where(
        out["ml_score"].notna(),
        0.75 * out["factor_score"] + 0.25 * out["ml_score"],
        out["factor_score"],
    )
    out = attach_walkforward_factor_score(out, specs, cluster_score_cols)
    out = attach_regime_factor_scores(out, specs, cluster_score_by_id, cluster_features)
    out["beta"] = pd.to_numeric(out.get("factor_beta_252_spy", 1.0), errors="coerce").fillna(1.0).clip(0.1, 3.0)
    out.attrs["feature_health_summary"] = feature_health_summary
    out.attrs["feature_health_cluster_score_cols"] = cluster_score_cols
    out.attrs["feature_health_cluster_features"] = cluster_features
    out.attrs["feature_health_cluster_weights"] = normalized_cluster_weights
    return out


def attach_regime_factor_scores(
    panel: pd.DataFrame,
    specs: list[dict],
    cluster_score_by_id: dict[str, str],
    cluster_features: dict[str, list[str]],
) -> pd.DataFrame:
    """Add simple regime-specific factor composites for core-satellite research."""
    out = panel.copy()
    all_features_by_cluster = {cid: set(features) for cid, features in cluster_features.items()}
    for spec in specs:
        cid = str(spec.get("cluster_id") or "")
        if cid in all_features_by_cluster:
            all_features_by_cluster[cid].add(str(spec.get("feature", "")))
    momentum_cols = [
        cluster_score_by_id[cid]
        for cid, features in all_features_by_cluster.items()
        if cid in cluster_score_by_id and any("factor_mom" in f or "factor_resid_mom" in f for f in features)
    ]
    reversal_cols = [
        cluster_score_by_id[cid]
        for cid, features in all_features_by_cluster.items()
        if cid in cluster_score_by_id
        and any(key in f for f in features for key in ("ret_1d", "ret_3d", "ret_5d", "ret_10d", "dist_ma", "liquidity", "illiquidity"))
    ]

    risk_on_parts: list[pd.Series] = []
    if momentum_cols:
        risk_on_parts.append(out[momentum_cols].mean(axis=1, skipna=True))
    if "factor_beta_252_spy" in out.columns:
        beta_rank = pd.to_numeric(out["factor_beta_252_spy"], errors="coerce").groupby(out["date"]).rank(pct=True)
        risk_on_parts.append(beta_rank)
        out["factor_defensive_score"] = 1.0 - beta_rank
    else:
        out["factor_defensive_score"] = out["factor_walkforward_score"]
    if risk_on_parts:
        out["factor_risk_on_score"] = pd.concat(risk_on_parts, axis=1).mean(axis=1, skipna=True)
    else:
        out["factor_risk_on_score"] = out["factor_walkforward_score"]

    if reversal_cols:
        reversal_score = out[reversal_cols].mean(axis=1, skipna=True)
        out["factor_defensive_score"] = pd.concat([out["factor_defensive_score"], reversal_score], axis=1).mean(axis=1, skipna=True)
    out["factor_risk_on_score"] = out["factor_risk_on_score"].fillna(out["factor_walkforward_score"])
    out["factor_defensive_score"] = out["factor_defensive_score"].fillna(out["factor_walkforward_score"])
    return out


def attach_walkforward_factor_score(
    panel: pd.DataFrame,
    specs: list[dict],
    cluster_score_cols: list[str],
    *,
    lookback_days: int = 504,
    min_periods: int = 126,
    top_k: int = 8,
) -> pd.DataFrame:
    """Add a walk-forward score that selects clusters, not raw features."""
    if not cluster_score_cols:
        panel["factor_walkforward_score"] = panel["factor_score"]
        return panel

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    ic_by_cluster: dict[str, pd.Series] = {}
    for raw_col in cluster_score_cols:
        daily_ic = (
            panel[["date", raw_col, "forward_return"]]
            .dropna()
            .groupby("date")
            .apply(lambda g: g[raw_col].corr(g["forward_return"], method="spearman") if g[raw_col].nunique() > 1 else np.nan)
        )
        daily_ic = daily_ic.reindex(dates).astype(float)
        ic_by_cluster[raw_col] = daily_ic.rolling(lookback_days, min_periods=min_periods).mean().shift(HORIZON_DAYS)

    ic_frame = pd.DataFrame(ic_by_cluster, index=dates)
    score_parts: list[pd.Series] = []
    for dt, group in panel.groupby("date", sort=False):
        if dt in ic_frame.index:
            row_ic = ic_frame.loc[dt].dropna()
        else:
            row_ic = pd.Series(dtype=float)
        if row_ic.empty:
            score_parts.append(pd.Series(panel.loc[group.index, "factor_score"], index=group.index))
            continue
        selected = row_ic.abs().sort_values(ascending=False).head(top_k).index.tolist()
        if not selected:
            score_parts.append(pd.Series(panel.loc[group.index, "factor_score"], index=group.index))
            continue
        components: list[pd.Series] = []
        for raw_col in selected:
            sign = float(row_ic.get(raw_col, 0.0))
            raw_rank = pd.to_numeric(group[raw_col], errors="coerce")
            components.append(raw_rank if sign >= 0 else 1.0 - raw_rank)
        # IC-weighted average of selected clusters (not simple mean).
        # PLAIN ENGLISH: Clusters with stronger recent IC get more weight in
        # the composite.  A cluster with IC=0.08 contributes 2x more than one
        # with IC=0.04.  This makes the overlay score responsive to which
        # factors are currently working best, rather than treating all selected
        # clusters as equally informative.
        ic_magnitudes = row_ic.abs()[selected]
        ic_sum = float(ic_magnitudes.sum())
        if ic_sum > 1e-9 and len(components) > 1:
            ic_weights = (ic_magnitudes / ic_sum).values
            combined = pd.concat(components, axis=1)
            score_parts.append(combined.mul(ic_weights, axis=1).sum(axis=1, skipna=True))
        else:
            score_parts.append(pd.concat(components, axis=1).mean(axis=1, skipna=True))

    panel["factor_walkforward_score"] = pd.concat(score_parts).sort_index()
    panel["factor_walkforward_score"] = panel["factor_walkforward_score"].fillna(panel["factor_score"])
    return panel


def _normalise(weights: pd.Series, gross: float) -> pd.Series:
    total = float(weights.abs().sum())
    if total <= 0:
        return weights * 0.0
    return weights / total * gross


def build_weights(day: pd.DataFrame, shape: str, exposure: str, weighting: str, sector_mode: str, score_col: str) -> pd.Series:
    day = day.dropna(subset=[score_col, "forward_return"]).copy()
    if day.empty:
        return pd.Series(dtype=float)
    if sector_mode == "sector_neutral":
        day["_rank_score"] = day.groupby("sector")[score_col].rank(pct=True)
    else:
        day["_rank_score"] = day[score_col].rank(pct=True)

    n = len(day)
    if shape == "top5":
        top_count = 5
    elif shape == "top10":
        top_count = 10
    elif "decile" in shape:
        top_count = max(1, int(np.ceil(n * 0.10)))
    else:
        top_count = max(1, int(np.ceil(n * 0.20)))
    top = day.nlargest(top_count, "_rank_score")
    bottom = day.nsmallest(top_count, "_rank_score")
    weights = pd.Series(0.0, index=day.index)

    if exposure == "long_only" or not shape.startswith("top_bottom"):
        selected = top
        if sector_mode == "sector_neutral" and not selected.empty:
            sector_budget = 1.0 / max(selected["sector"].nunique(), 1)
            for _sector, group in selected.groupby("sector"):
                if weighting == "score":
                    raw = (group["_rank_score"] - group["_rank_score"].min() + 0.01).clip(lower=0.01)
                    weights.loc[group.index] = _normalise(raw, sector_budget)
                else:
                    weights.loc[group.index] = sector_budget / len(group)
        elif not selected.empty:
            if weighting == "score":
                raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
                weights.loc[selected.index] = _normalise(raw, 1.0)
            else:
                weights.loc[selected.index] = 1.0 / len(selected)
        return weights[weights != 0.0]

    gross = MAX_GROSS_EXPOSURE
    long_gross = gross / 2.0
    short_gross = gross / 2.0
    if exposure == "beta_neutral":
        long_beta = float(top["beta"].mean()) if not top.empty else 1.0
        short_beta = float(bottom["beta"].mean()) if not bottom.empty else 1.0
        if long_beta > 0 and short_beta > 0:
            long_gross = gross * short_beta / (long_beta + short_beta)
            short_gross = gross - long_gross

    if sector_mode == "sector_neutral":
        sectors = sorted(set(top["sector"]) | set(bottom["sector"]))
        for sector in sectors:
            t = top[top["sector"] == sector]
            b = bottom[bottom["sector"] == sector]
            if t.empty or b.empty:
                continue
            lg = long_gross / len(sectors)
            sg = short_gross / len(sectors)
            if weighting == "score":
                weights.loc[t.index] = _normalise((t["_rank_score"] - t["_rank_score"].min() + 0.01), lg)
                weights.loc[b.index] = -_normalise((b["_rank_score"].max() - b["_rank_score"] + 0.01), sg)
            else:
                weights.loc[t.index] = lg / len(t)
                weights.loc[b.index] = -sg / len(b)
    else:
        if not top.empty:
            if weighting == "score":
                weights.loc[top.index] = _normalise((top["_rank_score"] - top["_rank_score"].min() + 0.01), long_gross)
            else:
                weights.loc[top.index] = long_gross / len(top)
        if not bottom.empty:
            if weighting == "score":
                weights.loc[bottom.index] = -_normalise((bottom["_rank_score"].max() - bottom["_rank_score"] + 0.01), short_gross)
            else:
                weights.loc[bottom.index] = -short_gross / len(bottom)
    return weights[weights != 0.0]


def portfolio_stats(equity: pd.Series, periods_per_year: float) -> dict:
    rets = equity.pct_change().dropna()
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
    years = max(len(rets) / periods_per_year, 0.01)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)) if len(rets) else 0.0
    dd = equity / equity.cummax() - 1.0
    return {
        "total_return_pct": round(total * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(float(dd.min()) * 100.0, 2) if len(dd) else 0.0,
    }


def run_one(panel: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.DataFrame, dict]:
    if config["score_source"] == "factor_plus_model":
        score_col = "factor_plus_model_score"
    elif config["score_source"] == "factor_walkforward":
        score_col = "factor_walkforward_score"
    else:
        score_col = "factor_score"
    dates = sorted(panel["date"].unique())
    rebalance_dates = dates[::HORIZON_DAYS]
    equity = INITIAL_CAPITAL
    prev = pd.Series(dtype=float)
    rows: list[dict] = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0

    for dt in rebalance_dates:
        day = panel[panel["date"] == dt]
        weights = build_weights(day, config["shape"], config["exposure"], config["weighting"], config["sector_mode"], score_col)
        if "feature_health_overlay_allowed" in day.columns and not bool(day["feature_health_overlay_allowed"].iloc[0]):
            weights = pd.Series(dtype=float)
        aligned = pd.concat([prev.rename("prev"), weights.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        cost = turnover * SLIPPAGE_BASE_PCT
        realized = float((day.loc[weights.index, "forward_return"] * weights).sum()) if not weights.empty else 0.0
        strategy_ret = realized - cost
        equity *= 1.0 + strategy_ret
        total_turnover += turnover
        total_cost += cost
        rows.append({"date": pd.Timestamp(dt) + pd.tseries.offsets.BDay(HORIZON_DAYS), "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "n_positions": int((weights != 0).sum()),
            "gross_exposure": float(weights.abs().sum()) if not weights.empty else 0.0,
            "net_exposure": float(weights.sum()) if not weights.empty else 0.0,
            "turnover": turnover,
            "cost": cost,
            "period_return": strategy_ret,
        })
        prev = weights

    equity_series = pd.DataFrame(rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    trades = pd.DataFrame(trade_rows)
    metrics = {
        "turnover_pct": round(total_turnover * 100.0, 2),
        "estimated_cost_pct": round(total_cost * 100.0, 4),
        "avg_gross_exposure": round(float(trades["gross_exposure"].mean()), 3) if not trades.empty else 0.0,
        "avg_net_exposure": round(float(trades["net_exposure"].mean()), 3) if not trades.empty else 0.0,
        "n_rebalances": int(len(trades)),
    }
    return equity_series, trades, metrics


def benchmark_equity(index: pd.DatetimeIndex) -> pd.DataFrame:
    prices = _load_etf_price_frame(index, ["SPY", "QQQ"])
    out = pd.DataFrame(index=index)
    out["SPY"] = INITIAL_CAPITAL * prices["SPY"] / prices["SPY"].iloc[0]
    out["QQQ"] = INITIAL_CAPITAL * prices["QQQ"] / prices["QQQ"].iloc[0]
    out["BLEND"] = INITIAL_CAPITAL * (0.60 * prices["SPY"] / prices["SPY"].iloc[0] + 0.40 * prices["QQQ"] / prices["QQQ"].iloc[0])
    return out.ffill().bfill()


def compare_to_benchmarks(equity: pd.Series, bench: pd.DataFrame) -> dict:
    comps: dict[str, dict] = {}
    strategy_ret = equity.pct_change().fillna(0.0)
    for symbol in ("SPY", "QQQ", "BLEND"):
        b = bench[symbol].reindex(equity.index).ffill().bfill()
        bm_total = float(b.iloc[-1] / b.iloc[0] - 1.0)
        st_total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        b_ret = b.pct_change().fillna(0.0)
        comps[symbol] = {
            "benchmark_return_pct": round(bm_total * 100.0, 2),
            "alpha_pct": round((st_total - bm_total) * 100.0, 2),
            "nw_tstat_vs_benchmark": round(_newey_west_tstat(strategy_ret - b_ret), 3),
        }
    return comps


def subperiod_metrics(equity: pd.Series, bench: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for name, (start, end) in SUBPERIODS.items():
        eq = equity.loc[(equity.index >= start) & (equity.index <= end)]
        if len(eq) < 3:
            out[name] = {"data_available": False}
            continue
        b = bench["BLEND"].reindex(eq.index).ffill().bfill()
        eq_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        bm_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
        out[name] = {
            "data_available": True,
            "return_pct": round(eq_ret * 100.0, 2),
            "benchmark_return_pct": round(bm_ret * 100.0, 2),
            "alpha_pct": round((eq_ret - bm_ret) * 100.0, 2),
        }
    return out


def gate_metrics(metrics: dict, bench_stats: dict, subperiods: dict, yearly_alpha: pd.Series) -> dict:
    alpha_blend = float(metrics["benchmark_comparisons"]["BLEND"]["alpha_pct"])
    alpha_spy = float(metrics["benchmark_comparisons"]["SPY"]["alpha_pct"])
    alpha_qqq = float(metrics["benchmark_comparisons"]["QQQ"]["alpha_pct"])
    qqq_dd = float(bench_stats["QQQ"]["max_drawdown_pct"])
    qqq_return = float(bench_stats["QQQ"]["total_return_pct"])
    stable = all(v.get("alpha_pct", -999.0) > 0 for v in subperiods.values() if v.get("data_available"))
    positive_alpha_years = yearly_alpha[yearly_alpha > 0]
    if alpha_blend > 0 and not positive_alpha_years.empty:
        concentration = float(positive_alpha_years.max() / max(positive_alpha_years.sum(), 1e-9))
    else:
        concentration = 1.0
    return {
        "alpha_vs_spy_pass": alpha_spy > 0,
        "alpha_vs_qqq_pass": alpha_qqq > 0,
        "alpha_vs_blend_after_cost_pass": alpha_blend > 0,
        "nw_tstat_vs_blend_pass": float(metrics["benchmark_comparisons"]["BLEND"]["nw_tstat_vs_benchmark"]) > 0,
        "subperiod_stability_pass": bool(stable),
        "year_alpha_concentration_pass": bool(concentration <= 0.50),
        "turnover_adjusted_alpha_pass": alpha_blend > 0,
        "drawdown_vs_qqq_pass": bool(float(metrics["max_drawdown_pct"]) >= qqq_dd or float(metrics["total_return_pct"]) > qqq_return),
        "max_positive_year_alpha_share": round(concentration, 3),
    }


def evaluate_config(panel: pd.DataFrame, config: dict) -> tuple[dict, pd.Series, pd.DataFrame]:
    equity, trades, extra = run_one(panel, config)
    periods_per_year = 252.0 / HORIZON_DAYS
    stats = portfolio_stats(equity, periods_per_year)
    bench = benchmark_equity(pd.DatetimeIndex(equity.index))
    bench_stats = {sym: portfolio_stats(bench[sym], periods_per_year) for sym in bench.columns}
    comps = compare_to_benchmarks(equity, bench)
    subs = subperiod_metrics(equity, bench)
    strat_rets = equity.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity.index.year).sum() * 100.0
    metrics = {
        **config,
        **stats,
        **extra,
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "yearly_alpha_pct": {str(k): round(float(v), 2) for k, v in yearly_alpha.items()},
    }
    gates = gate_metrics(metrics, bench_stats, subs, yearly_alpha)
    health = dict(getattr(panel, "attrs", {}).get("feature_health_summary") or {"feature_health_gate_pass": True})
    gates["feature_health_gate_pass"] = bool(health.get("feature_health_gate_pass", True))
    gates["all_pass"] = all(v for k, v in gates.items() if k.endswith("_pass"))
    metrics["alpha_factor_gate_results"] = gates
    metrics.update({
        "feature_health_gate_pass": bool(health.get("feature_health_gate_pass", True)),
        "feature_health_gate_reasons": health.get("feature_health_gate_reasons", []),
        "raw_feature_count": int(health.get("raw_feature_count", 0) or 0),
        "active_cluster_count": int(health.get("active_cluster_count", 0) or 0),
        "effective_cluster_count": int(health.get("effective_cluster_count", 0) or 0),
        "quarantined_features": health.get("quarantined_features", []),
        "watchlist_features": health.get("watchlist_features", []),
        "max_cluster_weight": float(health.get("max_cluster_weight", 0.0) or 0.0),
    })
    metrics["paper_ready"] = False
    return metrics, equity, trades


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    specs = load_feature_specs()
    panel = attach_scores(load_factor_panel(specs), specs, load_prediction_scores())

    rows: list[dict] = []
    best: tuple[dict, pd.Series, pd.DataFrame] | None = None
    score_sources = tuple(dict.fromkeys((*SCORE_SOURCES, "factor_walkforward")))
    for score_source in score_sources:
        for shape in TOP_N_SHAPES:
            for exposure in EXPOSURE_MODES:
                if exposure in {"long_short", "beta_neutral"} and not shape.startswith("top_bottom"):
                    continue
                for weighting in WEIGHTING_MODES:
                    for sector_mode in SECTOR_MODES:
                        config = {
                            "horizon_days": HORIZON_DAYS,
                            "score_source": score_source,
                            "shape": shape,
                            "exposure": exposure,
                            "weighting": weighting,
                            "sector_mode": sector_mode,
                        }
                        metrics, equity, trades = evaluate_config(panel, config)
                        comps = metrics["benchmark_comparisons"]
                        gates = metrics["alpha_factor_gate_results"]
                        score_components = robustness_score_components({**metrics, **gates})
                        rows.append({
                            **config,
                            "total_return_pct": metrics["total_return_pct"],
                            "cagr_pct": metrics["cagr_pct"],
                            "sharpe": metrics["sharpe"],
                            "max_drawdown_pct": metrics["max_drawdown_pct"],
                            "alpha_vs_spy_pct": comps["SPY"]["alpha_pct"],
                            "alpha_vs_qqq_pct": comps["QQQ"]["alpha_pct"],
                            "alpha_vs_blend_pct": comps["BLEND"]["alpha_pct"],
                            "nw_tstat_vs_blend": comps["BLEND"]["nw_tstat_vs_benchmark"],
                            "turnover_pct": metrics["turnover_pct"],
                            "estimated_cost_pct": metrics["estimated_cost_pct"],
                            "avg_gross_exposure": metrics["avg_gross_exposure"],
                            "feature_health_gate_pass": gates.get("feature_health_gate_pass", True),
                            "active_cluster_count": metrics.get("active_cluster_count", 0),
                            "effective_cluster_count": metrics.get("effective_cluster_count", 0),
                            "max_cluster_weight": metrics.get("max_cluster_weight", 0.0),
                            "feature_health_gate_reasons": "; ".join(metrics.get("feature_health_gate_reasons", [])),
                            "subperiod_stability_pass": gates["subperiod_stability_pass"],
                            "paper_ready": gates["all_pass"],
                            **score_components,
                        })
                        score_rank = (bool(gates["all_pass"]), float(score_components["robustness_score"]))
                        if best is None or score_rank > best[4]:
                            best = (metrics, equity, trades, score_components, score_rank)

    grid = pd.DataFrame(rows).sort_values(
        ["paper_ready", "robustness_score", "sharpe", "max_drawdown_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    grid_path = Path(SIGNAL_DIR) / "alpha_factor_grid.csv"
    grid.to_csv(grid_path, index=False)

    if best is None:
        raise SystemExit("No alpha factor configs evaluated.")
    best_metrics, best_equity, best_trades, best_score_components, _best_score_rank = best
    best_factor = grid[grid["score_source"] == "factor_only"].head(1)
    best_factor_ml = grid[grid["score_source"] == "factor_plus_model"].head(1)
    ml_value_add = {
        "tested": bool(not best_factor.empty and not best_factor_ml.empty),
        "ml_kept": False,
    }
    if not best_factor.empty and not best_factor_ml.empty:
        f = best_factor.iloc[0]
        m = best_factor_ml.iloc[0]
        ml_value_add.update({
            "best_factor_only_alpha_vs_blend_pct": round(float(f["alpha_vs_blend_pct"]), 2),
            "best_factor_plus_model_alpha_vs_blend_pct": round(float(m["alpha_vs_blend_pct"]), 2),
            "alpha_delta_pct": round(float(m["alpha_vs_blend_pct"] - f["alpha_vs_blend_pct"]), 2),
            "sharpe_delta": round(float(m["sharpe"] - f["sharpe"]), 3),
            "drawdown_delta_pct": round(float(m["max_drawdown_pct"] - f["max_drawdown_pct"]), 2),
        })
        ml_value_add["ml_kept"] = bool(
            ml_value_add["alpha_delta_pct"] > 0
            and (ml_value_add["sharpe_delta"] > 0 or ml_value_add["drawdown_delta_pct"] >= 0)
        )
    best_metrics["selected_features"] = specs
    best_metrics["grid_rows"] = int(len(grid))
    best_metrics["best_config_source"] = "best_robustness_score_after_cost"
    best_metrics.update(best_score_components)
    best_metrics["ml_value_add"] = ml_value_add
    best_metrics["paper_ready"] = bool(best_metrics["alpha_factor_gate_results"]["all_pass"])

    pd.DataFrame({"equity": best_equity}).to_csv(Path(SIGNAL_DIR) / "alpha_factor_equity.csv")
    best_trades.to_csv(Path(SIGNAL_DIR) / "alpha_factor_trades.csv", index=False)
    with open(Path(SIGNAL_DIR) / "alpha_factor_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"Saved -> {grid_path}")
    print(f"Saved -> {Path(SIGNAL_DIR) / 'alpha_factor_metrics.json'}")
    print(f"Saved -> {Path(SIGNAL_DIR) / 'alpha_factor_equity.csv'}")
    print("\nTop alpha factor rows")
    print(grid.head(15).to_string(index=False))
    print("\nBest config gates:", "PASS" if best_metrics["paper_ready"] else "FAIL")


if __name__ == "__main__":
    main()
