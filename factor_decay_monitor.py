"""
factor_decay_monitor.py — recent health checks for the core-satellite overlay.

The production signal remains unchanged. This monitor answers whether the
factor overlay still has recent cross-sectional IC and realized overlay alpha,
so paper/real-capital gates can distinguish a healthy backtest from a decaying
edge.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from settings import LOG_DIR, SIGNAL_DIR


OUT_CSV = Path(SIGNAL_DIR) / "factor_decay_monitor.csv"
OUT_JSON = Path(LOG_DIR) / "factor_decay_monitor.json"
TRADES_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_trades.csv"
METRICS_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
MIN_WEAK_OVERLAY_ALPHA_PCT = -0.1  # warn if overlay alpha below -0.1% (noise floor)
OVERLAY_ALPHA_BLOCK_THRESHOLD_PCT = 0.0


def edge_health_status(row: dict | pd.Series) -> str:
    """Classify live factor edge health without overreacting to full-rank IC alone."""
    rank_ic = pd.to_numeric(pd.Series([row.get("daily_ic_mean")]), errors="coerce").iloc[0]
    top_excess = pd.to_numeric(pd.Series([row.get("top_bucket_excess_return_pct")]), errors="coerce").iloc[0]
    overlay_alpha = pd.to_numeric(pd.Series([row.get("overlay_alpha_sum_pct")]), errors="coerce").iloc[0]
    if pd.notna(overlay_alpha) and float(overlay_alpha) < OVERLAY_ALPHA_BLOCK_THRESHOLD_PCT:
        return "block"
    if pd.isna(top_excess) or float(top_excess) <= 0.0:
        return "warning"
    if pd.notna(overlay_alpha) and float(overlay_alpha) <= MIN_WEAK_OVERLAY_ALPHA_PCT:
        return "warning"
    if pd.notna(rank_ic) and float(rank_ic) < 0.0:
        return "advisory"
    return "pass"


def aggregate_edge_health_status(rows: list[dict]) -> str:
    statuses = [str(row.get("edge_health_status", "warning")) for row in rows]
    for status in ("block", "warning", "advisory"):
        if status in statuses:
            return status
    return "pass"


def _selected_config() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    metrics = json.loads(METRICS_PATH.read_text())
    keys = (
        "regime_mode",
        "score_source",
        "holding_days",
        "regime_ma_window",
        "regime_high_vol",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _safe_spearman(group: pd.DataFrame, score_col: str, return_col: str) -> float:
    sub = group[[score_col, return_col]].dropna()
    if len(sub) < 5 or sub[score_col].nunique() < 2 or sub[return_col].nunique() < 2:
        return np.nan
    return float(sub[score_col].corr(sub[return_col], method="spearman"))


def _regime_score_panel(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = panel.copy()
    holding_days = int(config.get("holding_days", core.HORIZON_DAYS))
    return_col = "forward_return" if holding_days == core.HORIZON_DAYS else f"forward_return_{holding_days}d"
    if return_col not in out.columns:
        return_col = "forward_return"
    dates = pd.DatetimeIndex(sorted(out["date"].unique()))
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days) for dt in dates])
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in core.REGIME_PRESETS:
        regime_indicators = core._load_regime_indicators(dates, exit_dates, config)

    regime_by_date: dict[pd.Timestamp, str] = {}
    score_by_date: dict[pd.Timestamp, str] = {}
    dummy_weights = {"SPY": 0.0, "QQQ": 1.0}
    for dt in dates:
        eval_config = {**config, "core_weights": dummy_weights, "core_gross": 1.0, "overlay_gross": 0.0}
        regime, _core_weights, _core_gross, _overlay_gross = core._resolve_allocation(pd.Timestamp(dt), eval_config, regime_indicators)
        regime_by_date[pd.Timestamp(dt)] = regime
        score_by_date[pd.Timestamp(dt)] = core._score_col_for_regime(str(config.get("score_source", "regime_adaptive")), regime)

    out["_monitor_regime"] = out["date"].map(regime_by_date)
    out["_monitor_score_col"] = out["date"].map(score_by_date)
    score = pd.Series(np.nan, index=out.index, dtype=float)
    for score_col in sorted(set(score_by_date.values())):
        if score_col not in out.columns:
            continue
        mask = out["_monitor_score_col"].eq(score_col)
        score.loc[mask] = pd.to_numeric(out.loc[mask, score_col], errors="coerce")
    out["_monitor_score"] = score
    out["_monitor_return"] = pd.to_numeric(out[return_col], errors="coerce")
    return out.dropna(subset=["_monitor_score", "_monitor_return"])


def _ic_summary(scored: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = scored[(scored["date"] >= start) & (scored["date"] <= as_of)].copy()
    if recent.empty:
        return {
            "lookback_days": lookback_days,
            "daily_ic_mean": np.nan,
            "daily_ic_positive_rate": np.nan,
            "top_bucket_excess_return_pct": np.nan,
            "observations": 0,
        }
    daily_ic = recent.groupby("date").apply(lambda g: _safe_spearman(g, "_monitor_score", "_monitor_return"))
    top_returns = []
    for _dt, group in recent.groupby("date"):
        if len(group) < 10:
            continue
        ranks = group["_monitor_score"].rank(pct=True)
        top = group.loc[ranks >= 0.90, "_monitor_return"]
        rest = group.loc[ranks < 0.90, "_monitor_return"]
        if not top.empty and not rest.empty:
            top_returns.append(float(top.mean() - rest.mean()))
    return {
        "lookback_days": lookback_days,
        "daily_ic_mean": round(float(daily_ic.mean()), 5) if daily_ic.notna().any() else np.nan,
        "daily_ic_positive_rate": round(float((daily_ic.dropna() > 0).mean()), 4) if daily_ic.notna().any() else np.nan,
        "top_bucket_excess_return_pct": round(float(np.nanmean(top_returns) * 100.0), 4) if top_returns else np.nan,
        "observations": int(len(recent)),
    }


def _overlay_summary(trades: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    if trades.empty:
        return {
            "lookback_days": lookback_days,
            "overlay_periods": 0,
            "overlay_return_sum_pct": np.nan,
            "overlay_hit_rate": np.nan,
            "overlay_alpha_sum_pct": np.nan,
        }
    frame = trades.copy()
    frame["exit_date"] = pd.to_datetime(frame.get("exit_date"), errors="coerce")
    frame["factor_overlay_return"] = pd.to_numeric(frame.get("factor_overlay_return"), errors="coerce")
    frame["core_return"] = pd.to_numeric(frame.get("core_return"), errors="coerce")
    frame = frame.dropna(subset=["exit_date", "factor_overlay_return"])
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = frame[(frame["exit_date"] >= start) & (frame["exit_date"] <= as_of)]
    if recent.empty:
        return {
            "lookback_days": lookback_days,
            "overlay_periods": 0,
            "overlay_return_sum_pct": np.nan,
            "overlay_hit_rate": np.nan,
            "overlay_alpha_sum_pct": np.nan,
        }
    overlay = recent["factor_overlay_return"]
    return {
        "lookback_days": lookback_days,
        "overlay_periods": int(len(recent)),
        "overlay_return_sum_pct": round(float(overlay.sum() * 100.0), 4),
        "overlay_hit_rate": round(float((overlay > 0).mean()), 4),
        "overlay_alpha_sum_pct": round(float(overlay.sum() * 100.0), 4),
    }


def _top_contributor(trades: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    if trades.empty or "overlay_weights_json" not in trades.columns:
        return {"lookback_days": lookback_days, "top_selected_ticker": "", "top_selected_count": 0}
    frame = trades.copy()
    frame["exit_date"] = pd.to_datetime(frame.get("exit_date"), errors="coerce")
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = frame[(frame["exit_date"] >= start) & (frame["exit_date"] <= as_of)]
    counts: dict[str, int] = {}
    for raw in recent["overlay_tickers"].fillna("").astype(str):
        for ticker in [t for t in raw.split(",") if t]:
            counts[ticker] = counts.get(ticker, 0) + 1
    if not counts:
        return {"lookback_days": lookback_days, "top_selected_ticker": "", "top_selected_count": 0}
    ticker, count = max(counts.items(), key=lambda kv: kv[1])
    return {"lookback_days": lookback_days, "top_selected_ticker": ticker, "top_selected_count": int(count)}


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    config = _selected_config()
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))
    scored = _regime_score_panel(panel, config)
    as_of = pd.Timestamp(scored["date"].max())
    trades = pd.read_csv(TRADES_PATH) if TRADES_PATH.exists() else pd.DataFrame()

    rows = []
    for lookback in (60, 120):
        row = {
            **_ic_summary(scored, lookback, as_of),
            **{k: v for k, v in _overlay_summary(trades, lookback, as_of).items() if k != "lookback_days"},
            **{k: v for k, v in _top_contributor(trades, lookback, as_of).items() if k != "lookback_days"},
        }
        row["as_of"] = str(as_of.date())
        row["rank_ic_warning"] = bool(pd.notna(row["daily_ic_mean"]) and float(row["daily_ic_mean"]) < 0)
        row["overlay_alpha_warning"] = bool(pd.notna(row["overlay_alpha_sum_pct"]) and float(row["overlay_alpha_sum_pct"]) < 0)
        row["edge_health_status"] = edge_health_status(row)
        row["warning"] = bool(row["edge_health_status"] in {"warning", "block"})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    edge_status = aggregate_edge_health_status(rows)
    real_capital_block = edge_status == "block"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "factor_decay_monitor",
        "as_of": str(as_of.date()),
        "selected_score_source": config.get("score_source"),
        "rows": rows,
        "edge_health_status": edge_status,
        "warning": bool(edge_status == "warning"),
        "advisory": bool(edge_status == "advisory"),
        "real_capital_block": real_capital_block,
        "reason": (
            "recent overlay alpha is negative"
            if real_capital_block
            else "top-bucket edge is weak/non-positive"
            if edge_status == "warning"
            else "rank IC is weak/negative but top-bucket edge and overlay alpha remain positive"
            if edge_status == "advisory"
            else "factor decay monitor has no real-capital blocking warning"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"Factor decay monitor written -> {OUT_CSV}")
    print(f"Detailed report -> {OUT_JSON}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
