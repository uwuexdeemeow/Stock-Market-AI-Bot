from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from settings import MODEL_DIR

QUALITY_REPORT = os.path.join(MODEL_DIR, "model_quality_report.csv")

MIN_DIRECTION_EDGE_PP = 2.0
MIN_RETURN_SPEARMAN = 0.0
MIN_WALKFORWARD_TRADES = 10
MIN_AFTER_COST_SHARPE = 0.5
MAX_DRAWDOWN_PCT = -25.0

PREAPPROVED_WEAK_TICKERS = {"CAT", "GOOGL", "BAC", "AMD", "MSFT", "TSLA"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_train_summary(ticker: str) -> dict:
    path = os.path.join(MODEL_DIR, f"{ticker}_train_summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def trade_quality_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        return {
            "walkforward_trades": 0,
            "walkforward_win_rate": None,
            "walkforward_total_pnl": 0.0,
            "walkforward_sharpe": 0.0,
            "walkforward_max_drawdown_pct": 0.0,
        }

    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0).astype(float)
    trade_returns = pnl / 10_000.0
    sharpe = 0.0
    if len(trade_returns) > 1 and float(trade_returns.std()) > 0:
        sharpe = float((trade_returns.mean() / trade_returns.std()) * np.sqrt(252 / 10))

    equity = 10_000.0 + pnl.cumsum()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)

    return {
        "walkforward_trades": int(len(trades)),
        "walkforward_win_rate": round(float((pnl > 0).mean()), 4),
        "walkforward_total_pnl": round(float(pnl.sum()), 2),
        "walkforward_sharpe": round(sharpe, 3),
        "walkforward_max_drawdown_pct": round(float(drawdown.min()) * 100.0, 2) if len(drawdown) else 0.0,
    }


def evaluate_model_quality(
    ticker: str,
    train_summary: dict | None = None,
    trades: pd.DataFrame | None = None,
    extra_metrics: dict | None = None,
) -> dict:
    summary = train_summary or load_train_summary(ticker)
    trade_metrics = trade_quality_metrics(trades if trades is not None else pd.DataFrame())

    direction_accuracy = _safe_float(summary.get("direction_accuracy"))
    baseline_up_rate = _safe_float(summary.get("baseline_up_rate"), 50.0)
    direction_edge_pp = round(direction_accuracy - baseline_up_rate, 2)
    return_eval = summary.get("return_model_eval", {}) or {}
    return_spearman = _safe_float(return_eval.get("spearman_corr"), 0.0)

    reasons: list[str] = []
    if direction_edge_pp < MIN_DIRECTION_EDGE_PP:
        reasons.append(f"direction_edge {direction_edge_pp:.2f}pp < {MIN_DIRECTION_EDGE_PP:.2f}pp")
    if return_spearman <= MIN_RETURN_SPEARMAN:
        reasons.append(f"return_spearman {return_spearman:.4f} <= {MIN_RETURN_SPEARMAN:.4f}")
    if trade_metrics["walkforward_trades"] < MIN_WALKFORWARD_TRADES:
        reasons.append(f"walkforward_trades {trade_metrics['walkforward_trades']} < {MIN_WALKFORWARD_TRADES}")
    if trade_metrics["walkforward_sharpe"] < MIN_AFTER_COST_SHARPE:
        reasons.append(f"walkforward_sharpe {trade_metrics['walkforward_sharpe']:.3f} < {MIN_AFTER_COST_SHARPE:.3f}")
    if trade_metrics["walkforward_max_drawdown_pct"] < MAX_DRAWDOWN_PCT:
        reasons.append(
            f"max_drawdown {trade_metrics['walkforward_max_drawdown_pct']:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%"
        )
    if extra_metrics:
        portfolio_sharpe = extra_metrics.get("portfolio_sharpe")
        if portfolio_sharpe is not None and _safe_float(portfolio_sharpe) < MIN_AFTER_COST_SHARPE:
            reasons.append(f"portfolio_sharpe {_safe_float(portfolio_sharpe):.3f} < {MIN_AFTER_COST_SHARPE:.3f}")
        portfolio_nw = extra_metrics.get("portfolio_nw_tstat_vs_cash")
        if portfolio_nw is not None and _safe_float(portfolio_nw) < 1.65:
            reasons.append(f"portfolio_nw_tstat {_safe_float(portfolio_nw):.3f} < 1.650")
        portfolio_dd = extra_metrics.get("portfolio_max_drawdown_pct")
        if portfolio_dd is not None and _safe_float(portfolio_dd) < MAX_DRAWDOWN_PCT:
            reasons.append(f"portfolio_max_drawdown {_safe_float(portfolio_dd):.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if ticker.upper() in PREAPPROVED_WEAK_TICKERS:
        reasons.append("listed as weak until retrained and backtested")

    approved = not reasons
    row = {
        "ticker": ticker.upper(),
        "approved_for_live": bool(approved),
        "approval_status": "approved" if approved else "rejected",
        "approval_reason": "all live-edge gates passed" if approved else "; ".join(reasons),
        "direction_accuracy": round(direction_accuracy, 2),
        "baseline_up_rate": round(baseline_up_rate, 2),
        "direction_edge_pp": direction_edge_pp,
        "return_spearman": round(return_spearman, 4),
        "threshold_used_fallback": bool(summary.get("threshold_used_fallback", True)),
        "confidence_threshold": _safe_float(summary.get("confidence_threshold"), 57.5),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        **trade_metrics,
    }
    if extra_metrics:
        for key, value in extra_metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
    return row


def upsert_quality_report(rows: list[dict]) -> pd.DataFrame:
    os.makedirs(MODEL_DIR, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(QUALITY_REPORT):
        try:
            old_df = pd.read_csv(QUALITY_REPORT)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        out = new_df
    else:
        old_df = old_df[~old_df["ticker"].astype(str).str.upper().isin(new_df["ticker"].astype(str).str.upper())]
        out = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    out = out.sort_values(["approved_for_live", "direction_edge_pp"], ascending=[False, False])
    out.to_csv(QUALITY_REPORT, index=False)
    return out


def update_scaler_metadata(ticker: str, quality_row: dict) -> None:
    path = os.path.join(MODEL_DIR, f"{ticker.upper()}_scaler.pkl")
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            metadata = pickle.load(f)
        metadata["model_quality"] = quality_row
        metadata["approved_for_live"] = bool(quality_row.get("approved_for_live", False))
        metadata["approval_status"] = quality_row.get("approval_status", "rejected")
        metadata["approval_reason"] = quality_row.get("approval_reason", "not evaluated")
        with open(path, "wb") as f:
            pickle.dump(metadata, f)
    except Exception:
        return


def read_quality_report() -> pd.DataFrame:
    if not os.path.exists(QUALITY_REPORT):
        return pd.DataFrame()
    try:
        return pd.read_csv(QUALITY_REPORT)
    except Exception:
        return pd.DataFrame()
