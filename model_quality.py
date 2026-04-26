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

MIN_WALKFORWARD_TRADES = 10
MIN_WALKFORWARD_TOTAL_PNL = 0.0
MIN_AFTER_COST_SHARPE = 0.5
MIN_PROFIT_FACTOR = 1.10
MIN_AVG_WIN_LOSS_RATIO = 1.0
MAX_DRAWDOWN_PCT = -25.0
MIN_PORTFOLIO_CAGR_PCT = 0.0

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
            "walkforward_avg_win": None,
            "walkforward_avg_loss": None,
            "walkforward_avg_win_loss_ratio": None,
            "walkforward_profit_factor": None,
            "walkforward_avg_expected_return": None,
            "walkforward_exposure_pct": 0.0,
        }

    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0).astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(abs(losses.mean())) if len(losses) else None
    avg_win_loss_ratio = (
        float(avg_win / avg_loss)
        if avg_win is not None and avg_loss is not None and avg_loss > 0
        else None
    )
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit <= 0 else 999.0)
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
        "walkforward_avg_win": round(avg_win, 2) if avg_win is not None else None,
        "walkforward_avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "walkforward_avg_win_loss_ratio": round(avg_win_loss_ratio, 3) if avg_win_loss_ratio is not None else None,
        "walkforward_profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "walkforward_avg_expected_return": round(
            float(pd.to_numeric(trades.get("expected_return", pd.Series(dtype=float)), errors="coerce").mean()),
            4,
        )
        if "expected_return" in trades.columns
        else None,
        "walkforward_exposure_pct": round(
            float(pd.to_numeric(trades.get("position_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
            2,
        )
        if "position_pct" in trades.columns
        else 0.0,
    }


def evaluate_model_quality(
    ticker: str,
    train_summary: dict | None = None,
    trades: pd.DataFrame | None = None,
    extra_metrics: dict | None = None,
) -> dict:
    summary = train_summary or load_train_summary(ticker)
    trade_metrics = trade_quality_metrics(trades if trades is not None else pd.DataFrame())
    quality_source = str((extra_metrics or {}).get("quality_source", ""))

    direction_accuracy = _safe_float(summary.get("direction_accuracy"))
    baseline_up_rate = _safe_float(summary.get("baseline_up_rate"), 50.0)
    direction_edge_pp = round(direction_accuracy - baseline_up_rate, 2)
    return_eval = summary.get("return_model_eval", {}) or {}
    return_spearman_raw = return_eval.get("spearman_corr")
    return_spearman = None if return_spearman_raw is None else _safe_float(return_spearman_raw, 0.0)

    reasons: list[str] = []
    if trade_metrics["walkforward_trades"] < MIN_WALKFORWARD_TRADES:
        reasons.append(f"walkforward_trades {trade_metrics['walkforward_trades']} < {MIN_WALKFORWARD_TRADES}")
    if trade_metrics["walkforward_total_pnl"] <= MIN_WALKFORWARD_TOTAL_PNL:
        reasons.append(
            f"walkforward_total_pnl {trade_metrics['walkforward_total_pnl']:.2f} <= {MIN_WALKFORWARD_TOTAL_PNL:.2f}"
        )
    if trade_metrics["walkforward_sharpe"] < MIN_AFTER_COST_SHARPE:
        reasons.append(f"walkforward_sharpe {trade_metrics['walkforward_sharpe']:.3f} < {MIN_AFTER_COST_SHARPE:.3f}")
    profit_factor = trade_metrics.get("walkforward_profit_factor")
    if profit_factor is None or _safe_float(profit_factor) < MIN_PROFIT_FACTOR:
        reasons.append(f"profit_factor {_safe_float(profit_factor):.3f} < {MIN_PROFIT_FACTOR:.3f}")
    avg_win_loss = trade_metrics.get("walkforward_avg_win_loss_ratio")
    if avg_win_loss is None or _safe_float(avg_win_loss) < MIN_AVG_WIN_LOSS_RATIO:
        reasons.append(f"avg_win_loss_ratio {_safe_float(avg_win_loss):.3f} < {MIN_AVG_WIN_LOSS_RATIO:.3f}")
    if trade_metrics["walkforward_max_drawdown_pct"] < MAX_DRAWDOWN_PCT:
        reasons.append(
            f"max_drawdown {trade_metrics['walkforward_max_drawdown_pct']:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%"
        )
    if extra_metrics:
        portfolio_cagr = extra_metrics.get("portfolio_cagr_pct")
        if portfolio_cagr is not None and _safe_float(portfolio_cagr) <= MIN_PORTFOLIO_CAGR_PCT:
            reasons.append(f"portfolio_cagr {_safe_float(portfolio_cagr):.2f}% <= {MIN_PORTFOLIO_CAGR_PCT:.2f}%")
        portfolio_sharpe = extra_metrics.get("portfolio_sharpe")
        if portfolio_sharpe is not None and _safe_float(portfolio_sharpe) < MIN_AFTER_COST_SHARPE:
            reasons.append(f"portfolio_sharpe {_safe_float(portfolio_sharpe):.3f} < {MIN_AFTER_COST_SHARPE:.3f}")
        portfolio_nw = extra_metrics.get("portfolio_nw_tstat_vs_cash")
        if portfolio_nw is not None and _safe_float(portfolio_nw) < 1.65:
            reasons.append(f"portfolio_nw_tstat {_safe_float(portfolio_nw):.3f} < 1.650")
        portfolio_dd = extra_metrics.get("portfolio_max_drawdown_pct")
        if portfolio_dd is not None and _safe_float(portfolio_dd) < MAX_DRAWDOWN_PCT:
            reasons.append(f"portfolio_max_drawdown {_safe_float(portfolio_dd):.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if ticker.upper() in PREAPPROVED_WEAK_TICKERS and quality_source != "backtest_walkforward":
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
        "return_spearman": round(return_spearman, 4) if return_spearman is not None else None,
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
    sort_cols = [c for c in ["approved_for_live", "walkforward_profit_factor", "walkforward_total_pnl"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
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
