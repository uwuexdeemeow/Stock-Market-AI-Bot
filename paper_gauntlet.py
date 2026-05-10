from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpha_factor_backtest import benchmark_equity
from settings import LOG_DIR, PAPER_MODE_STRATEGY, SIGNAL_DIR


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
PAPER_TRADES = SIGNALS / "paper_trades.csv"
PAPER_EQUITY = SIGNALS / "paper_equity.csv"
PAPER_STATUS = SIGNALS / "paper_daily_status.json"
SURVIVORSHIP_STRESS = LOGS / "core_satellite_survivorship_audit.json"
EXECUTION_STRESS = LOGS / "core_satellite_execution_stress.json"
FACTOR_DECAY = LOGS / "factor_decay_monitor.json"

MIN_TRADES = 20
MIN_SHARPE = 0.5
MAX_DRAWDOWN_PCT = -10.0
MIN_WIN_RATE = 0.52
MIN_PORTFOLIO_EQUITY_DAYS = int(os.environ.get("PAPER_GAUNTLET_MIN_EQUITY_DAYS", "20"))
MIN_PORTFOLIO_FILL_RATE = float(os.environ.get("PAPER_GAUNTLET_MIN_FILL_RATE", "0.95"))
MAX_PORTFOLIO_CANCEL_RATE = float(os.environ.get("PAPER_GAUNTLET_MAX_CANCEL_RATE", "0.05"))
MAX_PORTFOLIO_DRIFT = float(os.environ.get("PAPER_GAUNTLET_MAX_DRIFT", "0.15"))
MAX_AVG_SLIPPAGE_BPS = float(os.environ.get("PAPER_GAUNTLET_MAX_AVG_SLIPPAGE_BPS", "10"))
DEFAULT_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2 or float(returns.std()) == 0.0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _paper_benchmark_comparisons(equity_df: pd.DataFrame, equity: pd.Series) -> dict:
    if equity_df.empty or len(equity) < 2:
        return {"data_available": False, "reason": "need_at_least_two_equity_snapshots"}
    if _paper_days(equity_df) < MIN_PORTFOLIO_EQUITY_DAYS:
        return {
            "data_available": False,
            "reason": f"paper_equity_days_less_than_{MIN_PORTFOLIO_EQUITY_DAYS}",
        }
    if "date" in equity_df.columns:
        dates = pd.to_datetime(equity_df["date"], errors="coerce")
    elif "timestamp" in equity_df.columns:
        dates = pd.to_datetime(equity_df["timestamp"], errors="coerce")
    else:
        return {"data_available": False, "reason": "missing_date_or_timestamp"}
    frame = pd.DataFrame({"date": dates, "equity": equity.reset_index(drop=True)}).dropna(subset=["date", "equity"])
    if len(frame) < 2:
        return {"data_available": False, "reason": "need_at_least_two_valid_equity_snapshots"}
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    idx = pd.DatetimeIndex(frame["date"].dt.normalize())
    paper_equity = pd.Series(frame["equity"].to_numpy(dtype=float), index=idx)
    try:
        bench = benchmark_equity(idx)
    except Exception as exc:
        return {"data_available": False, "reason": f"benchmark_load_failed: {exc}"}
    strategy_ret = float(paper_equity.iloc[-1] / paper_equity.iloc[0] - 1.0)
    out: dict[str, object] = {
        "data_available": True,
        "paper_return_pct": round(strategy_ret * 100.0, 2),
    }
    for symbol in ("SPY", "QQQ", "BLEND"):
        b = bench[symbol].reindex(paper_equity.index).ffill().bfill()
        bench_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
        out[f"{symbol.lower()}_return_pct"] = round(bench_ret * 100.0, 2)
        out[f"alpha_vs_{symbol.lower()}_pct"] = round((strategy_ret - bench_ret) * 100.0, 2)
    return out


def _paper_days(equity_df: pd.DataFrame) -> int:
    if equity_df.empty:
        return 0
    if "date" in equity_df.columns:
        return int(pd.to_datetime(equity_df["date"], errors="coerce").dt.date.nunique())
    if "timestamp" in equity_df.columns:
        return int(pd.to_datetime(equity_df["timestamp"], errors="coerce").dt.date.nunique())
    return int(len(equity_df))


def _order_fill_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "submitted_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "partial_orders": 0,
            "fill_rate": 0.0,
            "cancel_rate": 0.0,
        }

    submitted = trades
    if "submitted" in trades.columns:
        submitted = trades[trades["submitted"].astype(str).str.lower().isin(["true", "1", "yes"])]
    statuses = (
        submitted.get("fill_status", pd.Series(index=submitted.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    filled = int(statuses.eq("filled").sum())
    cancelled = int(statuses.eq("cancelled").sum())
    partial = int(statuses.eq("partial").sum() + statuses.eq("partially_filled").sum())
    total = int(len(submitted))
    return {
        "submitted_orders": total,
        "filled_orders": filled,
        "cancelled_orders": cancelled,
        "partial_orders": partial,
        "fill_rate": float((filled + 0.5 * partial) / total) if total else 0.0,
        "cancel_rate": float(cancelled / total) if total else 0.0,
    }


def _execution_slippage_bps(row: pd.Series | dict) -> float:
    try:
        reference = float(row.get("price", 0.0) or 0.0)
        fill_price = float(row.get("broker_dealt_avg_price", 0.0) or 0.0)
    except Exception:
        return np.nan
    if reference <= 0 or fill_price <= 0:
        return np.nan
    action = str(row.get("action", "")).upper().strip()
    if action == "BUY":
        return float((fill_price - reference) / reference * 10_000.0)
    if action == "SELL":
        return float((reference - fill_price) / reference * 10_000.0)
    return np.nan


def _slippage_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"avg_slippage_bps": None, "worst_slippage_bps": None, "slippage_observations": 0}
    filled = trades[trades.get("fill_status", pd.Series("", index=trades.index)).astype(str).str.lower().eq("filled")]
    if filled.empty:
        return {"avg_slippage_bps": None, "worst_slippage_bps": None, "slippage_observations": 0}
    if "execution_slippage_bps" in filled.columns:
        bps = pd.to_numeric(filled["execution_slippage_bps"], errors="coerce").dropna()
    else:
        bps = filled.apply(_execution_slippage_bps, axis=1).dropna()
    return {
        "avg_slippage_bps": None if bps.empty else round(float(bps.mean()), 3),
        "worst_slippage_bps": None if bps.empty else round(float(bps.max()), 3),
        "slippage_observations": int(len(bps)),
    }


def _signal_timezone():
    try:
        return ZoneInfo(str(DEFAULT_SIGNAL_TIMEZONE))
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_signal_predicted_at(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(_signal_timezone())
    return out.tz_convert("UTC")


def _filter_current_signal_trades(trades: pd.DataFrame, status: dict) -> pd.DataFrame:
    if trades.empty or "submitted_at" not in trades.columns:
        return trades
    signal_ts = _parse_signal_predicted_at(status.get("signal_predicted_at", ""))
    if pd.isna(signal_ts):
        return trades
    submitted_ts = pd.to_datetime(trades["submitted_at"], errors="coerce", utc=True)
    return trades[submitted_ts >= signal_ts].copy()


def _is_portfolio_strategy(status: dict) -> bool:
    strategy = str(status.get("strategy") or PAPER_MODE_STRATEGY).strip().lower()
    return strategy in {"core_satellite_alpha", "etf_rotation"}


def _survivorship_stress_status() -> dict:
    report = _read_json(SURVIVORSHIP_STRESS)
    rows = report.get("rows", []) if report else []
    by_scenario = {str(row.get("scenario")): row for row in rows if isinstance(row, dict)}
    stressed = by_scenario.get("watchlist_plus_failed_audit_tickers", {})
    delta = by_scenario.get("delta_stressed_minus_base", {})
    if not report or not stressed:
        return {
            "data_available": False,
            "stressed_paper_ready": False,
            "reason": "missing core_satellite_survivorship_audit.json",
        }
    return {
        "data_available": True,
        "stressed_paper_ready": bool(stressed.get("paper_ready", False)),
        "available_audit_tickers": report.get("available_audit_tickers", []),
        "missing_audit_tickers": report.get("missing_audit_tickers", []),
        "audit_rebalance_selections": int(float(stressed.get("audit_rebalance_selections", 0) or 0)),
        "total_return_delta_pct": float(delta.get("total_return_pct", 0.0) or 0.0),
        "max_drawdown_delta_pct": float(delta.get("max_drawdown_pct", 0.0) or 0.0),
    }


def _execution_stress_status() -> dict:
    report = _read_json(EXECUTION_STRESS)
    rows = [row for row in report.get("rows", []) if isinstance(row, dict) and not str(row.get("scenario", "")).startswith("delta_")]
    if not report or not rows:
        return {"data_available": False, "stress_pass": False, "reason": "missing core_satellite_execution_stress.json"}
    failed = [
        row for row in rows
        if not bool(row.get("paper_ready", False))
        or float(row.get("alpha_vs_qqq_pct", -999.0) or -999.0) <= 0
        or float(row.get("alpha_vs_blend_pct", -999.0) or -999.0) <= 0
    ]
    return {
        "data_available": True,
        "stress_pass": not failed,
        "scenarios": len(rows),
        "failed_scenarios": len(failed),
        "worst_drawdown_pct": min(float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in rows),
        "worst_alpha_vs_blend_pct": min(float(row.get("alpha_vs_blend_pct", 0.0) or 0.0) for row in rows),
    }


def _factor_decay_status() -> dict:
    report = _read_json(FACTOR_DECAY)
    if not report:
        return {"data_available": False, "real_capital_block": True, "reason": "missing factor_decay_monitor.json"}
    return {
        "data_available": True,
        "warning": bool(report.get("warning", False)),
        "real_capital_block": bool(report.get("real_capital_block", False)),
        "reason": report.get("reason"),
        "as_of": report.get("as_of"),
    }


def _evaluate_portfolio_paper(
    trades: pd.DataFrame,
    equity_df: pd.DataFrame,
    status: dict,
    equity: pd.Series,
    daily_returns: pd.Series,
    total_return_pct: float,
    sharpe: float,
    max_dd: float,
) -> dict:
    current_signal_trades = _filter_current_signal_trades(trades, status)
    fill_stats = _order_fill_stats(current_signal_trades)
    fill_stats_scope = "current_signal"
    paper_days = _paper_days(equity_df)
    strategy = str(status.get("strategy") or PAPER_MODE_STRATEGY)
    max_drift_abs = float(status.get("max_drift_abs", np.nan))
    expected_order_count = int(float(status.get("order_count", 0) or 0))
    if (
        fill_stats["submitted_orders"] <= 0
        and expected_order_count <= 0
        and (not np.isfinite(max_drift_abs) or max_drift_abs <= MAX_PORTFOLIO_DRIFT)
    ):
        historical_fill_stats = _order_fill_stats(trades)
        if historical_fill_stats["submitted_orders"] > 0:
            fill_stats = historical_fill_stats
            fill_stats_scope = "historical_position_aligned"
    slippage_stats = _slippage_stats(current_signal_trades if not current_signal_trades.empty else trades)
    paper_benchmarks = _paper_benchmark_comparisons(equity_df, equity)
    survivorship_stress = _survivorship_stress_status() if strategy == "core_satellite_alpha" else {"data_available": True}
    execution_stress = _execution_stress_status() if strategy == "core_satellite_alpha" else {"data_available": True, "stress_pass": True}
    factor_decay = _factor_decay_status() if strategy == "core_satellite_alpha" else {"data_available": True, "real_capital_block": False}

    reasons: list[str] = []
    if not bool(status.get("paper_ready", False)):
        reasons.append("strategy paper_ready is false")
    if not bool(status.get("gates_all_pass", False)):
        reasons.append("strategy backtest gates_all_pass is false")
    if not bool(status.get("freshness_ok", True)):
        reasons.append("paper signal freshness check failed")
    if paper_days < MIN_PORTFOLIO_EQUITY_DAYS:
        reasons.append(f"paper_equity_days {paper_days} < {MIN_PORTFOLIO_EQUITY_DAYS}")
    if fill_stats["submitted_orders"] <= 0:
        reasons.append("no submitted paper orders found")
    elif fill_stats["fill_rate"] < MIN_PORTFOLIO_FILL_RATE:
        reasons.append(f"fill_rate {fill_stats['fill_rate']:.3f} < {MIN_PORTFOLIO_FILL_RATE:.3f}")
    if fill_stats["cancel_rate"] > MAX_PORTFOLIO_CANCEL_RATE:
        reasons.append(f"cancel_rate {fill_stats['cancel_rate']:.3f} > {MAX_PORTFOLIO_CANCEL_RATE:.3f}")
    if np.isfinite(max_drift_abs) and max_drift_abs > MAX_PORTFOLIO_DRIFT:
        reasons.append(f"max_drift_abs {max_drift_abs:.3f} > {MAX_PORTFOLIO_DRIFT:.3f}")
    if (
        slippage_stats.get("avg_slippage_bps") is not None
        and float(slippage_stats["avg_slippage_bps"]) > MAX_AVG_SLIPPAGE_BPS
    ):
        reasons.append(f"avg_slippage_bps {float(slippage_stats['avg_slippage_bps']):.3f} > {MAX_AVG_SLIPPAGE_BPS:.3f}")
    if max_dd < MAX_DRAWDOWN_PCT:
        reasons.append(f"paper_max_drawdown {max_dd:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if paper_days >= MIN_PORTFOLIO_EQUITY_DAYS and sharpe < MIN_SHARPE:
        reasons.append(f"paper_sharpe {sharpe:.3f} < {MIN_SHARPE:.3f}")
    if strategy == "core_satellite_alpha" and not bool(survivorship_stress.get("data_available", False)):
        reasons.append("core-satellite survivorship stress report is missing")
    elif strategy == "core_satellite_alpha" and not bool(survivorship_stress.get("stressed_paper_ready", True)):
        reasons.append("core-satellite failed-name survivorship stress failed strategy gates")
    if strategy == "core_satellite_alpha" and not bool(execution_stress.get("data_available", False)):
        reasons.append("core-satellite execution stress report is missing")
    elif strategy == "core_satellite_alpha" and not bool(execution_stress.get("stress_pass", True)):
        reasons.append("core-satellite execution stress failed strategy gates")
    if strategy == "core_satellite_alpha" and not bool(factor_decay.get("data_available", False)):
        reasons.append("factor decay monitor is missing")
    elif strategy == "core_satellite_alpha" and bool(factor_decay.get("real_capital_block", False)):
        reasons.append(f"factor decay blocks real capital: {factor_decay.get('reason', 'unknown reason')}")

    return {
        "status": "passed" if not reasons else "failed",
        "approved_for_real_capital": not reasons,
        "reason": "portfolio paper gauntlet passed" if not reasons else "; ".join(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": strategy,
        "portfolio_gauntlet": True,
        "paper_equity_days": paper_days,
        "paper_equity_rows": int(len(equity_df)),
        "paper_trades": int(len(trades)),
        "current_signal_paper_trades": int(len(current_signal_trades)),
        "fill_stats_scope": fill_stats_scope,
        **fill_stats,
        **slippage_stats,
        "total_return_pct": round(total_return_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "current_regime": status.get("current_regime"),
        "paper_ready": bool(status.get("paper_ready", False)),
        "gates_all_pass": bool(status.get("gates_all_pass", False)),
        "freshness_ok": bool(status.get("freshness_ok", True)),
        "current_gross_exposure": status.get("current_gross_exposure"),
        "target_gross_exposure": status.get("target_gross_exposure"),
        "max_drift_abs": None if not np.isfinite(max_drift_abs) else round(max_drift_abs, 4),
        "paper_benchmark_comparisons": paper_benchmarks,
        "survivorship_stress": survivorship_stress,
        "execution_stress": execution_stress,
        "factor_decay": factor_decay,
        "recent_equity": float(equity.iloc[-1]) if not equity.empty else None,
        "daily_return_observations": int(len(daily_returns)),
    }


def evaluate_paper() -> dict:
    trades = _read_csv(PAPER_TRADES)
    equity_df = _read_csv(PAPER_EQUITY)
    status = _read_json(PAPER_STATUS)

    if trades.empty and equity_df.empty:
        return {
            "status": "no_data",
            "approved_for_real_capital": False,
            "reason": "No paper_trades.csv or paper_equity.csv found.",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    pnl = _numeric_series(trades, ["net_pnl", "pnl", "realized_pnl"])
    trade_returns = _numeric_series(trades, ["return_pct", "net_return_pct", "return"])
    if trade_returns.empty and not pnl.empty:
        trade_returns = pnl / 10_000.0

    equity = _numeric_series(equity_df, ["equity", "portfolio_value", "account_value"])
    if equity.empty and not pnl.empty:
        equity = 10_000.0 + pnl.cumsum()

    daily_returns = equity.pct_change().dropna() if not equity.empty else trade_returns
    total_pnl = float(pnl.sum()) if not pnl.empty else float(equity.iloc[-1] - equity.iloc[0]) if len(equity) > 1 else 0.0
    total_return_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0) if len(equity) > 1 else 0.0
    win_rate = float((pnl > 0).mean()) if not pnl.empty else 0.0
    sharpe = _sharpe(daily_returns)
    max_dd = _max_drawdown(equity)

    if _is_portfolio_strategy(status):
        return _evaluate_portfolio_paper(
            trades=trades,
            equity_df=equity_df,
            status=status,
            equity=equity,
            daily_returns=daily_returns,
            total_return_pct=total_return_pct,
            sharpe=sharpe,
            max_dd=max_dd,
        )

    reasons: list[str] = []
    if len(trades) < MIN_TRADES:
        reasons.append(f"paper_trades {len(trades)} < {MIN_TRADES}")
    if sharpe < MIN_SHARPE:
        reasons.append(f"paper_sharpe {sharpe:.3f} < {MIN_SHARPE:.3f}")
    if max_dd < MAX_DRAWDOWN_PCT:
        reasons.append(f"paper_max_drawdown {max_dd:.2f}% < {MAX_DRAWDOWN_PCT:.2f}%")
    if win_rate < MIN_WIN_RATE:
        reasons.append(f"paper_win_rate {win_rate:.3f} < {MIN_WIN_RATE:.3f}")

    return {
        "status": "passed" if not reasons else "failed",
        "approved_for_real_capital": not reasons,
        "reason": "paper gauntlet passed" if not reasons else "; ".join(reasons),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_trades": int(len(trades)),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
    }


def main() -> None:
    result = evaluate_paper()
    LOGS.mkdir(parents=True, exist_ok=True)
    out_path = LOGS / f"paper_gauntlet_{datetime.now().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Paper gauntlet written -> {out_path}")
    print(f"Status: {result['status']} | approved_for_real_capital={result['approved_for_real_capital']}")
    print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
