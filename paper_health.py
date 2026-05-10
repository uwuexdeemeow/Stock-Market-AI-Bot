from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import paper_gauntlet
from settings import DATA_DIR, LOG_DIR, SECTOR_MAP, SIGNAL_DIR, WATCHLIST


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
PAPER_TRADES = SIGNALS / "paper_trades.csv"
PAPER_EQUITY = SIGNALS / "paper_equity.csv"
PAPER_STATUS = SIGNALS / "paper_daily_status.json"
PAPER_HEALTH = SIGNALS / "paper_health.json"
CORE_ORDER_PLAN = SIGNALS / "core_satellite_alpha_orders.csv"
MAX_SLIPPAGE_WARN_BPS = float(__import__("os").environ.get("PAPER_HEALTH_MAX_AVG_SLIPPAGE_BPS", "10"))
MAX_DRAWDOWN_WARN_PCT = float(__import__("os").environ.get("PAPER_HEALTH_MAX_DRAWDOWN_WARN_PCT", "-5"))
MAX_TICKER_WEIGHT_WARN = float(__import__("os").environ.get("PAPER_HEALTH_MAX_TICKER_WEIGHT", "0.35"))
MAX_SECTOR_WEIGHT_WARN = float(__import__("os").environ.get("PAPER_HEALTH_MAX_SECTOR_WEIGHT", "0.50"))
MAX_CORE_TICKER_WEIGHT_WARN = float(__import__("os").environ.get("PAPER_HEALTH_MAX_CORE_TICKER_WEIGHT", "0.65"))
STALE_OPEN_ORDER_MINUTES = float(__import__("os").environ.get("PAPER_HEALTH_STALE_OPEN_ORDER_MINUTES", "60"))
MAX_EQUITY_DAILY_MOVE_WARN_PCT = float(__import__("os").environ.get("PAPER_HEALTH_MAX_EQUITY_DAILY_MOVE_WARN_PCT", "25"))
CORE_TICKERS = {
    item.strip().upper()
    for item in __import__("os").environ.get("PAPER_HEALTH_CORE_TICKERS", "SPY,QQQ,TQQQ").split(",")
    if item.strip()
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _paper_days(equity: pd.DataFrame) -> int:
    if equity.empty:
        return 0
    if "date" in equity.columns:
        return int(pd.to_datetime(equity["date"], errors="coerce").dt.date.nunique())
    if "timestamp" in equity.columns:
        return int(pd.to_datetime(equity["timestamp"], errors="coerce").dt.date.nunique())
    return int(len(equity))


def _execution_slippage(row: pd.Series | dict) -> tuple[float, float]:
    try:
        reference = float(row.get("price", 0.0) or 0.0)
        fill_price = float(row.get("broker_dealt_avg_price", 0.0) or 0.0)
        quantity = float(row.get("broker_dealt_qty", row.get("delta_shares", 0.0)) or 0.0)
    except Exception:
        return np.nan, np.nan
    if reference <= 0 or fill_price <= 0 or quantity == 0:
        return np.nan, np.nan
    action = str(row.get("action", "")).upper().strip()
    if action == "BUY":
        per_share = fill_price - reference
    elif action == "SELL":
        per_share = reference - fill_price
    else:
        return np.nan, np.nan
    return float(per_share / reference * 10_000.0), float(per_share * abs(quantity))


def _slippage_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "filled_orders_with_slippage": 0,
            "avg_slippage_bps": None,
            "total_slippage_dollars": None,
        }
    filled = trades[trades.get("fill_status", pd.Series("", index=trades.index)).astype(str).str.lower().eq("filled")].copy()
    if filled.empty:
        return {
            "filled_orders_with_slippage": 0,
            "avg_slippage_bps": None,
            "total_slippage_dollars": None,
        }
    if "execution_slippage_bps" not in filled.columns or "execution_slippage_dollars" not in filled.columns:
        slips = filled.apply(_execution_slippage, axis=1, result_type="expand")
        filled["execution_slippage_bps"] = pd.to_numeric(slips[0], errors="coerce")
        filled["execution_slippage_dollars"] = pd.to_numeric(slips[1], errors="coerce")
    bps = pd.to_numeric(filled["execution_slippage_bps"], errors="coerce").dropna()
    dollars = pd.to_numeric(filled["execution_slippage_dollars"], errors="coerce").dropna()
    return {
        "filled_orders_with_slippage": int(len(bps)),
        "avg_slippage_bps": None if bps.empty else round(float(bps.mean()), 3),
        "median_slippage_bps": None if bps.empty else round(float(bps.median()), 3),
        "worst_slippage_bps": None if bps.empty else round(float(bps.max()), 3),
        "total_slippage_dollars": None if dollars.empty else round(float(dollars.sum()), 2),
    }


def _current_order_lifecycle(trades: pd.DataFrame, status: dict) -> list[dict]:
    current = paper_gauntlet._filter_current_signal_trades(trades, status)
    if current.empty:
        return []
    rows: list[dict] = []
    for _, row in current.iterrows():
        requested = pd.to_numeric(pd.Series([row.get("broker_requested_qty", row.get("broker_qty", row.get("delta_shares", 0)))]), errors="coerce").fillna(0.0).iloc[0]
        filled = pd.to_numeric(pd.Series([row.get("broker_dealt_qty", 0)]), errors="coerce").fillna(0.0).iloc[0]
        unfilled = row.get("unfilled_qty", None)
        if pd.isna(unfilled):
            unfilled = max(float(requested) - float(filled), 0.0)
        rows.append({
            "ticker": str(row.get("ticker", "")).upper(),
            "action": str(row.get("action", "")).upper(),
            "broker_order_id": str(row.get("broker_order_id", "")),
            "fill_status": str(row.get("fill_status", "unknown")).lower(),
            "broker_order_status": str(row.get("broker_order_status", "")),
            "requested_qty": round(float(requested), 6),
            "filled_qty": round(float(filled), 6),
            "unfilled_qty": round(float(unfilled), 6),
            "submitted_at": str(row.get("submitted_at", "")),
            "broker_updated_time": str(row.get("broker_updated_time", "")),
        })
    return rows


def _drift_breakdown(orders: pd.DataFrame, *, top_n: int = 8) -> list[dict]:
    if orders.empty or "drift_abs" not in orders.columns:
        return []
    frame = orders.copy()
    frame["drift_abs"] = pd.to_numeric(frame["drift_abs"], errors="coerce")
    frame = frame.dropna(subset=["drift_abs"]).sort_values("drift_abs", ascending=False)
    rows: list[dict] = []
    for _, row in frame.head(int(top_n)).iterrows():
        rows.append({
            "ticker": str(row.get("ticker", "")).upper(),
            "action": str(row.get("action", "")),
            "reason": str(row.get("reason", "")),
            "current_weight": round(float(row.get("current_weight", 0.0) or 0.0), 6),
            "target_weight": round(float(row.get("target_weight", 0.0) or 0.0), 6),
            "drift_abs": round(float(row.get("drift_abs", 0.0) or 0.0), 6),
            "order_value": round(float(row.get("order_value", 0.0) or 0.0), 2),
        })
    return rows


def _stale_open_order_alerts(lifecycle: list[dict], *, now: pd.Timestamp | None = None) -> list[dict]:
    if not lifecycle:
        return []
    now_ts = now or pd.Timestamp.now(tz=timezone.utc)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(timezone.utc)
    alerts: list[dict] = []
    for row in lifecycle:
        status = str(row.get("fill_status", "")).lower().strip()
        if status not in {"open", "partial", "partially_filled", "unknown"}:
            continue
        submitted_at = pd.to_datetime(row.get("submitted_at", ""), errors="coerce", utc=True)
        if pd.isna(submitted_at):
            continue
        age_minutes = float((now_ts - pd.Timestamp(submitted_at)).total_seconds() / 60.0)
        if age_minutes >= STALE_OPEN_ORDER_MINUTES:
            alerts.append({
                "ticker": str(row.get("ticker", "")).upper(),
                "action": str(row.get("action", "")).upper(),
                "fill_status": status,
                "age_minutes": round(age_minutes, 1),
                "unfilled_qty": row.get("unfilled_qty"),
                "broker_order_id": str(row.get("broker_order_id", "")),
                "threshold_minutes": round(float(STALE_OPEN_ORDER_MINUTES), 1),
            })
    return alerts


def _equity_risk(equity: pd.DataFrame) -> dict:
    if equity.empty or "equity" not in equity.columns:
        return {"paper_return_pct": None, "paper_drawdown_pct": None, "paper_drawdown_warning": False}
    values = pd.to_numeric(equity["equity"], errors="coerce").dropna()
    if len(values) < 2:
        return {"paper_return_pct": None, "paper_drawdown_pct": None, "paper_drawdown_warning": False}
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) * 100.0
    drawdown = float(values.iloc[-1] / values.cummax().iloc[-1] - 1.0) * 100.0
    return {
        "paper_return_pct": round(total_return, 3),
        "paper_drawdown_pct": round(drawdown, 3),
        "paper_drawdown_warning": bool(drawdown <= MAX_DRAWDOWN_WARN_PCT),
    }


def _equity_sanity_checks(equity: pd.DataFrame) -> dict:
    warnings: list[str] = []
    if equity.empty:
        return {"ok": False, "warnings": ["missing_equity_history"], "max_abs_daily_move_pct": None}
    if "equity" not in equity.columns:
        return {"ok": False, "warnings": ["missing_equity_column"], "max_abs_daily_move_pct": None}
    values = pd.to_numeric(equity["equity"], errors="coerce")
    if values.isna().any():
        warnings.append("non_numeric_equity_values")
    clean = values.dropna()
    if clean.empty:
        return {"ok": False, "warnings": warnings + ["no_valid_equity_values"], "max_abs_daily_move_pct": None}
    if (clean <= 0).any():
        warnings.append("nonpositive_equity_value")
    date_col = "date" if "date" in equity.columns else "timestamp" if "timestamp" in equity.columns else ""
    if not date_col:
        warnings.append("missing_equity_date_or_timestamp")
    else:
        dates = pd.to_datetime(equity[date_col], errors="coerce")
        if dates.isna().any():
            warnings.append("invalid_equity_dates")
        valid_dates = dates.dropna()
        if valid_dates.dt.date.duplicated().any():
            warnings.append("duplicate_equity_dates")
    returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    max_abs_move = None if returns.empty else round(float(returns.abs().max() * 100.0), 3)
    if max_abs_move is not None and max_abs_move > MAX_EQUITY_DAILY_MOVE_WARN_PCT:
        warnings.append(f"large_equity_move_{max_abs_move:.2f}%")
    return {
        "ok": not warnings,
        "warnings": warnings,
        "max_abs_daily_move_pct": max_abs_move,
        "threshold_pct": float(MAX_EQUITY_DAILY_MOVE_WARN_PCT),
    }


def _position_concentration(status: dict) -> dict:
    values = {
        str(k).upper(): float(v or 0.0)
        for k, v in dict(status.get("position_values", {}) or {}).items()
    }
    equity = float(status.get("account_equity", 0.0) or 0.0)
    if equity <= 0 or not values:
        return {
            "ticker_weights": {},
            "core_ticker_weights": {},
            "overlay_ticker_weights": {},
            "sector_weights": {},
            "overlay_sector_weights": {},
            "max_ticker_weight": 0.0,
            "max_ticker": "",
            "max_core_ticker_weight": 0.0,
            "max_core_ticker": "",
            "max_overlay_ticker_weight": 0.0,
            "max_overlay_ticker": "",
            "max_sector_weight": 0.0,
            "max_sector": "",
            "max_overlay_sector_weight": 0.0,
            "max_overlay_sector": "",
            "ticker_concentration_warning": False,
            "core_ticker_concentration_warning": False,
            "overlay_ticker_concentration_warning": False,
            "sector_concentration_warning": False,
            "overlay_sector_concentration_warning": False,
        }
    ticker_weights = {ticker: value / equity for ticker, value in sorted(values.items())}
    core_weights = {ticker: weight for ticker, weight in ticker_weights.items() if ticker in CORE_TICKERS}
    overlay_weights = {ticker: weight for ticker, weight in ticker_weights.items() if ticker not in CORE_TICKERS}
    sector_values: dict[str, float] = {}
    overlay_sector_values: dict[str, float] = {}
    for ticker, value in values.items():
        sector = SECTOR_MAP.get(ticker, "OTHER")
        sector_values[sector] = sector_values.get(sector, 0.0) + abs(value)
        if ticker not in CORE_TICKERS:
            overlay_sector_values[sector] = overlay_sector_values.get(sector, 0.0) + abs(value)
    sector_weights = {sector: value / equity for sector, value in sorted(sector_values.items())}
    overlay_sector_weights = {sector: value / equity for sector, value in sorted(overlay_sector_values.items())}
    max_ticker, max_ticker_weight = max(ticker_weights.items(), key=lambda item: abs(item[1]))
    max_sector, max_sector_weight = max(sector_weights.items(), key=lambda item: abs(item[1]))
    max_core_ticker, max_core_ticker_weight = ("", 0.0)
    if core_weights:
        max_core_ticker, max_core_ticker_weight = max(core_weights.items(), key=lambda item: abs(item[1]))
    max_overlay_ticker, max_overlay_ticker_weight = ("", 0.0)
    if overlay_weights:
        max_overlay_ticker, max_overlay_ticker_weight = max(overlay_weights.items(), key=lambda item: abs(item[1]))
    max_overlay_sector, max_overlay_sector_weight = ("", 0.0)
    if overlay_sector_weights:
        max_overlay_sector, max_overlay_sector_weight = max(overlay_sector_weights.items(), key=lambda item: abs(item[1]))
    core_warning = bool(abs(max_core_ticker_weight) > MAX_CORE_TICKER_WEIGHT_WARN)
    overlay_ticker_warning = bool(abs(max_overlay_ticker_weight) > MAX_TICKER_WEIGHT_WARN)
    overlay_sector_warning = bool(abs(max_overlay_sector_weight) > MAX_SECTOR_WEIGHT_WARN)
    return {
        "ticker_weights": {k: round(float(v), 6) for k, v in ticker_weights.items()},
        "core_ticker_weights": {k: round(float(v), 6) for k, v in core_weights.items()},
        "overlay_ticker_weights": {k: round(float(v), 6) for k, v in overlay_weights.items()},
        "sector_weights": {k: round(float(v), 6) for k, v in sector_weights.items()},
        "overlay_sector_weights": {k: round(float(v), 6) for k, v in overlay_sector_weights.items()},
        "max_ticker_weight": round(float(max_ticker_weight), 6),
        "max_ticker": max_ticker,
        "max_core_ticker_weight": round(float(max_core_ticker_weight), 6),
        "max_core_ticker": max_core_ticker,
        "max_overlay_ticker_weight": round(float(max_overlay_ticker_weight), 6),
        "max_overlay_ticker": max_overlay_ticker,
        "max_sector_weight": round(float(max_sector_weight), 6),
        "max_sector": max_sector,
        "max_overlay_sector_weight": round(float(max_overlay_sector_weight), 6),
        "max_overlay_sector": max_overlay_sector,
        "ticker_concentration_warning": overlay_ticker_warning,
        "core_ticker_concentration_warning": core_warning,
        "overlay_ticker_concentration_warning": overlay_ticker_warning,
        "sector_concentration_warning": overlay_sector_warning,
        "overlay_sector_concentration_warning": overlay_sector_warning,
    }


def _open_position_attribution(status: dict, trades: pd.DataFrame) -> dict:
    positions = {str(k).upper(): float(v or 0.0) for k, v in dict(status.get("positions", {}) or {}).items()}
    position_values = {str(k).upper(): float(v or 0.0) for k, v in dict(status.get("position_values", {}) or {}).items()}
    if not positions or trades.empty:
        return {
            "data_available": False,
            "reason": "need_current_positions_and_filled_trades",
        }
    filled_buys = trades[
        trades.get("fill_status", pd.Series("", index=trades.index)).astype(str).str.lower().eq("filled")
        & trades.get("action", pd.Series("", index=trades.index)).astype(str).str.upper().eq("BUY")
    ].copy()
    if filled_buys.empty:
        return {"data_available": False, "reason": "no_filled_buy_orders"}
    attribution: dict[str, dict] = {}
    core_pnl = 0.0
    overlay_pnl = 0.0
    skipped_unpriced: list[str] = []
    for ticker, qty in positions.items():
        if qty <= 0 or ticker not in position_values:
            continue
        position_value = float(position_values.get(ticker, 0.0) or 0.0)
        if position_value <= 0:
            skipped_unpriced.append(ticker)
            continue
        rows = filled_buys[filled_buys["ticker"].astype(str).str.upper().eq(ticker)]
        if rows.empty:
            continue
        avg_fill = pd.to_numeric(rows["broker_dealt_avg_price"], errors="coerce").dropna()
        if avg_fill.empty:
            continue
        fill_price = float(avg_fill.iloc[-1])
        current_price = float(position_value / qty) if qty else 0.0
        if current_price <= 0:
            skipped_unpriced.append(ticker)
            continue
        pnl = (current_price - fill_price) * qty
        sleeve = "core" if ticker in {"SPY", "QQQ"} else "overlay"
        if sleeve == "core":
            core_pnl += pnl
        else:
            overlay_pnl += pnl
        attribution[ticker] = {
            "sleeve": sleeve,
            "shares": round(qty, 6),
            "fill_price": round(fill_price, 4),
            "current_price": round(current_price, 4),
            "open_pnl": round(float(pnl), 2),
        }
    return {
        "data_available": bool(attribution),
        "core_open_pnl": round(float(core_pnl), 2),
        "overlay_open_pnl": round(float(overlay_pnl), 2),
        "total_open_position_pnl": round(float(core_pnl + overlay_pnl), 2),
        "by_ticker": attribution,
        "skipped_unpriced_tickers": sorted(set(skipped_unpriced)),
    }


def _factor_data_status() -> dict:
    latest_dates: list[pd.Timestamp] = []
    missing = []
    for ticker in WATCHLIST:
        path = Path(DATA_DIR) / f"{ticker.upper()}.parquet"
        if not path.exists():
            missing.append(ticker.upper())
            continue
        try:
            frame = pd.read_parquet(path, columns=["Close"])
            idx = pd.to_datetime(frame.index, errors="coerce")
            if len(idx) and not pd.isna(idx.max()):
                latest_dates.append(pd.Timestamp(idx.max()).normalize())
        except Exception:
            missing.append(ticker.upper())
    latest = max(latest_dates) if latest_dates else pd.NaT
    completed_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    age_days = None
    if not pd.isna(latest):
        age_days = int(len(pd.bdate_range(latest + pd.tseries.offsets.BDay(1), completed_day)))
    return {
        "latest_data_date": None if pd.isna(latest) else str(latest.date()),
        "age_trading_days": age_days,
        "missing_or_unreadable_count": len(missing),
        "missing_or_unreadable_sample": missing[:10],
    }


def _readiness_flags(health: dict) -> dict:
    slippage = health.get("slippage", {}) or {}
    concentration = health.get("concentration", {}) or {}
    return {
        "strategy_ready": bool(health.get("paper_ready") and health.get("gates_all_pass")),
        "signal_fresh": bool(health.get("freshness_ok")),
        "broker_synced": bool(health.get("submitted_orders", 0) > 0 and health.get("filled_orders", 0) > 0),
        "account_aligned": bool(
            health.get("max_drift_abs") is not None
            and float(health.get("max_drift_abs") or 0.0) <= 0.03
            and abs(float(health.get("current_gross_exposure") or 0.0) - float(health.get("target_gross_exposure") or 0.0)) <= 0.05
        ),
        "slippage_ok": bool(
            slippage.get("avg_slippage_bps") is None
            or float(slippage.get("avg_slippage_bps") or 0.0) <= MAX_SLIPPAGE_WARN_BPS
        ),
        "drawdown_ok": not bool(health.get("equity_risk", {}).get("paper_drawdown_warning", False)),
        "concentration_ok": not bool(
            concentration.get("overlay_ticker_concentration_warning", False)
            or concentration.get("overlay_sector_concentration_warning", False)
            or concentration.get("core_ticker_concentration_warning", False)
        ),
        "real_capital_allowed": bool(health.get("approved_for_real_capital", False)),
    }


def _go_live_scorecard(health: dict) -> dict:
    slippage = health.get("slippage", {}) or {}
    flags = health.get("readiness_flags", {}) or {}
    criteria = [
        {
            "name": "strategy_ready",
            "passed": bool(flags.get("strategy_ready", False)),
            "actual": bool(flags.get("strategy_ready", False)),
            "target": True,
        },
        {
            "name": "signal_fresh",
            "passed": bool(flags.get("signal_fresh", False)),
            "actual": bool(flags.get("signal_fresh", False)),
            "target": True,
        },
        {
            "name": "no_current_signal_open_orders",
            "passed": int(health.get("current_signal_open_orders", 0) or 0) == 0,
            "actual": int(health.get("current_signal_open_orders", 0) or 0),
            "target": 0,
        },
        {
            "name": "broker_synced",
            "passed": bool(flags.get("broker_synced", False)),
            "actual": bool(flags.get("broker_synced", False)),
            "target": True,
        },
        {
            "name": "paper_equity_days",
            "passed": int(health.get("paper_equity_days", 0) or 0) >= int(paper_gauntlet.MIN_PORTFOLIO_EQUITY_DAYS),
            "actual": int(health.get("paper_equity_days", 0) or 0),
            "target": int(paper_gauntlet.MIN_PORTFOLIO_EQUITY_DAYS),
        },
        {
            "name": "fill_rate",
            "passed": float(health.get("fill_rate", 0.0) or 0.0) >= float(paper_gauntlet.MIN_PORTFOLIO_FILL_RATE),
            "actual": round(float(health.get("fill_rate", 0.0) or 0.0), 4),
            "target": float(paper_gauntlet.MIN_PORTFOLIO_FILL_RATE),
        },
        {
            "name": "max_drift_abs",
            "passed": health.get("max_drift_abs") is not None
            and float(health.get("max_drift_abs") or 0.0) <= float(paper_gauntlet.MAX_PORTFOLIO_DRIFT),
            "actual": health.get("max_drift_abs"),
            "target": float(paper_gauntlet.MAX_PORTFOLIO_DRIFT),
        },
        {
            "name": "avg_slippage_bps",
            "passed": slippage.get("avg_slippage_bps") is None
            or float(slippage.get("avg_slippage_bps") or 0.0) <= float(paper_gauntlet.MAX_AVG_SLIPPAGE_BPS),
            "actual": slippage.get("avg_slippage_bps"),
            "target": float(paper_gauntlet.MAX_AVG_SLIPPAGE_BPS),
        },
        {
            "name": "drawdown_ok",
            "passed": bool(flags.get("drawdown_ok", True)),
            "actual": bool(flags.get("drawdown_ok", True)),
            "target": True,
        },
        {
            "name": "concentration_ok",
            "passed": bool(flags.get("concentration_ok", True)),
            "actual": bool(flags.get("concentration_ok", True)),
            "target": True,
        },
    ]
    passed = sum(1 for item in criteria if item["passed"])
    return {
        "passed": int(passed),
        "total": int(len(criteria)),
        "ready": bool(passed == len(criteria) and health.get("approved_for_real_capital", False)),
        "criteria": criteria,
    }


def build_health() -> dict:
    trades = _read_csv(PAPER_TRADES)
    equity = _read_csv(PAPER_EQUITY)
    orders = _read_csv(CORE_ORDER_PLAN)
    status = _read_json(PAPER_STATUS)
    gauntlet = paper_gauntlet.evaluate_paper()
    current_signal_trades = paper_gauntlet._filter_current_signal_trades(trades, status)

    fill_status = (
        trades.get("fill_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )
    current_signal_fill_status = (
        current_signal_trades.get("fill_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )
    submitted = trades
    if "submitted" in trades.columns:
        submitted = trades[trades["submitted"].astype(str).str.lower().isin(["true", "1", "yes"])]

    slippage = _slippage_summary(trades)
    order_lifecycle = _current_order_lifecycle(trades, status)
    stale_open_orders = _stale_open_order_alerts(order_lifecycle)
    health = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": status.get("strategy", gauntlet.get("strategy")),
        "paper_ready": bool(status.get("paper_ready", False)),
        "gates_all_pass": bool(status.get("gates_all_pass", False)),
        "freshness_ok": bool(status.get("freshness_ok", True)),
        "freshness_issues": status.get("freshness_issues", []),
        "paper_equity_days": _paper_days(equity),
        "paper_equity_rows": int(len(equity)),
        "paper_trades": int(len(trades)),
        "submitted_orders": int(len(submitted)),
        "fill_status_counts": fill_status,
        "current_signal_fill_status_counts": current_signal_fill_status,
        "current_signal_open_orders": int(
            current_signal_fill_status.get("open", 0)
            + current_signal_fill_status.get("unknown", 0)
            + current_signal_fill_status.get("partial", 0)
            + current_signal_fill_status.get("partially_filled", 0)
        ),
        "current_order_lifecycle": order_lifecycle,
        "stale_open_order_alerts": stale_open_orders,
        "drift_breakdown": _drift_breakdown(orders),
        "current_signal_paper_trades": int(gauntlet.get("current_signal_paper_trades", 0) or 0),
        "fill_stats_scope": gauntlet.get("fill_stats_scope"),
        "filled_orders": int(gauntlet.get("filled_orders", 0) or 0),
        "fill_rate": float(gauntlet.get("fill_rate", 0.0) or 0.0),
        "cancel_rate": float(gauntlet.get("cancel_rate", 0.0) or 0.0),
        "account_equity": status.get("account_equity"),
        "current_gross_exposure": status.get("current_gross_exposure"),
        "target_gross_exposure": status.get("target_gross_exposure"),
        "max_drift_abs": status.get("max_drift_abs"),
        "gauntlet_status": gauntlet.get("status"),
        "approved_for_real_capital": bool(gauntlet.get("approved_for_real_capital", False)),
        "gauntlet_reason": gauntlet.get("reason"),
        "slippage": slippage,
        "slippage_warning": bool(
            slippage.get("avg_slippage_bps") is not None
            and float(slippage.get("avg_slippage_bps") or 0.0) > MAX_SLIPPAGE_WARN_BPS
        ),
        "equity_risk": _equity_risk(equity),
        "equity_sanity": _equity_sanity_checks(equity),
        "concentration": _position_concentration(status),
        "open_position_attribution": _open_position_attribution(status, trades),
        "factor_data": _factor_data_status(),
    }
    health["readiness_flags"] = _readiness_flags(health)
    health["go_live_scorecard"] = _go_live_scorecard(health)
    return health


def print_health(health: dict) -> None:
    print("Paper Health")
    print("-" * 72)
    print(f"Strategy:              {health.get('strategy')}")
    print(f"Paper ready/gates:     {health.get('paper_ready')} / {health.get('gates_all_pass')}")
    print(f"Freshness:             {health.get('freshness_ok')} {health.get('freshness_issues') or ''}")
    print(f"Equity days:           {health.get('paper_equity_days')} rows={health.get('paper_equity_rows')}")
    print(
        f"Orders:                submitted={health.get('submitted_orders')} "
        f"filled={health.get('filled_orders')} fill_rate={health.get('fill_rate'):.3f} "
        f"scope={health.get('fill_stats_scope')}"
    )
    print(f"Exposure:              current={health.get('current_gross_exposure')} target={health.get('target_gross_exposure')} max_drift={health.get('max_drift_abs')}")
    slip = health.get("slippage", {})
    print(
        "Slippage:              "
        f"avg={slip.get('avg_slippage_bps')} bps "
        f"worst={slip.get('worst_slippage_bps')} bps "
        f"total=${slip.get('total_slippage_dollars')}"
    )
    risk = health.get("equity_risk", {})
    print(f"Equity risk:           return={risk.get('paper_return_pct')}% drawdown={risk.get('paper_drawdown_pct')}% warning={risk.get('paper_drawdown_warning')}")
    equity_sanity = health.get("equity_sanity", {})
    if equity_sanity and not equity_sanity.get("ok", True):
        print(f"Equity sanity:         warnings={equity_sanity.get('warnings')}")
    concentration = health.get("concentration", {})
    print(
        "Concentration:         "
        f"max_ticker={concentration.get('max_ticker')}:{concentration.get('max_ticker_weight')} "
        f"max_overlay={concentration.get('max_overlay_ticker')}:{concentration.get('max_overlay_ticker_weight')} "
        f"max_overlay_sector={concentration.get('max_overlay_sector')}:{concentration.get('max_overlay_sector_weight')}"
    )
    attribution = health.get("open_position_attribution", {})
    if attribution.get("data_available"):
        print(
            "Open P&L attribution:  "
            f"core=${attribution.get('core_open_pnl')} "
            f"overlay=${attribution.get('overlay_open_pnl')} "
            f"total=${attribution.get('total_open_position_pnl')}"
        )
    factor_data = health.get("factor_data", {})
    print(f"Factor data:           latest={factor_data.get('latest_data_date')} age_bdays={factor_data.get('age_trading_days')}")
    print(f"Readiness:             {health.get('readiness_flags')}")
    lifecycle = health.get("current_order_lifecycle", []) or []
    if lifecycle:
        print("Current orders:")
        view = pd.DataFrame(lifecycle)[[
            "ticker",
            "action",
            "fill_status",
            "requested_qty",
            "filled_qty",
            "unfilled_qty",
            "broker_order_id",
        ]]
        print(view.to_string(index=False))
    drift = health.get("drift_breakdown", []) or []
    if drift:
        print("Top drift:")
        view = pd.DataFrame(drift)[["ticker", "action", "current_weight", "target_weight", "drift_abs", "order_value"]]
        print(view.to_string(index=False))
    stale = health.get("stale_open_order_alerts", []) or []
    if stale:
        print("Stale open orders:")
        view = pd.DataFrame(stale)[["ticker", "action", "age_minutes", "unfilled_qty", "broker_order_id"]]
        print(view.to_string(index=False))
    scorecard = health.get("go_live_scorecard", {}) or {}
    if scorecard:
        print(f"Go-live scorecard:     {scorecard.get('passed')}/{scorecard.get('total')} ready={scorecard.get('ready')}")
    print(f"Gauntlet:              {health.get('gauntlet_status')} approved={health.get('approved_for_real_capital')}")
    print(f"Reason:                {health.get('gauntlet_reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Moomoo paper-trading health.")
    parser.add_argument("--json", action="store_true", help="Print the health summary as JSON.")
    args = parser.parse_args()

    health = build_health()
    PAPER_HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    LOGS.mkdir(parents=True, exist_ok=True)
    dated = LOGS / f"paper_health_{datetime.now().strftime('%Y%m%d')}.json"
    dated.write_text(json.dumps(health, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print_health(health)
        print(f"\nSaved -> {PAPER_HEALTH}")
        print(f"Snapshot -> {dated}")


if __name__ == "__main__":
    main()
