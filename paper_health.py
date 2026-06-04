from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import alpaca_paper_gauntlet
from factor_data_health import trading_day_age
from safe_io import atomic_write_json
from signal_freshness import latest_completed_us_trading_day, validate_signal_freshness
from settings import DATA_DIR, LOG_DIR, SECTOR_MAP, SIGNAL_DIR, WATCHLIST


SIGNALS = Path(SIGNAL_DIR)
LOGS = Path(LOG_DIR)
# ── Broker-specific file paths ─────────────────────────────────────────
# PLAIN ENGLISH: Alpaca writes these paper-trading files.  This health
# report turns those raw logs into a summary for the daily runner.
PAPER_TRADES = SIGNALS / "alpaca_paper_log.csv"
PAPER_EQUITY = SIGNALS / "alpaca_paper_equity.csv"
PAPER_STATUS = SIGNALS / "alpaca_daily_status.json"
PAPER_HEALTH = SIGNALS / "alpaca_paper_health.json"
CORE_SIGNAL = SIGNALS / "core_satellite_alpha_signal.csv"
CORE_ORDER_PLAN = SIGNALS / "core_satellite_alpha_orders.csv"


MAX_SLIPPAGE_WARN_BPS = float(__import__("os").environ.get("PAPER_HEALTH_MAX_AVG_SLIPPAGE_BPS", "10"))
MAX_DRAWDOWN_WARN_PCT = float(__import__("os").environ.get("PAPER_HEALTH_MAX_DRAWDOWN_WARN_PCT", "-5"))
MAX_SIGNAL_AGE_HOURS = float(__import__("os").environ.get("ALPACA_MAX_SIGNAL_AGE_HOURS", "24.0"))
MAX_FACTOR_AGE_TRADING_DAYS = int(__import__("os").environ.get("ALPACA_MAX_FACTOR_AGE_TRADING_DAYS", "5"))
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
DEFAULT_SIGNAL_TIMEZONE = __import__("os").environ.get(
    "PAPER_SIGNAL_TIMEZONE",
    __import__("os").environ.get("TZ", "Asia/Singapore"),
)

# ── Backtest-vs-live drift detection ──────────────────────────────────
# Path to the walkforward results that contain the backtested OOS metrics
# (Sharpe, CAGR, drawdown, alpha) for the approved config.
WALKFORWARD_RESULTS = SIGNALS / "core_satellite_nested_walkforward.json"
# How many trading days of live data before we start checking drift.
# Too few days and the comparison is meaningless noise.
DRIFT_MIN_LIVE_DAYS = int(
    __import__("os").environ.get("PAPER_HEALTH_DRIFT_MIN_LIVE_DAYS", "10")
)
# Multiplier thresholds: if live drawdown is worse than backtest worst by
# this factor, we flag a warning.  E.g. 1.5 means 50% worse than the
# worst OOS drawdown triggers a warning.
DRIFT_DD_WARN_MULT = float(
    __import__("os").environ.get("PAPER_HEALTH_DRIFT_DD_WARN_MULT", "1.5")
)
# If the annualised live Sharpe ratio falls below this fraction of the
# backtest mean OOS Sharpe, flag a warning.  E.g. 0.3 means the live
# Sharpe is less than 30% of the backtested average — something is off.
DRIFT_SHARPE_WARN_FRAC = float(
    __import__("os").environ.get("PAPER_HEALTH_DRIFT_SHARPE_WARN_FRAC", "0.3")
)
# If live annualised return underperforms the backtest mean CAGR by more
# than this many percentage points, flag a warning.
DRIFT_CAGR_UNDERPERFORM_PCT = float(
    __import__("os").environ.get("PAPER_HEALTH_DRIFT_CAGR_UNDERPERFORM_PCT", "20")
)


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


def _first_nonempty(row: pd.Series | dict, *keys: str, default: object = None) -> object:
    """Return the first usable value from a row that may use old or new column names."""
    for key in keys:
        value = row.get(key, None)
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
            return value
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert CSV text/numbers into a float without crashing on blanks."""
    try:
        return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])
    except Exception:
        return float(default)


def _to_bool(value: object) -> bool:
    """Convert common CSV/JSON truthy values into a real boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _signal_gate_status(signal: pd.DataFrame, status: dict) -> dict:
    """Read broker-submit readiness from the signal file, with status fallback.

    PLAIN ENGLISH: `alpaca_daily_status.json` is mainly an account snapshot, so
    it may not contain strategy gate fields.  The signal CSV is the same file
    the broker submit script reads before sending orders.
    """
    row: dict = {}
    if not signal.empty:
        row = signal.iloc[0].to_dict()

    paper_ready = _to_bool(_first_nonempty(row, "paper_ready", default=status.get("paper_ready")))
    gates_all_pass = _to_bool(_first_nonempty(row, "gates_all_pass", default=status.get("gates_all_pass")))
    medium_risk_review_pass = _to_bool(
        _first_nonempty(row, "medium_risk_review_pass", default=status.get("medium_risk_review_pass"))
    )
    block_reasons: list[str] = []
    if not paper_ready:
        block_reasons.append("paper_ready=false")
    if not gates_all_pass:
        block_reasons.append("gates_all_pass=false")
    if not medium_risk_review_pass:
        block_reasons.append("medium_risk_review_pass=false")
    return {
        "strategy": _first_nonempty(row, "strategy", "paper_signal_type", default=status.get("strategy")),
        "paper_ready": paper_ready,
        "gates_all_pass": gates_all_pass,
        "medium_risk_review_pass": medium_risk_review_pass,
        "strategy_ready": paper_ready and gates_all_pass and medium_risk_review_pass,
        "readiness_block_reasons": block_reasons,
        "readiness_source": str(CORE_SIGNAL.name if row else PAPER_STATUS.name),
    }


def _signal_freshness_status(signal: pd.DataFrame, status: dict, *, now: datetime | None = None) -> dict:
    """Validate signal age from the signal itself instead of trusting defaults."""
    if not signal.empty:
        row = signal.iloc[0]
        ok, issues = validate_signal_freshness(
            row,
            max_signal_age_hours=MAX_SIGNAL_AGE_HOURS,
            max_factor_age_trading_days=MAX_FACTOR_AGE_TRADING_DAYS,
            now=now,
        )
        return {
            "freshness_ok": bool(ok),
            "freshness_issues": issues,
            "freshness_source": CORE_SIGNAL.name,
        }

    fallback_issues = status.get("freshness_issues", ["missing_signal_file"])
    if not isinstance(fallback_issues, list):
        fallback_issues = [str(fallback_issues)]
    return {
        "freshness_ok": bool(status.get("freshness_ok", False)),
        "freshness_issues": fallback_issues,
        "freshness_source": PAPER_STATUS.name,
    }


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
        reference = _to_float(_first_nonempty(row, "price", "limit_price"))
        fill_price = _to_float(_first_nonempty(row, "broker_dealt_avg_price", "filled_avg_price"))
        quantity = _to_float(_first_nonempty(row, "broker_dealt_qty", "filled_qty", "delta_shares", "quantity"))
    except Exception:
        return np.nan, np.nan
    if reference <= 0 or fill_price <= 0 or quantity == 0:
        return np.nan, np.nan
    action = str(_first_nonempty(row, "action", "side", default="")).upper().strip()
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
    status = trades.get("fill_status", pd.Series("", index=trades.index)).astype(str).str.lower().str.strip()
    filled_qty = (
        pd.to_numeric(trades["broker_dealt_qty"], errors="coerce")
        if "broker_dealt_qty" in trades.columns
        else pd.to_numeric(trades.get("filled_qty", pd.Series(0.0, index=trades.index)), errors="coerce")
    ).fillna(0.0)
    filled = trades[status.isin({"filled", "partial", "partially_filled"}) | filled_qty.gt(0)].copy()
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


def _signal_timezone():
    """Return the timezone used by signal timestamps without an offset."""
    try:
        return ZoneInfo(str(DEFAULT_SIGNAL_TIMEZONE))
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_signal_predicted_at(value: object) -> pd.Timestamp:
    """Convert the signal time into UTC so order times can be compared."""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(_signal_timezone())
    return out.tz_convert("UTC")


def _filter_current_signal_trades(trades: pd.DataFrame, status: dict) -> pd.DataFrame:
    """Keep trade rows submitted after the current signal was generated."""
    if trades.empty or "submitted_at" not in trades.columns:
        return trades
    signal_ts = _parse_signal_predicted_at(status.get("signal_predicted_at", ""))
    if pd.isna(signal_ts):
        return trades
    submitted_ts = pd.to_datetime(trades["submitted_at"], errors="coerce", utc=True)
    return trades[submitted_ts >= signal_ts].copy()


def _submitted_trade_rows(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that actually reached Alpaca, excluding planned skips/errors.

    PLAIN ENGLISH: A skipped row is useful audit history, but it is not an
    Alpaca-submitted order.  Counting only accepted-looking rows keeps Telegram
    and dashboard order counts from saying "submitted" when the bot only
    planned and then safely skipped.
    """
    if trades.empty:
        return trades
    submitted_mask = pd.Series(True, index=trades.index)
    if "submitted" in trades.columns:
        submitted_mask = trades["submitted"].astype(str).str.lower().isin(["true", "1", "yes"])

    fill_status = (
        trades.get("fill_status", pd.Series("", index=trades.index))
        .astype(str)
        .str.lower()
        .str.strip()
    )
    order_id = trades.get("order_id", pd.Series("", index=trades.index)).astype(str)
    not_sent_mask = (
        fill_status.isin({"skipped", "submission_failed"})
        | order_id.str.startswith(("SKIPPED", "ERROR"), na=False)
    )
    return trades[submitted_mask & ~not_sent_mask].copy()


def _current_order_lifecycle(trades: pd.DataFrame, status: dict) -> list[dict]:
    current = _filter_current_signal_trades(trades, status)
    if current.empty:
        return []
    rows: list[dict] = []
    for _, row in current.iterrows():
        requested = _to_float(_first_nonempty(row, "broker_requested_qty", "broker_qty", "quantity", "delta_shares"))
        filled = _to_float(_first_nonempty(row, "broker_dealt_qty", "filled_qty"))
        unfilled = _first_nonempty(row, "unfilled_qty")
        if unfilled is None or pd.isna(unfilled):
            unfilled = max(float(requested) - float(filled), 0.0)
        action = str(_first_nonempty(row, "action", "side", default="")).upper()
        order_id = str(_first_nonempty(row, "broker_order_id", "order_id", default=""))
        broker_status = str(_first_nonempty(row, "broker_order_status", "order_status", "fill_status", default=""))
        rows.append({
            "ticker": str(row.get("ticker", "")).upper(),
            "action": action,
            "broker_order_id": order_id,
            "fill_status": str(row.get("fill_status", "unknown")).lower(),
            "broker_order_status": broker_status,
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


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _neutral_position_concentration() -> dict:
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


def _max_abs_item(weights: dict[str, float]) -> tuple[str, float]:
    if not weights:
        return "", 0.0
    return max(weights.items(), key=lambda item: abs(item[1]))


def _position_concentration(status: dict) -> dict:
    values: dict[str, float] = {}
    for ticker, raw_value in dict(status.get("position_values", {}) or {}).items():
        value = _finite_float(raw_value)
        if value is not None and value != 0.0:
            values[str(ticker).upper()] = value

    equity = _finite_float(status.get("account_equity", 0.0))
    if equity is None or equity <= 0 or not values:
        return _neutral_position_concentration()

    ticker_weights = {
        ticker: value / equity
        for ticker, value in sorted(values.items())
        if np.isfinite(value / equity)
    }
    if not ticker_weights:
        return _neutral_position_concentration()

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
    max_ticker, max_ticker_weight = _max_abs_item(ticker_weights)
    max_sector, max_sector_weight = _max_abs_item(sector_weights)
    max_core_ticker, max_core_ticker_weight = _max_abs_item(core_weights)
    max_overlay_ticker, max_overlay_ticker_weight = _max_abs_item(overlay_weights)
    max_overlay_sector, max_overlay_sector_weight = _max_abs_item(overlay_sector_weights)
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
    positions: dict[str, float] = {}
    for ticker, raw_qty in dict(status.get("positions", {}) or {}).items():
        qty = _finite_float(raw_qty)
        if qty is not None and qty > 0:
            positions[str(ticker).upper()] = qty
    position_values: dict[str, float] = {}
    for ticker, raw_value in dict(status.get("position_values", {}) or {}).items():
        value = _finite_float(raw_value)
        if value is not None:
            position_values[str(ticker).upper()] = value
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


def _factor_data_status(*, now: datetime | None = None) -> dict:
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
    completed_day = latest_completed_us_trading_day(now)
    age_days = None
    if not pd.isna(latest):
        age_days = trading_day_age(latest, now=completed_day)
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
        "strategy_ready": bool(
            health.get("paper_ready")
            and health.get("gates_all_pass")
            and health.get("medium_risk_review_pass")
        ),
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
        "backtest_drift_ok": not bool(
            health.get("backtest_vs_live_drift", {}).get("drift_detected", False)
        ),
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
            "passed": int(health.get("paper_equity_days", 0) or 0) >= int(alpaca_paper_gauntlet.MIN_TRADING_DAYS),
            "actual": int(health.get("paper_equity_days", 0) or 0),
            "target": int(alpaca_paper_gauntlet.MIN_TRADING_DAYS),
        },
        {
            "name": "fill_rate",
            "passed": float(health.get("fill_rate", 0.0) or 0.0) >= float(alpaca_paper_gauntlet.MIN_FILL_RATE),
            "actual": round(float(health.get("fill_rate", 0.0) or 0.0), 4),
            "target": float(alpaca_paper_gauntlet.MIN_FILL_RATE),
        },
        {
            "name": "max_drift_abs",
            "passed": health.get("max_drift_abs") is not None
            and float(health.get("max_drift_abs") or 0.0) <= float(alpaca_paper_gauntlet.MAX_DRIFT),
            "actual": health.get("max_drift_abs"),
            "target": float(alpaca_paper_gauntlet.MAX_DRIFT),
        },
        {
            "name": "avg_slippage_bps",
            "passed": slippage.get("avg_slippage_bps") is None
            or float(slippage.get("avg_slippage_bps") or 0.0) <= float(MAX_SLIPPAGE_WARN_BPS),
            "actual": slippage.get("avg_slippage_bps"),
            "target": float(MAX_SLIPPAGE_WARN_BPS),
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
        {
            "name": "backtest_drift_ok",
            "passed": bool(flags.get("backtest_drift_ok", True)),
            "actual": bool(flags.get("backtest_drift_ok", True)),
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


def _backtest_vs_live_drift(equity: pd.DataFrame) -> dict:
    """Compare live paper-trading performance against backtested expectations.

    Reads the walkforward results (OOS Sharpe, CAGR, drawdown) for the
    approved config and compares them to the actual paper equity curve.
    Returns a dict with the comparison metrics and any warnings.

    This is the "implementation shortfall" detector — it catches cases where
    the live strategy is silently underperforming what the backtest predicted,
    which could indicate data issues, execution problems, or regime change
    the model hasn't adapted to yet.
    """
    result: dict = {
        "data_available": False,
        "reason": "",
        "warnings": [],
    }

    # ── Load backtest expectations from walkforward results ──────────
    if not WALKFORWARD_RESULTS.exists():
        result["reason"] = "walkforward_results_not_found"
        return result
    try:
        wf = json.loads(WALKFORWARD_RESULTS.read_text(encoding="utf-8"))
    except Exception:
        result["reason"] = "walkforward_results_unreadable"
        return result

    # Pull the backtested OOS metrics — these are our "expected" numbers
    bt_sharpe = wf.get("mean_oos_sharpe")
    bt_cagr = wf.get("mean_oos_cagr_pct")
    bt_worst_dd = wf.get("worst_oos_max_drawdown_pct")
    bt_mean_dd = wf.get("mean_oos_max_drawdown_pct")
    if bt_sharpe is None or bt_cagr is None:
        result["reason"] = "walkforward_missing_oos_metrics"
        return result

    result["backtest_mean_oos_sharpe"] = float(bt_sharpe)
    result["backtest_mean_oos_cagr_pct"] = float(bt_cagr)
    result["backtest_worst_oos_drawdown_pct"] = float(bt_worst_dd) if bt_worst_dd is not None else None
    result["backtest_mean_oos_drawdown_pct"] = float(bt_mean_dd) if bt_mean_dd is not None else None

    # ── Compute live performance from equity curve ───────────────────
    if equity.empty or "equity" not in equity.columns:
        result["reason"] = "no_equity_data"
        return result

    eq = equity.copy()
    # Parse dates — the CSV has a 'date' or 'timestamp' column
    date_col = "date" if "date" in eq.columns else "timestamp" if "timestamp" in eq.columns else ""
    if not date_col:
        result["reason"] = "no_date_column_in_equity"
        return result
    eq["_dt"] = pd.to_datetime(eq[date_col], errors="coerce")
    eq = eq.dropna(subset=["_dt"]).sort_values("_dt")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"])

    live_days = len(eq)
    if live_days < DRIFT_MIN_LIVE_DAYS:
        result["reason"] = f"too_few_live_days ({live_days} < {DRIFT_MIN_LIVE_DAYS})"
        return result

    result["data_available"] = True
    result["reason"] = "ok"
    result["live_equity_days"] = live_days

    # Total return and annualised return
    start_equity = float(eq["equity"].iloc[0])
    end_equity = float(eq["equity"].iloc[-1])
    if start_equity <= 0:
        result["reason"] = "invalid_start_equity"
        result["data_available"] = False
        return result

    total_return = (end_equity / start_equity) - 1.0
    # Approximate trading days per year = 252
    years = live_days / 252.0
    if years > 0:
        live_cagr = ((1.0 + total_return) ** (1.0 / years) - 1.0) * 100.0
    else:
        live_cagr = 0.0

    # Live Sharpe (annualised) from daily returns
    daily_returns = eq["equity"].pct_change().dropna()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily_returns) >= 2:
        live_sharpe = float(
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        ) if daily_returns.std() > 0 else 0.0
    else:
        live_sharpe = 0.0

    # Live max drawdown
    cummax = eq["equity"].cummax()
    drawdowns = (eq["equity"] / cummax - 1.0) * 100.0
    live_max_dd = float(drawdowns.min())

    result["live_total_return_pct"] = round(total_return * 100.0, 3)
    result["live_annualised_cagr_pct"] = round(live_cagr, 3)
    result["live_annualised_sharpe"] = round(live_sharpe, 3)
    result["live_max_drawdown_pct"] = round(live_max_dd, 3)

    # ── Compare and flag warnings ────────────────────────────────────
    warnings: list[str] = []

    # 1) Drawdown check: is live drawdown much worse than backtest worst?
    if bt_worst_dd is not None and bt_worst_dd < 0:
        dd_ratio = live_max_dd / bt_worst_dd  # both negative, ratio > 1 = worse
        result["drawdown_ratio_vs_worst_oos"] = round(dd_ratio, 3)
        if dd_ratio > DRIFT_DD_WARN_MULT:
            warnings.append(
                f"live_drawdown_exceeds_backtest: {live_max_dd:.1f}% vs "
                f"worst OOS {bt_worst_dd:.1f}% (ratio {dd_ratio:.2f}x, "
                f"threshold {DRIFT_DD_WARN_MULT:.1f}x)"
            )

    # 2) Sharpe check: is live Sharpe way below backtest average?
    if bt_sharpe > 0:
        sharpe_frac = live_sharpe / bt_sharpe
        result["sharpe_fraction_of_backtest"] = round(sharpe_frac, 3)
        if sharpe_frac < DRIFT_SHARPE_WARN_FRAC:
            warnings.append(
                f"live_sharpe_below_backtest: {live_sharpe:.2f} vs "
                f"backtest mean {bt_sharpe:.2f} "
                f"(ratio {sharpe_frac:.2f}, threshold {DRIFT_SHARPE_WARN_FRAC:.1f})"
            )

    # 3) CAGR check: is live annualised return way below backtest?
    cagr_gap = bt_cagr - live_cagr
    result["cagr_gap_pct"] = round(cagr_gap, 3)
    if cagr_gap > DRIFT_CAGR_UNDERPERFORM_PCT:
        warnings.append(
            f"live_cagr_underperforming: {live_cagr:.1f}% vs "
            f"backtest mean {bt_cagr:.1f}% "
            f"(gap {cagr_gap:.1f}pp, threshold {DRIFT_CAGR_UNDERPERFORM_PCT:.0f}pp)"
        )

    result["warnings"] = warnings
    result["drift_detected"] = bool(warnings)
    return result


def build_health() -> dict:
    trades = _read_csv(PAPER_TRADES)
    equity = _read_csv(PAPER_EQUITY)
    orders = _read_csv(CORE_ORDER_PLAN)
    signal = _read_csv(CORE_SIGNAL)
    status = _read_json(PAPER_STATUS)
    gate_status = _signal_gate_status(signal, status)
    freshness_status = _signal_freshness_status(signal, status)
    gauntlet = alpaca_paper_gauntlet.evaluate_alpaca_paper()
    current_signal_trades = _filter_current_signal_trades(trades, status)

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
    submitted = _submitted_trade_rows(trades)

    slippage = _slippage_summary(trades)
    order_lifecycle = _current_order_lifecycle(trades, status)
    stale_open_orders = _stale_open_order_alerts(order_lifecycle)
    health = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": gate_status.get("strategy") or status.get("strategy", gauntlet.get("strategy")),
        "paper_ready": bool(gate_status.get("paper_ready", False)),
        "gates_all_pass": bool(gate_status.get("gates_all_pass", False)),
        "medium_risk_review_pass": bool(gate_status.get("medium_risk_review_pass", False)),
        "strategy_ready": bool(gate_status.get("strategy_ready", False)),
        "readiness_block_reasons": gate_status.get("readiness_block_reasons", []),
        "readiness_source": gate_status.get("readiness_source"),
        "freshness_ok": bool(freshness_status.get("freshness_ok", False)),
        "freshness_issues": freshness_status.get("freshness_issues", []),
        "freshness_source": freshness_status.get("freshness_source"),
        "paper_equity_days": _paper_days(equity),
        "paper_equity_rows": int(len(equity)),
        "paper_trades": int(len(trades)),
        "submitted_orders": int(len(submitted)),
        "skipped_orders": int(fill_status.get("skipped", 0)),
        "submission_failed_orders": int(fill_status.get("submission_failed", 0)),
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
        "backtest_vs_live_drift": _backtest_vs_live_drift(equity),
    }
    health["readiness_flags"] = _readiness_flags(health)
    health["go_live_scorecard"] = _go_live_scorecard(health)
    return health


def print_health(health: dict) -> None:
    print("Paper Health")
    print("-" * 72)
    print(f"Strategy:              {health.get('strategy')}")
    print(
        "Submit gates:          "
        f"paper={health.get('paper_ready')} "
        f"gates={health.get('gates_all_pass')} "
        f"medium={health.get('medium_risk_review_pass')} "
        f"source={health.get('readiness_source')}"
    )
    if health.get("readiness_block_reasons"):
        print(f"Blocked by:            {', '.join(health.get('readiness_block_reasons') or [])}")
    print(
        f"Freshness:             {health.get('freshness_ok')} "
        f"source={health.get('freshness_source')} {health.get('freshness_issues') or ''}"
    )
    print(f"Equity days:           {health.get('paper_equity_days')} rows={health.get('paper_equity_rows')}")
    print(
        f"Orders:                submitted={health.get('submitted_orders')} "
        f"skipped={health.get('skipped_orders', 0)} "
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
    # ── Backtest-vs-live drift ────────────────────────────────────────
    bt_drift = health.get("backtest_vs_live_drift", {}) or {}
    if bt_drift.get("data_available"):
        print(
            f"Backtest drift:        "
            f"live_sharpe={bt_drift.get('live_annualised_sharpe')} "
            f"(bt={bt_drift.get('backtest_mean_oos_sharpe')}) "
            f"live_cagr={bt_drift.get('live_annualised_cagr_pct')}% "
            f"(bt={bt_drift.get('backtest_mean_oos_cagr_pct')}%) "
            f"live_dd={bt_drift.get('live_max_drawdown_pct')}% "
            f"(bt_worst={bt_drift.get('backtest_worst_oos_drawdown_pct')}%)"
        )
        drift_warnings = bt_drift.get("warnings", [])
        if drift_warnings:
            print(f"  ⚠ DRIFT WARNINGS:")
            for w in drift_warnings:
                print(f"    - {w}")
            # Send alert via all channels (Telegram, email, macOS)
            from notifications import send_alert as _notify
            _notify(
                "Backtest-vs-live drift detected:\n" + "\n".join(f"• {w}" for w in drift_warnings),
                title="Drift Alert",
                priority="warning",
            )
    elif bt_drift.get("reason"):
        print(f"Backtest drift:        {bt_drift.get('reason')}")
    print(f"Gauntlet:              {health.get('gauntlet_status')} approved={health.get('approved_for_real_capital')}")
    print(f"Reason:                {health.get('gauntlet_reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Alpaca paper-trading health.")
    parser.add_argument("--json", action="store_true", help="Print the health summary as JSON.")
    args = parser.parse_args()

    health = build_health()
    health["broker"] = "alpaca"
    atomic_write_json(health, PAPER_HEALTH)
    LOGS.mkdir(parents=True, exist_ok=True)
    dated = LOGS / f"alpaca_paper_health_{datetime.now().strftime('%Y%m%d')}.json"
    atomic_write_json(health, dated)
    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print_health(health)
        print(f"\nSaved -> {PAPER_HEALTH}")
        print(f"Snapshot -> {dated}")


if __name__ == "__main__":
    main()
