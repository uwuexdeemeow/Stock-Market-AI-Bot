from __future__ import annotations

"""
moomoo_paper_trading_complete.py
================================
Safer paper-trading wrapper.

Fixes:
- approved live universe enforced by default
- live SHORT orders disabled by default
- choose which trades to submit with include/exclude filters
- optional interactive confirmation per trade
- computes vol-aware recommended position size and borrow cost estimate
  so you can enter correct share counts directly into moomoo
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from risk_sizing import compute_position_size
from settings import (
    BORROW_COST_ANNUAL_DEFAULT,
    BORROW_COSTS,
    DATA_DIR,
    PAPER_MODE_STRATEGY,
    POSITION_SIZING_MODE,
    SINGLE_NAME_PAPER_TRADING_ENABLED,
)
from trade_rules import load_trade_rule, passes_trade_rule

SIGNAL_DIR = Path("signals")
SIGNALS_FILE = SIGNAL_DIR / "signals.csv"
APPROVED_TICKERS_FILE = SIGNAL_DIR / "approved_live_tickers.csv"
PORTFOLIO_METRICS_FILE = SIGNAL_DIR / "walkforward_long_only_147tickers_metrics.json"
TIMING_SIGNAL_FILE = SIGNAL_DIR / "spy_timing_signal.csv"
ETF_ROTATION_SIGNAL_FILE = SIGNAL_DIR / "etf_rotation_signal.csv"
CORE_SATELLITE_SIGNAL_FILE = SIGNAL_DIR / "core_satellite_alpha_signal.csv"
PAPER_STATE_FILE = SIGNAL_DIR / "paper_state.json"
PAPER_STATUS_FILE = SIGNAL_DIR / "paper_daily_status.json"
PAPER_EQUITY_FILE = SIGNAL_DIR / "paper_equity.csv"
PAPER_RECONCILIATION_FILE = SIGNAL_DIR / "paper_reconciliation.csv"

# Portfolio equity used for position sizing when no equity file exists.
# Update this to match your actual moomoo paper account balance.
DEFAULT_ACCOUNT_EQUITY = 10_000.0
DEFAULT_MIN_TRADE_VALUE = 25.0
DEFAULT_ETF_DRIFT_THRESHOLD = 0.03
DEFAULT_OVERLAY_DRIFT_THRESHOLD = 0.01
DEFAULT_LIMIT_OFFSET_BPS = 10.0
DEFAULT_MAX_SIGNAL_AGE_HOURS = 24.0
DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS = 5
DEFAULT_SYNC_AFTER_SUBMIT_LOOKBACK_DAYS = 3


def _normalise_ticker(ticker: str) -> str:
    return str(ticker).upper().strip()


def _moomoo_code(ticker: str) -> str:
    ticker = _normalise_ticker(ticker)
    if "." in ticker and ticker.split(".", 1)[0] in {"US", "HK", "SG", "AU", "CA", "JP"}:
        return ticker
    market = os.environ.get("MOOMOO_MARKET", "US").strip().upper() or "US"
    return f"{market}.{ticker}"


def _plain_ticker_from_code(code: str) -> str:
    code = str(code).upper().strip()
    return code.split(".", 1)[1] if "." in code else code


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker in sorted({_normalise_ticker(t) for t in tickers if str(t).strip()}):
        prices[ticker] = _fetch_price(ticker)
    return prices


def load_paper_state() -> dict:
    if not PAPER_STATE_FILE.exists():
        return {}
    try:
        with PAPER_STATE_FILE.open() as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def save_paper_state(state: dict) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PAPER_STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _parse_date(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert(None).normalize()


def _trading_days_since(value: object, today: pd.Timestamp | None = None) -> int | None:
    start = _parse_date(value)
    if start is None:
        return None
    end = (today or pd.Timestamp.now()).normalize()
    if end <= start:
        return 0
    return int(len(pd.bdate_range(start + pd.tseries.offsets.BDay(1), end)))


def us_regular_market_is_open(now: datetime | None = None) -> tuple[bool, str]:
    eastern = ZoneInfo("America/New_York")
    ts = (now or datetime.now(timezone.utc)).astimezone(eastern)
    if ts.weekday() >= 5:
        return False, f"US market closed: weekend ({ts.strftime('%Y-%m-%d %H:%M %Z')})"
    open_ts = ts.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ts = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    if ts < open_ts or ts >= close_ts:
        return False, f"US market closed: outside 09:30-16:00 ET ({ts.strftime('%Y-%m-%d %H:%M %Z')})"
    return True, f"US market open ({ts.strftime('%Y-%m-%d %H:%M %Z')})"


def validate_signal_freshness(
    signal: pd.Series,
    *,
    max_signal_age_hours: float,
    max_factor_age_trading_days: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    predicted_at = signal.get("predicted_at", "")
    predicted_ts = pd.to_datetime(predicted_at, errors="coerce")
    if pd.isna(predicted_ts):
        issues.append("missing_predicted_at")
    else:
        age_hours = (pd.Timestamp.now() - pd.Timestamp(predicted_ts)).total_seconds() / 3600.0
        if age_hours > float(max_signal_age_hours):
            issues.append(f"signal_age_{age_hours:.1f}h_gt_{float(max_signal_age_hours):.1f}h")

    latest_factor_date = signal.get("latest_factor_date", "")
    factor_ts = pd.to_datetime(latest_factor_date, errors="coerce")
    if pd.isna(factor_ts):
        issues.append("missing_latest_factor_date")
    else:
        age_days = len(pd.bdate_range(pd.Timestamp(factor_ts) + pd.tseries.offsets.BDay(1), pd.Timestamp.now().normalize()))
        if age_days > int(max_factor_age_trading_days):
            issues.append(f"factor_age_{age_days}_bdays_gt_{int(max_factor_age_trading_days)}")
    return not issues, issues


def single_name_paper_trading_allowed() -> tuple[bool, str]:
    """Gate single-name paper routing separately from index timing."""
    if PAPER_MODE_STRATEGY != "single_name" or not SINGLE_NAME_PAPER_TRADING_ENABLED:
        return False, f"single-name paper trading disabled; PAPER_MODE_STRATEGY={PAPER_MODE_STRATEGY}"
    if not PORTFOLIO_METRICS_FILE.exists():
        return False, f"missing raw stock gate metrics: {PORTFOLIO_METRICS_FILE}"
    try:
        with PORTFOLIO_METRICS_FILE.open() as f:
            metrics = json.load(f)
    except Exception as exc:
        return False, f"could not read raw stock gate metrics: {exc}"
    if not bool(metrics.get("raw_stock_gates_pass", False)):
        raw_ret = metrics.get("raw_strategy_total_return_pct")
        raw_sharpe = metrics.get("raw_strategy_sharpe_ratio_daily")
        return False, f"raw stock gates failed (return={raw_ret}, sharpe={raw_sharpe})"
    return True, "single-name paper trading enabled and raw stock gates passed"


def _print_signal_file(path: Path, cols: list[str], label: str) -> bool:
    if not path.exists():
        print(f"Missing {label} signal file: {path}")
        return False
    try:
        signal = pd.read_csv(path)
    except Exception:
        print(f"{label} signal exists but could not be read: {path}")
        return False
    show_cols = [c for c in cols if c in signal.columns]
    if show_cols:
        print(f"\nCurrent {label} allocation:")
        print(signal[show_cols].to_string(index=False))
    return True


def print_paper_allocation_hint() -> None:
    if PAPER_MODE_STRATEGY == "core_satellite_alpha":
        ok = _print_signal_file(
            CORE_SATELLITE_SIGNAL_FILE,
            [
                "paper_ready",
                "core_preset",
                "current_regime",
                "target_spy_weight",
                "target_qqq_weight",
                "target_cash_weight",
                "gross_exposure",
                "core_gross",
                "overlay_gross",
                "overlay_tickers",
                "gates_all_pass",
                "reason",
            ],
            "core-satellite alpha",
        )
        if not ok:
            print("Run `python3 core_satellite_alpha.py` to create the gated core-satellite allocation.")
        return
    if PAPER_MODE_STRATEGY == "etf_rotation":
        ok = _print_signal_file(
            ETF_ROTATION_SIGNAL_FILE,
            [
                "paper_ready",
                "regime",
                "target_spy_weight",
                "target_qqq_weight",
                "target_bil_weight",
                "target_ief_weight",
                "target_gld_weight",
                "target_cash_weight",
                "gross_exposure",
                "gates_all_pass",
            ],
            "ETF rotation",
        )
        if not ok:
            print("Run `python3 predict.py --ticker AAPL --etf-rotation` to create the ETF rotation allocation.")
        return
    ok = _print_signal_file(
        TIMING_SIGNAL_FILE,
        [
            "market_timing_signal",
            "target_spy_weight",
            "target_qqq_weight",
            "target_cash_weight",
            "bull_fraction",
            "threshold",
        ],
        "SPY/QQQ timing",
    )
    if not ok:
        print(f"Run `python3 predict.py --spy-timing` to create {TIMING_SIGNAL_FILE}.")


def load_approved_tickers() -> set[str]:
    if not APPROVED_TICKERS_FILE.exists():
        return set()
    try:
        df = pd.read_csv(APPROVED_TICKERS_FILE)
        if "ticker" not in df.columns:
            return set()
        return {str(t).upper().strip() for t in df["ticker"].dropna().tolist()}
    except Exception:
        return set()


def _fetch_vol(ticker: str) -> float:
    """3-month realized annual vol for ticker. Returns 0.20 on failure."""
    try:
        hist = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        closes = hist["Close"].dropna()
        if len(closes) >= 20:
            vol = float(closes.pct_change().tail(63).std() * np.sqrt(252))
            return max(0.05, min(1.0, vol))
    except Exception:
        pass
    return 0.20


def _fetch_price(ticker: str) -> float:
    """Latest close price. Returns 0.0 on failure."""
    try:
        hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                close = hist[("Close", ticker)] if ("Close", ticker) in hist.columns else hist["Close"].iloc[:, 0]
            else:
                close = hist["Close"]
            close = close.dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        pass
    try:
        local_path = Path(DATA_DIR) / f"{_normalise_ticker(ticker)}.parquet"
        if local_path.exists():
            df = pd.read_parquet(local_path, columns=["Close"])
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        pass
    return 0.0


def load_core_satellite_signal() -> pd.Series:
    if not CORE_SATELLITE_SIGNAL_FILE.exists():
        raise FileNotFoundError(f"Missing core-satellite signal file: {CORE_SATELLITE_SIGNAL_FILE}")
    df = pd.read_csv(CORE_SATELLITE_SIGNAL_FILE)
    if df.empty:
        raise RuntimeError(f"Core-satellite signal file is empty: {CORE_SATELLITE_SIGNAL_FILE}")
    row = df.iloc[0]
    if not bool(row.get("paper_ready", False)) or not bool(row.get("gates_all_pass", False)):
        reason = row.get("reason", "paper_ready/gates_all_pass false")
        raise RuntimeError(f"Core-satellite signal is not approved for paper trading: {reason}")
    return row


def core_satellite_target_weights(row: pd.Series) -> dict[str, float]:
    weights: dict[str, float] = {}
    spy_weight = float(row.get("target_spy_weight", 0.0) or 0.0)
    qqq_weight = float(row.get("target_qqq_weight", 0.0) or 0.0)
    if abs(spy_weight) > 1e-9:
        weights["SPY"] = weights.get("SPY", 0.0) + spy_weight
    if abs(qqq_weight) > 1e-9:
        weights["QQQ"] = weights.get("QQQ", 0.0) + qqq_weight

    overlay_raw = row.get("overlay_weights_json", "{}")
    try:
        overlay = json.loads(str(overlay_raw)) if pd.notna(overlay_raw) else {}
    except Exception as exc:
        raise RuntimeError(f"Could not parse overlay_weights_json: {exc}") from exc
    for ticker, weight in overlay.items():
        ticker = _normalise_ticker(ticker)
        weights[ticker] = weights.get(ticker, 0.0) + float(weight)
    return {ticker: weight for ticker, weight in weights.items() if abs(weight) > 1e-9}


def connect_moomoo_trade_context():
    try:
        from moomoo import OpenSecTradeContext
    except Exception as exc:
        raise RuntimeError("Moomoo SDK is not installed. Install with `pip install moomoo-api`.") from exc

    host = os.environ.get("MOOMOO_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("MOOMOO_PORT", "11111").strip() or "11111")
    market = os.environ.get("MOOMOO_MARKET", "US").strip().upper() or "US"
    firm = os.environ.get("MOOMOO_SECURITY_FIRM", "FUTUINC").strip().upper() or "FUTUINC"
    return OpenSecTradeContext(filter_trdmarket=market, host=host, port=port, security_firm=firm)


def _moomoo_constant(name: str, attr: str, fallback: str) -> str:
    try:
        module = __import__("moomoo", fromlist=[name])
        cls = getattr(module, name)
        return getattr(cls, attr)
    except Exception:
        return fallback


def fetch_moomoo_equity(ctx, fallback_equity: float | None = None) -> float:
    trd_env = _moomoo_constant("TrdEnv", "SIMULATE", "SIMULATE")
    ret, data = ctx.accinfo_query(trd_env=trd_env, currency="USD")
    if ret != 0:
        if fallback_equity and fallback_equity > 0:
            return float(fallback_equity)
        raise RuntimeError(f"Moomoo accinfo_query failed: {data}")
    if data is None or len(data) == 0:
        if fallback_equity and fallback_equity > 0:
            return float(fallback_equity)
        raise RuntimeError("Moomoo accinfo_query returned no account rows")
    row = data.iloc[0]
    for col in ("total_assets", "net_assets", "market_val", "cash"):
        if col in row and pd.notna(row[col]):
            value = float(row[col])
            if value > 0:
                return value
    if fallback_equity and fallback_equity > 0:
        return float(fallback_equity)
    raise RuntimeError(f"Could not infer account equity from Moomoo columns: {list(data.columns)}")


def fetch_moomoo_positions(ctx) -> dict[str, int]:
    trd_env = _moomoo_constant("TrdEnv", "SIMULATE", "SIMULATE")
    ret, data = ctx.position_list_query(trd_env=trd_env, refresh_cache=True)
    if ret != 0:
        raise RuntimeError(f"Moomoo position_list_query failed: {data}")
    if data is None or len(data) == 0:
        return {}
    code_col = "code" if "code" in data.columns else "stock_code" if "stock_code" in data.columns else None
    qty_col = "qty" if "qty" in data.columns else "quantity" if "quantity" in data.columns else None
    if code_col is None or qty_col is None:
        raise RuntimeError(f"Could not parse Moomoo positions columns: {list(data.columns)}")
    positions: dict[str, int] = {}
    for _, row in data.iterrows():
        ticker = _plain_ticker_from_code(row[code_col])
        qty = int(float(row[qty_col] or 0))
        if qty != 0:
            positions[ticker] = positions.get(ticker, 0) + qty
    return positions


def build_core_satellite_orders(
    *,
    equity: float,
    target_weights: dict[str, float],
    current_positions: dict[str, int],
    prices: dict[str, float],
    min_trade_value: float,
    limit_offset_bps: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    tickers = sorted(set(target_weights) | set(current_positions))
    for ticker in tickers:
        price = float(prices.get(ticker, 0.0) or 0.0)
        if price <= 0:
            rows.append({
                "ticker": ticker,
                "action": "SKIP",
                "reason": "missing_price",
                "price": price,
                "target_weight": target_weights.get(ticker, 0.0),
                "current_shares": current_positions.get(ticker, 0),
                "target_shares": 0,
                "delta_shares": 0,
                "limit_price": 0.0,
                "order_value": 0.0,
            })
            continue
        target_value = float(equity) * float(target_weights.get(ticker, 0.0))
        target_shares = int(np.floor(target_value / price)) if target_value >= 0 else int(np.ceil(target_value / price))
        current = int(current_positions.get(ticker, 0))
        delta = target_shares - current
        order_value = abs(delta) * price
        if delta == 0:
            action = "HOLD"
            reason = "already_at_target"
        elif order_value < min_trade_value:
            action = "SKIP"
            reason = "below_min_trade_value"
        else:
            action = "BUY" if delta > 0 else "SELL"
            reason = "rebalance_to_target"
        limit_price = 0.0
        if action == "BUY":
            limit_price = price * (1.0 + float(limit_offset_bps) / 10_000.0)
        elif action == "SELL":
            limit_price = price * (1.0 - float(limit_offset_bps) / 10_000.0)
        rows.append({
            "ticker": ticker,
            "action": action,
            "reason": reason,
            "price": round(price, 4),
            "target_weight": round(float(target_weights.get(ticker, 0.0)), 6),
            "target_value": round(target_value, 2),
            "current_shares": current,
            "target_shares": target_shares,
            "delta_shares": delta,
            "limit_price": round(limit_price, 4),
            "order_value": round(order_value, 2),
        })
    return pd.DataFrame(rows)


def add_drift_and_guards(
    orders: pd.DataFrame,
    *,
    equity: float,
    prices: dict[str, float],
    etf_drift_threshold: float,
    overlay_drift_threshold: float,
) -> pd.DataFrame:
    guarded = orders.copy()
    first_rebalance = not bool(load_paper_state().get("last_rebalance_at"))
    current_values = []
    current_weights = []
    drift_abs = []
    drift_thresholds = []
    drift_ok = []
    guarded_actions = []
    guarded_reasons = []
    for _, row in guarded.iterrows():
        ticker = _normalise_ticker(row["ticker"])
        price = float(prices.get(ticker, 0.0) or 0.0)
        current_value = float(row.get("current_shares", 0) or 0) * price
        current_weight = current_value / float(equity) if equity > 0 else 0.0
        target_weight = float(row.get("target_weight", 0.0) or 0.0)
        drift = abs(target_weight - current_weight)
        threshold = etf_drift_threshold if ticker in {"SPY", "QQQ"} else overlay_drift_threshold
        is_exit = abs(target_weight) < 1e-9 and abs(current_weight) > 1e-9
        new_entry = first_rebalance and abs(target_weight) > 1e-9 and abs(current_weight) < 1e-9
        ok = bool(is_exit or new_entry or drift >= threshold)
        action = str(row.get("action", "SKIP"))
        reason = str(row.get("reason", ""))
        if action in {"BUY", "SELL"} and not ok:
            action = "SKIP"
            reason = "below_drift_threshold"
        current_values.append(round(current_value, 2))
        current_weights.append(round(current_weight, 6))
        drift_abs.append(round(drift, 6))
        drift_thresholds.append(round(threshold, 6))
        drift_ok.append(ok)
        guarded_actions.append(action)
        guarded_reasons.append(reason)
    guarded["current_value"] = current_values
    guarded["current_weight"] = current_weights
    guarded["drift_abs"] = drift_abs
    guarded["drift_threshold"] = drift_thresholds
    guarded["drift_ok"] = drift_ok
    guarded["action"] = guarded_actions
    guarded["reason"] = guarded_reasons
    return guarded


def evaluate_rebalance_guard(
    *,
    signal: pd.Series,
    orders: pd.DataFrame,
    state: dict,
    force: bool,
) -> tuple[bool, list[str]]:
    if force:
        return True, ["forced"]
    reasons: list[str] = []
    holding_days = int(float(signal.get("holding_days", 10) or 10))
    days_since = _trading_days_since(state.get("last_rebalance_at"))
    if days_since is None:
        reasons.append("first_rebalance")
    elif days_since >= holding_days:
        reasons.append(f"schedule_due_{days_since}_trading_days")

    current_regime = str(signal.get("current_regime", "") or "")
    last_regime = str(state.get("last_regime", "") or "")
    if last_regime and current_regime and current_regime != last_regime:
        reasons.append(f"regime_changed_{last_regime}_to_{current_regime}")

    executable = orders[orders["action"].isin(["BUY", "SELL"])]
    if not executable.empty:
        max_drift = float(executable["drift_abs"].max()) if "drift_abs" in executable.columns else 0.0
        reasons.append(f"drift_orders_{len(executable)}_max_{max_drift:.2%}")

    return bool(reasons), reasons


def print_order_summary(orders: pd.DataFrame, *, equity: float, guard_reasons: list[str], allowed: bool) -> None:
    executable = orders[orders["action"].isin(["BUY", "SELL"])]
    buy_value = float(executable.loc[executable["action"] == "BUY", "order_value"].sum()) if not executable.empty else 0.0
    sell_value = float(executable.loc[executable["action"] == "SELL", "order_value"].sum()) if not executable.empty else 0.0
    largest = float(executable["order_value"].max()) if not executable.empty else 0.0
    projected_gross = float(orders["target_weight"].abs().sum()) if "target_weight" in orders.columns else 0.0
    print("\nOrder Summary")
    print(f"  Submit allowed: {'YES' if allowed else 'NO'}")
    print(f"  Guard reasons:  {', '.join(guard_reasons) if guard_reasons else 'none'}")
    print(f"  Buy value:      ${buy_value:,.2f}")
    print(f"  Sell value:     ${sell_value:,.2f}")
    print(f"  Largest order:  ${largest:,.2f}")
    print(f"  Projected gross:{projected_gross:,.2f}x")
    print(f"  Equity:         ${equity:,.2f}")
    market_open, market_reason = us_regular_market_is_open()
    print(f"  Market hours:   {'OPEN' if market_open else 'CLOSED'} - {market_reason}")


def write_paper_status(
    *,
    signal: pd.Series,
    orders: pd.DataFrame,
    equity: float,
    current_positions: dict[str, int],
    prices: dict[str, float],
    guard_reasons: list[str],
    submit_allowed: bool,
    freshness_ok: bool,
    freshness_issues: list[str],
) -> dict:
    position_values = {
        ticker: round(float(qty) * float(prices.get(ticker, 0.0) or 0.0), 2)
        for ticker, qty in sorted(current_positions.items())
    }
    current_gross = sum(abs(v) for v in position_values.values()) / float(equity) if equity > 0 else 0.0
    target_gross = float(orders["target_weight"].abs().sum()) if "target_weight" in orders.columns else 0.0
    executable = orders[orders["action"].isin(["BUY", "SELL"])]
    skipped = orders[orders["action"] == "SKIP"]
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "core_satellite_alpha",
        "paper_ready": bool(signal.get("paper_ready", False)),
        "gates_all_pass": bool(signal.get("gates_all_pass", False)),
        "core_preset": str(signal.get("core_preset", "")),
        "score_source": str(signal.get("score_source", "")),
        "current_regime": str(signal.get("current_regime", "")),
        "signal_predicted_at": str(signal.get("predicted_at", "")),
        "account_equity": round(float(equity), 2),
        "current_gross_exposure": round(current_gross, 4),
        "target_gross_exposure": round(target_gross, 4),
        "max_drift_abs": round(float(orders["drift_abs"].max()), 6) if "drift_abs" in orders.columns and not orders.empty else 0.0,
        "order_count": int(len(executable)),
        "buy_value": round(float(executable.loc[executable["action"] == "BUY", "order_value"].sum()), 2) if not executable.empty else 0.0,
        "sell_value": round(float(executable.loc[executable["action"] == "SELL", "order_value"].sum()), 2) if not executable.empty else 0.0,
        "skipped_orders": int(len(skipped)),
        "submit_allowed": bool(submit_allowed),
        "market_open": us_regular_market_is_open()[0],
        "market_status": us_regular_market_is_open()[1],
        "freshness_ok": bool(freshness_ok),
        "freshness_issues": freshness_issues,
        "guard_reasons": guard_reasons,
        "positions": current_positions,
        "position_values": position_values,
    }
    PAPER_STATUS_FILE.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    equity_row = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "timestamp": status["generated_at"],
        "equity": round(float(equity), 2),
        "strategy": "core_satellite_alpha",
        "current_regime": status["current_regime"],
        "current_gross_exposure": status["current_gross_exposure"],
        "target_gross_exposure": status["target_gross_exposure"],
        "max_drift_abs": status["max_drift_abs"],
    }
    if PAPER_EQUITY_FILE.exists():
        equity_df = pd.read_csv(PAPER_EQUITY_FILE)
        equity_df = equity_df[equity_df.get("date", pd.Series(dtype=str)).astype(str) != equity_row["date"]].copy()
        equity_df = pd.concat([equity_df, pd.DataFrame([equity_row])], ignore_index=True)
    else:
        equity_df = pd.DataFrame([equity_row])
    equity_df.to_csv(PAPER_EQUITY_FILE, index=False)
    return status


def submit_core_satellite_orders(orders: pd.DataFrame) -> pd.DataFrame:
    from moomoo import OrderType, TrdEnv, TrdSide

    submitted_rows: list[dict] = []
    ctx = connect_moomoo_trade_context()
    try:
        executable = orders[orders["action"].isin(["BUY", "SELL"])].copy()
        sells = executable[executable["action"] == "SELL"]
        buys = executable[executable["action"] == "BUY"]
        for _, row in pd.concat([sells, buys], ignore_index=True).iterrows():
            side = TrdSide.BUY if row["action"] == "BUY" else TrdSide.SELL
            qty = int(abs(row["delta_shares"]))
            limit_price = float(row.get("limit_price", 0.0) or 0.0)
            if limit_price <= 0:
                raise RuntimeError(f"Missing limit price for {row['ticker']} {row['action']} order")
            ret, data = ctx.place_order(
                price=limit_price,
                qty=qty,
                code=_moomoo_code(row["ticker"]),
                trd_side=side,
                order_type=OrderType.NORMAL,
                trd_env=TrdEnv.SIMULATE,
                remark="core_satellite_alpha",
            )
            out = row.to_dict()
            out["submitted"] = bool(ret == 0)
            out["broker_response"] = str(data)
            if ret == 0 and hasattr(data, "iloc") and len(data) > 0:
                for col in ("order_id", "order_id_ex", "code", "qty", "trd_side", "order_type", "order_status"):
                    if col in data.columns:
                        out[f"broker_{col}"] = data.iloc[0][col]
            submitted_rows.append(out)
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    return pd.DataFrame(submitted_rows)


def _order_status_bucket(status: object, dealt_qty: object, qty: object) -> str:
    status_text = str(status or "").upper()
    try:
        dealt = float(dealt_qty or 0.0)
        requested = float(qty or 0.0)
    except Exception:
        dealt = 0.0
        requested = 0.0
    if requested > 0 and dealt >= requested:
        return "filled"
    if dealt > 0:
        return "partial"
    if any(key in status_text for key in ("CANCEL", "FAILED", "DISABLED", "DELETED", "REJECT")):
        return "cancelled"
    if any(key in status_text for key in ("SUBMIT", "WAIT", "QUEUE")):
        return "open"
    return "unknown"


def fetch_moomoo_order_history(ctx, *, start: str, end: str) -> pd.DataFrame:
    from moomoo import TrdEnv

    frames: list[pd.DataFrame] = []
    for label, kwargs in (
        ("open", {"trd_env": TrdEnv.SIMULATE, "refresh_cache": True}),
        ("history", {"start": start, "end": end, "trd_env": TrdEnv.SIMULATE}),
    ):
        try:
            if label == "open":
                ret, data = ctx.order_list_query(**kwargs)
            else:
                ret, data = ctx.history_order_list_query(**kwargs)
        except Exception:
            continue
        if ret == 0 and hasattr(data, "empty") and not data.empty:
            frame = data.copy()
            frame["_order_source"] = label
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "order_id" in out.columns:
        out["order_id"] = out["order_id"].astype(str)
        out = out.drop_duplicates("order_id", keep="first")
    return out


def reconcile_paper_trades(*, lookback_days: int = 14) -> pd.DataFrame:
    if not (SIGNAL_DIR / "paper_trades.csv").exists():
        raise FileNotFoundError("No paper_trades.csv found. Submit paper orders before syncing fills.")
    trades = pd.read_csv(SIGNAL_DIR / "paper_trades.csv")
    if trades.empty:
        raise RuntimeError("paper_trades.csv is empty.")

    end_dt = pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days=int(lookback_days))
    ctx = connect_moomoo_trade_context()
    try:
        orders = fetch_moomoo_order_history(
            ctx,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )
        equity = fetch_moomoo_equity(ctx, fallback_equity=None)
        positions = fetch_moomoo_positions(ctx)
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    if orders.empty:
        trades["fill_status"] = trades.get("fill_status", "unknown")
        trades["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        trades.to_csv(SIGNAL_DIR / "paper_trades.csv", index=False)
        return trades

    order_lookup = orders.set_index(orders["order_id"].astype(str), drop=False)
    updated_rows: list[dict] = []
    for _, row in trades.iterrows():
        out = row.to_dict()
        order_id = str(row.get("broker_order_id", "")).split(".")[0]
        if order_id and order_id in order_lookup.index:
            broker = order_lookup.loc[order_id]
            if isinstance(broker, pd.DataFrame):
                broker = broker.iloc[0]
            qty = broker.get("qty", row.get("broker_qty", row.get("delta_shares", 0)))
            dealt_qty = broker.get("dealt_qty", 0.0)
            avg_price = broker.get("dealt_avg_price", 0.0)
            out.update({
                "broker_order_status": broker.get("order_status", out.get("broker_order_status", "")),
                "broker_updated_time": broker.get("updated_time", ""),
                "broker_dealt_qty": float(dealt_qty or 0.0),
                "broker_dealt_avg_price": float(avg_price or 0.0),
                "broker_requested_qty": float(qty or 0.0),
                "fill_status": _order_status_bucket(broker.get("order_status", ""), dealt_qty, qty),
                "filled_value": round(float(dealt_qty or 0.0) * float(avg_price or 0.0), 2),
            })
            try:
                out["unfilled_qty"] = max(float(qty or 0.0) - float(dealt_qty or 0.0), 0.0)
            except Exception:
                out["unfilled_qty"] = np.nan
        else:
            out["fill_status"] = out.get("fill_status", "missing_broker_order")
        out["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        updated_rows.append(out)

    reconciled = pd.DataFrame(updated_rows)
    reconciled.to_csv(SIGNAL_DIR / "paper_trades.csv", index=False)
    reconciled.to_csv(PAPER_RECONCILIATION_FILE, index=False)

    signal = load_core_satellite_signal()
    freshness_ok, freshness_issues = validate_signal_freshness(
        signal,
        max_signal_age_hours=DEFAULT_MAX_SIGNAL_AGE_HOURS,
        max_factor_age_trading_days=DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS,
    )
    current_tickers = sorted(set(positions) | set(core_satellite_target_weights(signal)))
    prices = _fetch_prices(current_tickers)
    dummy_orders = build_core_satellite_orders(
        equity=equity,
        target_weights=core_satellite_target_weights(signal),
        current_positions=positions,
        prices=prices,
        min_trade_value=DEFAULT_MIN_TRADE_VALUE,
        limit_offset_bps=DEFAULT_LIMIT_OFFSET_BPS,
    )
    dummy_orders = add_drift_and_guards(
        dummy_orders,
        equity=equity,
        prices=prices,
        etf_drift_threshold=DEFAULT_ETF_DRIFT_THRESHOLD,
        overlay_drift_threshold=DEFAULT_OVERLAY_DRIFT_THRESHOLD,
    )
    allowed, reasons = evaluate_rebalance_guard(signal=signal, orders=dummy_orders, state=load_paper_state(), force=False)
    write_paper_status(
        signal=signal,
        orders=dummy_orders,
        equity=equity,
        current_positions=positions,
        prices=prices,
        guard_reasons=reasons,
        submit_allowed=allowed,
        freshness_ok=freshness_ok,
        freshness_issues=freshness_issues,
    )
    return reconciled


def print_reconciliation_summary(reconciled: pd.DataFrame) -> None:
    counts = reconciled.get("fill_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    print(f"Reconciled paper trades -> {SIGNAL_DIR / 'paper_trades.csv'}")
    print(f"Snapshot copy -> {PAPER_RECONCILIATION_FILE}")
    print(f"Fill status counts: {counts}")
    cols = [
        "ticker",
        "action",
        "broker_order_id",
        "broker_order_status",
        "broker_dealt_qty",
        "broker_dealt_avg_price",
        "fill_status",
        "unfilled_qty",
    ]
    cols = [c for c in cols if c in reconciled.columns]
    if cols:
        print("\n" + reconciled[cols].tail(20).to_string(index=False))


def run_core_satellite_submission(
    *,
    equity_override: float | None,
    min_trade_value: float,
    submit: bool,
    yes: bool,
    force: bool,
    etf_drift_threshold: float,
    overlay_drift_threshold: float,
    status_only: bool = False,
    allow_closed_market_submit: bool = False,
    limit_offset_bps: float = DEFAULT_LIMIT_OFFSET_BPS,
    allow_stale_signal: bool = False,
    max_signal_age_hours: float = DEFAULT_MAX_SIGNAL_AGE_HOURS,
    max_factor_age_trading_days: int = DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS,
    sync_after_submit: bool = True,
    sync_after_submit_lookback_days: int = DEFAULT_SYNC_AFTER_SUBMIT_LOOKBACK_DAYS,
) -> None:
    signal = load_core_satellite_signal()
    freshness_ok, freshness_issues = validate_signal_freshness(
        signal,
        max_signal_age_hours=max_signal_age_hours,
        max_factor_age_trading_days=max_factor_age_trading_days,
    )
    target_weights = core_satellite_target_weights(signal)
    state = load_paper_state()

    ctx = connect_moomoo_trade_context()
    try:
        equity = fetch_moomoo_equity(ctx, fallback_equity=equity_override)
        if equity_override and equity_override > 0:
            equity = float(equity_override)
        current_positions = fetch_moomoo_positions(ctx)
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    prices = _fetch_prices(sorted(set(target_weights) | set(current_positions)))

    orders = build_core_satellite_orders(
        equity=equity,
        target_weights=target_weights,
        current_positions=current_positions,
        prices=prices,
        min_trade_value=min_trade_value,
        limit_offset_bps=limit_offset_bps,
    )
    orders = add_drift_and_guards(
        orders,
        equity=equity,
        prices=prices,
        etf_drift_threshold=etf_drift_threshold,
        overlay_drift_threshold=overlay_drift_threshold,
    )
    submit_allowed, guard_reasons = evaluate_rebalance_guard(signal=signal, orders=orders, state=state, force=force)
    plan_path = SIGNAL_DIR / "core_satellite_alpha_orders.csv"
    orders.to_csv(plan_path, index=False)
    print(f"Saved core-satellite order plan -> {plan_path}")
    print(f"Account equity used: ${equity:,.2f}")

    display_cols = [
        "ticker",
        "action",
        "target_weight",
        "current_weight",
        "drift_abs",
        "current_shares",
        "target_shares",
        "delta_shares",
        "price",
        "limit_price",
        "order_value",
        "reason",
    ]
    print("\n" + orders[display_cols].to_string(index=False))
    print_order_summary(orders, equity=equity, guard_reasons=guard_reasons, allowed=submit_allowed)
    print(f"  Signal fresh:   {'YES' if freshness_ok else 'NO'}")
    if freshness_issues:
        print(f"  Freshness:      {', '.join(freshness_issues)}")
    status = write_paper_status(
        signal=signal,
        orders=orders,
        equity=equity,
        current_positions=current_positions,
        prices=prices,
        guard_reasons=guard_reasons,
        submit_allowed=submit_allowed,
        freshness_ok=freshness_ok,
        freshness_issues=freshness_issues,
    )
    print(f"Status written -> {PAPER_STATUS_FILE}")
    print(f"Equity snapshot -> {PAPER_EQUITY_FILE}")

    if status_only:
        print("\nStatus only. No orders were submitted.")
        return

    executable = orders[orders["action"].isin(["BUY", "SELL"])]
    if executable.empty:
        print("\nNo orders needed; account is already at target or below minimum trade size.")
        return
    if not submit:
        print("\nPreview only. Re-run with `--submit` to send these orders to Moomoo simulated trading.")
        return
    if not submit_allowed:
        print("\nBlocked: rebalance guard did not allow submission. Use --force to override deliberately.")
        return
    if not yes:
        print("\nBlocked: add `--yes` with `--submit` to confirm broker order submission.")
        return
    if not freshness_ok and not allow_stale_signal:
        print(f"\nBlocked: stale signal/data ({', '.join(freshness_issues)}).")
        print("Run `python3 core_satellite_alpha.py` and refresh data if needed, or use --allow-stale-signal deliberately.")
        return
    market_open, market_reason = us_regular_market_is_open()
    if not market_open and not allow_closed_market_submit:
        print(f"\nBlocked: {market_reason}. Orders are likely to cancel outside regular hours.")
        print("Re-run during regular US market hours, or use --allow-closed-market-submit deliberately.")
        return

    submitted = submit_core_satellite_orders(orders)
    submitted["submitted_at"] = datetime.now(timezone.utc).isoformat()
    submitted["strategy"] = "core_satellite_alpha"
    out_path = SIGNAL_DIR / "paper_trades.csv"
    if out_path.exists():
        prior = pd.read_csv(out_path)
        submitted = pd.concat([prior, submitted], ignore_index=True)
    submitted.to_csv(out_path, index=False)
    new_state = {
        **state,
        "last_rebalance_at": datetime.now(timezone.utc).isoformat(),
        "last_regime": str(signal.get("current_regime", "")),
        "last_signal_predicted_at": str(signal.get("predicted_at", "")),
        "last_order_count": int(len(executable)),
        "last_account_equity": round(float(equity), 2),
        "last_guard_reasons": guard_reasons,
        "last_order_plan": str(plan_path),
    }
    save_paper_state(new_state)
    print(f"\nSubmitted {len(executable)} Moomoo simulated orders. Logged -> {out_path}")
    print(f"Updated paper state -> {PAPER_STATE_FILE}")
    if sync_after_submit:
        print("\nSyncing submitted orders against Moomoo order history...")
        reconciled = reconcile_paper_trades(lookback_days=int(sync_after_submit_lookback_days))
        print_reconciliation_summary(reconciled)


def enrich_with_sizing(df: pd.DataFrame, equity: float) -> pd.DataFrame:
    """Add recommended_position_pct, recommended_shares, and est_borrow_cost columns."""
    rec_pcts, rec_shares, borrow_costs = [], [], []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        confidence = float(row.get("confidence", 60.0))
        expected_return = float(row.get("expected_return", 0.0))
        signal_quality = str(row.get("signal_quality", "MEDIUM")).upper()
        signal = str(row.get("signal", "LONG")).upper()
        hold_days = int(row.get("horizon_days", 5))

        asset_vol = _fetch_vol(ticker)
        price = float(row.get("price", 0.0)) or _fetch_price(ticker)

        if POSITION_SIZING_MODE == "vol_kelly":
            pct = compute_position_size(confidence, expected_return, signal_quality, equity, asset_vol)
        else:
            conf_scale = max((confidence - 50.0) / 50.0, 0.1)
            ret_scale = min(max(abs(expected_return) / 4.0, 0.3), 1.0)
            pct = min(0.30, max(0.15, 0.15 * conf_scale * ret_scale * 2.0))
        rec_pcts.append(round(pct * 100, 2))

        shares = int((equity * pct) / price) if price > 0 else 0
        rec_shares.append(shares)

        # Estimated borrow cost shown for shorts so you know the carry cost upfront.
        if signal == "SHORT":
            annual_rate = BORROW_COSTS.get(ticker, BORROW_COST_ANNUAL_DEFAULT)
            cost = equity * pct * annual_rate * (hold_days / 365.0)
            borrow_costs.append(round(cost, 2))
        else:
            borrow_costs.append(0.0)

    df = df.copy()
    df["recommended_position_pct"] = rec_pcts
    df["recommended_shares"] = rec_shares
    df["est_borrow_cost_usd"] = borrow_costs
    return df


def filter_signals(
    include_tickers: Optional[set[str]],
    exclude_tickers: Optional[set[str]],
    enforce_approved_universe: bool,
    allow_shorts: bool,
) -> pd.DataFrame:
    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(f"Missing signals file: {SIGNALS_FILE}")
    df = pd.read_csv(SIGNALS_FILE)
    if "ticker" not in df.columns:
        raise RuntimeError("signals.csv missing ticker column")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "actionable" in df.columns:
        df = df[df["actionable"].fillna(False).astype(bool)].copy()
    if include_tickers:
        include_tickers = {str(t).upper().strip() for t in include_tickers}
        df = df[df["ticker"].isin(include_tickers)].copy()
    if exclude_tickers:
        exclude_tickers = {str(t).upper().strip() for t in exclude_tickers}
        df = df[~df["ticker"].isin(exclude_tickers)].copy()
    if enforce_approved_universe:
        approved = load_approved_tickers()
        if not approved:
            df = df.iloc[0:0].copy()
        else:
            df = df[df["ticker"].isin(approved)].copy()
    if not allow_shorts and "signal" in df.columns:
        df = df[df["signal"].astype(str).str.upper() == "LONG"].copy()
    if not df.empty:
        keep_mask: list[bool] = []
        rejection_reasons: list[str] = []
        mode = "long_short" if allow_shorts else "long_only"
        for _, row in df.iterrows():
            ticker = str(row["ticker"]).upper().strip()
            ok, reason = passes_trade_rule(row, load_trade_rule(ticker), mode=mode)
            keep_mask.append(ok)
            rejection_reasons.append(reason or "")
        df = df.copy()
        df["paper_filter_rejection_reason"] = rejection_reasons
        df = df[pd.Series(keep_mask, index=df.index)].copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Safer Moomoo paper-trading filter")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--exclude-tickers", nargs="*", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-approved-universe", action="store_true")
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated paper orders to Moomoo simulated trading. Without this flag, only preview orders.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --submit to confirm broker order submission.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override rebalance guard. Still requires --submit --yes.",
    )
    parser.add_argument(
        "--allow-closed-market-submit",
        action="store_true",
        help="Allow submitting orders outside regular US market hours. Usually not recommended.",
    )
    parser.add_argument(
        "--allow-stale-signal",
        action="store_true",
        help="Allow submission even when signal/factor freshness checks fail. Usually not recommended.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Sync paper equity/status only; write paper_daily_status.json and paper_equity.csv without submitting orders.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Reconcile submitted paper orders against Moomoo simulated order history.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Order-history lookback window for --sync.",
    )
    parser.add_argument(
        "--no-sync-after-submit",
        action="store_true",
        help="Do not automatically reconcile Moomoo order history after a successful --submit.",
    )
    parser.add_argument(
        "--min-trade-value",
        type=float,
        default=DEFAULT_MIN_TRADE_VALUE,
        help="Skip rebalance orders smaller than this dollar value.",
    )
    parser.add_argument(
        "--limit-offset-bps",
        type=float,
        default=DEFAULT_LIMIT_OFFSET_BPS,
        help="Limit-order offset from last price in bps. Buys pay up, sells shade down.",
    )
    parser.add_argument(
        "--max-signal-age-hours",
        type=float,
        default=DEFAULT_MAX_SIGNAL_AGE_HOURS,
        help="Block submit if core_satellite_alpha_signal.csv is older than this many hours.",
    )
    parser.add_argument(
        "--max-factor-age-trading-days",
        type=int,
        default=DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS,
        help="Block submit if latest_factor_date is older than this many trading days.",
    )
    parser.add_argument(
        "--etf-drift-threshold",
        type=float,
        default=DEFAULT_ETF_DRIFT_THRESHOLD,
        help="Minimum ETF weight drift needed before trading that ETF.",
    )
    parser.add_argument(
        "--overlay-drift-threshold",
        type=float,
        default=DEFAULT_OVERLAY_DRIFT_THRESHOLD,
        help="Minimum overlay stock weight drift needed before trading that stock.",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Override account equity for sizing. By default, core-satellite uses Moomoo account equity.",
    )
    args = parser.parse_args()

    if args.sync:
        reconciled = reconcile_paper_trades(lookback_days=int(args.lookback_days))
        print_reconciliation_summary(reconciled)
        return

    if PAPER_MODE_STRATEGY == "core_satellite_alpha":
        run_core_satellite_submission(
            equity_override=args.equity,
            min_trade_value=float(args.min_trade_value),
            submit=bool(args.submit),
            yes=bool(args.yes),
            force=bool(args.force),
            etf_drift_threshold=float(args.etf_drift_threshold),
            overlay_drift_threshold=float(args.overlay_drift_threshold),
            status_only=bool(args.status),
            allow_closed_market_submit=bool(args.allow_closed_market_submit),
            limit_offset_bps=float(args.limit_offset_bps),
            allow_stale_signal=bool(args.allow_stale_signal),
            max_signal_age_hours=float(args.max_signal_age_hours),
            max_factor_age_trading_days=int(args.max_factor_age_trading_days),
            sync_after_submit=not bool(args.no_sync_after_submit),
            sync_after_submit_lookback_days=int(args.lookback_days or DEFAULT_SYNC_AFTER_SUBMIT_LOOKBACK_DAYS),
        )
        return

    allowed, reason = single_name_paper_trading_allowed()
    if not allowed:
        print(f"No single-name paper trades: {reason}")
        print_paper_allocation_hint()
        return

    df = filter_signals(
        include_tickers=set(args.tickers) if args.tickers else None,
        exclude_tickers=set(args.exclude_tickers) if args.exclude_tickers else None,
        enforce_approved_universe=not args.no_approved_universe,
        allow_shorts=args.allow_shorts,
    )

    if df.empty:
        print("No live trades remain after filtering.")
        return

    single_name_equity = float(args.equity) if args.equity and args.equity > 0 else DEFAULT_ACCOUNT_EQUITY
    print(f"Fetching vol data for {len(df)} tickers (may take a few seconds)...")
    df = enrich_with_sizing(df, equity=single_name_equity)

    out_path = SIGNAL_DIR / "signals_live_filtered.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved filtered live universe -> {out_path}")

    # Print a readable summary table
    display_cols = ["ticker", "signal", "confidence", "expected_return",
                    "recommended_position_pct", "recommended_shares", "est_borrow_cost_usd"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\n" + df[display_cols].to_string(index=False))

    if args.interactive:
        keep_rows = []
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", ""))
            signal = str(row.get("signal", ""))
            conf = row.get("confidence", None)
            shares = row.get("recommended_shares", "?")
            pct = row.get("recommended_position_pct", "?")
            print(f"\nCandidate: {ticker}  signal={signal}  confidence={conf}  -> {shares} shares ({pct}% of equity)")
            choice = input(f"Submit this trade for {ticker}? [y/N]: ").strip().lower()
            if choice in {"y", "yes"}:
                keep_rows.append(row)
        df = pd.DataFrame(keep_rows)
        df.to_csv(out_path, index=False)
        print(f"Saved interactively approved trades -> {out_path}")

    print("\nNext step: enter the recommended_shares from the CSV directly into moomoo.")
    print("Live shorts are disabled by default. Use --allow-shorts to enable.")


if __name__ == "__main__":
    main()
