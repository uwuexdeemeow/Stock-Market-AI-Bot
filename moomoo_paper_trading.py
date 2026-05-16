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
from safe_io import atomic_write_csv, atomic_write_json, atomic_write_text
from settings import (
    BORROW_COST_ANNUAL_DEFAULT,
    BORROW_COSTS,
    DATA_DIR,
    PAPER_MODE_STRATEGY,
    POSITION_SIZING_MODE,
    SINGLE_NAME_PAPER_TRADING_ENABLED,
)
from signal_freshness import (
    latest_completed_us_trading_day as _shared_latest_completed_us_trading_day,
    parse_signal_timestamp as _shared_parse_signal_timestamp,
    validate_signal_freshness as _shared_validate_signal_freshness,
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
MOOMOO_OPEN_ORDER_REPAIR_AFTER_SECONDS = float(os.environ.get("MOOMOO_OPEN_ORDER_REPAIR_AFTER_SECONDS", "120"))
MOOMOO_MAX_LIMIT_OFFSET_BPS = float(os.environ.get("MOOMOO_MAX_LIMIT_OFFSET_BPS", "50"))
MOOMOO_REPAIR_LIMIT_OFFSET_BPS = float(os.environ.get("MOOMOO_REPAIR_LIMIT_OFFSET_BPS", "25"))
DEFAULT_MAX_SIGNAL_AGE_HOURS = 24.0
DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS = 5
DEFAULT_SYNC_AFTER_SUBMIT_LOOKBACK_DAYS = 3
DEFAULT_MAX_SUBMIT_GROSS_EXPOSURE = 1.00
DEFAULT_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))
MAX_MISSING_PRICE_SUBMIT_ROWS = int(os.environ.get("PAPER_MAX_MISSING_PRICE_SUBMIT_ROWS", "0"))

# Stop-loss and drawdown-halt config
# PLAIN ENGLISH: After buying overlay stocks, we place a limit sell order at
# (purchase price - 8%) so the position automatically exits if it falls that far.
# MooOrderType.TRAILING_STOP is not reliably supported in simulate mode, so we
# use a static limit-sell at the stop price instead.
MOOMOO_TRAILING_STOP_PCT = float(os.environ.get("MOOMOO_TRAILING_STOP_PCT", "0.08"))
# Minimum price increase (%) before we bother updating a stop order.
# Avoids churning stop orders for tiny price moves.  E.g. 0.02 = only
# trail up when the new stop would be at least 2% higher than the old one.
MOOMOO_STOP_TRAIL_MIN_IMPROVEMENT = float(os.environ.get("MOOMOO_STOP_TRAIL_MIN_IMPROVEMENT", "0.02"))
MOOMOO_CORE_STOP_ENABLED = os.environ.get("MOOMOO_CORE_STOP", "1").strip().lower() in {"1", "true", "yes", "on"}
MOOMOO_CORE_STOP_TICKERS = {
    item.strip().upper()
    for item in os.environ.get("MOOMOO_CORE_STOP_TICKERS", "SPY,QQQ,TQQQ").split(",")
    if item.strip()
}
MOOMOO_CORE_STOP_PCT = float(os.environ.get("MOOMOO_CORE_STOP_PCT", "0.05"))
MOOMOO_TQQQ_STOP_PCT = float(os.environ.get("MOOMOO_TQQQ_STOP_PCT", "0.10"))
MOOMOO_CORE_STOP_LIMIT_BUFFER_PCT = float(os.environ.get("MOOMOO_CORE_STOP_LIMIT_BUFFER_PCT", "0.005"))
MOOMOO_CORE_STOP_REPLACE_BPS = float(os.environ.get("MOOMOO_CORE_STOP_REPLACE_BPS", "25"))
MOOMOO_CORE_STOP_REMARK_PREFIX = os.environ.get("MOOMOO_CORE_STOP_REMARK_PREFIX", "core_etf_stop")
MOOMOO_GUARD_STATE_FILE = SIGNAL_DIR / "moomoo_execution_guard_state.json"
# If the total account value drops more than this from its all-time high, halt
# all new order submissions until the account recovers.
MOOMOO_DD_HALT_PCT = float(os.environ.get("MOOMOO_DD_HALT_PCT", "0.12"))
MOOMOO_EQUITY_FILE = SIGNAL_DIR / "moomoo_paper_equity.csv"
ACTIVE_FILL_STATUSES = {"open", "partial", "partially_filled", "unknown", "submitted", "pending", "queued"}



def _round_us_limit_price(price: float, action: str) -> float:
    """Return a Moomoo-valid US limit price.

    Moomoo rejects US stock/ETF limit prices with too many decimal places.
    For normal US names above $1, the valid minimum tick is usually $0.01.
    We round BUY prices up and SELL prices down so the protective limit offset
    remains conservative after rounding.
    """
    try:
        value = float(price)
    except Exception:
        return 0.0
    if not np.isfinite(value) or value <= 0:
        return 0.0

    # US listed stocks/ETFs are normally quoted in cents. For rare sub-$1
    # instruments, allow 4 decimals. Your current universe is all >$1.
    decimals = 4 if value < 1.0 else 2
    scale = 10 ** decimals
    action = str(action).upper().strip()
    if action == "BUY":
        rounded = np.ceil(value * scale) / scale
    elif action == "SELL":
        rounded = np.floor(value * scale) / scale
    else:
        rounded = round(value, decimals)
    return float(f"{rounded:.{decimals}f}")


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


def _signal_timezone():
    try:
        return ZoneInfo(str(DEFAULT_SIGNAL_TIMEZONE))
    except Exception:
        return datetime.now().astimezone().tzinfo or timezone.utc


def parse_signal_timestamp(value: object) -> pd.Timestamp:
    return _shared_parse_signal_timestamp(value, default_timezone=DEFAULT_SIGNAL_TIMEZONE)


def current_signal_open_orders(signal: pd.Series | dict, trades_path: Path | None = None) -> pd.DataFrame:
    """Return active local paper orders submitted for the current signal."""
    path = Path(trades_path or (SIGNAL_DIR / "paper_trades.csv"))
    if not path.exists():
        return pd.DataFrame()
    try:
        trades = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if trades.empty or "submitted_at" not in trades.columns:
        return pd.DataFrame()
    signal_ts = parse_signal_timestamp(signal.get("predicted_at", ""))
    if pd.isna(signal_ts):
        return pd.DataFrame()
    submitted_ts = pd.to_datetime(trades["submitted_at"], errors="coerce", utc=True)
    current = trades[submitted_ts >= signal_ts].copy()
    if current.empty:
        return current
    if "submitted" in current.columns:
        current = current[current["submitted"].astype(str).str.lower().isin(["true", "1", "yes"])]
    statuses = (
        current.get("fill_status", pd.Series("unknown", index=current.index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    return current[statuses.isin(ACTIVE_FILL_STATUSES)].copy()


def latest_completed_us_trading_day(now: datetime | None = None) -> pd.Timestamp:
    return _shared_latest_completed_us_trading_day(now=now)


def validate_signal_freshness(
    signal: pd.Series,
    *,
    max_signal_age_hours: float,
    max_factor_age_trading_days: int,
) -> tuple[bool, list[str]]:
    return _shared_validate_signal_freshness(
        signal,
        max_signal_age_hours=max_signal_age_hours,
        max_factor_age_trading_days=max_factor_age_trading_days,
    )


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
                "target_tqqq_weight",
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
    """
    3-month realized annual vol for ticker. Returns 0.20 on failure.

    PLAIN ENGLISH: Uses the multi-source data_provider so it works even
    when yfinance is down.  Falls back to yahooquery then Stooq.
    """
    try:
        from data_provider import download_single, flatten_yf
        hist = flatten_yf(download_single(ticker, period="6mo"))
        closes = hist["Close"].dropna()
        if len(closes) >= 20:
            vol = float(closes.pct_change().tail(63).std() * np.sqrt(252))
            return max(0.05, min(1.0, vol))
    except Exception:
        pass
    return 0.20


def _fetch_price(ticker: str) -> float:
    """
    Latest close price. Returns 0.0 on failure.

    PLAIN ENGLISH: First checks local parquet cache (no network needed).
    If that fails, uses the multi-source data_provider which tries
    yfinance → yahooquery → Stooq automatically.
    """
    # Try local parquet first (fastest, no network)
    try:
        local_path = Path(DATA_DIR) / f"{_normalise_ticker(ticker)}.parquet"
        if local_path.exists():
            df = pd.read_parquet(local_path, columns=["Close"])
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        pass
    # Fall back to multi-source download
    try:
        from data_provider import download_single, flatten_yf
        hist = flatten_yf(download_single(ticker, period="5d"))
        if not hist.empty:
            close = hist["Close"].dropna()
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
    if not _truthy(row.get("paper_ready", False)) or not _truthy(row.get("gates_all_pass", False)):
        reason = row.get("reason", "paper_ready/gates_all_pass false")
        raise RuntimeError(f"Core-satellite signal is not approved for paper trading: {reason}")
    if not _truthy(row.get("medium_risk_review_pass", False)):
        raise RuntimeError(
            "Core-satellite signal is missing a passed medium-risk review. "
            "Run `python3 core_satellite_nested_walkforward.py --strategy core-alpha --publish-live-config` "
            "and then regenerate `python3 core_satellite_alpha.py`."
        )
    return row


def core_satellite_target_weights(row: pd.Series) -> dict[str, float]:
    weights: dict[str, float] = {}
    for ticker, column in (
        ("SPY", "target_spy_weight"),
        ("QQQ", "target_qqq_weight"),
        ("TQQQ", "target_tqqq_weight"),
    ):
        weight = float(row.get(column, 0.0) or 0.0)
        if abs(weight) > 1e-9:
            weights[ticker] = weights.get(ticker, 0.0) + weight

    overlay_raw = row.get("overlay_weights_json", "{}")
    try:
        overlay = json.loads(str(overlay_raw)) if pd.notna(overlay_raw) else {}
    except Exception as exc:
        raise RuntimeError(f"Could not parse overlay_weights_json: {exc}") from exc
    for ticker, weight in overlay.items():
        ticker = _normalise_ticker(ticker)
        weights[ticker] = weights.get(ticker, 0.0) + float(weight)
    return {ticker: weight for ticker, weight in weights.items() if abs(weight) > 1e-9}



def target_gross_exposure(target_weights: dict[str, float]) -> float:
    """Absolute target exposure implied by the generated portfolio weights."""
    return float(sum(abs(float(w)) for w in target_weights.values()))


def scale_target_weights_to_gross(
    target_weights: dict[str, float],
    *,
    max_gross_exposure: float,
) -> tuple[dict[str, float], float, bool]:
    """
    Scale target weights down proportionally when the model emits more gross
    exposure than the broker/account should submit.

    Example: QQQ 55% + satellites 70% = 125% gross. With max_gross_exposure=1.0,
    every target weight is multiplied by 0.8 so the submitted plan is 100% gross.
    """
    gross = target_gross_exposure(target_weights)
    max_gross = float(max_gross_exposure)
    if gross <= 0 or max_gross <= 0 or gross <= max_gross + 1e-9:
        return dict(target_weights), 1.0, False
    scale = max_gross / gross
    scaled = {ticker: float(weight) * scale for ticker, weight in target_weights.items()}
    return scaled, scale, True


def _safe_order_id(value: object) -> str:
    """Return a clean broker order id, or '' for missing/NaN values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null"}:
        return ""
    if "." in text and text.replace(".", "", 1).isdigit():
        text = text.split(".", 1)[0]
    return text


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _extract_order_id_from_response(response: object) -> str:
    """Best-effort parser for order_id/order_id_ex from a stringified broker response."""
    import re

    text = str(response or "")
    for pattern in (r"order_id[_ex]*\s+([0-9]{4,})", r"order_id['\"]?\s*[:=]\s*['\"]?([0-9]{4,})"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


# Helper functions for compact display/logging of broker responses
def _summarise_broker_response(value: object, *, max_len: int = 160) -> str:
    """Compact noisy Moomoo responses for readable terminal/CSV logs."""
    text = str(value or "").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if text in {"", "None", "nan", "NaN"}:
        return ""
    # Moomoo successful place_order responses are often full DataFrame strings.
    # The order ID/status columns are stored separately, so keep the response short.
    if "rows x" in text or "stock_name" in text or "order_status" in text:
        return "accepted_by_moomoo"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _clean_for_display(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a compact display frame without huge broker response blobs."""
    view = df[[c for c in cols if c in df.columns]].copy()
    for col in ("broker_response", "broker_reject_reason", "unlock_message"):
        if col in view.columns:
            view[col] = view[col].map(_summarise_broker_response)
    return view

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


def maybe_unlock_moomoo_trade(ctx) -> tuple[bool, str]:
    """Optionally unlock trading when MOOMOO_UNLOCK_PWD is provided.

    Simulated trading often works without this, but some OpenD/account setups still
    reject place_order until the trade context is unlocked. We do not hard-code a
    password; set MOOMOO_UNLOCK_PWD in your shell only if your Moomoo OpenD asks
    for trade unlock.
    """
    pwd = os.environ.get("MOOMOO_UNLOCK_PWD", "").strip()
    if not pwd:
        return False, "MOOMOO_UNLOCK_PWD not set; skipped trade unlock"
    try:
        ret, data = ctx.unlock_trade(pwd)
    except Exception as exc:
        return False, f"unlock_trade_exception: {exc!r}"
    return bool(ret == 0), str(data)


def _moomoo_constant(name: str, attr: str, fallback: str) -> str:
    try:
        module = __import__("moomoo", fromlist=[name])
        cls = getattr(module, name)
        return getattr(cls, attr)
    except Exception:
        return fallback


def _moomoo_ret_ok() -> object:
    try:
        module = __import__("moomoo", fromlist=["RET_OK"])
        return getattr(module, "RET_OK")
    except Exception:
        return 0


def _moomoo_order_value(row: pd.Series | dict, *names: str, default: object = "") -> object:
    for name in names:
        value = row.get(name, default) if hasattr(row, "get") else default
        if str(value or "").strip() and str(value).lower() != "nan":
            return value
    return default


def _moomoo_order_id(row: pd.Series | dict) -> str:
    return _safe_order_id(_moomoo_order_value(row, "order_id", "order_id_ex", default=""))


def _moomoo_order_qty(row: pd.Series | dict) -> float:
    try:
        return float(_moomoo_order_value(row, "qty", "quantity", default=0.0) or 0.0)
    except Exception:
        return 0.0


def _moomoo_order_price(row: pd.Series | dict) -> float:
    try:
        return float(_moomoo_order_value(row, "aux_price", "price", default=0.0) or 0.0)
    except Exception:
        return 0.0


def _moomoo_core_stop_pct(ticker: str) -> float:
    return MOOMOO_TQQQ_STOP_PCT if _normalise_ticker(ticker) == "TQQQ" else MOOMOO_CORE_STOP_PCT


def _moomoo_core_stop_remark(ticker: str, stop_pct: float) -> str:
    safe_ticker = _normalise_ticker(ticker)
    return f"{MOOMOO_CORE_STOP_REMARK_PREFIX}_{safe_ticker}_{stop_pct * 100:.2f}pct"


def _is_moomoo_core_stop_order(row: pd.Series | dict, tickers: set[str] | None = None) -> bool:
    watch = MOOMOO_CORE_STOP_TICKERS if tickers is None else {_normalise_ticker(t) for t in tickers}
    ticker = _broker_row_ticker(row)
    side = str(_moomoo_order_value(row, "trd_side", "side", default="")).upper()
    remark = str(_moomoo_order_value(row, "remark", default=""))
    return ticker in watch and "SELL" in side and remark.startswith(MOOMOO_CORE_STOP_REMARK_PREFIX)


def _query_moomoo_open_orders(ctx) -> pd.DataFrame:
    from moomoo import TrdEnv

    ret, data = ctx.order_list_query(trd_env=TrdEnv.SIMULATE, refresh_cache=True)
    if ret != _moomoo_ret_ok() or data is None or not hasattr(data, "empty") or data.empty:
        return pd.DataFrame()
    return data.copy()


def _cancel_moomoo_order(ctx, order_id: str) -> tuple[bool, str]:
    from moomoo import ModifyOrderOp, TrdEnv

    try:
        ret, data = ctx.modify_order(ModifyOrderOp.CANCEL, str(order_id), 0, 0, trd_env=TrdEnv.SIMULATE)
    except Exception as exc:
        return False, repr(exc)
    return bool(ret == _moomoo_ret_ok()), str(data)


def trail_overlay_stop_orders(ctx, *, logger=print) -> dict:
    """Trail up existing overlay stop-loss orders to lock in gains.

    PLAIN ENGLISH: Moomoo simulate mode doesn't support real trailing stops.
    So we fake trailing stops by running this function each day:

    1. Find all open SELL limit orders whose remark contains "stop_loss"
    2. For each one, look up the stock's current price
    3. Compute what the stop SHOULD be: current_price * (1 - 8%)
    4. If the new stop is higher than the existing stop by at least 2%,
       cancel the old order and place a new one at the higher price

    This way, if a stock rallies 30%, the stop trails up from the original
    purchase price to 8% below the new high.  It only moves UP, never DOWN.

    Returns a summary dict with counts of updated, skipped, and failed stops.
    """
    from moomoo import OrderType as MooOrderType, TrdEnv, TrdSide

    ETF_SKIP = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}
    summary = {"checked": 0, "trailed_up": 0, "skipped": 0, "failed": 0, "details": []}

    # Step 1: Get all open orders
    open_orders = _query_moomoo_open_orders(ctx)
    if open_orders.empty:
        logger("  No open orders found — nothing to trail.")
        return summary

    # Step 2: Get current positions and prices
    try:
        positions, prices = fetch_moomoo_positions_and_prices(ctx)
    except Exception as exc:
        logger(f"  Could not fetch positions/prices: {exc}")
        return summary

    # Step 3: Filter to stop-loss sell orders (identified by remark)
    remark_col = "remark" if "remark" in open_orders.columns else None
    if remark_col is None:
        # Try to find any column with remark-like data
        for col in ("order_remark", "memo", "note"):
            if col in open_orders.columns:
                remark_col = col
                break
    if remark_col is None:
        logger("  No remark column in order data — cannot identify stop orders.")
        return summary

    # Filter for stop-loss orders: SELL side, remark contains "stop_loss"
    side_col = "trd_side" if "trd_side" in open_orders.columns else "order_side" if "order_side" in open_orders.columns else None
    status_col = "order_status" if "order_status" in open_orders.columns else None
    price_col = "price" if "price" in open_orders.columns else "order_price" if "order_price" in open_orders.columns else None
    code_col = "code" if "code" in open_orders.columns else "stock_code" if "stock_code" in open_orders.columns else None
    qty_col = "qty" if "qty" in open_orders.columns else "quantity" if "quantity" in open_orders.columns else None
    id_col = "order_id" if "order_id" in open_orders.columns else None

    if not all([side_col, price_col, code_col, qty_col, id_col]):
        logger(f"  Missing required columns in order data: {list(open_orders.columns)}")
        return summary

    for _, order in open_orders.iterrows():
        remark = str(order.get(remark_col, "")).lower()
        if "stop_loss" not in remark:
            continue

        # Check it's a sell order that's still open
        side = str(order.get(side_col, "")).upper()
        if "SELL" not in side:
            continue
        if status_col:
            status = str(order.get(status_col, "")).upper().replace(" ", "").replace("_", "")
            # Skip filled/cancelled orders
            if any(done in status for done in ("FILLEDALL", "CANCELLED", "DELETED", "FAILED")):
                continue

        ticker = _plain_ticker_from_code(str(order.get(code_col, "")))
        if ticker in ETF_SKIP:
            continue

        summary["checked"] += 1
        old_stop_price = float(order.get(price_col, 0))
        order_qty = int(float(order.get(qty_col, 0)))
        order_id = str(order.get(id_col, ""))

        if old_stop_price <= 0 or order_qty <= 0:
            continue

        # Get current price for this ticker
        current_price = prices.get(ticker, 0.0)
        if current_price <= 0:
            summary["skipped"] += 1
            continue

        # Compute where the trailing stop SHOULD be
        new_stop_price = round(current_price * (1.0 - MOOMOO_TRAILING_STOP_PCT), 2)

        # Only trail UP — never move the stop down
        if new_stop_price <= old_stop_price:
            summary["skipped"] += 1
            summary["details"].append({
                "ticker": ticker, "action": "skipped_no_improvement",
                "old_stop": old_stop_price, "new_stop": new_stop_price,
                "current_price": current_price,
            })
            continue

        # Only update if improvement is meaningful (avoid churn)
        improvement = (new_stop_price - old_stop_price) / old_stop_price
        if improvement < MOOMOO_STOP_TRAIL_MIN_IMPROVEMENT:
            summary["skipped"] += 1
            summary["details"].append({
                "ticker": ticker, "action": "skipped_below_min_improvement",
                "old_stop": old_stop_price, "new_stop": new_stop_price,
                "improvement_pct": round(improvement * 100, 2),
            })
            continue

        # Cancel old order, place new one at higher stop price
        ok, cancel_msg = _cancel_moomoo_order(ctx, order_id)
        if not ok:
            logger(f"    ✗ {ticker}: failed to cancel old stop (order {order_id}): {cancel_msg}")
            summary["failed"] += 1
            continue

        try:
            ret, data = ctx.place_order(
                price=new_stop_price,
                qty=order_qty,
                code=_moomoo_code(ticker),
                trd_side=TrdSide.SELL,
                order_type=MooOrderType.NORMAL,
                trd_env=TrdEnv.SIMULATE,
                remark=f"stop_loss_{MOOMOO_TRAILING_STOP_PCT*100:.0f}pct",
            )
            if ret == 0:
                logger(f"    ↑ {ticker}: stop trailed ${old_stop_price:.2f} → ${new_stop_price:.2f} "
                       f"(+{improvement*100:.1f}%, current=${current_price:.2f})")
                summary["trailed_up"] += 1
                summary["details"].append({
                    "ticker": ticker, "action": "trailed_up",
                    "old_stop": old_stop_price, "new_stop": new_stop_price,
                    "current_price": current_price,
                    "improvement_pct": round(improvement * 100, 2),
                })
            else:
                logger(f"    ✗ {ticker}: cancel OK but new stop order failed: {data}")
                summary["failed"] += 1
        except Exception as exc:
            logger(f"    ✗ {ticker}: trail-up failed: {exc}")
            summary["failed"] += 1

    return summary


def _submit_moomoo_core_stop_order(ctx, ticker: str, qty: int, reference_price: float) -> tuple[bool, str, dict]:
    """Submit a Moomoo static stop-limit sell for one core ETF position.

    PLAIN ENGLISH: This is not a trailing stop. It is a broker-side conditional
    stop-limit order. If Moomoo paper rejects STOP_LIMIT, we log the rejection
    and do NOT fall back to a normal sell limit, because a normal sell limit
    below market can execute immediately.
    """
    from moomoo import OrderType, TrdEnv, TrdSide

    stop_pct = _moomoo_core_stop_pct(ticker)
    trigger_price = _round_us_limit_price(float(reference_price) * (1.0 - stop_pct), "SELL")
    limit_price = _round_us_limit_price(trigger_price * (1.0 - MOOMOO_CORE_STOP_LIMIT_BUFFER_PCT), "SELL")
    details = {
        "ticker": _normalise_ticker(ticker),
        "qty": int(qty),
        "reference_price": round(float(reference_price), 4),
        "trigger_price": trigger_price,
        "limit_price": limit_price,
        "stop_pct": stop_pct,
    }
    if int(qty) <= 0 or trigger_price <= 0 or limit_price <= 0:
        return False, "invalid_stop_inputs", details

    kwargs = {
        "price": limit_price,
        "qty": int(qty),
        "code": _moomoo_code(ticker),
        "trd_side": TrdSide.SELL,
        "order_type": OrderType.STOP_LIMIT,
        "trd_env": TrdEnv.SIMULATE,
        "aux_price": trigger_price,
        "remark": _moomoo_core_stop_remark(ticker, stop_pct),
    }
    try:
        module = __import__("moomoo", fromlist=["TimeInForce"])
        time_in_force = getattr(getattr(module, "TimeInForce"), "GTC", None)
        if time_in_force is not None:
            kwargs["time_in_force"] = time_in_force
    except Exception:
        pass

    try:
        ret, data = ctx.place_order(**kwargs)
    except Exception as exc:
        return False, repr(exc), details
    if ret != _moomoo_ret_ok():
        return False, str(data), details
    order_id = ""
    if hasattr(data, "iloc") and len(data) > 0:
        order_id = _safe_order_id(data.iloc[0].get("order_id", "")) or _safe_order_id(data.iloc[0].get("order_id_ex", ""))
    details["order_id"] = order_id
    return True, str(data), details


def _load_moomoo_guard_state() -> dict:
    if not MOOMOO_GUARD_STATE_FILE.exists():
        return {}
    try:
        state = json.loads(MOOMOO_GUARD_STATE_FILE.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save_moomoo_guard_state(state: dict) -> None:
    MOOMOO_GUARD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state, MOOMOO_GUARD_STATE_FILE)


def cancel_moomoo_core_etf_stops(
    ctx,
    *,
    tickers: set[str] | None = None,
    dry_run: bool = False,
    logger=print,
) -> list[dict]:
    watch = MOOMOO_CORE_STOP_TICKERS if tickers is None else {_normalise_ticker(t) for t in tickers}
    open_orders = _query_moomoo_open_orders(ctx)
    actions: list[dict] = []
    if open_orders.empty:
        return actions
    for _, order in open_orders.iterrows():
        if not _is_moomoo_core_stop_order(order, watch):
            continue
        oid = _moomoo_order_id(order)
        ticker = _broker_row_ticker(order)
        if dry_run:
            ok, response = True, "dry_run"
        else:
            ok, response = _cancel_moomoo_order(ctx, oid)
        status = "dry_run" if dry_run else "cancelled" if ok else "failed"
        row = {"action": "cancel_stop", "ticker": ticker, "order_id": oid, "status": status, "response": response}
        actions.append(row)
        logger(f"{status}: Moomoo core ETF stop {ticker} order={oid[:12]}")
    return actions


def repair_moomoo_core_etf_stops(
    ctx,
    *,
    tickers: set[str] | None = None,
    dry_run: bool = False,
    replace: bool = False,
    use_high_water: bool = True,
    logger=print,
) -> dict:
    result = {"checked": [], "cancelled": [], "submitted": [], "skipped": [], "errors": []}
    if not MOOMOO_CORE_STOP_ENABLED:
        result["skipped"].append({"reason": "disabled"})
        logger("Moomoo core ETF stop guard disabled")
        return result

    watch = MOOMOO_CORE_STOP_TICKERS if tickers is None else {_normalise_ticker(t) for t in tickers}
    state = _load_moomoo_guard_state()
    hwm = state.setdefault("core_etf_hwm", {})
    positions, position_prices = fetch_moomoo_positions_and_prices(ctx)
    open_orders = _query_moomoo_open_orders(ctx)

    for ticker in sorted(watch):
        qty = int(positions.get(ticker, 0) or 0)
        ticker_stops = pd.DataFrame()
        if not open_orders.empty:
            mask = open_orders.apply(lambda row: _is_moomoo_core_stop_order(row, {ticker}), axis=1)
            ticker_stops = open_orders[mask].copy()
        result["checked"].append({"ticker": ticker, "position_qty": qty, "stop_count": int(len(ticker_stops))})

        if qty <= 0:
            hwm.pop(ticker, None)
            for _, stop in ticker_stops.iterrows():
                oid = _moomoo_order_id(stop)
                if dry_run:
                    ok, response = True, "dry_run"
                else:
                    ok, response = _cancel_moomoo_order(ctx, oid)
                status = "dry_run" if dry_run else "cancelled" if ok else "failed"
                result["cancelled"].append({"ticker": ticker, "order_id": oid, "status": status, "reason": "no_position"})
                if not ok:
                    result["errors"].append({"ticker": ticker, "order_id": oid, "error": response})
                logger(f"{status}: stale Moomoo core ETF stop {ticker} order={oid[:12]} (no position)")
            continue

        current_price = float(position_prices.get(ticker, 0.0) or 0.0)
        if current_price <= 0:
            current_price = _fetch_price(ticker)
        if current_price <= 0:
            result["errors"].append({"ticker": ticker, "error": "missing_price"})
            logger(f"failed: missing price for Moomoo core ETF stop {ticker}")
            continue

        previous_hwm = float(hwm.get(ticker, 0.0) or 0.0)
        reference_price = max(previous_hwm, current_price) if use_high_water else current_price
        stop_pct = _moomoo_core_stop_pct(ticker)
        candidate_trigger = _round_us_limit_price(reference_price * (1.0 - stop_pct), "SELL")
        if candidate_trigger >= current_price:
            logger(
                f"warning: Moomoo {ticker} high-water stop trigger ${candidate_trigger:.2f} "
                f"is at/above current price ${current_price:.2f}; resetting stop reference to current price"
            )
            reference_price = current_price
            candidate_trigger = _round_us_limit_price(reference_price * (1.0 - stop_pct), "SELL")
        hwm[ticker] = round(reference_price, 4)
        desired_trigger = _round_us_limit_price(reference_price * (1.0 - stop_pct), "SELL")
        replace_threshold = 1.0 + (MOOMOO_CORE_STOP_REPLACE_BPS / 10_000.0)
        existing_qty = float(ticker_stops["qty"].astype(float).sum()) if not ticker_stops.empty and "qty" in ticker_stops.columns else 0.0
        existing_trigger = max((_moomoo_order_price(row) for _, row in ticker_stops.iterrows()), default=0.0)
        qty_matches = abs(existing_qty - float(qty)) < 1e-6
        stop_fresh = existing_trigger > 0 and desired_trigger <= existing_trigger * replace_threshold

        if not ticker_stops.empty and qty_matches and stop_fresh and not replace:
            result["skipped"].append({"ticker": ticker, "reason": "already_protected", "trigger_price": existing_trigger})
            logger(f"protected: Moomoo {ticker} qty={qty} trigger=${existing_trigger:.2f}")
            continue

        cancel_failed = False
        for _, stop in ticker_stops.iterrows():
            oid = _moomoo_order_id(stop)
            if dry_run:
                ok, response = True, "dry_run"
            else:
                ok, response = _cancel_moomoo_order(ctx, oid)
            status = "dry_run" if dry_run else "cancelled" if ok else "failed"
            result["cancelled"].append({"ticker": ticker, "order_id": oid, "status": status, "reason": "replace"})
            logger(f"{status}: outdated Moomoo core ETF stop {ticker} order={oid[:12]}")
            if not ok:
                cancel_failed = True
                result["errors"].append({"ticker": ticker, "order_id": oid, "error": response})
        if cancel_failed:
            logger(f"skipped: replacement Moomoo core ETF stop for {ticker} because cancel failed")
            continue

        if dry_run:
            ok, response, details = True, "dry_run", {
                "ticker": ticker,
                "qty": qty,
                "reference_price": round(reference_price, 4),
                "trigger_price": desired_trigger,
                "limit_price": _round_us_limit_price(desired_trigger * (1.0 - MOOMOO_CORE_STOP_LIMIT_BUFFER_PCT), "SELL"),
                "stop_pct": stop_pct,
                "order_id": "DRY_RUN",
            }
        else:
            ok, response, details = _submit_moomoo_core_stop_order(ctx, ticker, qty, reference_price)
        if ok:
            result["submitted"].append(details)
            logger(f"submitted: Moomoo core ETF stop {ticker} qty={qty} trigger=${details['trigger_price']:.2f}")
        else:
            details["error"] = response
            result["errors"].append(details)
            logger(f"failed: submit Moomoo core ETF stop {ticker}: {response}")

    _save_moomoo_guard_state(state)
    return result


def run_moomoo_execution_guard(*, dry_run: bool = False, replace: bool = False) -> dict:
    """
    Run the laptop-side Moomoo ETF protection guard once.

    PLAIN ENGLISH: This refreshes core ETF stop-limit orders while the laptop is
    awake. The broker-side STOP_LIMIT orders remain at Moomoo after the script
    exits, but they are static, so this guard can move them up when prices rise.
    """
    ctx = connect_moomoo_trade_context()
    try:
        _unlock_ok, _unlock_msg = maybe_unlock_moomoo_trade(ctx)
        result = repair_moomoo_core_etf_stops(
            ctx,
            dry_run=dry_run,
            replace=replace,
            use_high_water=True,
            logger=lambda msg: print(f"  {msg}"),
        )
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    print(
        "Moomoo execution guard: "
        f"checked={len(result['checked'])} "
        f"submitted={len(result['submitted'])} "
        f"cancelled={len(result['cancelled'])} "
        f"errors={len(result['errors'])}"
    )
    if result["errors"]:
        print("Moomoo guard errors:")
        for row in result["errors"]:
            print(f"  - {row}")
    return result


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


def _check_moomoo_drawdown(ctx, halt_pct: float = MOOMOO_DD_HALT_PCT) -> tuple[bool, float]:
    """
    Check whether the Moomoo account has hit the portfolio drawdown halt threshold.

    PLAIN ENGLISH: Reads the account's current total equity from Moomoo, then
    looks at the all-time high equity stored in our local CSV log.  If today's
    value is more than halt_pct (e.g. 12%) below that peak, we return is_halted=True
    and the caller should skip all new order submissions.

    Returns (is_halted: bool, current_drawdown: float).
    Drawdown is negative when the account is below peak (e.g. -0.10 = down 10%).
    """
    try:
        current_equity = fetch_moomoo_equity(ctx)
    except Exception:
        return False, 0.0

    # Log this equity snapshot so we can track the running peak
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_row = {"date": today_str, "equity": round(current_equity, 2)}
    if MOOMOO_EQUITY_FILE.exists():
        try:
            eq_df = pd.read_csv(MOOMOO_EQUITY_FILE)
            eq_df = eq_df[eq_df.get("date", pd.Series(dtype=str)).astype(str) != today_str].copy()
            eq_df = pd.concat([eq_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            eq_df = pd.DataFrame([new_row])
    else:
        eq_df = pd.DataFrame([new_row])
    try:
        atomic_write_csv(eq_df, MOOMOO_EQUITY_FILE)
    except Exception:
        pass

    if eq_df.empty or "equity" not in eq_df.columns:
        return False, 0.0
    peak = float(eq_df["equity"].max())
    if peak <= 0:
        return False, 0.0
    drawdown = (current_equity - peak) / peak
    return drawdown < -halt_pct, drawdown


def fetch_moomoo_positions_and_prices(ctx) -> tuple[dict[str, int], dict[str, float]]:
    trd_env = _moomoo_constant("TrdEnv", "SIMULATE", "SIMULATE")
    ret, data = ctx.position_list_query(trd_env=trd_env, refresh_cache=True)
    if ret != 0:
        raise RuntimeError(f"Moomoo position_list_query failed: {data}")
    if data is None or len(data) == 0:
        return {}, {}
    code_col = "code" if "code" in data.columns else "stock_code" if "stock_code" in data.columns else None
    qty_col = "qty" if "qty" in data.columns else "quantity" if "quantity" in data.columns else None
    if code_col is None or qty_col is None:
        raise RuntimeError(f"Could not parse Moomoo positions columns: {list(data.columns)}")
    positions: dict[str, int] = {}
    prices: dict[str, float] = {}
    for _, row in data.iterrows():
        ticker = _plain_ticker_from_code(row[code_col])
        qty = int(float(row[qty_col] or 0))
        if qty != 0:
            positions[ticker] = positions.get(ticker, 0) + qty
            for price_col in ("nominal_price", "last_price", "latest_price", "market_price", "price"):
                if price_col in data.columns and pd.notna(row.get(price_col)):
                    price = float(row.get(price_col) or 0.0)
                    if price > 0:
                        prices[ticker] = price
                        break
            if ticker not in prices:
                for value_col in ("market_val", "market_value", "val", "position_market_value"):
                    if value_col in data.columns and pd.notna(row.get(value_col)):
                        value = float(row.get(value_col) or 0.0)
                        if value > 0:
                            prices[ticker] = value / abs(qty)
                            break
    return positions, prices


def fetch_moomoo_positions(ctx) -> dict[str, int]:
    positions, _prices = fetch_moomoo_positions_and_prices(ctx)
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
        # Round to nearest share to minimize drift from target weight.
        # Floor/ceil systematically under-allocates.
        target_shares = round(target_value / price)
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
        raw_limit_price = 0.0
        if action == "BUY":
            raw_limit_price = price * (1.0 + float(limit_offset_bps) / 10_000.0)
        elif action == "SELL":
            raw_limit_price = price * (1.0 - float(limit_offset_bps) / 10_000.0)
        limit_price = _round_us_limit_price(raw_limit_price, action)
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
            "raw_limit_price": round(raw_limit_price, 6),
            "limit_price": limit_price,
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


def evaluate_broker_submit_preflight(
    *,
    signal: pd.Series,
    orders: pd.DataFrame,
    freshness_ok: bool,
    freshness_issues: list[str],
    submit_allowed: bool,
    allow_stale_signal: bool,
    allow_closed_market_submit: bool,
    allow_leverage_submit: bool,
    allow_submit_with_open_orders: bool,
    max_submit_gross_exposure: float,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not bool(signal.get("paper_ready", False)):
        blockers.append("strategy paper_ready is false")
    if not bool(signal.get("gates_all_pass", False)):
        blockers.append("strategy gates_all_pass is false")
    if not freshness_ok and not allow_stale_signal:
        blockers.append(f"stale signal/data: {', '.join(freshness_issues)}")
    if not submit_allowed:
        blockers.append("rebalance guard did not allow submission")

    target_gross = float(orders["target_weight"].abs().sum()) if "target_weight" in orders.columns else 0.0
    if target_gross > float(max_submit_gross_exposure) + 1e-9 and not allow_leverage_submit:
        blockers.append(
            f"target gross {target_gross:.3f} exceeds max submit gross {float(max_submit_gross_exposure):.3f}"
        )

    missing_price = orders[
        orders.get("reason", pd.Series("", index=orders.index)).astype(str).eq("missing_price")
    ]
    if len(missing_price) > MAX_MISSING_PRICE_SUBMIT_ROWS:
        blockers.append(f"missing prices for {len(missing_price)} order rows")

    market_open, market_reason = us_regular_market_is_open()
    if not market_open and not allow_closed_market_submit:
        blockers.append(market_reason)
    open_orders = current_signal_open_orders(signal)
    if len(open_orders) > 0 and not allow_submit_with_open_orders:
        tickers = ",".join(open_orders.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().head(8).tolist())
        blockers.append(f"current signal already has {len(open_orders)} open paper orders: {tickers}")
    return not blockers, blockers


def build_submit_readiness_report(
    *,
    signal: pd.Series,
    orders: pd.DataFrame,
    freshness_ok: bool,
    freshness_issues: list[str],
    submit_allowed: bool,
    max_submit_gross_exposure: float = DEFAULT_MAX_SUBMIT_GROSS_EXPOSURE,
    allow_submit_with_open_orders: bool = False,
) -> dict:
    market_open, market_reason = us_regular_market_is_open()
    executable = orders[orders["action"].isin(["BUY", "SELL"])] if not orders.empty else pd.DataFrame()
    missing_price = orders[
        orders.get("reason", pd.Series("", index=orders.index)).astype(str).eq("missing_price")
    ] if not orders.empty else pd.DataFrame()
    target_gross = float(orders["target_weight"].abs().sum()) if "target_weight" in orders.columns else 0.0
    open_orders = current_signal_open_orders(signal)
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(signal.get("paper_ready", False)):
        blockers.append("strategy paper_ready is false")
    if not bool(signal.get("gates_all_pass", False)):
        blockers.append("strategy gates_all_pass is false")
    if not freshness_ok:
        blockers.append(f"stale signal/data: {', '.join(freshness_issues)}")
    if not submit_allowed:
        blockers.append("rebalance guard did not allow submission")
    if target_gross > float(max_submit_gross_exposure) + 1e-9:
        blockers.append(f"target gross {target_gross:.3f} exceeds max submit gross {float(max_submit_gross_exposure):.3f}")
    if len(missing_price) > MAX_MISSING_PRICE_SUBMIT_ROWS:
        blockers.append(f"missing prices for {len(missing_price)} order rows")
    if not market_open:
        blockers.append(market_reason)
    if len(open_orders) > 0 and not allow_submit_with_open_orders:
        tickers = ",".join(open_orders.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().head(8).tolist())
        blockers.append(f"current signal already has {len(open_orders)} open paper orders: {tickers}")
    if executable.empty:
        warnings.append("no executable BUY/SELL rows")
    return {
        "can_submit_during_regular_market": not blockers and not executable.empty,
        "blockers": blockers,
        "warnings": warnings,
        "market_open": bool(market_open),
        "market_status": market_reason,
        "executable_orders": int(len(executable)),
        "buy_value": round(float(executable.loc[executable["action"] == "BUY", "order_value"].sum()), 2) if not executable.empty else 0.0,
        "sell_value": round(float(executable.loc[executable["action"] == "SELL", "order_value"].sum()), 2) if not executable.empty else 0.0,
        "target_gross_exposure": round(float(target_gross), 6),
        "missing_price_rows": int(len(missing_price)),
        "current_signal_open_orders": int(len(open_orders)),
    }


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
    submit_readiness = build_submit_readiness_report(
        signal=signal,
        orders=orders,
        freshness_ok=freshness_ok,
        freshness_issues=freshness_issues,
        submit_allowed=submit_allowed,
    )
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
        "submit_readiness": submit_readiness,
    }
    atomic_write_json(status, PAPER_STATUS_FILE)

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
    atomic_write_csv(equity_df, PAPER_EQUITY_FILE)
    return status


def _submit_single_order(row: pd.Series, ctx, unlock_ok: bool, unlock_msg: str) -> dict:
    """
    Submit one order to Moomoo and return the result dict.

    PLAIN ENGLISH: Takes a single order row (BUY or SELL), sends it to
    Moomoo's paper trading API, and captures the broker's response
    (accepted, rejected, order ID, etc.).
    """
    from moomoo import OrderType, TrdEnv, TrdSide

    side = TrdSide.BUY if row["action"] == "BUY" else TrdSide.SELL
    qty = int(abs(row["delta_shares"]))
    limit_price = float(row.get("limit_price", 0.0) or 0.0)
    moomoo_code = _moomoo_code(row["ticker"])
    out = row.to_dict()
    out.update({
        "submitted": False,
        "broker_response": "",
        "broker_reject_reason": "",
        "broker_order_id": "",
        "broker_ret": np.nan,
        "broker_code": moomoo_code,
        "broker_requested_qty": qty,
        "broker_requested_price": limit_price,
        "unlock_attempted": bool(os.environ.get("MOOMOO_UNLOCK_PWD", "").strip()),
        "unlock_ok": bool(unlock_ok),
        "unlock_message": unlock_msg,
    })
    if qty <= 0:
        out["broker_reject_reason"] = "zero_quantity"
        return out
    if limit_price <= 0:
        out["broker_reject_reason"] = "missing_limit_price"
        return out
    # Final broker-safety normalization. Never send 4-decimal prices
    # for normal US stocks/ETFs because Moomoo rejects them as invalid.
    valid_limit_price = _round_us_limit_price(limit_price, str(row["action"]))
    out["broker_requested_price_raw"] = limit_price
    out["broker_requested_price"] = valid_limit_price
    limit_price = valid_limit_price
    try:
        ret, data = ctx.place_order(
            price=limit_price,
            qty=qty,
            code=moomoo_code,
            trd_side=side,
            order_type=OrderType.NORMAL,
            trd_env=TrdEnv.SIMULATE,
            remark="core_satellite_alpha",
        )
    except Exception as exc:
        out["broker_response"] = repr(exc)
        out["broker_reject_reason"] = "place_order_exception"
        return out

    out["broker_ret"] = ret
    out["submitted"] = bool(ret == 0)
    out["broker_response"] = str(data)
    if ret != 0:
        out["broker_reject_reason"] = str(data)
    if ret == 0 and hasattr(data, "iloc") and len(data) > 0:
        for col in ("order_id", "order_id_ex", "code", "qty", "trd_side", "order_type", "order_status"):
            if col in data.columns:
                out[f"broker_{col}"] = data.iloc[0][col]
        out["broker_order_id"] = _safe_order_id(out.get("broker_order_id")) or _safe_order_id(out.get("broker_order_id_ex"))
        if "order_status" in data.columns:
            out["broker_order_status"] = data.iloc[0]["order_status"]
    if not out.get("broker_order_id"):
        out["broker_order_id"] = _extract_order_id_from_response(out.get("broker_response", ""))
    if bool(out.get("submitted")) and not out.get("broker_order_id"):
        out["broker_reject_reason"] = "accepted_but_no_order_id_in_response"
    return out


def _open_order_repair_candidates(trades: pd.DataFrame, *, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    now_ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    status = trades.get("fill_status", pd.Series("", index=trades.index)).fillna("").astype(str).str.lower()
    submitted = trades.get("submitted", pd.Series(False, index=trades.index)).astype(str).str.lower().isin({"true", "1", "yes"})
    submitted_at = pd.to_datetime(trades.get("submitted_at", pd.Series("", index=trades.index)), errors="coerce", utc=True)
    age_seconds = (now_ts - submitted_at).dt.total_seconds()
    attempts = pd.to_numeric(trades.get("repair_attempt", pd.Series(0, index=trades.index)), errors="coerce").fillna(0)
    has_order_id = trades.get("broker_order_id", pd.Series("", index=trades.index)).map(_safe_order_id).ne("")
    actions = trades.get("action", pd.Series("", index=trades.index)).astype(str).str.upper().isin({"BUY", "SELL"})
    order_type = trades.get("broker_order_type", pd.Series("NORMAL", index=trades.index)).fillna("NORMAL").astype(str).str.upper()
    normal_order = order_type.eq("NORMAL") | order_type.eq("")
    mask = (
        status.isin(ACTIVE_FILL_STATUSES)
        & submitted
        & has_order_id
        & actions
        & normal_order
        & attempts.lt(1)
        & submitted_at.notna()
        & age_seconds.ge(MOOMOO_OPEN_ORDER_REPAIR_AFTER_SECONDS)
    )
    return trades.loc[mask].copy()


def _repair_limit_price(row: pd.Series, *, base_offset_bps: float = DEFAULT_LIMIT_OFFSET_BPS) -> tuple[float, float]:
    action = str(row.get("action", "")).upper().strip()
    reference = float(row.get("price", 0.0) or 0.0)
    if reference <= 0:
        reference = float(row.get("broker_requested_price", row.get("limit_price", 0.0)) or 0.0)
    offset_bps = min(float(base_offset_bps) + float(MOOMOO_REPAIR_LIMIT_OFFSET_BPS), float(MOOMOO_MAX_LIMIT_OFFSET_BPS))
    if action == "BUY":
        raw = reference * (1.0 + offset_bps / 10_000.0)
    elif action == "SELL":
        raw = reference * (1.0 - offset_bps / 10_000.0)
    else:
        raw = 0.0
    return offset_bps, _round_us_limit_price(raw, action)


def repair_open_paper_orders(*, now: pd.Timestamp | None = None, dry_run: bool = False) -> pd.DataFrame:
    path = SIGNAL_DIR / "paper_trades.csv"
    if not path.exists():
        raise FileNotFoundError("No paper_trades.csv found. Submit paper orders before repairing open orders.")
    trades = pd.read_csv(path)
    candidates = _open_order_repair_candidates(trades, now=now)
    if candidates.empty:
        return pd.DataFrame()

    ctx = connect_moomoo_trade_context()
    unlock_ok, unlock_msg = maybe_unlock_moomoo_trade(ctx)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repair_rows: list[dict] = []
    try:
        for idx, row in candidates.iterrows():
            parent_id = _safe_order_id(row.get("broker_order_id", ""))
            if dry_run:
                cancel_ok, cancel_response = True, "dry_run"
            else:
                cancel_ok, cancel_response = _cancel_moomoo_order(ctx, parent_id)
            trades.loc[idx, "repair_reason"] = "stale_open_order"
            trades.loc[idx, "repair_child_order_id"] = ""
            trades.loc[idx, "repair_cancel_ok"] = bool(cancel_ok)
            trades.loc[idx, "repair_cancel_response"] = cancel_response
            if not cancel_ok:
                continue

            repair = row.copy()
            prior_attempt = int(float(row.get("repair_attempt", 0) or 0))
            repair["repair_parent_order_id"] = parent_id
            repair["repair_attempt"] = prior_attempt + 1
            repair["repair_reason"] = "stale_open_order"
            repair["submission_batch_id"] = f"{batch_id}_repair"
            offset_bps, limit_price = _repair_limit_price(repair)
            repair["repair_limit_offset_bps"] = offset_bps
            repair["limit_price"] = limit_price
            repair["raw_limit_price"] = limit_price
            repair["submitted_at"] = datetime.now(timezone.utc).isoformat()
            if dry_run:
                out = repair.to_dict()
                out.update({"submitted": False, "broker_response": "dry_run", "broker_order_id": "", "fill_status": "dry_run"})
            else:
                out = _submit_single_order(repair, ctx, unlock_ok, unlock_msg)
                out["submitted_at"] = repair["submitted_at"]
                out["submission_batch_id"] = repair["submission_batch_id"]
                out["fill_status"] = "pending" if out.get("submitted") else "submission_failed"
            out["repair_parent_order_id"] = parent_id
            out["repair_attempt"] = prior_attempt + 1
            out["repair_reason"] = "stale_open_order"
            out["repair_limit_offset_bps"] = offset_bps
            repair_rows.append(out)
            child_id = _safe_order_id(out.get("broker_order_id", ""))
            if child_id:
                trades.loc[idx, "repair_child_order_id"] = child_id
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    repaired = pd.DataFrame(repair_rows)
    if not repaired.empty:
        combined = pd.concat([trades, repaired], ignore_index=True, sort=False)
        atomic_write_csv(combined, path)
    else:
        atomic_write_csv(trades, path)
    return repaired


# How long to wait for sell proceeds before placing buy orders (seconds).
# Moomoo paper accounts sometimes delay releasing buying power after sells fill.
SELL_SETTLE_WAIT_SECONDS = 30
SELL_SETTLE_MAX_POLLS = 6        # 6 polls × 10s = 60s max wait
SELL_SETTLE_POLL_INTERVAL = 10   # seconds between status checks


def _wait_for_sells_to_fill(ctx, sell_order_ids: list[str], max_polls: int = SELL_SETTLE_MAX_POLLS) -> bool:
    """
    Poll Moomoo until all sell orders have filled, or timeout.

    PLAIN ENGLISH: After submitting sell orders, we need to wait for them
    to actually fill before placing buy orders.  Otherwise the buying power
    from the sells hasn't been released yet and the buys will fail.

    Returns True if all sells filled, False if timed out.
    """
    import time
    from moomoo import TrdEnv

    if not sell_order_ids:
        return True

    print(f"\n  Waiting for {len(sell_order_ids)} sell order(s) to fill before placing buys...")

    for poll in range(max_polls):
        time.sleep(SELL_SETTLE_POLL_INTERVAL)
        try:
            ret, data = ctx.order_list_query(trd_env=TrdEnv.SIMULATE, refresh_cache=True)
        except Exception:
            continue

        if ret != 0 or not hasattr(data, "empty") or data.empty:
            continue

        # Check if our sell orders are filled
        if "order_id" in data.columns:
            data["order_id"] = data["order_id"].astype(str)
            our_sells = data[data["order_id"].isin(sell_order_ids)]
            if not our_sells.empty and "order_status" in our_sells.columns:
                statuses = our_sells["order_status"].astype(str).str.upper().str.replace(" ", "").str.replace("_", "")
                all_filled = all(
                    any(done in s for done in ("FILLEDALL", "ALLFILLED", "DEALTALL", "ALLDEALT"))
                    for s in statuses
                )
                pending = len(statuses) - sum(
                    any(done in s for done in ("FILLEDALL", "ALLFILLED", "DEALTALL", "ALLDEALT"))
                    for s in statuses
                )
                print(f"    Poll {poll+1}/{max_polls}: {len(statuses)-pending}/{len(statuses)} sells filled")
                if all_filled:
                    print(f"  All sells filled. Waiting {SELL_SETTLE_WAIT_SECONDS}s for buying power to settle...")
                    time.sleep(SELL_SETTLE_WAIT_SECONDS)
                    return True

    print(f"  ⚠ Timeout waiting for sells to fill. Proceeding with buys anyway.")
    return False


def submit_core_satellite_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Submit orders to Moomoo paper trading: sells first, wait for fills, then buys.

    PLAIN ENGLISH: This is the main order submission function.  It works in
    three phases to avoid the buying-power problem where buys get cancelled
    because sell proceeds haven't settled yet:

    Phase 1: Submit all SELL orders
    Phase 2: Wait for sells to fill (up to 60 seconds)
    Phase 3: Submit all BUY orders (now with buying power available)
    """
    submitted_rows: list[dict] = []
    ctx = connect_moomoo_trade_context()
    unlock_ok, unlock_msg = maybe_unlock_moomoo_trade(ctx)
    try:
        # ── Portfolio drawdown halt check ─────────────────────────────────
        # PLAIN ENGLISH: If the account has dropped too far from its peak value,
        # stop all new orders immediately.  This mirrors the same protection in
        # alpaca_paper_trading.py.  The halt lifts when equity recovers above the
        # threshold on a subsequent run.
        dd_halted, current_dd = _check_moomoo_drawdown(ctx)
        if dd_halted:
            print(f"\n  🛑 PORTFOLIO DRAWDOWN HALT: account is {current_dd*100:.1f}% below peak")
            print(f"     Threshold: {MOOMOO_DD_HALT_PCT*100:.0f}%")
            print(f"     No new orders until drawdown recovers.")
            return pd.DataFrame(submitted_rows)
        elif current_dd < -0.05:
            print(f"  ⚠  Moomoo drawdown: {current_dd*100:.1f}% from peak "
                  f"(halt at {MOOMOO_DD_HALT_PCT*100:.0f}%)")

        executable = orders[orders["action"].isin(["BUY", "SELL"])].copy()
        sells = executable[executable["action"] == "SELL"]
        buys = executable[executable["action"] == "BUY"]
        core_order_tickers = {
            _normalise_ticker(ticker)
            for ticker in executable.get("ticker", pd.Series(dtype=str)).astype(str)
            if _normalise_ticker(ticker) in MOOMOO_CORE_STOP_TICKERS
        }
        if MOOMOO_CORE_STOP_ENABLED and core_order_tickers:
            print(f"\n  Clearing Moomoo core ETF stops before rebalance: {', '.join(sorted(core_order_tickers))}")
            cancel_actions = cancel_moomoo_core_etf_stops(
                ctx,
                tickers=core_order_tickers,
                logger=lambda msg: print(f"    {msg}"),
            )
            if any(row.get("status") == "failed" for row in cancel_actions):
                print("  ✗ Could not cancel all affected Moomoo core ETF stops. Aborting rebalance.")
                print("     Existing stop orders may reserve shares and cause sell orders to fail.")
                return pd.DataFrame(submitted_rows)

        # ── Phase 1: Submit all SELL orders first ──────────────────────
        # PLAIN ENGLISH: We sell first to free up buying power.
        sell_order_ids: list[str] = []
        if not sells.empty:
            print(f"\n  Phase 1: Submitting {len(sells)} SELL order(s)...")
        for _, row in sells.iterrows():
            out = _submit_single_order(row, ctx, unlock_ok, unlock_msg)
            submitted_rows.append(out)
            if out.get("broker_order_id"):
                sell_order_ids.append(str(out["broker_order_id"]))

        # ── Phase 2: Wait for sells to fill before placing buys ───────
        # PLAIN ENGLISH: Moomoo paper accounts sometimes take a moment to
        # release buying power after sells fill.  If we place buys too fast,
        # they sit unfilled all day and get auto-cancelled at market close.
        if sell_order_ids and not buys.empty:
            _wait_for_sells_to_fill(ctx, sell_order_ids)

        # ── Phase 3: Submit all BUY orders ────────────────────────────
        if not buys.empty:
            print(f"\n  Phase 3: Submitting {len(buys)} BUY order(s)...")
        for _, row in buys.iterrows():
            out = _submit_single_order(row, ctx, unlock_ok, unlock_msg)
            submitted_rows.append(out)

        # ── Phase 3.5: Trail up existing stop orders ─────────────────
        # PLAIN ENGLISH: Before placing NEW stops, update any existing stop
        # orders that are now too low.  If a stock has rallied since we
        # placed the stop, move the stop up to lock in gains.  This gives
        # us "fake trailing stops" on Moomoo simulate mode.
        print("\n  Phase 3.5: Trailing up existing overlay stop orders...")
        trail_result = trail_overlay_stop_orders(ctx, logger=lambda m: print(f"  {m}"))
        if trail_result["trailed_up"] > 0:
            print(f"    Trailed up {trail_result['trailed_up']} stop(s), "
                  f"skipped {trail_result['skipped']}, failed {trail_result['failed']}")
        elif trail_result["checked"] > 0:
            print(f"    Checked {trail_result['checked']} stop(s) — none needed trailing up")

        # ── Phase 4: Submit stop-loss orders for overlay stocks ────────
        # PLAIN ENGLISH: After buying individual stocks, place a limit SELL
        # order at (purchase price - 8%).  If the stock falls to that price,
        # the order fills and limits the loss.
        #
        # We use MooOrderType.NORMAL (limit sell) instead of TRAILING_STOP
        # because Moomoo's simulate environment does not reliably support
        # TRAILING_STOP order types — they fail silently.  A static limit-sell
        # at the stop price is simpler and fires reliably in simulate mode.
        # ETFs (SPY, QQQ, TQQQ) are excluded — regime switching handles those.
        ETF_SKIP = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}
        overlay_buys = buys[~buys["ticker"].isin(ETF_SKIP)]
        if not overlay_buys.empty:
            from moomoo import OrderType as MooOrderType, TrdEnv, TrdSide
            print(f"\n  Phase 4: Placing stop-loss orders ({MOOMOO_TRAILING_STOP_PCT*100:.0f}%) "
                  f"on {len(overlay_buys)} overlay stocks...")
            for _, row in overlay_buys.iterrows():
                try:
                    qty = int(abs(row["delta_shares"]))
                    limit_price = float(row.get("limit_price", 0.0) or 0.0)
                    stop_price = round(limit_price * (1.0 - MOOMOO_TRAILING_STOP_PCT), 2)
                    if qty > 0 and stop_price > 0:
                        ret, data = ctx.place_order(
                            price=stop_price,
                            qty=qty,
                            code=_moomoo_code(row["ticker"]),
                            trd_side=TrdSide.SELL,
                            order_type=MooOrderType.NORMAL,
                            trd_env=TrdEnv.SIMULATE,
                            remark=f"stop_loss_{MOOMOO_TRAILING_STOP_PCT*100:.0f}pct",
                        )
                        if ret == 0:
                            print(f"    ✓ STOP SELL {row['ticker']} qty={qty} "
                                  f"at ${stop_price:.2f} ({MOOMOO_TRAILING_STOP_PCT*100:.0f}% below ${limit_price:.2f})")
                        else:
                            print(f"    ✗ STOP SELL {row['ticker']}: {data}")
                except Exception as exc:
                    print(f"    ✗ STOP SELL {row['ticker']} FAILED: {exc}")

        # ── Phase 5: Repair durable core ETF stop-limit protection ──────
        # PLAIN ENGLISH: Moomoo paper does not reliably support trailing stops.
        # We use conditional STOP_LIMIT sell orders for core ETFs when the API
        # accepts them, and never fall back to a plain sell limit below market.
        if MOOMOO_CORE_STOP_ENABLED:
            print("\n  Phase 5: Repairing Moomoo core ETF stop-limit protection...")
            repair_moomoo_core_etf_stops(
                ctx,
                tickers=MOOMOO_CORE_STOP_TICKERS,
                logger=lambda msg: print(f"    {msg}"),
            )
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    return pd.DataFrame(submitted_rows)


def print_submission_summary(submitted: pd.DataFrame) -> None:
    if submitted.empty:
        print("\nNo rows returned from the submit function.")
        return

    submitted = submitted.copy()
    id_series = submitted.get("broker_order_id", pd.Series("", index=submitted.index)).map(_safe_order_id)
    accepted = submitted[submitted.get("submitted", pd.Series(False, index=submitted.index)).astype(bool)]
    with_ids = submitted[id_series.ne("")]
    rejected = submitted[~submitted.index.isin(with_ids.index)]

    print("\nSubmission result")
    print("-" * 72)
    print(f"Planned executable orders : {len(submitted)}")
    print(f"Accepted by Moomoo        : {len(accepted)}")
    print(f"Logged with broker IDs    : {len(with_ids)}")
    print(f"Rejected / no broker ID   : {len(rejected)}")

    cols = [
        "ticker",
        "action",
        "broker_code",
        "broker_requested_qty",
        "broker_requested_price",
        "submitted",
        "broker_order_id",
        "broker_order_status",
        "broker_ret",
        "broker_reject_reason",
    ]
    view = _clean_for_display(submitted, cols)
    if not view.empty:
        print("\nOrder submission table:")
        print(view.to_string(index=False))

    if not rejected.empty:
        reject_cols = ["ticker", "action", "broker_ret", "broker_reject_reason", "broker_response"]
        reject_view = _clean_for_display(rejected, reject_cols)
        print("\nRejected / no-ID rows:")
        print(reject_view.to_string(index=False))
        print("\nRejected rows are saved separately in signals/paper_trade_rejections.csv.")

def _order_status_bucket(status: object, dealt_qty: object, qty: object) -> str:
    status_text = str(status or "").upper()
    compact_status = status_text.replace("_", "").replace("-", "").replace(" ", "")
    try:
        dealt = float(dealt_qty or 0.0)
        requested = float(qty or 0.0)
    except Exception:
        dealt = 0.0
        requested = 0.0
    if requested > 0 and dealt >= requested:
        return "filled"
    if any(key in compact_status for key in ("FILLEDALL", "ALLFILLED", "DEALTALL", "ALLDEALT", "TRADEDALL", "ALLTRADED")):
        return "filled"
    if dealt > 0:
        return "partial"
    if any(key in compact_status for key in ("FILLED", "DEALT", "TRADED")):
        return "filled"
    if any(key in compact_status for key in ("CANCEL", "FAILED", "DISABLED", "DELETED", "REJECT", "INVALID")):
        return "cancelled"
    if any(key in compact_status for key in ("SUBMIT", "WAIT", "QUEUE", "PENDING", "UNSUBMITTED")):
        return "open"
    return "unknown"


def _broker_action_matches(action: object, broker_side: object) -> bool:
    action_text = str(action or "").upper().strip()
    side_text = str(broker_side or "").upper().strip()
    if action_text == "BUY":
        return "BUY" in side_text or side_text in {"B", "LONG"}
    if action_text == "SELL":
        return "SELL" in side_text or side_text in {"S", "SHORT"}
    return False


def _broker_row_ticker(row: pd.Series | dict) -> str:
    for col in ("code", "stock_code", "symbol", "ticker"):
        value = row.get(col, "") if hasattr(row, "get") else ""
        if str(value or "").strip():
            return _plain_ticker_from_code(str(value))
    return ""


def _float_close(left: object, right: object, *, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left or 0.0) - float(right or 0.0)) <= tolerance
    except Exception:
        return False


def _broker_order_time(row: pd.Series | dict) -> pd.Timestamp | None:
    for col in ("updated_time", "created_time", "submitted_time", "create_time"):
        value = row.get(col, "") if hasattr(row, "get") else ""
        if str(value or "").strip():
            ts = pd.to_datetime(value, errors="coerce", utc=True)
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    return None


def _find_broker_order_match(trade: pd.Series, orders: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Fallback match for broker rows when the local order id is missing or stale."""
    if orders.empty:
        return None, ""
    order_id = _safe_order_id(trade.get("broker_order_id", "")) or _extract_order_id_from_response(trade.get("broker_response", ""))
    if order_id and "order_id" in orders.columns:
        matches = orders[orders["order_id"].astype(str).map(_safe_order_id).eq(order_id)]
        if not matches.empty:
            return matches.iloc[0], "order_id"

    ticker = str(trade.get("ticker", "") or "").upper().strip()
    try:
        requested_qty = abs(float(trade.get("broker_qty", trade.get("delta_shares", 0.0)) or 0.0))
    except Exception:
        requested_qty = 0.0
    candidates: list[tuple[float, int]] = []
    submitted_at = pd.to_datetime(trade.get("submitted_at", ""), errors="coerce", utc=True)
    for idx, broker in orders.iterrows():
        if ticker and _broker_row_ticker(broker) != ticker:
            continue
        if not _broker_action_matches(trade.get("action", ""), broker.get("trd_side", broker.get("side", ""))):
            continue
        try:
            broker_qty = abs(float(broker.get("qty", 0.0) or 0.0))
        except Exception:
            broker_qty = 0.0
        if requested_qty > 0 and not _float_close(requested_qty, broker_qty, tolerance=max(1e-6, requested_qty * 0.001)):
            continue
        score = 0.0
        broker_time = _broker_order_time(broker)
        if not pd.isna(submitted_at) and broker_time is not None:
            score = abs((broker_time - submitted_at).total_seconds())
        candidates.append((score, int(idx)))
    if not candidates:
        return None, ""
    _, best_idx = min(candidates, key=lambda item: item[0])
    return orders.loc[best_idx], "ticker_side_qty"


def execution_slippage(row: pd.Series | dict) -> tuple[float, float]:
    """Return adverse execution slippage in bps and dollars versus plan price."""
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
    bps = per_share / reference * 10_000.0
    dollars = per_share * abs(quantity)
    return float(bps), float(dollars)


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
        positions, broker_position_prices = fetch_moomoo_positions_and_prices(ctx)
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    if orders.empty:
        trades["fill_status"] = trades.get("fill_status", "unknown")
        trades["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_csv(trades, SIGNAL_DIR / "paper_trades.csv")
        return trades

    updated_rows: list[dict] = []
    for _, row in trades.iterrows():
        out = row.to_dict()
        order_id = _safe_order_id(row.get("broker_order_id", "")) or _extract_order_id_from_response(row.get("broker_response", ""))
        if order_id:
            out["broker_order_id"] = order_id
        broker, match_type = _find_broker_order_match(row, orders)
        if broker is not None:
            if match_type != "order_id" and _safe_order_id(broker.get("order_id", "")):
                out["broker_order_id"] = _safe_order_id(broker.get("order_id", ""))
            out["reconcile_match_type"] = match_type
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
            slip_bps, slip_dollars = execution_slippage(out)
            out["execution_slippage_bps"] = slip_bps
            out["execution_slippage_dollars"] = slip_dollars
        else:
            existing = str(out.get("fill_status", "") or "").strip().lower()
            out["reconcile_match_type"] = "missing_order_id" if not order_id else "missing_broker_order"
            if not order_id:
                out["fill_status"] = "not_submitted_or_missing_order_id"
            elif existing in {"", "nan", "none"}:
                out["fill_status"] = "missing_broker_order"
        out["reconciled_at"] = datetime.now(timezone.utc).isoformat()
        updated_rows.append(out)

    reconciled = pd.DataFrame(updated_rows)
    atomic_write_csv(reconciled, SIGNAL_DIR / "paper_trades.csv")
    atomic_write_csv(reconciled, PAPER_RECONCILIATION_FILE)

    signal = load_core_satellite_signal()
    freshness_ok, freshness_issues = validate_signal_freshness(
        signal,
        max_signal_age_hours=DEFAULT_MAX_SIGNAL_AGE_HOURS,
        max_factor_age_trading_days=DEFAULT_MAX_FACTOR_AGE_TRADING_DAYS,
    )
    current_tickers = sorted(set(positions) | set(core_satellite_target_weights(signal)))
    tickers_to_price = sorted(set(current_tickers) - set(broker_position_prices))
    prices = _fetch_prices(tickers_to_price)
    for ticker, price in broker_position_prices.items():
        if float(price or 0.0) > 0:
            prices[ticker] = float(price)
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
        "execution_slippage_bps",
        "execution_slippage_dollars",
        "fill_status",
        "reconcile_match_type",
        "unfilled_qty",
    ]
    view = _clean_for_display(reconciled.tail(20), cols)
    if not view.empty:
        print("\nRecent reconciled orders:")
        print(view.to_string(index=False))


def run_core_satellite_submission(
    *,
    equity_override: float | None,
    min_trade_value: float,
    submit: bool,
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
    max_submit_gross_exposure: float = DEFAULT_MAX_SUBMIT_GROSS_EXPOSURE,
    allow_leverage_submit: bool = False,
    allow_submit_with_open_orders: bool = False,
    reset_paper_log: bool = False,
) -> None:
    signal = load_core_satellite_signal()
    freshness_ok, freshness_issues = validate_signal_freshness(
        signal,
        max_signal_age_hours=max_signal_age_hours,
        max_factor_age_trading_days=max_factor_age_trading_days,
    )
    raw_target_weights = core_satellite_target_weights(signal)
    raw_target_gross = target_gross_exposure(raw_target_weights)
    if allow_leverage_submit:
        target_weights = raw_target_weights
        target_scale = 1.0
        target_scaled = False
    else:
        target_weights, target_scale, target_scaled = scale_target_weights_to_gross(
            raw_target_weights,
            max_gross_exposure=max_submit_gross_exposure,
        )
    state = load_paper_state()

    ctx = connect_moomoo_trade_context()
    try:
        equity = fetch_moomoo_equity(ctx, fallback_equity=equity_override)
        if equity_override and equity_override > 0:
            equity = float(equity_override)
        current_positions, broker_position_prices = fetch_moomoo_positions_and_prices(ctx)
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    tickers_to_price = sorted((set(target_weights) | set(current_positions)) - set(broker_position_prices))
    prices = _fetch_prices(tickers_to_price)
    for ticker, price in broker_position_prices.items():
        if float(price or 0.0) > 0:
            prices[ticker] = float(price)

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
    atomic_write_csv(orders, plan_path)
    print(f"Saved core-satellite order plan -> {plan_path}")
    print(f"Account equity used: ${equity:,.2f}")
    print(f"Raw target gross: {raw_target_gross:.2f}x")
    if target_scaled:
        print(f"Scaled target weights by {target_scale:.4f} to respect max submit gross {float(max_submit_gross_exposure):.2f}x")
    elif allow_leverage_submit and raw_target_gross > float(max_submit_gross_exposure) + 1e-9:
        print(f"Leverage allowed: submitting raw target gross {raw_target_gross:.2f}x")

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
        if submit and MOOMOO_CORE_STOP_ENABLED:
            print("Checking Moomoo core ETF stop-limit protection...")
            run_moomoo_execution_guard(dry_run=False, replace=False)
        return
    if not submit:
        print("\nPreview only. Re-run with `--submit` to send these orders to Moomoo simulated trading.")
        return

    preflight_ok, preflight_blockers = evaluate_broker_submit_preflight(
        signal=signal,
        orders=orders,
        freshness_ok=freshness_ok,
        freshness_issues=freshness_issues,
        submit_allowed=submit_allowed,
        allow_stale_signal=allow_stale_signal,
        allow_closed_market_submit=allow_closed_market_submit,
        allow_leverage_submit=allow_leverage_submit,
        allow_submit_with_open_orders=allow_submit_with_open_orders,
        max_submit_gross_exposure=max_submit_gross_exposure,
    )
    if not preflight_ok:
        print("\nBlocked: broker submit preflight failed.")
        for blocker in preflight_blockers:
            print(f"  - {blocker}")
        print("Refresh data/signals or use the explicit override flag for the specific blocker.")
        return

    submitted = submit_core_satellite_orders(orders)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submitted["submitted_at"] = datetime.now(timezone.utc).isoformat()
    submitted["submission_batch_id"] = batch_id
    submitted["strategy"] = "core_satellite_alpha"
    submitted["raw_target_gross_exposure"] = round(float(raw_target_gross), 6)
    submitted["submitted_target_gross_exposure"] = round(float(target_gross_exposure(target_weights)), 6)
    submitted["target_weight_scale"] = round(float(target_scale), 8)
    out_path = SIGNAL_DIR / "paper_trades.csv"
    reject_path = SIGNAL_DIR / "paper_trade_rejections.csv"

    # Keep the main broker log clean. paper_trades.csv should contain only
    # accepted Moomoo orders with real broker_order_id values, because --sync
    # can only reconcile rows that exist in Moomoo order history. Failed
    # attempts are still useful, so preserve them separately.
    batch_rows = submitted.copy()
    broker_id_series = (
        batch_rows.get("broker_order_id", pd.Series("", index=batch_rows.index))
        .astype(str)
        .str.strip()
    )
    valid_id_mask = broker_id_series.ne("") & broker_id_series.str.lower().ne("nan")
    accepted_mask = batch_rows.get("submitted", pd.Series(False, index=batch_rows.index)).astype(bool)
    broker_logged_batch = batch_rows[accepted_mask & valid_id_mask].copy()
    broker_rejected_batch = batch_rows[~(accepted_mask & valid_id_mask)].copy()

    archive_dir = SIGNAL_DIR / "archive"
    if reset_paper_log:
        archive_dir.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            archive_path = archive_dir / f"paper_trades_before_{batch_id}.csv"
            out_path.replace(archive_path)
            print(f"Archived old paper trade log -> {archive_path}")
        if reject_path.exists():
            reject_archive_path = archive_dir / f"paper_trade_rejections_before_{batch_id}.csv"
            reject_path.replace(reject_archive_path)
            print(f"Archived old paper rejection log -> {reject_archive_path}")

    if out_path.exists() and not broker_logged_batch.empty:
        prior = pd.read_csv(out_path)
        broker_logged = pd.concat([prior, broker_logged_batch], ignore_index=True)
    else:
        broker_logged = broker_logged_batch

    if not broker_logged.empty:
        atomic_write_csv(broker_logged, out_path)
        print(f"Logged accepted broker orders -> {out_path}")
    elif not out_path.exists():
        # Create a CSV with the expected columns so --sync has a clear file state.
        atomic_write_csv(broker_logged, out_path)
        print(f"Created empty broker order log -> {out_path}")

    if not broker_rejected_batch.empty:
        if reject_path.exists():
            prior_rejects = pd.read_csv(reject_path)
            broker_rejections = pd.concat([prior_rejects, broker_rejected_batch], ignore_index=True)
        else:
            broker_rejections = broker_rejected_batch
        atomic_write_csv(broker_rejections, reject_path)
        print(f"Logged rejected/no-ID rows -> {reject_path}")

    print_submission_summary(batch_rows)
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
    print(f"\nAccepted broker orders in this batch: {len(broker_logged_batch)} / {len(batch_rows)}")
    print(f"Accepted order log -> {out_path}")
    if not broker_rejected_batch.empty:
        print(f"Rejected/no-ID log -> {reject_path}")
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
        "--force",
        action="store_true",
        help="Override rebalance guard. Still requires --submit.",
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
        "--repair-open-orders",
        action="store_true",
        help="Cancel stale open normal paper orders and resubmit once with a wider capped limit offset.",
    )
    parser.add_argument(
        "--execution-guard",
        action="store_true",
        help="Run Moomoo ETF protection guard once: repair core ETF stop-limit orders.",
    )
    parser.add_argument(
        "--guard-dry-run",
        action="store_true",
        help="With --execution-guard, show stop changes without submitting/cancelling orders.",
    )
    parser.add_argument(
        "--replace-protection",
        action="store_true",
        help="With --execution-guard, cancel/recreate matching core ETF protection even if it looks fresh.",
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
    parser.add_argument(
        "--max-submit-gross-exposure",
        type=float,
        default=DEFAULT_MAX_SUBMIT_GROSS_EXPOSURE,
        help="Broker-safety cap for submitted target gross exposure. Default 1.00 means no leverage.",
    )
    parser.add_argument(
        "--allow-leverage-submit",
        action="store_true",
        help="Submit raw target weights even if gross exposure is above --max-submit-gross-exposure.",
    )
    parser.add_argument(
        "--allow-submit-with-open-orders",
        action="store_true",
        help="Allow a new submit even when current-signal paper orders are still open. Usually not recommended.",
    )
    parser.add_argument(
        "--reset-paper-log",
        action="store_true",
        help="Archive the old signals/paper_trades.csv before logging this submit attempt.",
    )
    args = parser.parse_args()

    if args.execution_guard:
        run_moomoo_execution_guard(
            dry_run=bool(args.guard_dry_run),
            replace=bool(args.replace_protection),
        )
        return

    if args.sync:
        reconciled = reconcile_paper_trades(lookback_days=int(args.lookback_days))
        print_reconciliation_summary(reconciled)
        return

    if args.repair_open_orders:
        repaired = repair_open_paper_orders()
        print(f"Repaired open paper orders: {len(repaired)}")
        if not repaired.empty:
            view = _clean_for_display(
                repaired,
                ["ticker", "action", "repair_parent_order_id", "broker_order_id", "submitted", "broker_requested_price", "fill_status"],
            )
            print(view.to_string(index=False))
        return

    if PAPER_MODE_STRATEGY == "core_satellite_alpha":
        run_core_satellite_submission(
            equity_override=args.equity,
            min_trade_value=float(args.min_trade_value),
            submit=bool(args.submit),
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
            max_submit_gross_exposure=float(args.max_submit_gross_exposure),
            allow_leverage_submit=bool(args.allow_leverage_submit),
            allow_submit_with_open_orders=bool(args.allow_submit_with_open_orders),
            reset_paper_log=bool(args.reset_paper_log),
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
    atomic_write_csv(df, out_path)
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
        atomic_write_csv(df, out_path)
        print(f"Saved interactively approved trades -> {out_path}")

    print("\nNext step: enter the recommended_shares from the CSV directly into moomoo.")
    print("Live shorts are disabled by default. Use --allow-shorts to enable.")


if __name__ == "__main__":
    main()
