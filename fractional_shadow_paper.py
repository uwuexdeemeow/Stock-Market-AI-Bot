"""Track a $400 fractional-share portfolio without sending broker orders.

PLAIN ENGLISH:
The normal Alpaca paper account is intentionally left unchanged.  This script
reads the approved active signal, uses historical close prices as simulated
fills, and keeps a small pretend account containing cash and fractional shares.
It writes audit files only; it does not import a broker client or submit orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json, configure_console_output
from settings import DATA_DIR, SIGNAL_DIR


configure_console_output()

# These defaults represent the small account the user wants to test.  Every
# value can be changed for a separate experiment without touching live paper.
DEFAULT_INITIAL_EQUITY = 400.0
DEFAULT_MIN_ORDER_NOTIONAL = 1.0
DEFAULT_CASH_BUFFER_PCT = 0.005
DEFAULT_CASH_BUFFER_DOLLARS = 2.0
DEFAULT_SLIPPAGE_BPS = 10.0
# The small-account experiment uses a tighter one-percent band so rounding and
# cash do not hide a two-percent allocation miss during the evidence period.
DEFAULT_ETF_DRIFT = 0.01
DEFAULT_STOCK_DRIFT = 0.01
QUANTITY_DECIMALS = 9

# Current fee defaults are configurable because regulators can change them.
# SEC is expressed in basis points of sell value; TAF and CAT are per share.
DEFAULT_SEC_SELL_FEE_BPS = 0.206
DEFAULT_TAF_SELL_FEE_PER_SHARE = 0.000195
DEFAULT_CAT_FEE_PER_SHARE = 0.000003

SIGNAL_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"
STATE_PATH = Path(SIGNAL_DIR) / "fractional_shadow_state.json"
ORDERS_PATH = Path(SIGNAL_DIR) / "fractional_shadow_orders.csv"
EQUITY_PATH = Path(SIGNAL_DIR) / "fractional_shadow_equity.csv"
REPORT_PATH = Path(SIGNAL_DIR) / "fractional_shadow_report.json"
ETF_TICKERS = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}


def _utc_now() -> str:
    """Return one readable UTC timestamp for audit records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any, default: float = 0.0) -> float:
    """Turn a CSV or JSON value into a safe finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _truthy(value: Any) -> bool:
    """Understand common text versions of true, such as ``yes`` and ``1``."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a non-negative decimal setting and fall back when it is invalid."""
    value = _as_float(os.environ.get(name), default)
    return value if value >= minimum else float(default)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV safely; a missing or empty file means there is no history."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_active_signal(path: Path = SIGNAL_PATH) -> dict[str, Any]:
    """Load the newest approved active-paper signal, never a shadow candidate."""
    frame = _read_csv(Path(path))
    if frame.empty:
        raise RuntimeError(f"Active signal is missing or empty: {path}")
    row = frame.iloc[-1].to_dict()
    blockers = [
        name
        for name in ("paper_ready", "gates_all_pass", "medium_risk_review_pass")
        if not _truthy(row.get(name))
    ]
    if blockers:
        raise RuntimeError("Active signal is not paper-ready: " + ",".join(blockers))
    return row


def target_weights_from_signal(signal: dict[str, Any]) -> dict[str, float]:
    """Convert the signal's ETF columns and stock JSON into one weight map."""
    weights: dict[str, float] = {}
    for ticker, field in (
        ("SPY", "target_spy_weight"),
        ("QQQ", "target_qqq_weight"),
        ("TQQQ", "target_tqqq_weight"),
    ):
        weight = _as_float(signal.get(field))
        if weight > 0:
            weights[ticker] = weight

    raw_overlay = signal.get("overlay_weights_json")
    try:
        overlay = json.loads(str(raw_overlay)) if raw_overlay not in (None, "") else {}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Active signal overlay weights are invalid JSON") from exc
    if not isinstance(overlay, dict):
        raise RuntimeError("Active signal overlay weights must be a JSON object")
    for ticker, raw_weight in overlay.items():
        weight = _as_float(raw_weight)
        if weight > 0:
            symbol = str(ticker).upper().strip()
            if symbol:
                weights[symbol] = weights.get(symbol, 0.0) + weight

    gross = sum(weights.values())
    if gross <= 0 or gross > 1.000001:
        raise RuntimeError(f"Fractional shadow needs gross exposure in (0, 1], got {gross:.6f}")
    return {ticker: round(weight, 10) for ticker, weight in sorted(weights.items())}


def _signal_date(signal: dict[str, Any]) -> str:
    """Choose the completed price date attached to the signal."""
    for field in ("latest_factor_date", "predicted_at"):
        value = signal.get(field)
        if value not in (None, ""):
            parsed = pd.Timestamp(value)
            if not pd.isna(parsed):
                return parsed.date().isoformat()
    raise RuntimeError("Active signal has no usable price date")


def _latest_close(ticker: str, date_text: str, data_dir: Path) -> tuple[str, float]:
    """Return the last close on or before the signal date without lookahead."""
    path = Path(data_dir) / f"{ticker}.parquet"
    if not path.exists():
        raise RuntimeError(f"Missing price history for {ticker}: {path}")
    try:
        frame = pd.read_parquet(path, columns=["Close"])
    except Exception as exc:
        raise RuntimeError(f"Could not read price history for {ticker}: {exc}") from exc
    if frame.empty:
        raise RuntimeError(f"Price history is empty for {ticker}")

    close_raw = frame["Close"]
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0]
    close = pd.to_numeric(close_raw, errors="coerce")
    close.index = pd.to_datetime(close.index, errors="coerce", utc=True).tz_localize(None).normalize()
    close = close[(close > 0) & close.notna()]
    eligible = close[close.index <= pd.Timestamp(date_text).normalize()]
    if eligible.empty:
        raise RuntimeError(f"No non-lookahead price exists for {ticker} on {date_text}")
    used_date = pd.Timestamp(eligible.index[-1]).date().isoformat()
    return used_date, float(eligible.iloc[-1])


def load_prices(
    tickers: list[str] | set[str],
    date_text: str,
    *,
    data_dir: Path = Path(DATA_DIR),
) -> tuple[dict[str, float], dict[str, str]]:
    """Load every required price and record its source date for auditability."""
    prices: dict[str, float] = {}
    dates: dict[str, str] = {}
    for ticker in sorted(set(tickers)):
        used_date, close = _latest_close(ticker, date_text, Path(data_dir))
        prices[ticker] = close
        dates[ticker] = used_date
    return prices, dates


def _signal_key(signal: dict[str, Any], weights: dict[str, float]) -> str:
    """Create a stable identifier so rerunning one signal cannot trade twice."""
    payload = {
        "predicted_at": str(signal.get("predicted_at") or ""),
        "latest_factor_date": str(signal.get("latest_factor_date") or ""),
        "live_config_hash": str(signal.get("live_config_hash") or ""),
        "weights": weights,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _new_state(initial_equity: float) -> dict[str, Any]:
    """Create an empty pretend account with cash and no positions."""
    return {
        "schema_version": 1,
        "mode": "fractional_shadow_only",
        "real_capital_approved": False,
        "broker_orders_submitted": False,
        "initial_equity": round(float(initial_equity), 2),
        "cash": round(float(initial_equity), 10),
        "positions": {},
        "last_signal_key": "",
        "last_valuation_date": "",
        "cumulative_slippage_cost": 0.0,
        "cumulative_regulatory_fees": 0.0,
        "created_at": _utc_now(),
    }


def load_state(
    state_path: Path,
    equity_path: Path,
    *,
    initial_equity: float,
) -> dict[str, Any]:
    """Load account state and fail closed if history exists without its ledger."""
    state_path = Path(state_path)
    equity_path = Path(equity_path)
    if not state_path.exists():
        if not _read_csv(equity_path).empty:
            raise RuntimeError("Fractional equity history exists but account state is missing")
        return _new_state(initial_equity)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fractional shadow state is unreadable: {exc}") from exc
    if state.get("mode") != "fractional_shadow_only" or state.get("schema_version") != 1:
        raise RuntimeError("Fractional shadow state schema or mode is invalid")
    if _as_float(state.get("initial_equity")) <= 0 or _as_float(state.get("cash"), -1) < 0:
        raise RuntimeError("Fractional shadow state contains invalid account values")
    if abs(_as_float(state.get("initial_equity")) - float(initial_equity)) > 0.01:
        raise RuntimeError(
            "Existing fractional shadow uses different initial equity; "
            "use separate state/equity/order/report paths for a new experiment"
        )
    if not isinstance(state.get("positions"), dict):
        raise RuntimeError("Fractional shadow positions are invalid")
    for ticker, position in state["positions"].items():
        if not isinstance(position, dict) or _as_float(position.get("quantity"), -1) < 0:
            raise RuntimeError(f"Fractional shadow position is invalid for {ticker}")
    return state


def _round_quantity(quantity: float) -> float:
    """Match Alpaca's documented maximum of nine fractional-share decimals."""
    return round(max(0.0, float(quantity)), QUANTITY_DECIMALS)


def _regulatory_fee_raw(side: str, quantity: float, trade_value: float, cfg: dict[str, float]) -> float:
    """Estimate government/industry fees before Alpaca's daily cent rounding."""
    cat = quantity * cfg["cat_fee_per_share"]
    if side != "sell":
        return cat
    sec = trade_value * cfg["sec_sell_fee_bps"] / 10_000.0
    taf = quantity * cfg["taf_sell_fee_per_share"]
    return cat + sec + taf


def _ceil_cents(value: float) -> float:
    """Round a positive daily fee total upward to the nearest cent."""
    if value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def _position_quantities(state: dict[str, Any]) -> dict[str, float]:
    """Extract clean quantities from the JSON position ledger."""
    out: dict[str, float] = {}
    for ticker, item in (state.get("positions") or {}).items():
        quantity = _as_float((item or {}).get("quantity"))
        if quantity > 0:
            out[str(ticker).upper()] = quantity
    return out


def _portfolio_equity(state: dict[str, Any], prices: dict[str, float]) -> tuple[float, float]:
    """Mark cash and positions to the supplied completed-day prices."""
    invested = 0.0
    for ticker, quantity in _position_quantities(state).items():
        if ticker not in prices:
            raise RuntimeError(f"Missing valuation price for held position {ticker}")
        invested += quantity * prices[ticker]
    return _as_float(state.get("cash")) + invested, invested


def _order_id(signal_key: str, ticker: str, side: str) -> str:
    """Create a deterministic pretend order identifier for audit and deduping."""
    return f"frac-shadow-{signal_key[:10]}-{ticker.lower()}-{side}"


def _append_orders(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append new pretend fills while preserving all earlier experiments."""
    if not rows:
        return
    existing = _read_csv(Path(path))
    new_rows = pd.DataFrame(rows)
    if not existing.empty and "order_id" in existing.columns:
        new_rows = new_rows[~new_rows["order_id"].astype(str).isin(set(existing["order_id"].astype(str)))]
    out = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    atomic_write_csv(out, path, index=False)


def _append_equity(path: Path, row: dict[str, Any]) -> None:
    """Write one row per valuation date, replacing an idempotent rerun."""
    existing = _read_csv(Path(path))
    if not existing.empty and "valuation_date" in existing.columns:
        existing = existing[existing["valuation_date"].astype(str) != str(row["valuation_date"])]
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    if "valuation_date" in out.columns:
        out = out.sort_values("valuation_date", kind="stable")
    atomic_write_csv(out, path, index=False)


def run_fractional_shadow(
    *,
    signal_path: Path = SIGNAL_PATH,
    state_path: Path = STATE_PATH,
    orders_path: Path = ORDERS_PATH,
    equity_path: Path = EQUITY_PATH,
    report_path: Path = REPORT_PATH,
    data_dir: Path = Path(DATA_DIR),
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
) -> dict[str, Any]:
    """Mark and rebalance the pretend fractional account, then write evidence."""
    if initial_equity <= 0:
        raise RuntimeError("Fractional shadow initial equity must be positive")
    signal = load_active_signal(Path(signal_path))
    weights = target_weights_from_signal(signal)
    valuation_date = _signal_date(signal)
    state = load_state(Path(state_path), Path(equity_path), initial_equity=initial_equity)

    # Existing holdings must also be priced when the signal removes a ticker.
    held_tickers = set(_position_quantities(state))
    prices, price_dates = load_prices(set(weights) | held_tickers, valuation_date, data_dir=Path(data_dir))
    equity_before, invested_before = _portfolio_equity(state, prices)
    signal_key = _signal_key(signal, weights)
    repeated_signal = str(state.get("last_signal_key") or "") == signal_key

    cfg = {
        "minimum_order_notional": _env_float(
            "FRACTIONAL_SHADOW_MIN_ORDER_NOTIONAL", DEFAULT_MIN_ORDER_NOTIONAL
        ),
        "cash_buffer_pct": _env_float("FRACTIONAL_SHADOW_CASH_BUFFER_PCT", DEFAULT_CASH_BUFFER_PCT),
        "cash_buffer_dollars": _env_float(
            "FRACTIONAL_SHADOW_CASH_BUFFER_DOLLARS", DEFAULT_CASH_BUFFER_DOLLARS
        ),
        "slippage_bps": _env_float("FRACTIONAL_SHADOW_SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS),
        "etf_drift": _env_float("FRACTIONAL_SHADOW_ETF_DRIFT", DEFAULT_ETF_DRIFT),
        "stock_drift": _env_float("FRACTIONAL_SHADOW_STOCK_DRIFT", DEFAULT_STOCK_DRIFT),
        "sec_sell_fee_bps": _env_float(
            "FRACTIONAL_SHADOW_SEC_SELL_FEE_BPS", DEFAULT_SEC_SELL_FEE_BPS
        ),
        "taf_sell_fee_per_share": _env_float(
            "FRACTIONAL_SHADOW_TAF_SELL_FEE_PER_SHARE", DEFAULT_TAF_SELL_FEE_PER_SHARE
        ),
        "cat_fee_per_share": _env_float(
            "FRACTIONAL_SHADOW_CAT_FEE_PER_SHARE", DEFAULT_CAT_FEE_PER_SHARE
        ),
    }

    cash_buffer = max(cfg["cash_buffer_dollars"], equity_before * cfg["cash_buffer_pct"])
    investable_equity = max(0.0, equity_before - cash_buffer)
    current_qty = _position_quantities(state)
    planned: list[dict[str, Any]] = []
    if not repeated_signal:
        for ticker in sorted(set(weights) | set(current_qty)):
            price = prices[ticker]
            target_value = weights.get(ticker, 0.0) * investable_equity
            target_qty = _round_quantity(target_value / price)
            delta_qty = target_qty - current_qty.get(ticker, 0.0)
            current_weight = current_qty.get(ticker, 0.0) * price / equity_before if equity_before else 0.0
            drift = abs(weights.get(ticker, 0.0) - current_weight)
            threshold = cfg["etf_drift"] if ticker in ETF_TICKERS else cfg["stock_drift"]
            notional = abs(delta_qty) * price
            if abs(delta_qty) <= 10 ** (-QUANTITY_DECIMALS) or drift < threshold:
                continue
            if notional + 1e-9 < cfg["minimum_order_notional"]:
                continue
            planned.append({
                "ticker": ticker,
                "side": "buy" if delta_qty > 0 else "sell",
                "quantity": _round_quantity(abs(delta_qty)),
                "reference_price": price,
                "target_weight": weights.get(ticker, 0.0),
                "current_weight_before": current_weight,
                "drift_before": drift,
            })

    # Sells release cash first. Larger trades go first for deterministic output.
    planned.sort(key=lambda item: (0 if item["side"] == "sell" else 1, -item["quantity"] * item["reference_price"]))
    order_rows: list[dict[str, Any]] = []
    raw_regulatory_fees = 0.0
    slippage_cost = 0.0
    cash = _as_float(state.get("cash"))
    positions = dict(state.get("positions") or {})
    run_at = _utc_now()

    for order in planned:
        ticker = order["ticker"]
        side = order["side"]
        reference = float(order["reference_price"])
        direction = 1.0 if side == "buy" else -1.0
        fill_price = reference * (1.0 + direction * cfg["slippage_bps"] / 10_000.0)
        quantity = float(order["quantity"])

        if side == "buy":
            spendable = max(0.0, cash - cash_buffer)
            affordable = _round_quantity(spendable / fill_price)
            quantity = min(quantity, affordable)
            if quantity * reference + 1e-9 < cfg["minimum_order_notional"]:
                continue
        else:
            quantity = min(quantity, _as_float((positions.get(ticker) or {}).get("quantity")))
            if quantity <= 0:
                continue

        trade_value = quantity * fill_price
        fee_raw = _regulatory_fee_raw(side, quantity, trade_value, cfg)
        raw_regulatory_fees += fee_raw
        order_slippage = quantity * abs(fill_price - reference)
        slippage_cost += order_slippage
        old = positions.get(ticker) or {"quantity": 0.0, "avg_cost": 0.0}
        old_qty = _as_float(old.get("quantity"))
        old_avg = _as_float(old.get("avg_cost"))

        if side == "buy":
            new_qty = _round_quantity(old_qty + quantity)
            avg_cost = ((old_qty * old_avg) + trade_value) / new_qty if new_qty else 0.0
            positions[ticker] = {"quantity": new_qty, "avg_cost": round(avg_cost, 10)}
            cash -= trade_value
        else:
            new_qty = _round_quantity(old_qty - quantity)
            cash += trade_value
            if new_qty <= 10 ** (-QUANTITY_DECIMALS):
                positions.pop(ticker, None)
            else:
                positions[ticker] = {"quantity": new_qty, "avg_cost": old_avg}

        order_rows.append({
            "order_id": _order_id(signal_key, ticker, side),
            "simulated_at": run_at,
            "valuation_date": valuation_date,
            "signal_key": signal_key,
            "mode": "fractional_shadow_only",
            "ticker": ticker,
            "side": side,
            "order_type": "market",
            "time_in_force": "day",
            "quantity": round(quantity, QUANTITY_DECIMALS),
            "notional": round(trade_value, 6),
            "reference_price": round(reference, 6),
            "simulated_fill_price": round(fill_price, 6),
            "slippage_bps": cfg["slippage_bps"],
            "slippage_cost": round(order_slippage, 8),
            "regulatory_fee_raw": round(fee_raw, 8),
            "commission": 0.0,
            "target_weight": round(order["target_weight"], 10),
            "drift_before": round(order["drift_before"], 10),
            "broker_order_submitted": False,
        })

    # Alpaca accrues regulatory charges during the day, then rounds the daily
    # total to cents. Deduct it once so many tiny orders do not overstate fees.
    regulatory_fees = _ceil_cents(raw_regulatory_fees)
    cash -= regulatory_fees
    if cash < -0.005:
        raise RuntimeError(f"Fractional shadow cash became negative after fees: {cash:.6f}")
    cash = max(0.0, cash)

    state["cash"] = round(cash, 10)
    state["positions"] = positions
    state["last_signal_key"] = signal_key
    state["last_valuation_date"] = valuation_date
    state["last_run_at"] = run_at
    state["last_price_dates"] = price_dates
    state["cumulative_slippage_cost"] = round(
        _as_float(state.get("cumulative_slippage_cost")) + slippage_cost, 10
    )
    state["cumulative_regulatory_fees"] = round(
        _as_float(state.get("cumulative_regulatory_fees")) + regulatory_fees, 10
    )
    atomic_write_json(state, state_path)
    _append_orders(Path(orders_path), order_rows)

    equity_after, invested_after = _portfolio_equity(state, prices)
    actual_weights = {
        ticker: round(_as_float(item.get("quantity")) * prices[ticker] / equity_after, 10)
        for ticker, item in sorted(positions.items())
        if equity_after > 0 and ticker in prices
    }
    target_gaps = {
        ticker: round(actual_weights.get(ticker, 0.0) - weights.get(ticker, 0.0), 10)
        for ticker in sorted(set(weights) | set(actual_weights))
    }
    max_target_gap = max((abs(value) for value in target_gaps.values()), default=0.0)

    equity_history = _read_csv(Path(equity_path))
    previous_equity = None
    if not equity_history.empty and "valuation_date" in equity_history.columns:
        older = equity_history[equity_history["valuation_date"].astype(str) < valuation_date]
        if not older.empty:
            previous_equity = _as_float(older.iloc[-1].get("equity"))
    daily_return = (equity_after / previous_equity - 1.0) if previous_equity else 0.0
    equity_row = {
        "valuation_date": valuation_date,
        "timestamp": run_at,
        "mode": "fractional_shadow_only",
        "initial_equity": round(_as_float(state.get("initial_equity")), 2),
        "equity": round(equity_after, 6),
        "cash": round(cash, 6),
        "invested_value": round(invested_after, 6),
        "cash_weight": round(cash / equity_after, 10) if equity_after else 0.0,
        "gross_exposure": round(invested_after / equity_after, 10) if equity_after else 0.0,
        "daily_return_pct": round(daily_return * 100.0, 6),
        "total_return_pct": round((equity_after / _as_float(state.get("initial_equity")) - 1.0) * 100.0, 6),
        "orders_today": len(order_rows),
        "slippage_cost_today": round(slippage_cost, 8),
        "regulatory_fees_today": round(regulatory_fees, 8),
        "cumulative_slippage_cost": state["cumulative_slippage_cost"],
        "cumulative_regulatory_fees": state["cumulative_regulatory_fees"],
        "max_target_weight_gap": round(max_target_gap, 10),
        "signal_key": signal_key,
        "signal_predicted_at": signal.get("predicted_at"),
        "live_config_hash": signal.get("live_config_hash"),
        "paper_ready": True,
        "broker_orders_submitted": False,
    }
    _append_equity(Path(equity_path), equity_row)

    report = {
        "schema_version": 1,
        "generated_at": run_at,
        "status": "ok",
        "mode": "fractional_shadow_only",
        "capital": {
            "initial_equity": round(_as_float(state.get("initial_equity")), 2),
            "current_equity": round(equity_after, 6),
            "cash": round(cash, 6),
            "invested_value": round(invested_after, 6),
        },
        "signal": {
            "signal_key": signal_key,
            "predicted_at": signal.get("predicted_at"),
            "valuation_date": valuation_date,
            "live_config_hash": signal.get("live_config_hash"),
            "repeated_signal": repeated_signal,
        },
        "execution": {
            "orders_simulated": len(order_rows),
            "broker_orders_submitted": False,
            "minimum_order_notional": cfg["minimum_order_notional"],
            "quantity_precision_decimals": QUANTITY_DECIMALS,
            "order_type": "market",
            "time_in_force": "day",
            "slippage_bps": cfg["slippage_bps"],
            "slippage_cost_today": round(slippage_cost, 8),
            "regulatory_fees_today": round(regulatory_fees, 8),
            "regulatory_fees_raw_today": round(raw_regulatory_fees, 8),
            "commission_today": 0.0,
            "fee_model": {
                "sec_sell_fee_bps": cfg["sec_sell_fee_bps"],
                "taf_sell_fee_per_share": cfg["taf_sell_fee_per_share"],
                "cat_fee_per_share": cfg["cat_fee_per_share"],
                "daily_rounding": "ceil_to_cent",
            },
        },
        "allocation": {
            "target_weights": weights,
            "actual_weights": actual_weights,
            "target_weight_gaps": target_gaps,
            "max_absolute_target_weight_gap": round(max_target_gap, 10),
            "cash_weight": round(cash / equity_after, 10) if equity_after else 0.0,
            "positions": positions,
        },
        "data": {"price_dates": price_dates, "no_lookahead": True},
        "safety": {
            "shadow_only": True,
            "real_capital_approved": False,
            "broker_credentials_required": False,
            "broker_fractionability_verified": False,
        },
        "promotion_blockers": [
            "shadow_evidence_epoch_incomplete",
            "broker_fractionability_not_live_verified",
            "fractional_protective_stop_path_not_implemented",
            "real_capital_not_approved",
        ],
        "sources": {
            "active_signal": str(signal_path),
            "state": str(state_path),
            "orders": str(orders_path),
            "equity": str(equity_path),
        },
    }
    atomic_write_json(report, report_path)
    return report


def main() -> int:
    """Run the safe fractional tracker from a terminal or GitHub Actions."""
    parser = argparse.ArgumentParser(description="Track a broker-free $400 fractional shadow account.")
    parser.add_argument("--signal-path", type=Path, default=SIGNAL_PATH)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--orders-path", type=Path, default=ORDERS_PATH)
    parser.add_argument("--equity-path", type=Path, default=EQUITY_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=_env_float("FRACTIONAL_SHADOW_INITIAL_EQUITY", DEFAULT_INITIAL_EQUITY, minimum=0.01),
    )
    args = parser.parse_args()
    report = run_fractional_shadow(
        signal_path=args.signal_path,
        state_path=args.state_path,
        orders_path=args.orders_path,
        equity_path=args.equity_path,
        report_path=args.report_path,
        data_dir=args.data_dir,
        initial_equity=args.initial_equity,
    )
    capital = report["capital"]
    allocation = report["allocation"]
    print(
        "Fractional shadow ready: "
        f"equity=${capital['current_equity']:.2f}, cash=${capital['cash']:.2f}, "
        f"orders={report['execution']['orders_simulated']}, "
        f"max_gap={allocation['max_absolute_target_weight_gap']:.2%}"
    )
    print("Safety: shadow only; zero broker orders submitted; real capital remains unapproved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
