from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json
from settings import DEFAULT_FIXED_CONFIDENCE_THRESHOLD, MODEL_DIR, RETURN_HORIZON_DAYS

TRADE_RULE_REPORT = os.path.join(MODEL_DIR, "trade_rule_report.csv")


@dataclass
class TradeRule:
    ticker: str
    confidence_threshold: float = DEFAULT_FIXED_CONFIDENCE_THRESHOLD
    min_expected_return: float = 0.25
    allowed_qualities: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
    exit_horizon_days: int = RETURN_HORIZON_DAYS
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.08
    max_position_pct: float = 0.15
    allow_shorts: bool = False
    avoid_bear_regime: bool = False
    optimized_at: str | None = None
    optimization_score: float | None = None

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["allowed_qualities"] = list(self.allowed_qualities)
        return data


def default_trade_rule(ticker: str) -> TradeRule:
    return TradeRule(ticker=ticker.upper())


def trade_rule_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.upper()}_trade_rules.json")


def load_trade_rule(ticker: str) -> TradeRule:
    path = trade_rule_path(ticker)
    if not os.path.exists(path):
        return default_trade_rule(ticker)
    try:
        with open(path) as f:
            data = json.load(f)
        data["allowed_qualities"] = tuple(data.get("allowed_qualities", ("MEDIUM", "HIGH")))
        return TradeRule(**{**default_trade_rule(ticker).to_json_dict(), **data})
    except Exception:
        return default_trade_rule(ticker)


def save_trade_rule(rule: TradeRule) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = trade_rule_path(rule.ticker)
    data = rule.to_json_dict()
    data["optimized_at"] = data.get("optimized_at") or datetime.now().isoformat(timespec="seconds")
    atomic_write_json(data, path)
    return path


def passes_trade_rule(row: pd.Series | dict, rule: TradeRule, mode: str = "long_only") -> tuple[bool, str | None]:
    signal = str(row.get("signal", "")).upper()
    confidence = float(row.get("confidence", 0.0) or 0.0)
    expected_return = float(row.get("expected_return", 0.0) or 0.0)

    if confidence < rule.confidence_threshold:
        return False, "confidence_below_rule"
    if signal == "LONG" and expected_return < rule.min_expected_return:
        return False, "expected_return_below_rule"
    if signal == "SHORT" and abs(expected_return) < rule.min_expected_return:
        return False, "expected_return_below_rule"
    if signal == "SHORT" and (mode == "long_only" or not rule.allow_shorts):
        return False, "short_not_allowed"
    return True, None


def _nearest_loc(index: pd.DatetimeIndex, dt) -> int:
    ts = pd.Timestamp(dt)
    pos = index.searchsorted(ts)
    if pos >= len(index):
        return len(index) - 1
    return int(pos)


def resolve_rule_exit(hist: pd.DataFrame, row: pd.Series | dict, rule: TradeRule) -> tuple[pd.Timestamp, float, int, str]:
    """Return exit date/price using fixed horizon plus optional stop/take-profit."""
    hist = hist.copy()
    hist.index = pd.DatetimeIndex(hist.index)
    if hist.empty:
        raise ValueError("Price history required for rule exits")
    hist = hist.sort_index()
    if hist.index.has_duplicates:
        raise ValueError("Duplicate price dates in rule exit history")

    entry_date = pd.Timestamp(row.get("entry_date"))
    signal = str(row.get("signal", "LONG")).upper()
    entry_price = float(row.get("open_next", row.get("entry_price", 0.0)) or 0.0)
    if not np.isfinite(entry_price) or entry_price <= 0 or signal not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid entry price/direction at {entry_date}")
    if entry_date not in hist.index:
        raise ValueError(f"Missing exact entry session: {entry_date}")
    entry_pos = int(hist.index.get_loc(entry_date))
    exit_pos = entry_pos + int(rule.exit_horizon_days)
    if exit_pos >= len(hist):
        raise ValueError("Incomplete rule-exit horizon")
    future = hist.iloc[entry_pos: exit_pos + 1]
    if future.empty:
        px = float(hist["Close"].iloc[min(entry_pos, len(hist) - 1)])
        return hist.index[min(entry_pos, len(hist) - 1)], px, 0, "no_future_rows"

    stop_px = None
    take_px = None
    if entry_price > 0:
        if signal == "LONG":
            stop_px = entry_price * (1.0 - rule.stop_loss_pct) if rule.stop_loss_pct > 0 else None
            take_px = entry_price * (1.0 + rule.take_profit_pct) if rule.take_profit_pct > 0 else None
        else:
            stop_px = entry_price * (1.0 + rule.stop_loss_pct) if rule.stop_loss_pct > 0 else None
            take_px = entry_price * (1.0 - rule.take_profit_pct) if rule.take_profit_pct > 0 else None

    # Entry is at the open, so the entry day's high/low can trigger an exit.
    for dt, bar in future.iterrows():
        open_price = float(bar.get("Open", np.nan))
        high = float(bar.get("High", np.nan))
        low = float(bar.get("Low", np.nan))
        close = float(bar.get("Close", np.nan))
        if not all(np.isfinite(x) and x > 0 for x in (open_price, high, low, close)):
            raise ValueError(f"Invalid OHLC on {dt}")
        # The open happens first on both possible daily paths. A profit gap
        # already reaches its limit before any later low/high can hit a stop.
        if take_px is not None and ((signal == "LONG" and open_price >= take_px) or (signal == "SHORT" and open_price <= take_px)):
            return pd.Timestamp(dt), float(take_px), int((pd.Timestamp(dt) - entry_date).days), "take_profit"
        if signal == "LONG":
            if stop_px is not None and low <= stop_px:
                return pd.Timestamp(dt), float(min(open_price, stop_px) if signal == "LONG" else max(open_price, stop_px)), int((pd.Timestamp(dt) - entry_date).days), "stop_loss"
            if take_px is not None and high >= take_px:
                return pd.Timestamp(dt), float(take_px), int((pd.Timestamp(dt) - entry_date).days), "take_profit"
        else:
            if stop_px is not None and high >= stop_px:
                return pd.Timestamp(dt), float(min(open_price, stop_px) if signal == "LONG" else max(open_price, stop_px)), int((pd.Timestamp(dt) - entry_date).days), "stop_loss"
            if take_px is not None and low <= take_px:
                return pd.Timestamp(dt), float(take_px), int((pd.Timestamp(dt) - entry_date).days), "take_profit"
        if pd.Timestamp(dt) == future.index[-1]:
            return pd.Timestamp(dt), close, int((pd.Timestamp(dt) - entry_date).days), "time_exit"

    last = future.iloc[-1]
    dt = pd.Timestamp(future.index[-1])
    return dt, float(last.get("Close", entry_price)), int((dt - entry_date).days), "time_exit"


def append_rule_report(rows: Iterable[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(list(rows))
    if new_df.empty:
        return new_df
    if os.path.exists(TRADE_RULE_REPORT):
        try:
            old_df = pd.read_csv(TRADE_RULE_REPORT)
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()
    if not old_df.empty and "ticker" in old_df.columns:
        old_df = old_df[~old_df["ticker"].astype(str).str.upper().isin(new_df["ticker"].astype(str).str.upper())]
    out = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    out = out.sort_values(["approved_candidate", "score"], ascending=[False, False])
    atomic_write_csv(out, TRADE_RULE_REPORT, index=False)
    return out
