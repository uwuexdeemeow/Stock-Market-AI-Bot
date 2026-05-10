from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

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
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
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
        exit_date = pd.Timestamp(row.get("exit_date", row.get("entry_date")))
        return exit_date, float(row.get("exit_future_close", row.get("open_next", 0.0))), 0, "missing_history"

    entry_date = pd.Timestamp(row.get("entry_date"))
    signal = str(row.get("signal", "LONG")).upper()
    entry_price = float(row.get("open_next", row.get("entry_price", 0.0)) or 0.0)
    entry_pos = _nearest_loc(hist.index, entry_date)
    exit_pos = min(entry_pos + int(rule.exit_horizon_days), len(hist) - 1)
    future = hist.iloc[entry_pos: exit_pos + 1]
    if future.empty:
        px = float(hist["Close"].iloc[min(entry_pos, len(hist) - 1)])
        return hist.index[min(entry_pos, len(hist) - 1)], px, 0, "no_future_rows"

    stop_px = None
    take_px = None
    if entry_price > 0:
        if signal == "LONG":
            stop_px = entry_price * (1.0 - max(rule.stop_loss_pct, 0.0))
            take_px = entry_price * (1.0 + max(rule.take_profit_pct, 0.0))
        else:
            stop_px = entry_price * (1.0 + max(rule.stop_loss_pct, 0.0))
            take_px = entry_price * (1.0 - max(rule.take_profit_pct, 0.0))

    for dt, bar in future.iloc[1:].iterrows():
        high = float(bar.get("High", bar.get("Close", np.nan)))
        low = float(bar.get("Low", bar.get("Close", np.nan)))
        close = float(bar.get("Close", entry_price))
        if signal == "LONG":
            if stop_px is not None and low <= stop_px:
                return pd.Timestamp(dt), float(stop_px), int((pd.Timestamp(dt) - entry_date).days), "stop_loss"
            if take_px is not None and high >= take_px:
                return pd.Timestamp(dt), float(take_px), int((pd.Timestamp(dt) - entry_date).days), "take_profit"
        else:
            if stop_px is not None and high >= stop_px:
                return pd.Timestamp(dt), float(stop_px), int((pd.Timestamp(dt) - entry_date).days), "stop_loss"
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
    out.to_csv(TRADE_RULE_REPORT, index=False)
    return out
