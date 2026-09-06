"""Daily shares/cash simulation and exact recorded-event accounting; no broker I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np
import pandas as pd

from execution_model import realistic_fill_price, commission
from settings import SLIPPAGE_BASE_PCT
from paper_policy import PaperPolicy, rebalance_deltas, cash_reserve

LEDGER_VERSION = "daily-ledger-v1"


@dataclass
class LedgerResult:
    events: pd.DataFrame
    holdings: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict
    data_quality: list[dict] = field(default_factory=list)


def daily_metrics(equity: pd.Series) -> dict:
    """Use daily returns and elapsed years; expose frequency instead of guessing."""
    if equity.empty or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Daily equity must be finite and positive")
    returns = equity.pct_change(fill_method=None).dropna()
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    growth = float(equity.iloc[-1] / equity.iloc[0])
    return {"total_return_pct": (growth - 1) * 100,
            "cagr_pct": (growth ** (1 / years) - 1) * 100 if years > 0 else None,
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else None,
            "max_drawdown_pct": float((equity / equity.cummax() - 1).min() * 100),
            "valuation_frequency": "daily_close", "intraday_drawdown_measured": False,
            "ledger_version": LEDGER_VERSION}


class Account:
    """All money/share changes go through one event recorder."""
    def __init__(self, cash: float, holdings: dict | None = None, version: str = LEDGER_VERSION):
        if not math.isfinite(cash) or cash < 0:
            raise ValueError("Opening cash is required and cannot be negative")
        self.cash = float(cash)
        self.shares = dict(holdings or {})
        if any(not math.isfinite(float(q)) or q < 0 for q in self.shares.values()):
            raise ValueError("Opening holdings must be finite and nonnegative")
        self.events = []
        self.seen = set()
        self.pending = {}
        self.receivables = {}
        self.version = version

    def record(self, timestamp, kind, source, **fields):
        self.events.append({"timestamp": pd.Timestamp(timestamp), "kind": kind, "source": source,
                            "strategy_version": self.version, "decision_id": fields.pop("decision_id", ""),
                            "order_id": fields.pop("order_id", ""), **fields})

    def fill(self, timestamp, ticker, quantity, price, fee, *, event_id, source, reference=None, decision_id="", order_id=""):
        # Fill quantity is signed: purchases consume cash; sales release it.
        if event_id in self.seen:
            raise ValueError(f"Duplicate fill event: {event_id}")
        if not all(math.isfinite(float(x)) for x in (quantity, price, fee)) or price <= 0 or fee < 0 or quantity == 0:
            raise ValueError(f"Invalid fill {event_id}")
        next_shares = self.shares.get(ticker, 0.) + quantity
        next_cash = self.cash - quantity * price - fee
        if next_shares < -1e-8 or next_cash < -.005:
            raise ValueError(f"Fill violates long-only cash/share conservation: {event_id}")
        self.cash = next_cash
        self.shares[ticker] = max(0., next_shares)
        self.seen.add(event_id)
        if order_id in self.pending:
            self.pending[order_id] = max(0., self.pending[order_id] - abs(quantity))
        self.record(timestamp, "fill", source, ticker=ticker, quantity=quantity, price=price, fee=fee,
                    event_id=event_id, decision_id=decision_id, order_id=order_id, cash=self.cash,
                    execution_cost=quantity * (price - reference) if reference is not None else None)

    def mark(self, prices: dict) -> float:
        total = self.cash + sum(self.receivables.values())
        for ticker, quantity in self.shares.items():
            if quantity:
                price = prices.get(ticker)
                if price is None or not math.isfinite(price) or price <= 0:
                    raise ValueError(f"Missing valuation price for {ticker}")
                total += quantity * price
        return total

    def action(self, timestamp, event: dict):
        key = str(event["event_id"])
        if key in self.seen:
            raise ValueError(f"Duplicate corporate action: {key}")
        ticker = event["ticker"]
        amount = float(event["value"])
        if not math.isfinite(amount) or amount < 0 or (event["kind"] == "split" and amount == 0):
            raise ValueError(f"Invalid corporate action: {key}")
        if event["kind"] == "split":
            self.shares[ticker] = self.shares.get(ticker, 0) * amount
        elif event["kind"] == "dividend":
            # Dividend events supply documented entitlement, not payment-day holdings.
            entitlement = self.receivables.pop(str(event.get("entitlement_id", "")), None)
            self.cash += entitlement if entitlement is not None else float(event["entitled_shares"]) * amount
        elif event["kind"] == "symbol_change":
            new_ticker = str(event["new_ticker"])
            self.shares[new_ticker] = self.shares.get(new_ticker, 0.) + self.shares.pop(ticker, 0.)
        elif event["kind"] == "cash_liquidation":
            self.cash += self.shares.pop(ticker, 0.) * amount
        else:
            raise ValueError(f"Unsupported corporate action: {event['kind']}")
        self.seen.add(key)
        self.record(timestamp, event["kind"], event["source"], ticker=ticker, value=amount, event_id=key, cash=self.cash)


def trailing_outcome(bar: pd.Series, watermark: float, stop_pct: float) -> tuple[float | None, float, bool]:
    """Try both daily paths; choose the smaller exit/closing value for a long."""
    outcomes = []
    for path in (("Open", "High", "Low", "Close"), ("Open", "Low", "High", "Close")):
        high = watermark
        fill = None
        for field in path:
            price = float(bar[field])
            if price <= high * (1 - stop_pct):
                fill = price if field == "Open" else high * (1 - stop_pct)
                break
            high = max(high, price)
        outcomes.append((fill, high))
    chosen = min(outcomes, key=lambda result: result[0] if result[0] is not None else float(bar["Close"]))
    return chosen[0], chosen[1], outcomes[0] != outcomes[1]


def simulate_daily(bars: pd.DataFrame, target_at: Callable, *, start, end, initial_cash=100000.,
                   policy=PaperPolicy(), actions: pd.DataFrame | None = None, provenance: dict,
                   cost_stress=1., base_slippage_pct=SLIPPAGE_BASE_PCT, terminal_liquidation=True) -> LedgerResult:
    """Choose targets at yesterday's close, fill at today's open, mark every close."""
    from core_satellite_alpha import _nyse_sessions, _session_offset
    if provenance.get("adjustment_mode") != "raw_ohlcv" or provenance.get("actions_verified") is not True:
        raise ValueError("Verified raw prices and corporate-action coverage are required")
    if not math.isfinite(cost_stress) or not math.isfinite(base_slippage_pct) or cost_stress < 0 or base_slippage_pct < 0:
        raise ValueError("Costs cannot be negative")
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    if bars.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate daily price rows")
    prices = bars.set_index(["date", "ticker"]).sort_index()
    sessions = _nyse_sessions(start, end)
    if sessions.empty:
        raise ValueError("No evaluation sessions")
    account = Account(initial_cash)
    high_water = {}
    peak = initial_cash
    halted_at = None
    liquidate_next = False
    events = actions.to_dict("records") if actions is not None else []
    # Split ratios take effect before that session's per-share dividend rights.
    events.sort(key=lambda event: (event["kind"] == "dividend", pd.Timestamp(event["date"]), str(event["event_id"])))
    totals = {"turnover": 0., "fees": 0., "execution_cost": 0.}
    daily = [{"date": _session_offset(sessions[0], -1), "equity": initial_cash, "cash": initial_cash, "gross_exposure": 0.}]
    held_rows = []

    def execute(date, ticker, quantity, reference, pre_equity, reason, decision):
        # ADV excludes today's volume: the future full-day volume is not known at the open.
        symbols = {ticker}
        for event in sorted(events, key=lambda event: pd.Timestamp(event["date"]), reverse=True):
            if event["kind"] == "symbol_change" and pd.Timestamp(event["date"]) <= date and event["new_ticker"] in symbols:
                symbols.add(event["ticker"])
        history = pd.concat([prices.xs(symbol, level="ticker") for symbol in symbols if symbol in prices.index.get_level_values("ticker")])
        history = history.loc[history.index < date].sort_index().tail(20)
        adv = float(history.Volume.mean()) if not history.empty else float("nan")
        if not math.isfinite(adv) or adv <= 0:
            raise ValueError(f"Missing past volume for {ticker} at {date}")
        side = "buy" if quantity > 0 else "sell"
        model_price = realistic_fill_price(reference, abs(quantity), adv, side, base_slippage_pct=base_slippage_pct)
        fill_price = reference + cost_stress * (model_price - reference)
        fee = commission(abs(quantity)) * cost_stress
        if quantity > 0:
            reserve = cash_reserve(pre_equity, policy.cash_buffer_pct, policy.cash_buffer_min)
            # Recalculate costs after downsizing so the impact uses actual shares.
            affordable = min(quantity, int(max(0., account.cash - reserve) / (fill_price + fee / quantity)))
            if affordable != quantity:
                if affordable <= 0 or affordable * reference < policy.minimum_trade:
                    return
                return execute(date, ticker, affordable, reference, pre_equity, reason, decision)
        event_id = f"{date.isoformat()}:{len(account.events)}"
        source = "terminal_close_convention" if reason == "terminal" else "daily_bar_stop_proxy" if reason == "trailing_stop" else "daily_open_proxy"
        account.fill(date, ticker, quantity, fill_price, fee, event_id=event_id, source=source,
                     reference=reference, decision_id=str(decision), order_id=event_id)
        account.events[-1]["reason"] = reason
        totals["turnover"] += abs(quantity * fill_price) / pre_equity
        totals["fees"] += fee
        totals["execution_cost"] += quantity * (fill_price - reference)
        if quantity > 0:
            high_water[ticker] = max(high_water.get(ticker, fill_price), fill_price)
        if not account.shares[ticker]:
            high_water.pop(ticker, None)

    for date in sessions:
        previous = _session_offset(date, -1)
        targets = dict(target_at(previous, dict(account.shares), daily[-1]["equity"]))
        for event in events:
            if event["kind"] == "dividend" and pd.Timestamp(event.get("ex_date")) == date:
                entitlement_id = str(event["event_id"]) + ":entitlement"
                account.receivables[entitlement_id] = account.shares.get(event["ticker"], 0.) * float(event["value"])
                event["entitlement_id"] = entitlement_id
                account.record(date, "dividend_entitlement", event["source"], event_id=entitlement_id,
                               ticker=event["ticker"], amount=account.receivables[entitlement_id])
            if sessions[0] <= pd.Timestamp(event["date"]) <= date and str(event["event_id"]) not in account.seen:
                if event["kind"] == "dividend" and "entitlement_id" not in event:
                    # A flat-start portfolio has no rights to a dividend whose
                    # ex-date preceded the evaluation window.
                    event["entitled_shares"] = 0.
                account.action(date, event)
                if event["kind"] == "split" and event["ticker"] in high_water:
                    high_water[event["ticker"]] /= float(event["value"])
                elif event["kind"] == "symbol_change":
                    if event["ticker"] in high_water:
                        high_water[event["new_ticker"]] = high_water.pop(event["ticker"])
                    if event["ticker"] in targets:
                        targets[event["new_ticker"]] = targets.get(event["new_ticker"], 0.) + targets.pop(event["ticker"])
                elif event["kind"] == "cash_liquidation":
                    high_water.pop(event["ticker"], None)
        if date not in prices.index.get_level_values("date"):
            raise ValueError(f"Missing market session: {date}")
        day = prices.xs(date, level="date")
        relevant = set(targets) | {t for t, q in account.shares.items() if q}
        for ticker in relevant:
            if ticker not in day.index or not np.isfinite(day.loc[ticker, ["Open", "High", "Low", "Close"]].astype(float)).all():
                raise ValueError(f"Missing required OHLC for {ticker} at {date}")
            if (day.loc[ticker, ["Open", "High", "Low", "Close"]].astype(float) <= 0).any():
                raise ValueError(f"Invalid required OHLC for {ticker} at {date}")
            bar = day.loc[ticker]
            if bar.Low > min(bar.Open, bar.Close) or bar.High < max(bar.Open, bar.Close) or bar.Low > bar.High:
                raise ValueError(f"Inconsistent required OHLC for {ticker} at {date}")
        opens = day.Open.to_dict()
        pre_equity = account.mark(opens)
        # A resting protective stop sees an adverse opening gap before the
        # daily rebalance can buy more shares. Do not buy back that same day.
        for ticker, quantity in list(account.shares.items()):
            if quantity and ticker not in policy.etfs and policy.trailing_stop > 0 and opens[ticker] <= high_water[ticker] * (1 - policy.trailing_stop):
                execute(date, ticker, -quantity, opens[ticker], pre_equity, "trailing_stop", previous)
                targets.pop(ticker, None)
        if not liquidate_next and halted_at is not None and date > halted_at and pre_equity / peak - 1 > -policy.drawdown_halt * policy.halt_recovery_ratio:
            halted_at = None
            liquidate_next = False
        if liquidate_next:
            for ticker, quantity in list(account.shares.items()):
                if quantity:
                    execute(date, ticker, -quantity, opens[ticker], pre_equity, "drawdown_halt", previous)
            liquidate_next = False
        if halted_at is None:
            for ticker, quantity in rebalance_deltas(account.shares, targets, opens, account.mark(opens), policy):
                execute(date, ticker, quantity, opens[ticker], pre_equity, "rebalance", previous)
        beginning_exposure = sum(q * opens[t] for t, q in account.shares.items() if q) / pre_equity
        for ticker, quantity in list(account.shares.items()):
            if quantity and ticker not in policy.etfs and policy.trailing_stop > 0:
                fill, high, ambiguous = trailing_outcome(day.loc[ticker], high_water[ticker], policy.trailing_stop)
                high_water[ticker] = high
                if fill is not None:
                    execute(date, ticker, -quantity, fill, pre_equity, "trailing_stop", previous)
                if ambiguous:
                    account.record(date, "ambiguous_bar", "conservative_two_path", ticker=ticker)
        if date == sessions[-1] and terminal_liquidation:
            for ticker, quantity in list(account.shares.items()):
                if quantity:
                    execute(date, ticker, -quantity, float(day.loc[ticker, "Close"]), account.mark(day.Close.to_dict()), "terminal", previous)
        equity = account.mark(day.Close.to_dict())
        peak = max(peak, equity)
        if halted_at is None and equity / peak - 1 < -policy.drawdown_halt:
            halted_at, liquidate_next = date, True
            account.record(date, "halt", "daily_close", drawdown=equity / peak - 1)
        daily.append({"date": date, "equity": equity, "cash": account.cash, "gross_exposure": beginning_exposure,
                      "halted": halted_at is not None})
        held_rows.extend({"date": date, "ticker": ticker, "shares": quantity, "close": float(day.loc[ticker, "Close"]),
                          "stop_watermark": high_water.get(ticker), "stop_price": high_water[ticker] * (1 - policy.trailing_stop) if ticker in high_water and ticker not in policy.etfs else None}
                         for ticker, quantity in account.shares.items() if quantity)
    daily_frame = pd.DataFrame(daily).set_index("date")
    metrics = daily_metrics(daily_frame.equity)
    metrics.update(turnover_pct=totals["turnover"] * 100, estimated_cost_pct=(totals["fees"] + totals["execution_cost"]) / initial_cash * 100,
                   fees=totals["fees"], execution_cost=totals["execution_cost"], execution_mode="daily_open_proxy",
                   terminal_liquidation=terminal_liquidation, paper_execution_parity=False, cost_stress=cost_stress,
                   dividend_receivable=sum(account.receivables.values()))
    return LedgerResult(pd.DataFrame(account.events), pd.DataFrame(held_rows), daily_frame, metrics)


def replay_events(events: pd.DataFrame, *, opening_cash: float | None, opening_holdings: dict | None,
                  marks: pd.DataFrame | None = None, expected_cash=None, expected_holdings=None) -> LedgerResult:
    """Apply documented fills once; never assume an opening balance or unknown fees."""
    gaps = []
    known_changes = {}
    if opening_cash is None or opening_holdings is None:
        gaps.append({"reason": "opening_balances_missing"})
    if expected_cash is None or expected_holdings is None:
        gaps.append({"reason": "closing_balances_missing"})
    if "kind" in events:
        for event in events.loc[events.kind == "fill"].to_dict("records"):
            quantity = event.get("quantity")
            if quantity is not None and math.isfinite(float(quantity)):
                ticker = str(event.get("ticker", ""))
                known_changes[ticker] = known_changes.get(ticker, 0.) + float(quantity)
            if pd.isna(event.get("fee")):
                gaps.append({"reason": "fee_missing", "event_id": event.get("event_id")})
    # Unknown closing balances do not prevent reconstructing a documented
    # opening account, but unknown opening values or fees do.
    if any(gap["reason"] != "closing_balances_missing" for gap in gaps):
        return LedgerResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                            {"reconciled": False, "ledger_version": LEDGER_VERSION, "known_share_changes": known_changes,
                             "known_changes_are_closing_balances": False}, gaps)
    account = Account(opening_cash, opening_holdings)
    if "timestamp" not in events:
        raise ValueError("Recorded events need timestamps")
    if pd.to_datetime(events.timestamp, utc=True, errors="coerce").isna().any():
        raise ValueError("Invalid recorded event timestamp")
    mark_rows, held_rows = [], []
    ordered = events.sort_values("timestamp", kind="stable").to_dict("records")
    # Merge daily close marks into the chronological event stream. This avoids
    # valuing old dates with the final account's holdings.
    if marks is not None:
        for timestamp, group in marks.groupby("timestamp"):
            ordered.append({"timestamp": timestamp, "kind": "valuation", "prices": dict(zip(group.ticker, group.price))})
    ordered.sort(key=lambda event: pd.Timestamp(event["timestamp"]))
    for event in ordered:
        kind = event["kind"]
        if kind == "fill":
            if pd.isna(event.get("fee")):
                gaps.append({"reason": "fee_missing", "event_id": event.get("event_id")})
                continue
            account.fill(event["timestamp"], event["ticker"], event["quantity"], event["price"], event["fee"],
                         event_id=event["event_id"], source="recorded_fill", order_id=event.get("order_id", ""), decision_id=event.get("decision_id", ""))
        elif kind in {"split", "dividend", "symbol_change", "cash_liquidation"}:
            account.action(event["timestamp"], event)
        elif kind in {"submitted", "canceled", "rejected", "expired"}:
            account.pending[event["order_id"]] = float(event["quantity"]) if kind == "submitted" else 0.
            account.record(event["timestamp"], kind, "broker", order_id=event["order_id"])
        elif kind == "valuation":
            mark_rows.append({"date": pd.Timestamp(event["timestamp"]), "equity": account.mark(event["prices"]), "cash": account.cash})
            held_rows.extend({"date": pd.Timestamp(event["timestamp"]), "ticker": ticker, "shares": quantity}
                             for ticker, quantity in account.shares.items())
        else:
            raise ValueError(f"Unknown recorded event kind: {kind}")
    if expected_cash is not None and expected_holdings is not None:
        if not math.isfinite(float(expected_cash)) or any(not math.isfinite(float(q)) or q < 0 for q in expected_holdings.values()):
            raise ValueError("Closing balances must be finite with nonnegative shares")
        if abs(account.cash - expected_cash) > .01:
            gaps.append({"reason": "cash_mismatch", "actual": account.cash, "expected": expected_cash})
        for ticker in set(account.shares) | set(expected_holdings):
            if abs(account.shares.get(ticker, 0) - expected_holdings.get(ticker, 0)) > 1e-8:
                gaps.append({"reason": "shares_mismatch", "ticker": ticker})
    marked = pd.DataFrame(mark_rows)
    metrics = {"reconciled": not gaps, "cash": account.cash, "holdings": account.shares,
               "pending_orders": account.pending, "ledger_version": LEDGER_VERSION, "execution_mode": "recorded_fills",
               "performance_window": "provided_valuation_marks_only" if len(marked) >= 2 else "unavailable"}
    if len(marked) >= 2:
        metrics.update(daily_metrics(marked.set_index("date").equity))
        metrics.update(performance_start=str(marked.date.iloc[0]), performance_end=str(marked.date.iloc[-1]))
    return LedgerResult(pd.DataFrame(account.events), pd.DataFrame(held_rows), marked, metrics, gaps)
