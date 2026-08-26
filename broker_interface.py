"""
broker_interface.py - Shared broker shapes used by paper-trading adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "trailing_stop"]


@dataclass
class Order:
    ticker: str
    side: Side
    quantity: int
    type: OrderType = "market"
    limit_price: float | None = None
    stop_price: float | None = None  # for a fixed stop-loss order
    trail_percent: float | None = None  # for trailing_stop orders (e.g., 0.06 = 6%)
    client_id: str | None = None


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float


@dataclass
class Fill:
    order_id: str
    ticker: str
    side: Side
    quantity: int
    price: float
    commission: float


class Broker(ABC):
    @abstractmethod
    def get_equity(self) -> float: ...
    @abstractmethod
    def get_positions(self) -> list[Position]: ...
    @abstractmethod
    def place_order(self, order: Order) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...
    @abstractmethod
    def recent_fills(self, since_iso: str) -> list[Fill]: ...


class DryRunBroker(Broker):
    """Simple in-memory paper broker for local testing.

    By default market orders are immediately filled at the last known mark for
    the ticker. Set marks through set_mark_price() before placing orders.
    """

    def __init__(self, starting_equity: float = 100_000.0, commission_per_share: float = 0.005):
        self._cash = float(starting_equity)
        self._commission_per_share = float(commission_per_share)
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[tuple[str, Fill]] = []
        self._marks: dict[str, float] = {}

    def set_mark_price(self, ticker: str, price: float) -> None:
        self._marks[str(ticker).upper()] = float(price)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _commission(self, qty: int) -> float:
        return abs(int(qty)) * self._commission_per_share

    def get_equity(self) -> float:
        mtm = self._cash
        for pos in self._positions.values():
            mark = self._marks.get(pos.ticker, pos.avg_price)
            mtm += pos.quantity * mark
        return float(mtm)

    def get_positions(self) -> list[Position]:
        return [Position(ticker=p.ticker, quantity=p.quantity, avg_price=p.avg_price) for p in self._positions.values() if p.quantity != 0]

    def _apply_position_fill(self, ticker: str, signed_qty: int, price: float) -> None:
        prev = self._positions.get(ticker, Position(ticker=ticker, quantity=0, avg_price=0.0))
        new_qty = prev.quantity + signed_qty

        if new_qty == 0:
            self._positions.pop(ticker, None)
            return

        if prev.quantity == 0:
            self._positions[ticker] = Position(ticker=ticker, quantity=new_qty, avg_price=price)
            return

        if (prev.quantity > 0) == (signed_qty > 0):
            prev_abs = abs(prev.quantity)
            fill_abs = abs(signed_qty)
            total_cost = prev.avg_price * prev_abs + price * fill_abs
            self._positions[ticker] = Position(ticker=ticker, quantity=new_qty, avg_price=total_cost / abs(new_qty))
            return

        if (prev.quantity > 0) == (new_qty > 0):
            self._positions[ticker] = Position(ticker=ticker, quantity=new_qty, avg_price=prev.avg_price)
            return

        self._positions[ticker] = Position(ticker=ticker, quantity=new_qty, avg_price=price)

    def place_order(self, order: Order) -> str:
        if int(order.quantity) <= 0:
            raise ValueError("Order quantity must be positive")
        ticker = str(order.ticker).upper()
        px = float(order.limit_price if order.limit_price is not None else self._marks.get(ticker, 0.0))
        if px <= 0:
            raise ValueError(f"No executable price available for {ticker}; set a mark or limit_price first")

        oid = f"dry-{len(self._orders) + 1}"
        self._orders[oid] = Order(
            ticker=ticker,
            side=order.side,
            quantity=int(order.quantity),
            type=order.type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            trail_percent=order.trail_percent,
            client_id=order.client_id,
        )

        qty = int(order.quantity)
        signed = qty if order.side == "buy" else -qty
        fee = self._commission(qty)
        self._cash -= signed * px + fee

        self._apply_position_fill(ticker, signed, px)

        self._fills.append((self._now_iso(), Fill(order_id=oid, ticker=ticker, side=order.side, quantity=qty, price=px, commission=fee)))
        return oid

    def cancel_order(self, order_id: str) -> bool:
        return order_id in self._orders

    def recent_fills(self, since_iso: str) -> list[Fill]:
        try:
            cutoff = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        except Exception:
            cutoff = None
        out: list[Fill] = []
        for ts, fill in self._fills:
            if cutoff is None:
                out.append(fill)
                continue
            try:
                fill_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                out.append(fill)
                continue
            if fill_dt >= cutoff:
                out.append(fill)
        return out
