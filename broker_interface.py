"""
broker_interface.py — Abstract broker so the rest of the system isn't married
to Moomoo.

PLAIN ENGLISH:
Today we paper-trade via Moomoo. Tomorrow we might use Alpaca, IBKR, or a
crypto venue. If code that needs to "place an order" imports Moomoo
directly, we'll be rewriting everything on the switch. The fix is to
define a small abstract interface (`Broker`) here and have each concrete
adapter (MoomooBroker, AlpacaBroker, DryRunBroker) implement it.

The rest of the pipeline only ever talks to `Broker`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass
class Order:
    ticker: str
    side: Side
    quantity: int
    type: OrderType = "market"
    limit_price: float | None = None
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
    def place_order(self, order: Order) -> str: ...  # returns order_id
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...
    @abstractmethod
    def recent_fills(self, since_iso: str) -> list[Fill]: ...


class DryRunBroker(Broker):
    """For testing — accepts everything, stores nothing real."""
    def __init__(self, starting_equity: float = 100_000.0):
        self._equity = starting_equity
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._fills: list[Fill] = []

    def get_equity(self) -> float:
        return self._equity

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def place_order(self, order: Order) -> str:
        oid = f"dry-{len(self._orders)}"
        self._orders.append(order)
        return oid

    def cancel_order(self, order_id: str) -> bool:
        return True

    def recent_fills(self, since_iso: str) -> list[Fill]:
        return list(self._fills)
