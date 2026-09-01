"""Small compatibility layer between this bot and Alpaca's current SDK.

PLAIN ENGLISH: The bot was written around Alpaca's retired ``REST`` client.
This file translates those familiar calls into ``alpaca-py`` request objects,
so the safety-critical trading code can migrate without a large rewrite.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any


class _CompatObject:
    """Expose Alpaca enum fields as simple strings expected by old bot code."""

    def __init__(self, value: Any):
        self._value = value

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._value, name)
        return value.value if isinstance(value, Enum) else value


def _wrapped(value: Any) -> Any:
    """Wrap one Alpaca model while leaving plain values unchanged."""
    return _CompatObject(value) if hasattr(value, "model_fields") else value


class AlpacaPyRESTCompat:
    """Translate the legacy calls used by the bot into ``alpaca-py`` calls."""

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        *,
        trading_client: Any | None = None,
        data_client: Any | None = None,
    ) -> None:
        # Imports stay here so non-broker research scripts can run without the
        # optional broker package being loaded.
        paper = "paper-api.alpaca.markets" in str(base_url).lower()
        if trading_client is None or data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        self._trading = trading_client or TradingClient(  # type: ignore[name-defined]
            key_id, secret_key, paper=paper, url_override=base_url
        )
        self._data = data_client or StockHistoricalDataClient(key_id, secret_key)  # type: ignore[name-defined]

    def get_account(self) -> Any:
        return _wrapped(self._trading.get_account())

    def list_positions(self) -> list[Any]:
        return [_wrapped(item) for item in self._trading.get_all_positions()]

    def get_position(self, symbol: str) -> Any:
        return _wrapped(self._trading.get_open_position(symbol))

    def submit_order(self, **kwargs: Any) -> Any:
        """Build the correct typed Alpaca order request from legacy keywords."""
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopOrderRequest,
            TrailingStopOrderRequest,
        )

        order_type = str(kwargs.pop("type", "market")).lower()
        request_types = {
            "limit": LimitOrderRequest,
            "market": MarketOrderRequest,
            "stop": StopOrderRequest,
            "trailing_stop": TrailingStopOrderRequest,
        }
        if order_type not in request_types:
            raise ValueError(f"Unsupported Alpaca order type: {order_type}")
        request = request_types[order_type](**kwargs)
        return _wrapped(self._trading.submit_order(order_data=request))

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)

    def cancel_all_orders(self) -> Any:
        return self._trading.cancel_orders()

    def list_orders(self, **kwargs: Any) -> list[Any]:
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        values = dict(kwargs)
        status = str(values.pop("status", "open")).lower()
        direction = str(values.pop("direction", "desc")).lower()
        values["status"] = QueryOrderStatus(status)
        values["direction"] = Sort(direction)
        # Alpaca-py accepts datetimes rather than the old client's ISO strings.
        for key in ("after", "until"):
            if isinstance(values.get(key), str):
                values[key] = datetime.fromisoformat(values[key].replace("Z", "+00:00"))
        request = GetOrdersRequest(**values)
        return [_wrapped(item) for item in self._trading.get_orders(filter=request)]

    def get_clock(self) -> Any:
        return _wrapped(self._trading.get_clock())

    def get_order(self, order_id: str) -> Any:
        return _wrapped(self._trading.get_order_by_id(order_id))

    def get_snapshot(self, symbol: str) -> Any:
        from alpaca.data.requests import StockSnapshotRequest

        result = self._data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol))
        return result[str(symbol).upper()]

    def get_latest_trade(self, symbol: str) -> Any:
        from alpaca.data.requests import StockLatestTradeRequest

        result = self._data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return result[str(symbol).upper()]

    def get_bars(
        self,
        symbol: str,
        timeframe: Any,
        *,
        start: str | datetime,
        end: str | datetime,
        adjustment: str = "raw",
    ) -> Any:
        from alpaca.data.requests import StockBarsRequest

        def parsed(value: str | datetime) -> datetime:
            return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=parsed(start),
            end=parsed(end),
            adjustment=adjustment,
        )
        return self._data.get_stock_bars(request)


def minute_timeframe() -> Any:
    """Return Alpaca's one-minute bar marker without leaking SDK imports."""
    from alpaca.data.timeframe import TimeFrame

    return TimeFrame.Minute
