"""Small compatibility layer between this bot and Alpaca's current SDK.

PLAIN ENGLISH: The bot was written around Alpaca's retired ``REST`` client.
This file translates those familiar calls into ``alpaca-py`` request objects,
so the safety-critical trading code can migrate without a large rewrite.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

# The only trading host this project may talk to.  Alpaca's live-money host
# is api.alpaca.markets; it never passes the check below.
PAPER_TRADING_HOST = "paper-api.alpaca.markets"


def is_paper_endpoint(url: Any) -> bool:
    """True only when ``url`` is exactly Alpaca's paper trading host over HTTPS.

    PLAIN ENGLISH: the old check only asked "does the text contain
    paper-api.alpaca.markets?".  A web address such as
    https://paper-api.alpaca.markets.example.com contains that text but points
    somewhere else, and the bot would have sent its keys there.  Here the
    address is taken apart and the host name itself must match.
    """
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme.lower() == "https"
        and (parts.hostname or "") == PAPER_TRADING_HOST
        and port in (None, 443)
        and not parts.username
        and not parts.password
        and parts.path in ("", "/")
        and not parts.query
        and not parts.fragment
    )


class _CompatObject:
    """Expose Alpaca enum fields as simple strings expected by old bot code."""

    def __init__(self, value: Any):
        self._value = value

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._value, name)
        return value.value if isinstance(value, Enum) else value


def _wrapped(value: Any) -> Any:
    """Wrap one Alpaca model while leaving plain values unchanged."""
    # Pydantic describes fields on the model class. Reading them from an
    # individual account/order object is deprecated in the current SDK.
    return _CompatObject(value) if hasattr(type(value), "model_fields") else value


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
        data_feed: str = "iex",
    ) -> None:
        # PLAIN ENGLISH: this project is paper-only.  Refuse anything that is
        # not exactly the paper host, and tell the SDK "paper" in plain code
        # (never from a setting), as Alpaca's paper-trading guide asks.
        if not is_paper_endpoint(base_url):
            raise ValueError("Refusing non-paper Alpaca endpoint; real-capital trading is not approved")
        # Imports stay here so non-broker research scripts can run without the
        # optional broker package being loaded.
        if trading_client is None or data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        self._trading = trading_client or TradingClient(  # type: ignore[name-defined]
            key_id, secret_key, paper=True
        )
        self._data = data_client or StockHistoricalDataClient(key_id, secret_key)  # type: ignore[name-defined]
        # PLAIN ENGLISH: Keep the requested exchange feed explicit. Alpaca can
        # otherwise silently choose a feed from the account subscription, which
        # makes a recorded spread impossible to audit later.
        self._data_feed = str(data_feed or "iex").strip().lower()

    def _request_feed(self) -> Any:
        """Translate the configured feed name into Alpaca's typed enum."""
        from alpaca.data.enums import DataFeed

        try:
            return DataFeed(self._data_feed)
        except ValueError as exc:
            raise ValueError(f"Unsupported Alpaca data feed: {self._data_feed}") from exc

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

    def get_order_by_client_id(self, client_order_id: str) -> Any:
        """Find an order by the ID the bot gave it (used after a timeout)."""
        return _wrapped(self._trading.get_order_by_client_id(client_order_id))

    def get_snapshot(self, symbol: str) -> Any:
        from alpaca.data.requests import StockSnapshotRequest

        result = self._data.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbol, feed=self._request_feed())
        )
        return result[str(symbol).upper()]

    def get_latest_trade(self, symbol: str) -> Any:
        from alpaca.data.requests import StockLatestTradeRequest

        result = self._data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self._request_feed())
        )
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
