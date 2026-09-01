"""Tests for the current Alpaca SDK compatibility layer."""
from enum import Enum
from types import SimpleNamespace

from alpaca_sdk_adapter import AlpacaPyRESTCompat, _CompatObject


class ExampleStatus(Enum):
    FILLED = "filled"


def test_compat_object_turns_sdk_enum_into_plain_string():
    wrapped = _CompatObject(SimpleNamespace(status=ExampleStatus.FILLED))
    assert wrapped.status == "filled"


def test_adapter_routes_basic_account_position_and_cancel_calls():
    calls = []
    trading = SimpleNamespace(
        get_account=lambda: SimpleNamespace(equity="1000"),
        get_all_positions=lambda: [],
        cancel_order_by_id=lambda order_id: calls.append(order_id),
        cancel_orders=lambda: [],
    )
    adapter = AlpacaPyRESTCompat(
        "key", "secret", trading_client=trading, data_client=SimpleNamespace()
    )

    assert adapter.get_account().equity == "1000"
    assert adapter.list_positions() == []
    adapter.cancel_order("order-1")
    assert calls == ["order-1"]
