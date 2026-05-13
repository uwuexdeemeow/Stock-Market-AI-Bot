from __future__ import annotations

import pytest

from broker_interface import DryRunBroker, Order


def order(side: str, qty: int, price: float) -> Order:
    return Order(ticker="XYZ", side=side, quantity=qty, type="limit", limit_price=price)


def only_position(broker: DryRunBroker):
    positions = broker.get_positions()
    assert len(positions) == 1
    return positions[0]


def test_buy_more_into_long_weighted_averages_avg_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("buy", 100, 10.0))
    broker.place_order(order("buy", 50, 16.0))

    pos = only_position(broker)
    assert pos.quantity == 150
    assert pos.avg_price == pytest.approx(12.0)


def test_sell_part_of_long_keeps_avg_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("buy", 100, 10.0))
    broker.place_order(order("sell", 40, 12.0))

    pos = only_position(broker)
    assert pos.quantity == 60
    assert pos.avg_price == pytest.approx(10.0)


def test_sell_exact_long_removes_position():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("buy", 100, 10.0))
    broker.place_order(order("sell", 100, 12.0))

    assert broker.get_positions() == []


def test_sell_more_than_long_flips_to_short_at_fill_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("buy", 100, 10.0))
    broker.place_order(order("sell", 150, 12.0))

    pos = only_position(broker)
    assert pos.quantity == -50
    assert pos.avg_price == pytest.approx(12.0)


def test_sell_more_into_short_weighted_averages_avg_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("sell", 100, 20.0))
    broker.place_order(order("sell", 50, 14.0))

    pos = only_position(broker)
    assert pos.quantity == -150
    assert pos.avg_price == pytest.approx(18.0)


def test_buy_part_of_short_keeps_avg_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("sell", 100, 20.0))
    broker.place_order(order("buy", 40, 17.0))

    pos = only_position(broker)
    assert pos.quantity == -60
    assert pos.avg_price == pytest.approx(20.0)


def test_buy_more_than_short_flips_to_long_at_fill_price():
    broker = DryRunBroker(commission_per_share=0.0)

    broker.place_order(order("sell", 100, 20.0))
    broker.place_order(order("buy", 150, 12.0))

    pos = only_position(broker)
    assert pos.quantity == 50
    assert pos.avg_price == pytest.approx(12.0)
