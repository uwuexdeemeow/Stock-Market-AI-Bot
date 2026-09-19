from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class _FakeBroker:
    equity = 100_000.0
    buying_power = 25_000.0
    market_open = True
    account = SimpleNamespace(status="ACTIVE", account_blocked=False, trading_blocked=False)

    def __init__(self):
        self._api = SimpleNamespace(get_account=lambda: self.account)

    def get_equity(self):
        return self.equity

    def get_buying_power(self):
        return self.buying_power

    def is_market_open(self):
        return self.market_open


@pytest.fixture(autouse=True)
def _reset_fake_broker_state():
    """Keep broker-health tests independent when their order changes."""
    _FakeBroker.equity = 100_000.0
    _FakeBroker.buying_power = 25_000.0
    _FakeBroker.market_open = True
    _FakeBroker.account = SimpleNamespace(
        status="ACTIVE", account_blocked=False, trading_blocked=False
    )


def test_broker_health_rejects_nonfinite_equity(monkeypatch):
    import alpaca_paper_trading
    import broker_health

    _FakeBroker.equity = float("nan")
    monkeypatch.setattr(alpaca_paper_trading, "AlpacaBroker", _FakeBroker)

    result = broker_health.check_alpaca()

    assert result["healthy"] is False
    assert result["equity"] is None
    assert "invalid broker equity" in result["error"]


def test_broker_health_accepts_positive_finite_equity(monkeypatch):
    import alpaca_paper_trading
    import broker_health

    _FakeBroker.equity = 12345.67
    monkeypatch.setattr(alpaca_paper_trading, "AlpacaBroker", _FakeBroker)

    result = broker_health.check_alpaca()

    assert result["healthy"] is True
    assert result["equity"] == 12345.67
    assert result["buying_power"] == 25000.0
    assert result["account_status"] == "ACTIVE"
    assert result["market_open"] is True


def test_broker_health_rejects_trading_blocked_account(monkeypatch):
    """A reachable but restricted account must stop the daily run."""
    import alpaca_paper_trading
    import broker_health

    _FakeBroker.account = SimpleNamespace(
        status="ACTIVE", account_blocked=False, trading_blocked=True
    )
    monkeypatch.setattr(alpaca_paper_trading, "AlpacaBroker", _FakeBroker)

    result = broker_health.check_alpaca()

    assert result["healthy"] is False
    assert "blocked from trading" in result["error"]


def test_broker_health_rejects_invalid_buying_power(monkeypatch):
    """Broken account numbers are not a healthy broker response."""
    import alpaca_paper_trading
    import broker_health

    _FakeBroker.account = SimpleNamespace(
        status="ACTIVE", account_blocked=False, trading_blocked=False
    )
    _FakeBroker.buying_power = float("nan")
    monkeypatch.setattr(alpaca_paper_trading, "AlpacaBroker", _FakeBroker)

    result = broker_health.check_alpaca()

    assert result["healthy"] is False
    assert "invalid broker buying power" in result["error"]


def test_strict_cli_exits_nonzero_when_broker_is_unhealthy(monkeypatch, tmp_path):
    """The daily runner can fail fast instead of discovering an outage at submit."""
    import broker_health

    monkeypatch.setattr(
        broker_health,
        "check_all",
        lambda **_kwargs: {"all_healthy": False, "down_brokers": ["alpaca"], "brokers": {}},
    )
    monkeypatch.setattr(sys, "argv", ["broker_health.py", "--strict", "--json"])

    with pytest.raises(SystemExit) as caught:
        broker_health.main()

    assert caught.value.code == 1
