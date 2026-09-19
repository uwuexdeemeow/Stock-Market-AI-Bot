from __future__ import annotations

import sys

import pytest


class _FakeBroker:
    equity = 100_000.0

    def get_equity(self):
        return self.equity


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
