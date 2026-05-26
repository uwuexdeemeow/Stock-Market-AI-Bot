from __future__ import annotations


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
