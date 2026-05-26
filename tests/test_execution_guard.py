from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from broker_interface import Position


class FakeAPI:
    def __init__(self, orders=None, account=None):
        self.orders = list(orders or [])
        self.account = account or SimpleNamespace(equity="100000", last_equity="100000")

    def list_orders(self, **_kwargs):
        return list(self.orders)

    def get_account(self):
        return self.account


class FakeBroker:
    def __init__(self, positions=None, orders=None, account=None, market_open=True, fail_cancel_ids=None):
        self._api = FakeAPI(orders=orders, account=account)
        self.positions = list(positions or [])
        self.market_open = market_open
        self.fail_cancel_ids = set(fail_cancel_ids or [])
        self.cancelled = []
        self.placed = []

    def get_positions(self):
        return list(self.positions)

    def cancel_order(self, order_id):
        self.cancelled.append(str(order_id))
        if str(order_id) in self.fail_cancel_ids:
            return False
        return True

    def place_order(self, order):
        self.placed.append(order)
        return f"new-{len(self.placed)}"

    def get_equity(self):
        return float(self._api.account.equity)

    def is_market_open(self):
        return bool(self.market_open)


def _order(symbol, qty, order_type="trailing_stop", side="sell", trail_percent="5", submitted_at=None, oid="old-1"):
    return SimpleNamespace(
        id=oid,
        symbol=symbol,
        qty=str(qty),
        type=order_type,
        side=side,
        trail_percent=trail_percent,
        submitted_at=submitted_at,
    )


def test_repair_preserves_matching_core_stop(monkeypatch):
    import alpaca_protection as protection

    broker = FakeBroker(
        positions=[Position("SPY", 10, 500.0)],
        orders=[_order("SPY", 10, trail_percent="5", oid="stop-1")],
    )
    result = protection.repair_core_etf_protective_stops(
        broker,
        tickers={"SPY"},
        logger=lambda _msg: None,
    )

    assert result["submitted"] == []
    assert result["cancelled"] == []
    assert broker.placed == []
    assert broker.cancelled == []
    assert result["skipped"][0]["reason"] == "already_protected"


def test_repair_replaces_wrong_sized_core_stop():
    import alpaca_protection as protection

    broker = FakeBroker(
        positions=[Position("TQQQ", 7, 80.0)],
        orders=[_order("TQQQ", 5, trail_percent="10", oid="stop-tqqq")],
    )
    result = protection.repair_core_etf_protective_stops(
        broker,
        tickers={"TQQQ"},
        logger=lambda _msg: None,
    )

    assert broker.cancelled == ["stop-tqqq"]
    assert len(broker.placed) == 1
    assert broker.placed[0].ticker == "TQQQ"
    assert broker.placed[0].quantity == 7
    assert broker.placed[0].trail_percent == protection.TQQQ_PROTECTION_TRAIL_PCT
    assert result["submitted"][0]["ticker"] == "TQQQ"


def test_stale_guard_skips_trailing_stops_and_cancels_old_normal_order(monkeypatch):
    import execution_guard

    old_time = datetime.now(timezone.utc) - timedelta(minutes=120)
    broker = FakeBroker(
        orders=[
            _order("SPY", 10, order_type="trailing_stop", submitted_at=old_time, oid="protective"),
            _order("AAPL", 2, order_type="limit", side="buy", submitted_at=old_time, oid="stale-buy"),
        ]
    )
    alerts = []
    monkeypatch.setattr(execution_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(execution_guard, "log", lambda _msg: None)
    state = {"stale_alerted_order_ids": []}

    execution_guard.guard_stale_orders(broker, state, dry_run=False)

    assert broker.cancelled == ["stale-buy"]
    assert state["stale_alerted_order_ids"] == ["stale-buy"]
    assert len(alerts) == 1
    assert "AAPL" in alerts[0]


def test_repair_does_not_submit_replacement_when_cancel_fails():
    import alpaca_protection as protection

    broker = FakeBroker(
        positions=[Position("SPY", 12, 500.0)],
        orders=[_order("SPY", 10, trail_percent="5", oid="uncancellable")],
        fail_cancel_ids={"uncancellable"},
    )

    result = protection.repair_core_etf_protective_stops(
        broker,
        tickers={"SPY"},
        logger=lambda _msg: None,
    )

    assert broker.cancelled == ["uncancellable"]
    assert broker.placed == []
    assert result["errors"][0]["error"] == "could_not_cancel_existing_protective_stop"


def test_repair_skips_invalid_trailing_stop_config(monkeypatch):
    import alpaca_protection as protection

    monkeypatch.setattr(protection, "CORE_PROTECTION_TRAIL_PCT", float("nan"))
    broker = FakeBroker(
        positions=[Position("SPY", 12, 500.0)],
        orders=[],
    )

    result = protection.repair_core_etf_protective_stops(
        broker,
        tickers={"SPY"},
        logger=lambda _msg: None,
    )

    assert broker.placed == []
    assert broker.cancelled == []
    assert result["errors"] == [{"ticker": "SPY", "error": "invalid_core_trail_pct"}]


def test_pnl_halt_closed_market_alerts_without_liquidating(monkeypatch):
    import execution_guard

    account = SimpleNamespace(equity="91000", last_equity="100000")
    broker = FakeBroker(account=account, market_open=False)
    calls = []
    alerts = []
    monkeypatch.setattr(execution_guard, "_emergency_liquidate", lambda _broker: calls.append("liquidate"))
    monkeypatch.setattr(execution_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(execution_guard, "log", lambda _msg: None)
    state = {"date": "2026-05-12"}

    execution_guard.guard_intraday_pnl(
        broker,
        state,
        dry_run=False,
        market_open=False,
        force_market_closed=False,
    )

    assert calls == []
    assert state["pnl_halt_sent"] is not True
    assert state["pnl_halt_blocked_alert_sent"] is True
    assert alerts


def test_pnl_guard_skips_invalid_current_equity(monkeypatch):
    import execution_guard

    account = SimpleNamespace(equity="nan", last_equity="100000")
    broker = FakeBroker(account=account, market_open=True)
    calls = []
    alerts = []
    monkeypatch.setattr(execution_guard, "_emergency_liquidate", lambda _broker: calls.append("liquidate"))
    monkeypatch.setattr(execution_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(execution_guard, "log", lambda _msg: None)
    state = {"date": "2026-05-12", "baseline_equity": "not-a-number"}

    execution_guard.guard_intraday_pnl(
        broker,
        state,
        dry_run=False,
        market_open=True,
        force_market_closed=False,
    )

    assert calls == []
    assert alerts == ["Intraday P&L guard skipped: invalid current broker equity"]
    assert "intraday_high" not in state
