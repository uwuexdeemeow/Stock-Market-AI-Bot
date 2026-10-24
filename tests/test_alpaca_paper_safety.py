"""Tests for the 2026-10-25 paper-safety fixes (Alpaca paper-trading skill audit).

G1  paper endpoint is checked by host name, not by a text match
G2  a timed-out order is looked up by client order ID before any resend
G3  broker retries honour Retry-After / X-RateLimit-Reset
G4  PAPER_ONLY counts as an active paper account
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import alpaca_paper_trading as apt
from alpaca_sdk_adapter import AlpacaPyRESTCompat, is_paper_endpoint
from broker_interface import Order


# ── G1: paper endpoint ─────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets/",
    "https://PAPER-API.alpaca.markets",
    "https://paper-api.alpaca.markets:443",
])
def test_paper_endpoint_accepts_only_the_paper_host(url):
    assert is_paper_endpoint(url)


@pytest.mark.parametrize("url", [
    "https://api.alpaca.markets",                              # live money
    "https://paper-api.alpaca.markets.example.com",            # look-alike host
    "https://example.com/paper-api.alpaca.markets",            # text in the path
    "https://api.alpaca.markets/?x=paper-api.alpaca.markets",  # text in the query
    "https://user:pw@paper-api.alpaca.markets",                # credentials in URL
    "http://paper-api.alpaca.markets",                         # not HTTPS
    "https://paper-api.alpaca.markets:8443",                   # odd port
    "",
    None,
])
def test_paper_endpoint_rejects_everything_else(url):
    assert not is_paper_endpoint(url)


def test_adapter_refuses_non_paper_endpoint():
    with pytest.raises(ValueError, match="Refusing non-paper Alpaca endpoint"):
        AlpacaPyRESTCompat(
            "key", "secret", base_url="https://paper-api.alpaca.markets.example.com",
            trading_client=SimpleNamespace(), data_client=SimpleNamespace(),
        )


def test_broker_refuses_look_alike_paper_host(monkeypatch):
    monkeypatch.setattr(apt, "ALPACA_API_KEY", "paper-key")
    monkeypatch.setattr(apt, "ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets.example.com")
    with pytest.raises(ValueError, match="Refusing non-paper Alpaca endpoint"):
        apt.AlpacaBroker()


# ── G2: idempotent order submission ────────────────────────────────────────

class _FakeOrderApi:
    """Scripted submit results plus a client-order-ID lookup table."""

    def __init__(self, submit_results, known=None):
        self.submit_results = list(submit_results)
        self.known = dict(known or {})
        self.submits = []
        self.lookups = []

    def submit_order(self, **kwargs):
        self.submits.append(dict(kwargs))
        kwargs.pop("type", None)        # the real adapter consumes "type" too
        result = self.submit_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(id=result)

    def get_order_by_client_id(self, client_order_id):
        self.lookups.append(client_order_id)
        if client_order_id not in self.known:
            raise RuntimeError("order not found")
        return self.known[client_order_id]


def _broker(api):
    broker = apt.AlpacaBroker.__new__(apt.AlpacaBroker)
    broker._api = api
    return broker


def _order(client_id="20261102_QQQ_buy_56-a1"):
    return Order(ticker="QQQ", side="buy", quantity=56, type="limit",
                 limit_price=705.0, client_id=client_id)


def _found(qty=56, minutes_ago=0, symbol="QQQ", side="buy"):
    return SimpleNamespace(
        id="broker-1", symbol=symbol, side=side, qty=str(qty),
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(apt.time, "sleep", lambda seconds: None)


def test_timeout_that_reached_alpaca_returns_existing_order_without_resend():
    cid = "20261102_QQQ_buy_56-a1"
    api = _FakeOrderApi([TimeoutError("read timed out")], known={cid: _found()})
    assert _broker(api).place_order(_order(cid)) == "broker-1"
    assert len(api.submits) == 1
    assert api.lookups == [cid]


def test_timeout_that_never_arrived_is_resent_with_the_same_client_id():
    cid = "20261102_QQQ_buy_56-a1"
    api = _FakeOrderApi([TimeoutError("read timed out"), "broker-2"])
    assert _broker(api).place_order(_order(cid)) == "broker-2"
    assert [s["client_order_id"] for s in api.submits] == [cid, cid]
    assert api.submits[1]["type"] == "limit"   # the resend is still a full order


def test_duplicate_on_resend_means_first_try_arrived():
    cid = "20261102_QQQ_buy_56-a1"
    api = _FakeOrderApi(
        [ConnectionError("connection reset"), RuntimeError('{"message":"client_order_id must be unique"}')],
    )
    # First lookup (after the reset) misses; the second (after the duplicate) finds it.
    original = api.get_order_by_client_id

    def lookup(client_order_id):
        if len(api.lookups) >= 1:
            api.known[cid] = _found()
        return original(client_order_id)

    api.get_order_by_client_id = lookup
    assert _broker(api).place_order(_order(cid)) == "broker-1"
    assert len(api.submits) == 2


def test_duplicate_on_first_try_is_an_old_collision_and_raises():
    cid = "20261102_QQQ_buy_56-a1"
    api = _FakeOrderApi([RuntimeError("client_order_id must be unique")], known={cid: _found()})
    with pytest.raises(RuntimeError, match="must be unique"):
        _broker(api).place_order(_order(cid))
    assert api.lookups == []


def test_old_or_different_order_with_same_client_id_is_not_accepted():
    cid = "20261102_QQQ_buy_56-a1"
    for stale in (_found(minutes_ago=60), _found(qty=10), _found(symbol="BIL"), _found(side="sell")):
        api = _FakeOrderApi([TimeoutError("timed out"), "broker-new"], known={cid: stale})
        assert _broker(api).place_order(_order(cid)) == "broker-new"
        assert len(api.submits) == 2


def test_validation_error_is_not_retried():
    api = _FakeOrderApi([ValueError("insufficient buying power"), "never"])
    with pytest.raises(ValueError):
        _broker(api).place_order(_order())
    assert len(api.submits) == 1
    assert api.lookups == []


def test_order_without_client_id_gets_a_non_date_auto_id():
    api = _FakeOrderApi(["stop-1"])
    _broker(api).place_order(Order(ticker="QQQ", side="sell", quantity=5, type="stop", stop_price=600.0))
    cid = api.submits[0]["client_order_id"]
    assert cid.startswith("auto_") and len(cid) <= 48


def test_gives_up_after_max_retries_when_nothing_arrives():
    api = _FakeOrderApi([TimeoutError("timed out")] * 4)
    with pytest.raises(TimeoutError):
        _broker(api).place_order(_order())
    assert len(api.submits) == 4


# ── G3: Retry-After ────────────────────────────────────────────────────────

def _rate_limited(headers):
    exc = RuntimeError("429 rate limit")
    exc.response = SimpleNamespace(headers=headers)
    return exc


def test_retry_delay_uses_retry_after_header():
    assert apt._broker_retry_delay(_rate_limited({"Retry-After": "7"}), 0, 1.0) == 7.0


def test_retry_delay_uses_rate_limit_reset_time():
    reset = time.time() + 12
    delay = apt._broker_retry_delay(_rate_limited({"X-RateLimit-Reset": str(reset)}), 0, 1.0)
    assert 10.0 <= delay <= 12.0


def test_retry_delay_is_capped_and_falls_back_to_backoff():
    assert apt._broker_retry_delay(_rate_limited({"Retry-After": "9999"}), 0, 1.0) == 60.0
    assert apt._broker_retry_delay(RuntimeError("503"), 2, 1.0) == 4.0
    assert apt._broker_retry_delay(_rate_limited({"Retry-After": "soon"}), 1, 1.0) == 2.0


def test_broker_retry_sleeps_for_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(apt.time, "sleep", waits.append)
    calls = iter([_rate_limited({"Retry-After": "5"}), "ok"])

    @apt._broker_retry
    def read():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    assert read() == "ok"
    assert waits == [5.0]


# ── G4: PAPER_ONLY status ─────────────────────────────────────────────────

def _fake_adapter(status):
    class _Adapter:
        def __init__(self, **kwargs):
            pass

        def get_account(self):
            return SimpleNamespace(status=status, account_blocked=False, trading_blocked=False)
    return _Adapter


@pytest.mark.parametrize("status,ok", [("ACTIVE", True), ("PAPER_ONLY", True), ("ACCOUNT_CLOSED", False)])
def test_broker_accepts_paper_only_status(monkeypatch, status, ok):
    import alpaca_sdk_adapter

    monkeypatch.setattr(apt, "ALPACA_API_KEY", "paper-key")
    monkeypatch.setattr(apt, "ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(apt, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setattr(alpaca_sdk_adapter, "AlpacaPyRESTCompat", _fake_adapter(status))
    if ok:
        apt.AlpacaBroker()
    else:
        with pytest.raises(ConnectionError, match="not active"):
            apt.AlpacaBroker()
