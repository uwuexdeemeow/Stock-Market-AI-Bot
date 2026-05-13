from __future__ import annotations

import sys
import types

import pandas as pd


class FakeMoomooContext:
    def __init__(self, *, positions=None, orders=None, fail_cancel_ids=None):
        self.positions = pd.DataFrame(positions or [])
        self.orders = pd.DataFrame(orders or [])
        self.fail_cancel_ids = set(fail_cancel_ids or [])
        self.cancelled = []
        self.placed = []

    def position_list_query(self, **_kwargs):
        return 0, self.positions.copy()

    def order_list_query(self, **_kwargs):
        return 0, self.orders.copy()

    def modify_order(self, _op, order_id, _qty, _price, **_kwargs):
        self.cancelled.append(str(order_id))
        if str(order_id) in self.fail_cancel_ids:
            return 1, "cancel failed"
        return 0, pd.DataFrame([{"order_id": str(order_id)}])

    def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return 0, pd.DataFrame([{"order_id": f"new-{len(self.placed)}"}])


def _install_fake_moomoo(monkeypatch):
    fake = types.SimpleNamespace(
        RET_OK=0,
        TrdEnv=types.SimpleNamespace(SIMULATE="SIMULATE"),
        ModifyOrderOp=types.SimpleNamespace(CANCEL="CANCEL"),
        OrderType=types.SimpleNamespace(STOP_LIMIT="STOP_LIMIT", NORMAL="NORMAL"),
        TrdSide=types.SimpleNamespace(SELL="SELL"),
        TimeInForce=types.SimpleNamespace(GTC="GTC"),
    )
    monkeypatch.setitem(sys.modules, "moomoo", fake)


def _position(ticker="SPY", qty=10, price=100.0):
    return {"code": f"US.{ticker}", "qty": qty, "nominal_price": price}


def _core_stop(ticker="SPY", qty=10, trigger=95.0, order_id="old-stop"):
    return {
        "order_id": order_id,
        "code": f"US.{ticker}",
        "qty": qty,
        "trd_side": "SELL",
        "order_type": "STOP_LIMIT",
        "price": trigger * 0.995,
        "aux_price": trigger,
        "remark": f"core_etf_stop_{ticker}_5.00pct",
    }


def test_moomoo_repair_submits_stop_limit_for_unprotected_core_etf(monkeypatch, tmp_path):
    _install_fake_moomoo(monkeypatch)
    import moomoo_paper_trading as mpt

    monkeypatch.setattr(mpt, "MOOMOO_GUARD_STATE_FILE", tmp_path / "state.json")
    ctx = FakeMoomooContext(positions=[_position()])

    result = mpt.repair_moomoo_core_etf_stops(ctx, tickers={"SPY"}, logger=lambda _msg: None)

    assert result["errors"] == []
    assert len(ctx.placed) == 1
    order = ctx.placed[0]
    assert order["order_type"] == "STOP_LIMIT"
    assert order["aux_price"] == 95.0
    assert order["price"] == 94.52
    assert order["remark"].startswith("core_etf_stop_SPY")


def test_moomoo_repair_preserves_fresh_matching_core_stop(monkeypatch, tmp_path):
    _install_fake_moomoo(monkeypatch)
    import moomoo_paper_trading as mpt

    monkeypatch.setattr(mpt, "MOOMOO_GUARD_STATE_FILE", tmp_path / "state.json")
    ctx = FakeMoomooContext(positions=[_position()], orders=[_core_stop(trigger=95.0)])

    result = mpt.repair_moomoo_core_etf_stops(ctx, tickers={"SPY"}, logger=lambda _msg: None)

    assert result["submitted"] == []
    assert result["cancelled"] == []
    assert ctx.placed == []
    assert ctx.cancelled == []
    assert result["skipped"][0]["reason"] == "already_protected"


def test_moomoo_repair_moves_stop_up_when_high_water_rises(monkeypatch, tmp_path):
    _install_fake_moomoo(monkeypatch)
    import moomoo_paper_trading as mpt

    monkeypatch.setattr(mpt, "MOOMOO_GUARD_STATE_FILE", tmp_path / "state.json")
    ctx = FakeMoomooContext(
        positions=[_position(price=110.0)],
        orders=[_core_stop(trigger=95.0, order_id="stale-stop")],
    )

    result = mpt.repair_moomoo_core_etf_stops(ctx, tickers={"SPY"}, logger=lambda _msg: None)

    assert ctx.cancelled == ["stale-stop"]
    assert len(ctx.placed) == 1
    assert ctx.placed[0]["aux_price"] == 104.5
    assert result["submitted"][0]["trigger_price"] == 104.5


def test_moomoo_repair_does_not_replace_when_cancel_fails(monkeypatch, tmp_path):
    _install_fake_moomoo(monkeypatch)
    import moomoo_paper_trading as mpt

    monkeypatch.setattr(mpt, "MOOMOO_GUARD_STATE_FILE", tmp_path / "state.json")
    ctx = FakeMoomooContext(
        positions=[_position(qty=12, price=110.0)],
        orders=[_core_stop(qty=10, trigger=95.0, order_id="uncancellable")],
        fail_cancel_ids={"uncancellable"},
    )

    result = mpt.repair_moomoo_core_etf_stops(ctx, tickers={"SPY"}, logger=lambda _msg: None)

    assert ctx.cancelled == ["uncancellable"]
    assert ctx.placed == []
    assert result["errors"][0]["error"] == "cancel failed"


def test_moomoo_repair_does_not_submit_trigger_above_current_price(monkeypatch, tmp_path):
    _install_fake_moomoo(monkeypatch)
    import moomoo_paper_trading as mpt

    state_path = tmp_path / "state.json"
    state_path.write_text('{"core_etf_hwm": {"SPY": 100.0}}', encoding="utf-8")
    monkeypatch.setattr(mpt, "MOOMOO_GUARD_STATE_FILE", state_path)
    ctx = FakeMoomooContext(positions=[_position(price=90.0)])

    result = mpt.repair_moomoo_core_etf_stops(ctx, tickers={"SPY"}, logger=lambda _msg: None)

    assert result["errors"] == []
    assert len(ctx.placed) == 1
    assert ctx.placed[0]["aux_price"] == 85.5
    assert ctx.placed[0]["aux_price"] < 90.0
