"""Audit fixes H2 (emergency halt can restart) and H3 (no buy-back after a stop).

PLAIN ENGLISH: after an emergency sell-off the account is all cash, which can
never grow back to the old peak, so the halt must also clear when the market
is risk-on again, and drawdown must then be measured from the restart point.
A stock sold by its protective stop must not be bought straight back the next
morning; it sits out one holding period.  Fake brokers and files only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

import alpaca_paper_trading as apt
import core_satellite_alpha as core
from signal_freshness import latest_completed_us_trading_day


# ── H2: halt recovery ───────────────────────────────────────────────────────

def _flat_broker(equity: float) -> SimpleNamespace:
    return SimpleNamespace(
        get_positions=lambda: [],
        get_equity=lambda: equity,
        _api=SimpleNamespace(list_orders=lambda **_kwargs: []),
    )


def _halt(tmp_path, monkeypatch):
    sentinel = tmp_path / "alpaca_halt_active.txt"
    apt._write_halt_sentinel(
        sentinel,
        now=datetime(2026, 5, 12, 9, 40, tzinfo=timezone.utc),
        liquidation={"cancel_verified": True, "errors": []},
    )
    monkeypatch.setattr(apt, "_HALT_SENTINEL_FILE", sentinel)
    monkeypatch.setattr(apt, "_DRAWDOWN_PEAK_RESET_FILE", tmp_path / "peak_reset.json")
    monkeypatch.setattr(apt, "PORTFOLIO_DRAWDOWN_HALT_PCT", 0.12)
    return sentinel


def test_all_cash_account_restarts_when_market_is_risk_on(monkeypatch, tmp_path):
    sentinel = _halt(tmp_path, monkeypatch)
    # Still 10% below the old peak: the account alone could never clear this.
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.10))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (True, "regime_risk_on"))

    assert apt._maybe_auto_clear_halt(_flat_broker(88_000.0)) is True
    assert not sentinel.exists()
    assert apt._read_drawdown_peak_reset(tmp_path / "peak_reset.json")["equity"] == 88_000.0


def test_halt_stays_while_market_is_not_risk_on(monkeypatch, tmp_path):
    sentinel = _halt(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.10))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (False, "regime_risk_off"))

    assert apt._maybe_auto_clear_halt(_flat_broker(88_000.0)) is False
    assert sentinel.exists()
    assert not (tmp_path / "peak_reset.json").exists()


def test_halt_stays_if_restart_point_cannot_be_recorded(monkeypatch, tmp_path):
    sentinel = _halt(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.10))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (True, "regime_risk_on"))

    assert apt._maybe_auto_clear_halt(_flat_broker(float("nan"))) is False
    assert sentinel.exists()


# ── H2 tightening: minimum wait before a market-based restart ──────────────
# The halt in _halt() fired on Tuesday 2026-05-12.

def test_market_restart_waits_minimum_sessions_after_halt(monkeypatch, tmp_path):
    # Thursday: only 2 sessions (Wed, Thu) since the halt.  The slow regime
    # signal may still read risk_on right after a crash, so no restart yet.
    sentinel = _halt(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.10))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (True, "regime_risk_on"))

    thursday = datetime(2026, 5, 14, 14, 0, tzinfo=timezone.utc)
    assert apt._maybe_auto_clear_halt(_flat_broker(88_000.0), now=thursday) is False
    assert sentinel.exists()
    assert not (tmp_path / "peak_reset.json").exists()


def test_market_restart_allowed_after_minimum_sessions(monkeypatch, tmp_path):
    # Next Tuesday: 5 sessions (Wed, Thu, Fri, Mon, Tue) have passed.
    sentinel = _halt(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.10))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (True, "regime_risk_on"))

    tuesday = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    assert apt.HALT_MARKET_RESTART_MIN_SESSIONS == 5
    assert apt._maybe_auto_clear_halt(_flat_broker(88_000.0), now=tuesday) is True
    assert not sentinel.exists()


def test_account_recovery_path_does_not_wait(monkeypatch, tmp_path):
    # The minimum wait only applies to the market signal.  A real recovery
    # of the account itself (better than half the halt level) still clears
    # from the next day, as before.
    sentinel = _halt(tmp_path, monkeypatch)
    monkeypatch.setattr(apt, "check_portfolio_drawdown", lambda _broker: (False, -0.03))
    monkeypatch.setattr(apt, "_market_recovered_for_halt", lambda: (False, "regime_risk_off"))

    thursday = datetime(2026, 5, 14, 14, 0, tzinfo=timezone.utc)
    assert apt._maybe_auto_clear_halt(_flat_broker(97_000.0), now=thursday) is True
    assert not sentinel.exists()


def test_session_count_skips_weekends_and_holidays():
    friday_halt = datetime(2026, 5, 22, 18, 0, tzinfo=timezone.utc)
    # Memorial Day (Monday 2026-05-25) is an exchange holiday.
    assert apt._nyse_sessions_since(friday_halt, datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)) == 0
    assert apt._nyse_sessions_since(friday_halt, datetime(2026, 5, 26, 18, 0, tzinfo=timezone.utc)) == 1
    # Same day, or a clock before the halt, counts nothing.
    assert apt._nyse_sessions_since(friday_halt, friday_halt) == 0


def test_repeated_sell_off_keeps_the_original_halt_time(monkeypatch, tmp_path):
    # While halted, the daily run repeats the emergency sell-off.  That must
    # not re-stamp the halt time, or the waiting clock would restart daily.
    sentinel = _halt(tmp_path, monkeypatch)
    original_time, _state = apt._read_halt_sentinel(sentinel)
    broker = SimpleNamespace(cancel_all_orders=lambda: True, get_positions=lambda: [])

    apt._emergency_liquidate(broker)

    halt_time, state = apt._read_halt_sentinel(sentinel)
    assert halt_time == original_time
    # The retry itself is still recorded (fresh liquidation state).
    assert state["liquidation"]["cancel_verified"] is True


def test_drawdown_is_measured_from_the_restart_point(monkeypatch, tmp_path):
    equity_file = tmp_path / "equity.csv"
    pd.DataFrame({"date": ["2026-05-01", "2026-05-11", "2026-05-20"],
                  "equity": [100_000.0, 87_000.0, 88_000.0]}).to_csv(equity_file, index=False)
    monkeypatch.setattr(apt, "EQUITY_FILE", equity_file)
    reset_file = tmp_path / "peak_reset.json"
    monkeypatch.setattr(apt, "_DRAWDOWN_PEAK_RESET_FILE", reset_file)
    monkeypatch.setattr(apt, "PORTFOLIO_DRAWDOWN_HALT_PCT", 0.12)
    broker = SimpleNamespace(get_equity=lambda: 85_000.0)

    # Without a restart point, the May 1 peak makes this a 15% drawdown: halted.
    halted, dd = apt.check_portfolio_drawdown(broker)
    assert halted and round(dd, 3) == -0.15

    apt._write_drawdown_peak_reset(87_000.0, path=reset_file, now=datetime(2026, 5, 13, 14, tzinfo=timezone.utc))
    halted, dd = apt.check_portfolio_drawdown(broker)
    # Peak is now max(87,000 restart, 88,000 on May 20): 85k is 3.4% below.
    assert not halted and round(dd, 3) == round(85_000 / 88_000 - 1, 3)


def test_market_recovery_needs_a_fresh_risk_on_signal(tmp_path):
    now = datetime.now(timezone.utc)
    base = {"predicted_at": now.isoformat(timespec="minutes"),
            "latest_factor_date": str(latest_completed_us_trading_day(now=now)),
            "live_regime_refresh_failed": False}

    def check(**row):
        path = tmp_path / "signal.csv"
        pd.DataFrame([{**base, **row}]).to_csv(path, index=False)
        return apt._market_recovered_for_halt(path)

    assert check(current_regime="risk_on") == (True, "regime_risk_on")
    assert check(current_regime="neutral")[0] is False
    assert check(current_regime="risk_on", live_regime_refresh_failed=True)[0] is False
    assert check(current_regime="risk_on", predicted_at="2020-01-01T09:00+00:00")[0] is False
    assert apt._market_recovered_for_halt(tmp_path / "missing.csv")[0] is False


# ── H3: stop cooldown ──────────────────────────────────────────────────────

def test_recent_protective_exits_lists_only_filled_stop_sells():
    now = datetime(2026, 9, 25, 20, tzinfo=timezone.utc)
    orders = [
        SimpleNamespace(symbol="NVDA", type="trailing_stop", side="sell", filled_qty="10", filled_at="2026-09-24T18:00:00Z"),
        SimpleNamespace(symbol="AMD", type="trailing_stop", side="sell", filled_qty="0", filled_at=None),  # still open
        SimpleNamespace(symbol="MU", type="limit", side="sell", filled_qty="5", filled_at="2026-09-24T15:00:00Z"),  # rebalance
        SimpleNamespace(symbol="TSLA", type="stop", side="sell", filled_qty="3", filled_at="2026-06-01T15:00:00Z"),  # too old
    ]
    broker = SimpleNamespace(_api=SimpleNamespace(list_orders=lambda **_kwargs: orders))
    available, exits = apt._recent_protective_exits(broker, now=now)
    assert available is True
    assert [row["ticker"] for row in exits] == ["NVDA"]


def test_recent_protective_exits_reports_unavailable_broker():
    def boom(**_kwargs):
        raise RuntimeError("api down")
    available, exits = apt._recent_protective_exits(SimpleNamespace(_api=SimpleNamespace(list_orders=boom)))
    assert available is False and exits == []


def _status(tmp_path, exits, available=True):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"recent_protective_exits_available": available,
                                "recent_protective_exits": exits}))
    return path


def test_stopped_out_stock_sits_out_one_holding_period(tmp_path):
    path = _status(tmp_path, [
        {"ticker": "NVDA", "filled_at": "2026-09-24T18:00:00+00:00"},   # yesterday
        {"ticker": "AMD", "filled_at": "2026-08-03T15:00:00+00:00"},    # long ago
        {"ticker": "MU", "filled_at": "2026-09-24T18:00:00+00:00"},     # still held
    ])
    state = core._stop_cooldown_state(as_of=pd.Timestamp("2026-09-24"), cooldown_sessions=20,
                                      held_tickers={"MU"}, status_path=path)
    assert state["tickers"] == ["NVDA"]
    assert state["source"] == "alpaca_daily_status"
    # 20 sessions later the cooldown is over.
    later = core._stop_cooldown_state(as_of=pd.Timestamp("2026-10-22"), cooldown_sessions=20,
                                      held_tickers=set(), status_path=path)
    assert "NVDA" not in later["tickers"]


def test_cooldown_is_inactive_and_reported_without_exit_data(tmp_path):
    state = core._stop_cooldown_state(as_of=pd.Timestamp("2026-09-24"), cooldown_sessions=20,
                                      held_tickers=set(), status_path=_status(tmp_path, [], available=False))
    assert state == {"tickers": [], "source": "exits_unavailable", "details": []}
    missing = core._stop_cooldown_state(as_of=pd.Timestamp("2026-09-24"), cooldown_sessions=20,
                                        held_tickers=set(), status_path=tmp_path / "nope.json")
    assert missing["source"] == "status_unreadable"
