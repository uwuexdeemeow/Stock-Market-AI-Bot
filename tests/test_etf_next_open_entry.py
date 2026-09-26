"""Core ETFs are bought at the next session's Open, not the signal day's Close.

PLAIN ENGLISH: the signal is computed from the rebalance day's closing
prices, so nobody can also trade at that same close.  Stocks were already
bought at the next morning's Open; the core ETFs (SPY/QQQ) were bought at the
signal day's own Close, which quietly used information a live account can't
trade on.  These tests pin the fixed timing with a fake QQQ whose every
Open gaps up from the previous Close, so the two timings give different
numbers.
"""
from __future__ import annotations

import pandas as pd
import pytest

import core_satellite_alpha as alpha


def _fake_qqq(dates: pd.DatetimeIndex) -> pd.DataFrame:
    # Close rises 1% a day.  Overnight gaps repeat every 3 mornings
    # (+0.5%, -1.0%, +2.0%); the 10-day holding period is not a multiple of 3,
    # so the entry and exit gaps differ and Open-to-Open != Close-to-Close.
    close = pd.Series([100.0 * 1.01 ** i for i in range(len(dates))], index=dates)
    gaps = pd.Series([(1.005, 0.99, 1.02)[i % 3] for i in range(len(dates))], index=dates)
    open_ = close.shift(1) * gaps
    open_.iloc[0] = 100.0
    return pd.DataFrame({"Open": open_, "Close": close})


def _run(monkeypatch, *, entry_delay_days=0):
    sessions = alpha._nyse_sessions("2026-01-02", "2026-03-31")
    bars = _fake_qqq(sessions)
    panel = pd.DataFrame({"date": sessions, "ticker": "TEST", "forward_return": 0.0,
                          "forward_return_10d": 0.0, "forward_return_delay1_10d": 0.0})
    monkeypatch.setattr(alpha, "_resolve_allocation", lambda *_a: ("risk_on", {"QQQ": 1.0}, 1.0, 0.0))
    monkeypatch.setattr(alpha, "_select_sticky_holdings", lambda *_a, **_k: pd.DataFrame({"ticker": pd.Series(dtype=str)}))
    monkeypatch.setattr(alpha, "_sticky_overlay_weights", lambda *_a, **_k: pd.Series(dtype=float))
    monkeypatch.setattr(alpha, "_score_col_for_regime", lambda *_a: "score")
    monkeypatch.setattr(alpha, "_apply_concentration_overlay_target", lambda *_a: (0.0, 0.0, None))
    monkeypatch.setattr(alpha, "_core_tickers_for_config", lambda _cfg: ["QQQ"])
    monkeypatch.setattr(alpha, "_etf_open_close_bars", lambda _ticker: bars)
    # Normalised closes, like the real price cache.
    monkeypatch.setattr(
        alpha, "_cached_etf_prices",
        lambda index, _tickers: pd.DataFrame(
            {"QQQ": bars["Close"].reindex(pd.DatetimeIndex(index)).ffill() / 100.0}),
    )
    _equity, trades, _extra = alpha.run_core_satellite(
        panel,
        {"holding_days": 10, "regime_mode": "static", "score_source": "raw", "shape": "top5",
         "weighting": "score", "max_per_sector": 2, "core_weights": {"QQQ": 1.0},
         "core_gross": 1.0, "overlay_gross": 0.0, "entry_delay_days": entry_delay_days},
    )
    return bars, trades


def test_core_etf_is_held_open_to_open(monkeypatch):
    bars, trades = _run(monkeypatch)
    first = trades.iloc[0]
    signal_day = pd.Timestamp(first["date"])
    entry_day = alpha._session_offset(signal_day, 1)
    exit_day = pd.Timestamp(first["exit_date"])
    next_open_day = alpha._session_offset(exit_day, 1)
    expected = bars.at[next_open_day, "Open"] / bars.at[entry_day, "Open"] - 1.0
    old_same_close = bars.at[exit_day, "Close"] / bars.at[signal_day, "Close"] - 1.0
    assert first["core_return"] == pytest.approx(expected, rel=1e-9)
    assert abs(first["core_return"] - old_same_close) > 1e-4


def test_consecutive_periods_chain_without_a_gap(monkeypatch):
    # Period k ends at the very Open where period k+1 starts: no night skipped.
    _bars, trades = _run(monkeypatch)
    for (_, now), (_, nxt) in zip(trades.iloc[:-1].iterrows(), trades.iloc[1:].iterrows()):
        assert alpha._session_offset(pd.Timestamp(now["exit_date"]), 1) == alpha._session_offset(pd.Timestamp(nxt["date"]), 1)


def test_delay_stress_moves_the_etf_entry_one_more_day(monkeypatch):
    bars, trades = _run(monkeypatch, entry_delay_days=1)
    first = trades.iloc[0]
    entry_day = alpha._session_offset(pd.Timestamp(first["date"]), 2)
    exit_day = pd.Timestamp(first["exit_date"])
    next_open_day = alpha._session_offset(exit_day, 1)
    expected = bars.at[next_open_day, "Open"] / bars.at[entry_day, "Open"] - 1.0
    assert first["core_return"] == pytest.approx(expected, rel=1e-9)


def test_last_period_never_reads_past_the_window(monkeypatch):
    sessions = alpha._nyse_sessions("2026-01-02", "2026-01-30")
    bars = _fake_qqq(sessions)
    monkeypatch.setattr(alpha, "_etf_open_close_bars", lambda _t: bars)
    exit_day, end_ts = sessions[10], sessions[10]
    # The next Open is after the window, so the exit day's Close is used.
    assert alpha._etf_exit_price("QQQ", exit_day, end_ts) == bars.at[exit_day, "Close"]
    assert alpha._etf_exit_price("QQQ", exit_day, sessions[-1]) == bars.at[sessions[11], "Open"]


def test_missing_etf_bar_fails_loudly(monkeypatch):
    sessions = alpha._nyse_sessions("2026-01-02", "2026-01-30")
    monkeypatch.setattr(alpha, "_etf_open_close_bars", lambda _t: _fake_qqq(sessions).drop(sessions[3]))
    with pytest.raises(ValueError, match="no filling allowed"):
        alpha._etf_bar_price("QQQ", sessions[3], "Open")
