"""Audit regressions using synthetic prices and fake brokers; no orders leave tests."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import alpaca_paper_trading as trading
import backtest
import core_satellite_alpha as core
import alpha_factor_backtest as factors


def quote(bid=100, ask=100.1, **changes):
    result = {"bid_price": bid, "ask_price": ask, "spread_pct": 0,
              "quote_timestamp": datetime.now(timezone.utc).isoformat()}
    result.update(changes)
    return result


@pytest.mark.parametrize("bid,ask", [(0, 1), (-1, 1), (101, 100), (np.nan, 100), (100, np.inf), (None, 100)])
def test_invalid_submission_quotes(bid, ask):
    assert trading._submission_quote(quote(bid, ask), "MU")[1]


@pytest.mark.parametrize("stamp", [None, "bad", "2020-01-01T00:00:00Z",
                                    (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()])
def test_missing_stale_and_future_timestamps(stamp):
    assert trading._submission_quote(quote(quote_timestamp=stamp), "MU")[1] == "quote_stale"


def test_threshold_uses_prices_and_asset_type(monkeypatch):
    monkeypatch.setattr(trading, "MAX_SPREAD_PCT_ETF", .001)
    monkeypatch.setattr(trading, "MAX_SPREAD_PCT_OVERLAY", .005)
    assert trading._submission_quote(quote(99.9, 100.1), "SPY")[1].startswith("spread_guard")
    checked, reason = trading._submission_quote(quote(99.9, 100.1), "MU")
    assert reason == ""
    assert checked["spread_pct"] == pytest.approx(.002)
    assert trading._submission_quote(quote(100, 100), "SPY")[1] == ""


@pytest.mark.parametrize("two_stage", [True, False])
def test_narrow_planning_quote_then_wide_submission_blocks(monkeypatch, two_stage):
    monkeypatch.setattr(trading, "MAX_SPREAD_PCT_OVERLAY", .005)
    monkeypatch.setattr(trading, "TWO_STAGE_EXECUTION_ENABLED", two_stage)
    snapshots = iter([quote(), quote(100, 110)])
    sent = []
    broker = SimpleNamespace(get_quote_snapshot=lambda ticker: next(snapshots),
                             place_order=sent.append, cancel_order=lambda oid: True)
    order = {"ticker": "MU", "side": "buy", "quantity": 1, "price": 100}
    accepted, _, _ = trading._apply_spread_guard(broker, [order])
    assert accepted == [order]
    result = trading._submit_one_rebalance_order(broker, order, use_market_order=False, use_quote_limit=True)
    assert result.startswith("SKIPPED: spread_guard")
    assert sent == []


@pytest.mark.parametrize("raw,days,expected", [
    ([True, True, True, False, True, True, True], 3, [True]*7),
    ([True, False, False, False, True, True, True], 3, [True, True, True, False, False, False, True]),
    ([False, True, True, True, False, True], 3, [False, False, False, True, True, True]),
    ([True, False], 3, [True, True]),
    ([False, True], 1, [False, True]),
    ([], 3, []),
])
def test_consecutive_confirmation(raw, days, expected):
    assert core._confirm_regime_flag(pd.Series(raw, dtype=bool), days).tolist() == expected


def test_session_offsets_skip_holidays():
    assert core._session_offset("2024-12-02", 20) == pd.Timestamp("2024-12-31")
    assert core._session_offset("2024-12-24", 1) == pd.Timestamp("2024-12-26")


def test_loader_keeps_missing_future_and_actual_timestamps(tmp_path, monkeypatch):
    dates = core._nyse_sessions("2024-12-02", "2025-02-28").delete(3)
    frame = pd.DataFrame({"Open": 100., "Close": 101., "feature": 1.}, index=dates)
    frame.to_parquet(tmp_path / "TEST.parquet")
    monkeypatch.setattr(factors, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(factors, "WATCHLIST", ["TEST"])
    monkeypatch.setattr(factors, "apply_membership_if_complete", lambda panel: (panel, {}))
    panel = factors.load_factor_panel([{"feature": "feature"}], require_forward_returns=False)
    assert len(panel) == len(dates)
    assert pd.isna(panel.iloc[-1]["forward_return_20d"])
    assert panel.iloc[0]["forward_return_20d_end_date"] == dates[20]
    assert panel.iloc[0]["forward_return_delay1_20d_entry_date"] == dates[2]


def _strategy_fixture(monkeypatch):
    # Keep the real selector and return accounting while isolating market data.
    monkeypatch.setattr(core, "_cached_etf_prices", lambda index, tickers: pd.DataFrame(index=index))
    monkeypatch.setattr(core, "_resolve_allocation", lambda *args: ("risk_on", {}, 0., 1.))
    monkeypatch.setattr(core, "_score_col_for_regime", lambda *args: "score")
    monkeypatch.setattr(core, "_apply_concentration_overlay_target", lambda date, cg, og, ri, cfg: (og, og, None))
    config = {"holding_days": 20, "regime_mode": "static", "score_source": "raw",
              "shape": "top5", "weighting": "equal", "max_per_sector": 5}
    dates = core._nyse_sessions("2024-12-02", "2025-02-28")
    panel = pd.DataFrame({"date": dates, "ticker": "TEST", "sector": "OTHER", "score": 1.,
                          "forward_return_20d": .01})
    return panel, config


def test_missing_future_does_not_change_selection_and_stops_run(monkeypatch):
    panel, config = _strategy_fixture(monkeypatch)
    day = panel.iloc[:1].copy()
    kwargs = dict(score_col="score", return_col="forward_return_20d", shape="top5", exit_rank_floor=.8, max_per_sector=5)
    before = core._select_sticky_holdings(day, set(), **kwargs)
    day["forward_return_20d"] = np.nan
    after = core._select_sticky_holdings(day, set(), **kwargs)
    assert before.ticker.tolist() == after.ticker.tolist() == ["TEST"]
    panel.loc[0, "forward_return_20d"] = np.nan
    with pytest.raises(ValueError, match="TEST.*2024-12-02"):
        core.run_core_satellite(panel, config)


def test_actual_label_end_purges_whole_period(monkeypatch):
    panel, config = _strategy_fixture(monkeypatch)
    panel["forward_return_20d_end_date"] = pd.NaT
    panel.loc[0, "forward_return_20d_end_date"] = pd.Timestamp("2025-01-02")
    # Calendar exit is Dec 31, but this ticker's missing session moves its label.
    with pytest.raises(ValueError, match="No complete holding periods"):
        core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2024-12-31"))


def test_calendar_purges_december_holiday_crossing(monkeypatch):
    panel, config = _strategy_fixture(monkeypatch)
    with pytest.raises(ValueError, match="No complete holding periods"):
        core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2024-12-30"))
    _, trades, _ = core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2024-12-31"))
    assert trades.iloc[0]["exit_date"] == pd.Timestamp("2024-12-31")


def test_cache_alignment_order_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "_ETF_PRICE_FRAME_CACHE", {})
    dates = pd.date_range("2024-01-02", periods=3)
    pd.DataFrame({"Close": [100., 110., 120.]}, index=dates).to_parquet(tmp_path / "SPY.parquet")
    sparse = backtest._load_etf_price_frame(dates[[0, 2]], ["SPY"])
    dense = backtest._load_etf_price_frame(dates, ["SPY"])
    assert dense.index.equals(dates)
    assert dense.SPY.tolist() == pytest.approx([1, 1.1, 1.2])
    assert sparse.SPY.tolist() == pytest.approx([1, 1.2])
    dense.iloc[0, 0] = 999
    reversed_frame = backtest._load_etf_price_frame(dates[::-1], ["SPY"])
    assert reversed_frame.SPY.tolist() == pytest.approx([1.2, 1.1, 1])
    backtest._ETF_PRICE_FRAME_CACHE.clear()
    backtest._load_etf_price_frame(dates, ["SPY"])
    assert backtest._load_etf_price_frame(dates[[0, 2]], ["SPY"]).equals(sparse)


def test_cache_does_not_use_future_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "_ETF_PRICE_FRAME_CACHE", {})
    dates = pd.date_range("2024-01-02", periods=3)
    pd.DataFrame({"Close": [110., 120.]}, index=dates[1:]).to_parquet(tmp_path / "SPY.parquet")
    with pytest.raises(ValueError, match="SPY.*2024-01-02"):
        backtest._load_etf_price_frame(dates, ["SPY"])


def test_failed_download_can_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "_ETF_PRICE_FRAME_CACHE", {})
    dates = pd.date_range("2024-01-02", periods=2)
    responses = iter([pd.DataFrame(), pd.DataFrame({"Close": [100., 101.]}, index=dates)])
    monkeypatch.setattr(backtest, "_download_yfinance", lambda *args, **kwargs: next(responses))
    with pytest.raises(ValueError, match="unavailable"):
        backtest._load_etf_price_frame(dates, ["SPY"])
    assert backtest._load_etf_price_frame(dates, ["SPY"]).SPY.tolist() == [1., 1.01]


def test_early_exit_survives_purging_next_trade(monkeypatch):
    panel, config = _strategy_fixture(monkeypatch)
    panel["Open"] = 100.
    panel["Close"] = 110.
    # The long label is unavailable, but the actual short interval is known.
    panel["forward_return_20d"] = np.nan
    config.update(regime_mode=next(iter(core.REGIME_PRESETS)), early_rebalance_on_regime_change=True)
    monkeypatch.setattr(core, "_load_regime_indicators", lambda *args: pd.DataFrame())
    monkeypatch.setattr(core, "_resolve_allocation", lambda date, *args:
                        ("risk_on" if date < pd.Timestamp("2024-12-06") else "risk_off", {}, 0., 1.))
    _, trades, metrics = core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2024-12-10"))
    assert len(trades) == 1
    assert trades.iloc[0].exit_date == pd.Timestamp("2024-12-06")
    assert trades.iloc[0].factor_overlay_return == pytest.approx(.1 * trades.iloc[0].overlay_gross)
    assert metrics["purged_trailing_trade_count"] == 1


def test_delayed_fold_boundary_uses_exchange_exit(monkeypatch):
    panel, config = _strategy_fixture(monkeypatch)
    panel["forward_return_delay1_20d"] = .01
    config["entry_delay_days"] = 1
    with pytest.raises(ValueError, match="No complete holding periods"):
        core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2025-01-01"))
    _, trades, _ = core.run_core_satellite(panel, config, evaluation_end=pd.Timestamp("2025-01-02"))
    assert trades.iloc[0].exit_date == pd.Timestamp("2025-01-02")


@pytest.mark.parametrize("replacement", [quote(100, 110), quote(101, 100), quote(0, 100), quote(quote_timestamp=None)])
def test_blocked_replacement_preserves_partial_fill_and_reason(tmp_path, monkeypatch, replacement):
    monkeypatch.setattr(trading, "EXECUTION_STAGE1_WAIT_SECONDS", 0)
    monkeypatch.setattr(trading, "EXECUTION_CANCEL_WAIT_SECONDS", 0)
    monkeypatch.setattr(trading, "_send_submit_guard_alert", lambda *args, **kwargs: None)
    monkeypatch.setattr(trading, "PAPER_LOG_FILE", tmp_path / "log.csv")
    if replacement["quote_timestamp"] is not None:
        replacement = dict(replacement, quote_timestamp=datetime.now(timezone.utc).isoformat())
    snapshots = iter([quote(), replacement])
    sent = []
    broker = SimpleNamespace(
        get_quote_snapshot=lambda ticker: next(snapshots),
        place_order=lambda order: sent.append(order) or "first",
        cancel_order=lambda oid: True,
        _api=SimpleNamespace(get_order=lambda oid: SimpleNamespace(status="canceled", filled_qty=4, filled_avg_price=100.)),
    )
    order = {"ticker": "MU", "side": "buy", "quantity": 10, "price": 100., "trade_value": 1000., "target_weight": .1}
    result = trading._submit_two_stage_limit(broker, order)
    assert result == "first"
    assert len(sent) == 1
    assert order["filled_qty"] == 4
    assert order["fill_status"] == "partially_filled"
    assert order["stage2_block_reason"]
    trading.log_submission([order], [result])
    logged = pd.read_csv(tmp_path / "log.csv")
    assert logged.iloc[0].stage2_block_reason == order["stage2_block_reason"]


def test_causal_alignment_uses_interior_source_bar_and_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "_ETF_PRICE_FRAME_CACHE", {})
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    path = tmp_path / "SPY.parquet"
    frame = pd.DataFrame({"Close": [100., 110., 120.]}, index=dates)
    frame.to_parquet(path)
    requested = pd.to_datetime(["2024-01-02", "2024-01-04"])
    assert backtest._load_etf_price_frame(requested, ["SPY"]).SPY.tolist() == [1., 1.1]
    frame.loc[dates[1], "Close"] = 115.
    frame.to_parquet(path)
    assert core._cached_etf_prices(requested, ["SPY"]).SPY.tolist() == [1., 1.15]


def test_skipped_sell_cannot_unlock_buy_phase(monkeypatch):
    calls = []
    def skip_sell(broker, row, **kwargs):
        calls.append(row["ticker"])
        return trading._skip_order(row, "spread_guard:test")
    monkeypatch.setattr(trading, "_submit_one_rebalance_order", skip_sell)
    rows = [{"ticker": "MU", "side": "sell", "quantity": 1},
            {"ticker": "FCX", "side": "buy", "quantity": 1}]
    _, ids = trading.submit_rebalance_orders(SimpleNamespace(), rows, use_market_order=False, use_quote_limit=True)
    assert calls == ["MU"]
    assert ids[1] == "SKIPPED: sell_submission_failed"
