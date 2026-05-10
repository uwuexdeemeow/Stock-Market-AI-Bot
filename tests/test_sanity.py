"""
test_sanity.py — Smoke tests runnable with `python -m pytest tests/`.

PLAIN ENGLISH:
These aren't exhaustive — they are a minimum bar. If any of these break,
don't trade. Every test should take < 1 second.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from labels import forward_return, vol_normalized_return, triple_barrier
from risk_sizing import vol_target_size, fractional_kelly, position_size_with_stop
from execution_model import realistic_fill_price, commission, capacity_warning
from data_validation import validate_price_frame
from core_satellite_alpha import _paper_signal_timestamp, _scale_paper_targets_to_gross
from paper_health import (
    _current_order_lifecycle,
    _drift_breakdown,
    _equity_sanity_checks,
    _execution_slippage,
    _go_live_scorecard,
    _open_position_attribution,
    _position_concentration,
    _stale_open_order_alerts,
)
import moomoo_paper_trading
from moomoo_paper_trading import (
    _find_broker_order_match,
    _order_status_bucket,
    build_submit_readiness_report,
    evaluate_broker_submit_preflight,
)
import paper_gauntlet
import daily_paper_check
from daily_paper_check import _after_fill_verdict, _prune_snapshots, _verdict
from refresh_etf_data import _validate_etf_frame
from config_health import _requirement_ok


def _price_frame(n=100, start="2024-01-01"):
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e6, 5e6, n),
    }, index=idx)


def test_forward_return_is_leak_free():
    df = _price_frame()
    r = forward_return(df["Close"], horizon=5)
    # last 5 rows must be NaN — nothing to look forward to
    assert r.tail(5).isna().all()


def test_vol_normalized_scales_with_vol():
    df = _price_frame()
    r = vol_normalized_return(df["Close"], horizon=5, vol_window=20).dropna()
    assert np.isfinite(r).any()


def test_triple_barrier_outputs_in_set():
    df = _price_frame()
    lab = triple_barrier(df["Close"], df["High"], df["Low"], max_hold=5).dropna()
    assert set(lab.unique()).issubset({-1.0, 0.0, 1.0})


def test_risk_sizing_nonnegative():
    assert vol_target_size(100_000, 0.20) > 0
    assert vol_target_size(100_000, 0.0) == 0
    # fractional_kelly takes p_win (win probability), not an edge offset.
    # p_win <= 0.5 → no edge → returns 0.  p_win > 0.5 → returns positive fraction.
    assert fractional_kelly(0.40) == 0   # 40% win rate → below 50% → no bet
    assert fractional_kelly(0.50) == 0   # 50% win rate → coin flip → no bet
    assert 0 < fractional_kelly(0.60) < 1  # 60% win rate → positive edge → size up
    assert fractional_kelly(0.60) < fractional_kelly(0.70)  # more edge → larger bet
    assert position_size_with_stop(100_000, 100, 2.0) > 0


def test_execution_costs_sane():
    px = realistic_fill_price(100.0, 1_000, 1_000_000, "buy")
    assert px > 100  # buying costs more than mid
    assert commission(100) > 0
    assert capacity_warning(100_000, 1_000_000) is True


def test_validation_rejects_bad():
    import pytest
    good = _price_frame()
    validate_price_frame(good, "TEST", max_lag_days=10_000)  # bypass freshness in tests

    bad = good.copy()
    bad.loc[bad.index[5], "Close"] = -1
    with pytest.raises(ValueError):
        validate_price_frame(bad, "TEST", max_lag_days=10_000)


def test_core_satellite_paper_scaling_caps_gross():
    overlay = pd.Series({"AMAT": 0.233333, "C": 0.138866, "MU": 0.327801})
    spy, qqq, paper_overlay, raw_gross, scale, scaled = _scale_paper_targets_to_gross(
        target_spy=0.0,
        target_qqq=0.55,
        overlay=overlay,
        max_gross=1.0,
    )
    assert scaled is True
    assert round(raw_gross, 6) == 1.25
    assert round(scale, 6) == 0.8
    assert round(abs(spy) + abs(qqq) + float(paper_overlay.abs().sum()), 6) == 1.0


def test_signal_timestamp_is_timezone_aware():
    ts = pd.Timestamp(_paper_signal_timestamp())
    assert ts.tzinfo is not None


def test_gauntlet_counts_local_signal_fills(monkeypatch):
    monkeypatch.setattr(paper_gauntlet, "DEFAULT_SIGNAL_TIMEZONE", "Asia/Singapore")
    trades = pd.DataFrame({
        "submitted": [True],
        "submitted_at": ["2026-05-04T14:57:29+00:00"],
        "fill_status": ["filled"],
    })
    status = {"signal_predicted_at": "2026-05-04 22:45"}
    current = paper_gauntlet._filter_current_signal_trades(trades, status)
    assert len(current) == 1
    assert paper_gauntlet._order_fill_stats(current)["fill_rate"] == 1.0


def test_execution_slippage_is_adverse_by_side():
    buy_bps, buy_dollars = _execution_slippage({
        "action": "BUY",
        "price": 100.0,
        "broker_dealt_avg_price": 100.10,
        "broker_dealt_qty": 10,
    })
    sell_bps, sell_dollars = _execution_slippage({
        "action": "SELL",
        "price": 100.0,
        "broker_dealt_avg_price": 99.90,
        "broker_dealt_qty": 10,
    })
    assert round(buy_bps, 6) == 10.0
    assert round(sell_bps, 6) == 10.0
    assert round(buy_dollars, 6) == 1.0
    assert round(sell_dollars, 6) == 1.0


def test_health_concentration_separates_core_etfs_from_overlay_stocks():
    concentration = _position_concentration({
        "account_equity": 100_000,
        "position_values": {
            "QQQ": 44_000,
            "MU": 20_000,
            "CAT": 12_000,
        },
    })
    assert concentration["max_ticker"] == "QQQ"
    assert concentration["max_core_ticker"] == "QQQ"
    assert concentration["max_overlay_ticker"] == "MU"
    assert concentration["overlay_ticker_concentration_warning"] is False
    assert concentration["core_ticker_concentration_warning"] is False


def test_moomoo_status_bucket_accepts_common_filled_variants():
    assert _order_status_bucket("FILLED_ALL", 0, 0) == "filled"
    assert _order_status_bucket("ALL_TRADED", 0, 0) == "filled"
    assert _order_status_bucket("SUBMITTED", 0, 10) == "open"
    assert _order_status_bucket("CANCELLED_ALL", 0, 10) == "cancelled"


def test_reconcile_fallback_matches_by_ticker_side_and_qty():
    trade = pd.Series({
        "ticker": "MU",
        "action": "BUY",
        "broker_qty": 25,
        "submitted_at": "2026-05-08T13:30:00+00:00",
    })
    orders = pd.DataFrame([{
        "order_id": "123456789",
        "code": "US.MU",
        "trd_side": "BUY",
        "qty": 25,
        "updated_time": "2026-05-08 13:31:00",
    }])
    match, match_type = _find_broker_order_match(trade, orders)
    assert match is not None
    assert match_type == "ticker_side_qty"
    assert str(match["order_id"]) == "123456789"


def test_open_position_attribution_skips_zero_priced_positions():
    attribution = _open_position_attribution(
        {
            "positions": {"QQQ": 10, "MU": 5},
            "position_values": {"QQQ": 0.0, "MU": 550.0},
        },
        pd.DataFrame([
            {"ticker": "QQQ", "action": "BUY", "fill_status": "filled", "broker_dealt_avg_price": 100.0},
            {"ticker": "MU", "action": "BUY", "fill_status": "filled", "broker_dealt_avg_price": 100.0},
        ]),
    )
    assert "QQQ" in attribution["skipped_unpriced_tickers"]
    assert "QQQ" not in attribution["by_ticker"]
    assert attribution["by_ticker"]["MU"]["open_pnl"] == 50.0


def test_daily_check_verdict_waits_for_open_orders():
    verdict, reasons = _verdict(
        {
            "current_signal_open_orders": 2,
            "approved_for_real_capital": False,
            "readiness_flags": {
                "strategy_ready": True,
                "signal_fresh": True,
                "broker_synced": True,
                "account_aligned": False,
                "slippage_ok": True,
                "drawdown_ok": True,
                "concentration_ok": True,
            },
        },
        [{"name": "health", "status": "ok"}],
    )
    assert verdict == "WAIT_FOR_OPEN_ORDERS"
    assert "current_signal_open_orders=2" in reasons


def test_after_fill_verdict_distinguishes_open_and_aligned():
    open_verdict, _ = _after_fill_verdict(
        {"current_signal_open_orders": 1, "readiness_flags": {"broker_synced": True, "account_aligned": False}},
        [{"name": "sync", "status": "ok"}],
    )
    aligned_verdict, _ = _after_fill_verdict(
        {"current_signal_open_orders": 0, "readiness_flags": {"broker_synced": True, "account_aligned": True}},
        [{"name": "sync", "status": "ok"}],
    )
    assert open_verdict == "STILL_OPEN"
    assert aligned_verdict == "ALIGNED"


def test_prune_snapshots_removes_only_old_timestamped_dirs(tmp_path, monkeypatch):
    root = tmp_path / "paper_snapshots"
    old_dir = root / "20250101T000000Z"
    new_dir = root / "20260501T000000Z"
    odd_dir = root / "keep_me"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    odd_dir.mkdir()
    monkeypatch.setattr(daily_paper_check, "LOGS", tmp_path)
    result = _prune_snapshots(keep_days=30, now=datetime(2026, 5, 9, tzinfo=timezone.utc))
    assert not old_dir.exists()
    assert new_dir.exists()
    assert odd_dir.exists()
    assert str(old_dir) in result["removed"]


def test_submit_readiness_blocks_over_gross_and_closed_market(monkeypatch):
    monkeypatch.setattr(moomoo_paper_trading, "us_regular_market_is_open", lambda: (False, "closed for test"))
    report = build_submit_readiness_report(
        signal=pd.Series({"paper_ready": True, "gates_all_pass": True}),
        orders=pd.DataFrame([{
            "ticker": "QQQ",
            "action": "BUY",
            "target_weight": 1.25,
            "order_value": 1000.0,
            "reason": "rebalance_to_target",
        }]),
        freshness_ok=True,
        freshness_issues=[],
        submit_allowed=True,
        max_submit_gross_exposure=1.0,
    )
    assert report["can_submit_during_regular_market"] is False
    assert any("target gross" in item for item in report["blockers"])
    assert "closed for test" in report["blockers"]


def test_submit_preflight_blocks_current_signal_open_orders(monkeypatch):
    monkeypatch.setattr(moomoo_paper_trading, "us_regular_market_is_open", lambda: (True, "open for test"))
    monkeypatch.setattr(
        moomoo_paper_trading,
        "current_signal_open_orders",
        lambda signal: pd.DataFrame([{"ticker": "CAT", "fill_status": "open"}]),
    )
    ok, blockers = evaluate_broker_submit_preflight(
        signal=pd.Series({"paper_ready": True, "gates_all_pass": True}),
        orders=pd.DataFrame([{
            "ticker": "CAT",
            "action": "BUY",
            "target_weight": 1.0,
            "order_value": 1000.0,
            "reason": "rebalance_to_target",
        }]),
        freshness_ok=True,
        freshness_issues=[],
        submit_allowed=True,
        allow_stale_signal=False,
        allow_closed_market_submit=False,
        allow_leverage_submit=False,
        allow_submit_with_open_orders=False,
        max_submit_gross_exposure=1.0,
    )
    assert ok is False
    assert any("open paper orders" in blocker for blocker in blockers)


def test_current_order_lifecycle_summarizes_open_quantity(monkeypatch):
    monkeypatch.setattr(paper_gauntlet, "DEFAULT_SIGNAL_TIMEZONE", "Asia/Singapore")
    trades = pd.DataFrame([{
        "ticker": "CAT",
        "action": "BUY",
        "broker_order_id": "123",
        "fill_status": "open",
        "broker_requested_qty": 139,
        "broker_dealt_qty": 0,
        "submitted_at": "2026-05-08T13:35:35+00:00",
    }])
    lifecycle = _current_order_lifecycle(trades, {"signal_predicted_at": "2026-05-08T21:17+08:00"})
    assert lifecycle[0]["ticker"] == "CAT"
    assert lifecycle[0]["fill_status"] == "open"
    assert lifecycle[0]["unfilled_qty"] == 139


def test_drift_breakdown_ranks_largest_drift_first():
    breakdown = _drift_breakdown(pd.DataFrame([
        {"ticker": "QQQ", "action": "SKIP", "current_weight": 0.42, "target_weight": 0.44, "drift_abs": 0.02, "order_value": 10},
        {"ticker": "INTC", "action": "BUY", "current_weight": 0.0, "target_weight": 0.18, "drift_abs": 0.18, "order_value": 100},
    ]))
    assert breakdown[0]["ticker"] == "INTC"
    assert breakdown[0]["drift_abs"] == 0.18


def test_stale_open_order_alerts_only_active_old_orders():
    alerts = _stale_open_order_alerts(
        [
            {
                "ticker": "CAT",
                "action": "BUY",
                "fill_status": "open",
                "submitted_at": "2026-05-08T13:00:00+00:00",
                "unfilled_qty": 139,
                "broker_order_id": "1",
            },
            {
                "ticker": "MU",
                "action": "SELL",
                "fill_status": "filled",
                "submitted_at": "2026-05-08T13:00:00+00:00",
                "unfilled_qty": 0,
                "broker_order_id": "2",
            },
        ],
        now=pd.Timestamp("2026-05-08T15:00:00+00:00"),
    )
    assert len(alerts) == 1
    assert alerts[0]["ticker"] == "CAT"
    assert alerts[0]["age_minutes"] == 120.0


def test_go_live_scorecard_tracks_progress_not_just_fail():
    scorecard = _go_live_scorecard({
        "paper_equity_days": 3,
        "current_signal_open_orders": 2,
        "fill_rate": 0.6,
        "max_drift_abs": 0.18,
        "approved_for_real_capital": False,
        "slippage": {"avg_slippage_bps": 1.0},
        "readiness_flags": {
            "strategy_ready": True,
            "signal_fresh": True,
            "broker_synced": True,
            "drawdown_ok": True,
            "concentration_ok": True,
        },
    })
    assert scorecard["passed"] < scorecard["total"]
    names = {item["name"]: item for item in scorecard["criteria"]}
    assert names["paper_equity_days"]["passed"] is False
    assert names["strategy_ready"]["passed"] is True


def test_equity_sanity_flags_large_moves_and_duplicates():
    sanity = _equity_sanity_checks(pd.DataFrame({
        "date": ["2026-05-01", "2026-05-01", "2026-05-02"],
        "equity": [100.0, 130.0, 90.0],
    }))
    assert sanity["ok"] is False
    assert "duplicate_equity_dates" in sanity["warnings"]
    assert any(item.startswith("large_equity_move") for item in sanity["warnings"])


def test_etf_validator_rejects_flat_recent_close():
    idx = pd.bdate_range(start="2025-01-01", periods=260)
    frame = pd.DataFrame({"Close": [100.0] * 260}, index=idx)
    report = _validate_etf_frame(frame, symbol="SPY", max_age_business_days=10_000)
    assert report["ok"] is False
    assert "flat_recent_close" in report["issues"]


def test_config_health_requirement_detects_installed_package():
    check = _requirement_ok("yfinance", ">=0")
    assert check["ok"] is True
    assert check["installed"]
