"""Audit medium fixes M1, M2, M3, M4 and M6 (FULL_AUDIT_2026-09-26.md).

PLAIN ENGLISH: small fake inputs check each fix:
  M1  replacement stocks also pass the news check;
  M2  trading cost is measured against the price when the order was sent;
  M3  trades are costed from real (drifted) holdings, ETFs included;
  M4  drawdown sees the dip inside a 20-day period;
  M6  a missing number blocks approval instead of passing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core_satellite_alpha as csa
import core_satellite_nested_walkforward as nested
import execution_cost_calibration as calibration
import robustness_review


# ── M1 ───────────────────────────────────────────────────────────────────────

def test_sentiment_veto_also_checks_replacements(monkeypatch):
    day = pd.DataFrame({
        "date": pd.Timestamp("2026-09-25"),
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "sector": ["S1", "S2", "S3", "S4", "S5"],
        "score": [1.0, 0.9, 0.8, 0.7, 0.6],
    })
    news = {"AAA": -0.8, "DDD": -0.9}   # DDD would be the first replacement
    calls = []

    def fake_news(tickers):
        calls.append(list(tickers))
        return {t: news.get(t, 0.1) for t in tickers}

    monkeypatch.setattr(csa, "SENTIMENT_VETO_ENABLED", True)
    monkeypatch.setattr(csa, "_fetch_live_sentiment", fake_news)
    kwargs = dict(score_col="score", shape="top3", exit_rank_floor=0.0, max_per_sector=5, earnings_blackout_days=0)
    selected = csa._select_sticky_holdings(day, set(), return_col=None, **kwargs)
    final, scores = csa._apply_sentiment_veto(selected, day, **kwargs)

    assert sorted(final["ticker"]) == ["BBB", "CCC", "EEE"]
    assert "DDD" in scores               # the replacement was checked
    assert len(calls) == 3               # first picks, then each replacement


# ── M2 ───────────────────────────────────────────────────────────────────────

def test_cost_calibration_uses_arrival_shortfall_and_skips_stops(tmp_path):
    orders = [{"symbol": "MU", "side": "buy", "order_type": "limit",
               "slippage_bps": 1.0, "arrival_shortfall_bps": 30.0, "filled_qty": 10, "fill_price": 100.0} for _ in range(20)]
    orders += [{"symbol": "MU", "side": "sell", "order_type": "trailing_stop",
                "slippage_bps": 500.0, "arrival_shortfall_bps": 500.0, "filled_qty": 10, "fill_price": 100.0} for _ in range(5)]
    out = calibration.build_execution_cost_calibration({"orders": orders}, data_dir=tmp_path)
    assert out["recommended_one_way_slippage_bps"] == 30.0   # not the ~1 bp fill-minute number
    assert out["sample_count"] == 20                          # stops excluded from both
    assert out["status"] == "ready"
    assert out["arrival_shortfall_samples"] == 20


def test_cost_calibration_falls_back_for_old_fills(tmp_path):
    orders = [{"symbol": "MU", "side": "buy", "order_type": "limit", "slippage_bps": 12.0, "filled_qty": 1, "fill_price": 50.0}] * 3
    out = calibration.build_execution_cost_calibration({"orders": orders}, data_dir=tmp_path)
    assert out["fill_minute_fallback_samples"] == 3
    assert out["status"] == "collecting"


# ── M3 ───────────────────────────────────────────────────────────────────────

def test_first_rebalance_pays_for_etf_and_stock_purchases():
    core = csa._core_target_weights(0.75, {"QQQ": 1.0, "SPY": 0.0}, ["SPY", "QQQ"])
    stock, etf = csa._rebalance_trades(pd.Series(dtype=float), pd.Series({"AAA": 0.25}), core)
    assert (round(stock, 6), round(etf, 6)) == (0.25, 0.75)


def test_unchanged_targets_still_trade_the_drift():
    overlay = pd.Series({"AAA": 0.5})
    core = pd.Series({"QQQ": 0.5})
    # AAA +20%, QQQ flat: portfolio +10% before costs.
    held = csa._drifted_weights(overlay, pd.Series({"AAA": 0.20}), core, {"QQQ": 0.0}, 0.10)
    assert round(held["AAA"], 6) == round(0.6 / 1.1, 6)
    stock, etf = csa._rebalance_trades(held, overlay, core)
    # Rebalancing back to 50/50 sells some AAA and buys some QQQ.
    assert stock > 0.04 and etf > 0.04


# ── M4 ───────────────────────────────────────────────────────────────────────

def test_intra_period_marks_see_a_dip_that_recovers():
    dates = csa._nyse_sessions(pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-31"))
    closes = pd.DataFrame({"AAA": np.r_[np.full(5, 100.0), np.full(5, 60.0), np.full(len(dates) - 10, 100.0)]}, index=dates)
    opens = closes.copy()
    marks = csa._intra_period_marks(
        equity_start=100_000.0, cost=0.0, overlay=pd.Series({"AAA": 1.0}),
        stock_entry_date=dates[0], core_target=pd.Series(dtype=float), etf_entry_date=dates[0],
        exit_date=dates[-1], price_pivots=(opens, closes), daily_etf_prices=None,
    )
    daily = pd.Series(dict(marks))
    assert daily.min() == 60_000.0
    # The period's two end points (100k -> 100k) would show no drawdown at all.
    assert round(csa._max_drawdown_pct(daily), 1) == -40.0


# ── M6 ───────────────────────────────────────────────────────────────────────

def test_missing_selector_uplift_and_drawdown_block_approval():
    result = {"valid": True, "strategy": "core-alpha", "frozen_baseline_available": True,
              "selector_sharpe_uplift_vs_baseline": None,
              "selector_alpha_hit_uplift_vs_baseline": None}
    reasons = nested.approval_status(result)["reasons"]
    for reason in ("selector_sharpe_uplift_missing", "selector_alpha_hit_uplift_missing",
                   "mean_oos_drawdown_missing", "worst_oos_drawdown_missing"):
        assert reason in reasons


def test_stress_review_fails_when_a_drawdown_is_missing():
    execution = {"rows": [{"scenario": "base", "paper_ready": True,
                           "alpha_vs_qqq_pct": 5.0, "alpha_vs_blend_pct": 5.0}]}
    review = robustness_review.evaluate_medium_risk_review(
        survivorship={}, execution=execution, factor_decay={"edge_health_status": "pass"})
    assert review["execution_stress_review"]["pass"] is False
    execution["rows"][0]["max_drawdown_pct"] = -20.0
    review = robustness_review.evaluate_medium_risk_review(
        survivorship={}, execution=execution, factor_decay={"edge_health_status": "pass"})
    assert review["execution_stress_review"]["pass"] is True


def test_survivorship_review_fails_without_delta_numbers():
    survivorship = {"rows": [{"scenario": "watchlist_plus_failed_audit_tickers", "audit_rebalance_selections": 0}],
                    "survivorship_adjusted_score": 1.0}
    review = robustness_review.evaluate_medium_risk_review(
        survivorship=survivorship, execution={}, factor_decay={})
    assert review["survivorship_review"]["pass"] is False
