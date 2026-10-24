"""Tests for the pre-registered H-multi-asset check (fake prices only).

PLAIN ENGLISH: the check compares a QQQ/TLT/GLD mix (with and without
borrowing) against rule V.  These tests pin the mix weights, the
volatility scaling, V matching the older H-index-risk engine, the trading
costs, the loan interest and the pass/fail gates.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ma = _load("multi_asset_check", "research_evidence/multi_asset_20261025/multi_asset_check.py")
ir = _load("index_risk_check_for_ma", "research_evidence/index_risk_20261016/index_risk_check.py")


def _walk(n, vol, seed, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, vol / np.sqrt(252), n))
    return pd.Series(close, index=days)


def _trade(closes: dict) -> dict:
    """Open = Close for simplicity; one frame per ETF."""
    return {s: pd.DataFrame({"Open": c.to_numpy(), "Close": c.to_numpy()}, index=c.index) for s, c in closes.items()}


def test_mix_weights_are_inverse_vol_and_scaled_to_target():
    risky = pd.DataFrame({"QQQ": _walk(200, 0.30, 1), "TLT": _walk(200, 0.15, 2), "GLD": _walk(200, 0.15, 3)})
    day = risky.index[-1]
    w = ma.mix_target(risky, day, cap=1.0)
    returns = risky.pct_change().dropna().tail(63)
    vols = returns.apply(ma.annual_vol)
    base = (1 / vols) / (1 / vols).sum()
    # Each risky weight is proportional to base, and nothing is above 100%.
    k = w["QQQ"] / base["QQQ"]
    assert w["TLT"] / base["TLT"] == pytest.approx(k)
    assert sum(w[s] for s in ma.RISKY) + w["BIL"] == pytest.approx(1.0)
    mix_vol = ma.annual_vol(returns.to_numpy() @ base.to_numpy())
    assert k == pytest.approx(min(1.0, 0.20 / mix_vol))


def test_borrowing_cap_holds_no_bil_and_never_exceeds_two():
    calm = pd.DataFrame({s: _walk(200, 0.03, i) for i, s in enumerate(ma.RISKY)})
    w = ma.mix_target(calm, calm.index[-1], cap=2.0)
    assert sum(w[s] for s in ma.RISKY) == pytest.approx(2.0)   # very calm: hits the cap
    assert w["BIL"] == 0.0
    w1 = ma.mix_target(calm, calm.index[-1], cap=1.0)
    assert sum(w1[s] for s in ma.RISKY) == pytest.approx(1.0)


def test_v_target_matches_h_index_risk_rule():
    qqq = _walk(300, 0.35, 7)
    for day in qqq.index[[70, 150, 299]]:
        assert ma.v_target(qqq, day)["QQQ"] == pytest.approx(ir.vol_weight(qqq, day))


def test_v_simulation_matches_h_index_risk_engine():
    qqq, bil = _walk(400, 0.30, 11), _walk(400, 0.01, 12)
    flat = pd.Series(100.0, index=qqq.index)
    trade = _trade({"QQQ": qqq, "TLT": flat, "GLD": flat, "BIL": bil})
    signal_days = ma.month_end_sessions(qqq.index)[:-1]
    prices = {"qqq_close": qqq}
    targets = ma.target_table("V", prices, signal_days)
    start, end = qqq.index[100], qqq.index[-1]
    # Costs off: the price moves and trade timing must agree exactly.  (With
    # costs on, this engine charges both the QQQ and the BIL leg, as the
    # pre-registration says; H-index-risk charged the QQQ leg only.)
    new = ma.simulate(trade, targets, start, end, cost_bps=0)
    old = ir.simulate(trade["QQQ"], trade["BIL"], targets["QQQ"], start, end, cost_bps=0)
    assert new["equity"].to_numpy() == pytest.approx(old["equity"].to_numpy(), rel=1e-12)


def test_first_purchase_and_rebalance_costs():
    days = pd.bdate_range("2021-01-01", periods=30)
    flat = pd.Series(100.0, index=days)
    trade = _trade({s: flat for s in (*ma.RISKY, "BIL")})
    targets = pd.DataFrame([{"QQQ": 0.5, "TLT": 0.25, "GLD": 0.25, "BIL": 0.0},
                            {"QQQ": 0.4, "TLT": 0.3, "GLD": 0.3, "BIL": 0.0}],
                           index=[days[0], days[10]])
    sim = ma.simulate(trade, targets, days[1], days[-1], cost_bps=10)
    # Buy everything (pay on 100%), then change 0.1+0.05+0.05 = 0.2 of weight.
    assert sim["equity"].iloc[-1] == pytest.approx((1 - 0.001) * (1 - 0.001 * 0.2))


def test_loan_interest_is_bil_plus_spread():
    days = pd.bdate_range("2021-01-01", periods=11)
    flat = pd.Series(100.0, index=days)
    bil = pd.Series(100.0 * 1.0001 ** np.arange(11), index=days)     # BIL earns 1 bp a day
    trade = _trade({"QQQ": flat, "TLT": flat, "GLD": flat, "BIL": bil})
    targets = pd.DataFrame([{"QQQ": 1.0, "TLT": 0.5, "GLD": 0.5, "BIL": 0.0}], index=[days[0]])
    sim = ma.simulate(trade, targets, days[1], days[-1], cost_bps=0, borrow_spread=0.0252)
    # Gross 2: the loan equals the starting money and grows 9 days at (1 bp + 1 bp).
    debt_end = (1.0001 + 0.0001) ** 9
    assert sim["equity"].iloc[-1] == pytest.approx(2.0 - debt_end, rel=1e-9)
    assert sim["interest_paid"] == pytest.approx(debt_end - 1.0, rel=1e-9)


def test_signal_trades_at_next_open_and_stress_one_later():
    days = pd.bdate_range("2021-01-29", periods=6)        # 2021-01-29 is a month end
    qqq = pd.Series([100, 100, 110, 121, 121, 121], index=days, dtype=float)
    flat = pd.Series(100.0, index=days)
    trade = _trade({"QQQ": qqq, "TLT": flat, "GLD": flat, "BIL": flat})
    targets = pd.DataFrame([{"QQQ": 0.0, "TLT": 0, "GLD": 0, "BIL": 1.0},
                            {"QQQ": 1.0, "TLT": 0, "GLD": 0, "BIL": 0.0}],
                           index=[days[0] - pd.Timedelta(days=1), days[0]])
    on_time = ma.simulate(trade, targets, days[0], days[-1], cost_bps=0)
    late = ma.simulate(trade, targets, days[0], days[-1], cost_bps=0, delay=1)
    assert on_time["equity"].iloc[-1] == pytest.approx(1.21)   # bought at the 100 Open on day 1
    assert late["equity"].iloc[-1] == pytest.approx(1.10)      # bought at the 110 Open on day 2


def test_gates_follow_the_preregistration():
    base = {"cagr_pct": 11.0, "sharpe": 0.60, "max_drawdown_pct": -38.0}
    decision = {"V": base,
                "M": {"cagr_pct": 8.0, "sharpe": 0.71, "max_drawdown_pct": -30.0},
                "ML": {"cagr_pct": 12.1, "sharpe": 0.62, "max_drawdown_pct": -42.5}}
    stress = {"V": base,
              "M": {"cagr_pct": 7.0, "sharpe": 0.69, "max_drawdown_pct": -31.0},
              "ML": {"cagr_pct": 11.5, "sharpe": 0.55, "max_drawdown_pct": -45.0}}
    subperiods = {"a": {"V": 1, "ML": 2}, "b": {"V": 1, "ML": 2}, "c": {"V": 3, "ML": 2}}
    gates = ma.judge(decision, stress, subperiods)
    assert gates["M"]["passes_decision"] is False          # stress Sharpe edge only 0.09
    assert gates["ML"]["L1_more_return"] and gates["ML"]["L2_drawdown"] and gates["ML"]["L4_subperiods"]
    assert gates["ML"]["L5_robust"] is False               # stress Sharpe below V's
    assert ma.judge_exam({"V": base, "ML": {"sharpe": 0.41, "max_drawdown_pct": -43.0}}, "ML")["passes"]


def test_period_cagr_uses_prior_close():
    days = pd.bdate_range("2020-12-30", "2022-01-03")
    curve = pd.Series(1.0, index=days)
    curve[curve.index >= "2021-01-01"] = 1.1
    assert ma.period_cagr(curve, "2021-01-01", "2021-12-31") > 9.0
