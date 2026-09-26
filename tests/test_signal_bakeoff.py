"""Pre-registered signal bake-off (research_evidence/signal_bakeoff_20260926).

PLAIN ENGLISH: the bake-off compares new signal ideas under one fixed rule.
These tests use small fake tables, never real market data.  They check that
each idea's score is built the way the pre-registration says, that idea B
cannot peek at the future, that the sector-ETF rows use the same "buy next
open, sell 20 closes later" labels as the stock rows, and that the judge picks
a winner (or none) exactly by the written rule.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "signal_bakeoff", ROOT / "research_evidence" / "signal_bakeoff_20260926" / "signal_bakeoff.py")
bakeoff = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bakeoff)


# ── Idea A ─────────────────────────────────────────────────────────────────

def test_momentum_score_averages_ranks_and_skips_unknown_values():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4),
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "factor_mom_12_1": [0.30, 0.10, -0.20, 0.0],          # DDD: 0.0 = not known yet
        "factor_resid_mom_sector_12_1": [0.20, 0.25, -0.10, 0.05],
    })
    score = bakeoff.momentum_score(panel)
    assert np.isnan(score.iloc[3])
    # AAA ranks 1st and 2nd -> best average; CCC is last on both.
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]


# ── Idea B ─────────────────────────────────────────────────────────────────

def _b_panel(n_days: int = 1850, n_tickers: int = 30, seed: int = 0) -> pd.DataFrame:
    """Fake panel: 'good' predicts late returns before 2016; 'future' only after."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    rows = []
    for i, date in enumerate(dates):
        good = rng.normal(size=n_tickers)
        future = rng.normal(size=n_tickers)
        noise = rng.normal(size=n_tickers)
        late = np.where(date.year < 2016, good, future) + 0.3 * noise
        end = dates[min(i + 21, len(dates) - 1)]
        for t in range(n_tickers):
            rows.append({
                "date": date, "ticker": f"T{t}",
                "good": good[t], "bad_sign": -good[t], "future": future[t], "noise_feat": rng.normal(),
                "forward_return_delay1_20d": late[t], "forward_return_20d": late[t],
                "forward_return_delay1_20d_end_date": end,
            })
    return pd.DataFrame(rows)


def test_b_selects_predictive_features_and_keeps_their_direction():
    panel = _b_panel()
    train = panel[panel["date"] < "2015-06-01"]
    chosen = bakeoff.select_b_features(train, pool=("good", "bad_sign", "future", "noise_feat"))
    names = {row["feature"]: row for row in chosen}
    assert "good" in names and names["good"]["late_ic"] > 0
    assert "bad_sign" in names and names["bad_sign"]["late_ic"] < 0
    assert "future" not in names and "noise_feat" not in names


def test_b_year_score_uses_only_labels_finished_before_the_year(monkeypatch):
    panel = _b_panel()
    monkeypatch.setattr(bakeoff, "B_FEATURE_POOL", ("good", "future"))
    seen_max_end = {}
    real_select = bakeoff.select_b_features

    def spy(train, pool=bakeoff.B_FEATURE_POOL):
        year = int(train["date"].max().year) + 1 if not train.empty else None
        if not train.empty:
            seen_max_end[year] = pd.Timestamp(train["forward_return_delay1_20d_end_date"].max())
        return real_select(train, pool=("good", "future"))

    monkeypatch.setattr(bakeoff, "select_b_features", spy)
    score, chosen = bakeoff.horizon_matched_score(panel, [2014, 2016])
    for year, max_end in seen_max_end.items():
        assert max_end < pd.Timestamp(f"{year}-01-01")
    # 'future' only starts predicting in 2016, so 2016's choice can't know it.
    assert [row["feature"] for row in chosen["2016"]] == ["good"]
    in_2016 = score[panel["date"].dt.year == 2016]
    assert len(in_2016) > 0 and in_2016.notna().all()
    # Scores are daily ranks between 0 and 1.
    assert in_2016.between(0, 1).all()
    # Years without enough history get no score (no overlay picks).
    assert score[panel["date"].dt.year == 2010].isna().all()


# ── Idea C ─────────────────────────────────────────────────────────────────

def _etf_frame(start: str, n: int, drift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(drift + 0.01 * rng.normal(size=n)))
    return pd.DataFrame({"Open": close * (1 + 0.001 * rng.normal(size=n)), "High": close, "Low": close,
                         "Close": close, "Volume": 1e6}, index=idx)


def test_etf_labels_match_the_stock_panel_loader(tmp_path, monkeypatch):
    import alpha_factor_backtest as afb

    frame = _etf_frame("2012-01-02", 400, 0.0005, 1)
    frame["dummy_feature"] = 1.0
    frame.to_parquet(tmp_path / "AAA.parquet")
    monkeypatch.setattr(afb, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(afb, "WATCHLIST", ["AAA"])
    monkeypatch.setattr(afb, "apply_membership_if_complete", lambda panel: (panel, {}))
    stock = afb.load_factor_panel([{"feature": "dummy_feature"}], require_forward_returns=False)

    etf = bakeoff.etf_rotation_panel({"AAA": frame}, frame.index.min(), frame.index.max())
    for label in ("forward_return_20d", "forward_return_delay1_20d"):
        for col in (label, f"{label}_end_date", f"{label}_entry_date"):
            pd.testing.assert_series_equal(
                etf[col].reset_index(drop=True), stock[col].reset_index(drop=True),
                check_names=False, check_dtype=False)


def test_etf_below_its_200_day_average_cannot_be_picked():
    up = _etf_frame("2009-01-02", 600, 0.002, 2)
    down = _etf_frame("2009-01-02", 600, -0.002, 3)
    flat = _etf_frame("2009-01-02", 600, 0.0, 4)
    panel = bakeoff.etf_rotation_panel({"UPP": up, "DWN": down, "FLT": flat}, "2011-01-01", "2011-06-30")
    last = panel[panel["date"] == panel["date"].max()].set_index("ticker")
    assert np.isnan(last.loc["DWN", bakeoff.ENGINE_SCORE_COL])
    assert last.loc["UPP", bakeoff.ENGINE_SCORE_COL] > 0
    assert panel["date"].min() >= pd.Timestamp("2011-01-01")


# ── Configs ────────────────────────────────────────────────────────────────

INCUMBENT = {"score_source": "regime_adaptive", "shape": "top3", "weighting": "sticky_score",
             "overlay_gross": 0.5, "max_per_sector": 2, "earnings_blackout_days": 5,
             "regime_preset": {"risk_on": {"overlay_gross": 0.5}}, "holding_days": 20}


def test_configs_change_only_the_preregistered_keys():
    assert bakeoff.idea_config("R0", INCUMBENT) == INCUMBENT
    zero = {"risk_on": {"overlay_gross": 0.0}}
    e = bakeoff.idea_config("E", INCUMBENT, zero)
    assert e["overlay_gross"] == 0.0 and e["regime_preset"] == zero
    for idea, shape in (("A", "top10"), ("B", "top10"), ("C", "top3")):
        cfg = bakeoff.idea_config(idea, INCUMBENT)
        assert cfg["score_source"] == "factor_walkforward" and cfg["shape"] == shape
        assert cfg["weighting"] == "equal" and cfg["holding_days"] == 20
        assert cfg["regime_preset"] == INCUMBENT["regime_preset"]   # same core and regimes
    assert INCUMBENT["shape"] == "top3"   # the incumbent dict itself is untouched


# ── Judge ──────────────────────────────────────────────────────────────────

def _rows(idea: str, on_time: float, late: float, spread: float = 10.0, n: int = 20) -> list[dict]:
    rows = []
    for delay, base in ((0, on_time), (1, late)):
        for k in range(n):
            rows.append({"idea": idea, "delay": delay, "offset": k,
                         "decision_alpha_vs_qqq_pct": base + spread * (k / (n - 1) - 0.5),
                         "diagnostic_alpha_vs_qqq_pct": 1.0, "decision_max_drawdown_pct": -20.0,
                         "turnover_pct": 500.0})
    return rows


def test_judge_picks_the_best_late_alpha_among_eligible_candidates():
    rows = (_rows("E", -50, -50) + _rows("R0", 300, 250, spread=120) + _rows("R1", 100, 98)
            + _rows("A", 120, 118) + _rows("B", 150, 140) + _rows("C", 30, 29))
    result = bakeoff.judge(rows)
    assert result["summary"]["B"]["gates"]["G3_delay_cost_under_limit"] is False   # costs 10 points
    assert result["eligible"] == ["A", "C"]
    # C (29) is below half of A (118), so A wins despite C having no survivor bias.
    assert result["winner"] == "A"


def test_judge_prefers_survivor_free_c_when_it_is_close_enough():
    rows = _rows("E", -50, -50) + _rows("A", 100, 98) + _rows("C", 60, 59)
    assert bakeoff.judge(rows)["winner"] == "C"


def test_judge_names_no_winner_when_a_run_loses_to_qqq_or_runs_are_missing():
    rows = _rows("E", -50, -50) + _rows("A", 20, 19, spread=60) + _rows("C", 50, 49)[:-1]
    result = bakeoff.judge(rows)
    assert result["summary"]["A"]["gates"]["G1_every_run_beats_qqq"] is False
    assert result["summary"]["C"]["gates"]["G0_all_runs_present"] is False
    assert result["winner"] is None and result["eligible"] == []


def test_judge_requires_beating_the_no_overlay_control():
    rows = _rows("E", 200, 200) + _rows("A", 120, 118)
    result = bakeoff.judge(rows)
    assert result["summary"]["A"]["gates"]["G4_beats_no_overlay_control"] is False
    assert result["winner"] is None


def test_run_one_reports_both_windows(monkeypatch):
    dates = pd.bdate_range("2012-01-02", "2024-12-31", freq="20B")
    equity = pd.Series(np.linspace(100, 300, len(dates)), index=dates)

    class FakeCore:
        @staticmethod
        def run_core_satellite(panel, cfg, evaluation_start, evaluation_end):
            assert cfg["entry_delay_days"] == 1
            return equity, pd.DataFrame(), {"turnover_pct": 123.0}

        @staticmethod
        def benchmark_equity(index):
            flat = pd.Series(100.0, index=index)
            return pd.DataFrame({"SPY": flat, "QQQ": flat, "BLEND": flat})

        from core_satellite_alpha import _holdout_comparisons as _hc
        _holdout_comparisons = staticmethod(_hc)

    sessions = pd.bdate_range("2012-01-02", periods=30)
    row = bakeoff.run_one(FakeCore, pd.DataFrame(), {}, sessions, 3, 1, dates[-1])
    assert row["offset"] == 3 and row["delay"] == 1
    assert row["decision_alpha_vs_qqq_pct"] > 0 and row["diagnostic_alpha_vs_qqq_pct"] > 0
    assert row["turnover_pct"] == 123.0


def test_etf_dates_with_a_time_zone_are_accepted():
    frame = _etf_frame("2009-01-02", 400, 0.001, 5)
    frame.index = frame.index.tz_localize("America/New_York")
    panel = bakeoff.etf_rotation_panel({"TZZ": frame}, "2010-01-01", "2010-06-30")
    assert panel["date"].dt.tz is None and not panel.empty


def test_dry_run_output_check_catches_missing_runs_and_bad_flags():
    spec = importlib.util.spec_from_file_location(
        "dry_run", ROOT / "research_evidence" / "signal_bakeoff_20260926" / "dry_run_fake_data.py")
    dry_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry_run)
    rows = [{"idea": idea} for idea in ("R0", "R1", "E", "A", "B", "C") for _ in range(4)]
    good = {"rows": rows, "valid_full_test": False, "approves_trading": False,
            "result": {"summary": {idea: {"gates": {}} for idea in ("A", "B", "C")}}}
    assert dry_run.check_output(good, 2) == []
    bad = {**good, "rows": rows[:-1], "approves_trading": True}
    assert len(dry_run.check_output(bad, 2)) == 2
