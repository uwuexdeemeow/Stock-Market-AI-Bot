"""Rebalance-calendar robustness report.

PLAIN ENGLISH: these tests use a fake backtest, so they never touch market
data.  They check that each offset really starts the calendar later, that a
broken offset does not stop the others, and that the summary ranks the usual
calendar correctly.
"""
from __future__ import annotations

import pandas as pd

import core_satellite_calendar_robustness as cal


def _panel(n_days: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({"date": list(dates) * 2, "ticker": ["AAA"] * n_days + ["BBB"] * n_days})


def _metrics(holdout_alpha: float) -> dict:
    return {
        "benchmark_comparisons": {"QQQ": {"alpha_pct": 1.0}, "BLEND": {"alpha_pct": 2.0}},
        "holdout_2023_2026": {"strategy_return_pct": 10.0, "alpha_vs_qqq_pct": holdout_alpha,
                              "alpha_vs_blend_pct": holdout_alpha},
        "total_return_pct": 100.0, "cagr_pct": 5.0, "sharpe": 1.0, "max_drawdown_pct": -20.0,
    }


def test_each_offset_starts_the_calendar_later_and_errors_do_not_stop_the_run():
    seen_first_dates = []

    def fake_evaluate(panel, config):
        first = pd.to_datetime(panel["date"]).min()
        seen_first_dates.append(first)
        if len(seen_first_dates) == 2:
            raise ValueError("data gap")
        return _metrics(float(len(seen_first_dates))), None, None

    rows = cal.run_offsets(_panel(), {}, [0, 1, 2], evaluate=fake_evaluate)

    assert [d.strftime("%Y-%m-%d") for d in seen_first_dates] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert rows["status"].tolist() == ["ok", "error", "ok"]
    assert "data gap" in rows.loc[1, "error"]


def test_offset_past_the_data_is_an_error_not_a_crash():
    rows = cal.run_offsets(_panel(2), {}, [5], evaluate=lambda p, c: (_metrics(1.0), None, None))
    assert rows["status"].tolist() == ["error"]


def test_summary_ranks_usual_calendar_and_counts_losing_calendars():
    rows = pd.DataFrame([
        {"offset": 0, "status": "ok", "holdout_alpha_vs_qqq_pct": 30.0},
        {"offset": 1, "status": "ok", "holdout_alpha_vs_qqq_pct": 50.0},
        {"offset": 2, "status": "ok", "holdout_alpha_vs_qqq_pct": -10.0},
        {"offset": 3, "status": "ok", "holdout_alpha_vs_qqq_pct": 0.0},
        {"offset": 4, "status": "error", "holdout_alpha_vs_qqq_pct": None},
    ])
    summary = cal.summarize(rows)
    assert summary["offsets_ok"] == 4 and summary["offsets_failed"] == 1
    assert summary["usual_calendar_holdout_rank"] == 2
    assert summary["holdout_alpha_vs_qqq_nonpositive_share"] == 0.5
    assert summary["holdout_alpha_vs_qqq_pct"]["median"] == 15.0
    assert summary["holdout_alpha_vs_qqq_pct"]["min"] == -10.0


def test_telegram_message_shows_median_worst_and_live_rank():
    summary = {
        "offsets_ok": 19, "offsets_failed": 1,
        "holdout_alpha_vs_qqq_pct": {"median": 25.85, "min": -71.93, "max": 71.78},
        "usual_calendar_holdout_alpha_vs_qqq_pct": 58.09, "usual_calendar_holdout_rank": 5,
        "holdout_alpha_vs_qqq_nonpositive_share": 0.2632,
        "max_drawdown_pct": {"median": -30.97, "min": -34.08},
    }
    text = cal.telegram_message(summary)
    assert "median 25.85%" in text and "worst -71.93%" in text
    assert "rank 5 of 19" in text and "not beating QQQ: 26%" in text
