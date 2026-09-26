"""
tranche_preview.py — research-only preview for hypothesis H-tranche.

PLAIN ENGLISH: the incumbent strategy rebalances every 20 trading days on one
fixed calendar.  Which day that calendar starts on turns out to change the
2023-2026 result enormously ("timing luck").  This script estimates what
happens if the money is split into 4 equal slices ("tranches") that each
rebalance on a different day (5 days apart).

How: run the unchanged incumbent once per calendar start day (offset 0-19),
with on-time fills and with fills one day late.  A 4-tranche book is the
average of 4 of those runs (offsets k, k+5, k+10, k+15).  Each slice starts
with 25% of the money and then drifts; there is no netting of trades between
slices, so costs are slightly overstated (the conservative direction).

Pre-registered pass rule (Documentation/DELAY_STRESS_PAPER_ADVISORY.md):
  1. all 10 books (k = 0-4, on-time and late) beat QQQ over 2023-2026, and
  2. the on-time books' 2023-2026 alpha spread (max - min) is under 73 points.

Run from the project root (needs local data/ and signals/):
    python research_evidence/phase_luck_20260926/tranche_preview.py
Output: research_evidence/phase_luck_20260926/tranche_preview.json
Nothing is published and no official report is written.
"""
import json
import sys
from pathlib import Path

import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))
from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from validation_bundle import load_approved_research_config

OUT_DIR = Path(__file__).resolve().parent
TRANCHES = 4          # number of slices
GAP = 5               # trading days between slice calendars (20 / 4)
PASS_MAX_SPREAD = 73.0

# The incumbent = the currently approved research config.  Unchanged.
config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
holding_days = int(config.get("holding_days", 20))

# The "panel" is the big table of stocks x dates with features and scores.
specs = load_feature_specs()
panel = core._ensure_robust_score_columns(attach_scores(
    load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
sessions = core._nyse_sessions(dates.min(), dates.max())   # real trading days
end = sessions[-30]   # same end for every run, so the last labels are complete


def single_calendar_equity(offset: int, delay: int) -> pd.Series:
    """Equity curve of the incumbent when its calendar starts `offset` days in."""
    cfg = {**config, "entry_delay_days": delay}
    equity, _trades, _extra = core.run_core_satellite(
        panel, cfg, evaluation_start=sessions[offset], evaluation_end=end)
    return equity


def alpha_vs_qqq(equity: pd.Series, start: str, stop: str) -> float:
    """Strategy return minus QQQ return over [start, stop], in % points."""
    bench = core.benchmark_equity(pd.DatetimeIndex(equity.index))
    return float(core._holdout_comparisons(equity, bench, start=start, end=stop)["alpha_vs_qqq_pct"])


rows = []
for delay in (0, 1):
    # One run per possible calendar start day.
    curves = {k: single_calendar_equity(k, delay) for k in range(holding_days)}
    # Put all curves on a common daily calendar (carry the last value forward
    # between rebalance dates) starting from the latest first date.
    first = max(c.index[0] for c in curves.values())
    daily = sessions[(sessions >= first) & (sessions <= end)]
    aligned = {k: c.reindex(c.index.union(daily)).ffill().reindex(daily) for k, c in curves.items()}
    for k in range(GAP):
        members = [k + GAP * i for i in range(TRANCHES)]
        # Each slice starts with the same money: normalise to 1, then average.
        book = sum(aligned[m] / aligned[m].iloc[0] for m in members) / TRANCHES
        row = {
            "delay": delay,
            "calendar_k": k,
            "offsets": members,
            "holdout_alpha_vs_qqq_pct": round(alpha_vs_qqq(book, "2023-01-01", "2026-12-31"), 2),
            "pre_2013_2022_alpha_vs_qqq_pct": round(alpha_vs_qqq(book, "2013-01-01", "2022-12-31"), 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

on_time = [r["holdout_alpha_vs_qqq_pct"] for r in rows if r["delay"] == 0]
all_positive = all(r["holdout_alpha_vs_qqq_pct"] > 0 for r in rows)
spread = max(on_time) - min(on_time)
result = {
    "hypothesis": "H-tranche",
    "research_only": True,
    "approves_trading": False,
    "end_date": str(end.date()),
    "books": rows,
    "all_books_positive_holdout_alpha": all_positive,
    "on_time_holdout_alpha_spread_pts": round(spread, 2),
    "max_allowed_spread_pts": PASS_MAX_SPREAD,
    "preview_pass": bool(all_positive and spread < PASS_MAX_SPREAD),
}
(OUT_DIR / "tranche_preview.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "books"}, indent=1))
