"""
phase_luck.py — rebalance-day ("timing luck") diagnostic for the incumbent.

PLAIN ENGLISH: the incumbent strategy rebalances every 20 trading days.  The
official stress test compares "buy on the planned day" with "buy one day
late".  This script asks a wider question: how much does the result change
just because of WHICH day the 20-day calendar starts on?  It runs the
unchanged incumbent 40 times: calendar start offsets 0-19, each with on-time
fills (delay 0) and one-day-late fills (delay 1).  Nothing is tuned.

Run from the project root (needs local data/ and signals/):
    python research_evidence/phase_luck_20260926/phase_luck.py [N_OFFSETS] [OUT_JSON]
Output: one JSON row per run with 2023-2026 ("holdout") and 2013-2022 alpha
versus QQQ, in % points.  Research only: nothing is published.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))
from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
import core_satellite_alpha as core
from validation_bundle import load_approved_research_config

offsets = range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "phase_luck.json"

# The incumbent = the currently approved research config.  Unchanged.
config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
# The "panel" is the big table of stocks x dates with features and scores.
specs = load_feature_specs()
panel = core._ensure_robust_score_columns(attach_scores(
    load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
sessions = core._nyse_sessions(dates.min(), dates.max())   # real trading days
hd = int(config.get("holding_days", 20))
print("config shape", config.get("shape"), "hd", hd, "sessions", sessions[0].date(), sessions[-1].date(), flush=True)

rows = []
for delay in (0, 1):
    for k in offsets:
        t0 = time.time()
        cfg = {**config, "entry_delay_days": delay}
        start = sessions[k]
        # Same fixed end for every run so the last labels are always complete.
        equity, trades, extra = core.run_core_satellite(panel, cfg, evaluation_start=start, evaluation_end=sessions[-30])
        bench = core.benchmark_equity(pd.DatetimeIndex(equity.index))
        hold = core._holdout_comparisons(equity, bench, start="2023-01-01", end="2026-12-31")
        pre = core._holdout_comparisons(equity, bench, start="2013-01-01", end="2022-12-31")
        row = {"delay": delay, "offset": k, "start": str(start.date()),
               "holdout_alpha_vs_qqq_pct": hold.get("alpha_vs_qqq_pct"),
               "holdout_return_pct": hold.get("strategy_return_pct"),
               "pre_alpha_vs_qqq_pct": pre.get("alpha_vs_qqq_pct"),
               "secs": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Save after every run so a crash keeps the finished rows.
        out_path.write_text(json.dumps({"config": {k2: v for k2, v in config.items() if isinstance(v, (str, int, float, bool))}, "rows": rows}, indent=1, default=str))
