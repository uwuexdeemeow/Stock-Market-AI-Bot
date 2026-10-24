"""
compare.py — check the independent Alpaca re-build (run.py) against the
project's own rule V code and data.

PLAIN ENGLISH: two things could make the project's V numbers wrong:
  1. the CODE (a bug, or using tomorrow's price by mistake), or
  2. the DATA (data/QQQ.parquet, data/BIL.parquet from the project's
     provider).
We run a small 2 x 2 table to pull them apart:

    project code  + project data   (what H-index-risk reported)
    project code  + Alpaca data    (same code, other data   -> data effect)
    run.py code   + Alpaca data    (other code, same data   -> code effect)

and we compare the month-end QQQ weights one by one.  We also check that
the live signal's calendar (v_rule_signal.py) picks the same signal and
trade days as Alpaca's own exchange calendar.

How to run (from the project root, after run.py):
    python research_evidence/v_rule_alpaca_crosscheck_20261025/compare.py
Writes compare.json next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_evidence" / "index_risk_20261016"))

import index_risk_check as irc          # noqa: E402  project's pre-registered test
import v_rule_helper                    # noqa: E402  the live signal's weight maths
import v_rule_signal                    # noqa: E402  the live signal's calendar
import run as alpaca_run                # noqa: E402  this folder's independent build


def alpaca_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alpaca bars in the shape the project code expects (Open/Close columns)."""
    bars, _ = alpaca_run.load_bars()
    out = []
    for s in ("QQQ", "BIL"):
        out.append(bars[s].rename(columns={"open": "Open", "close": "Close"})[["Open", "Close"]])
    return out[0], out[1]


def main() -> dict:
    sig = pd.read_csv(HERE / "signals.csv", parse_dates=["signal_day", "trade_day"])
    run_equity = pd.read_csv(HERE / "equity.csv", parse_dates=["date"], index_col="date")["equity"]

    p_qqq, p_bil = irc.load_prices(str(ROOT / "data"))
    a_qqq, a_bil = alpaca_frames()

    # Common window: first run.py trade day -> last day both data sets have.
    start = run_equity.index[0]
    end = min(p_bil.index[-1], a_bil.index[-1])
    signal_days = pd.DatetimeIndex(sig["signal_day"])

    # 1) Month-end weights, one by one.
    w = pd.DataFrame({
        "run_py_alpaca": sig.set_index("signal_day")["qqq_weight"],
        "project_code_alpaca_data": [irc.vol_weight(a_qqq["Close"], d) for d in signal_days],
        "live_helper_alpaca_data": [v_rule_helper.qqq_weight(a_qqq["Close"].loc[:d])["qqq_weight"] for d in signal_days],
        "project_code_project_data": [irc.vol_weight(p_qqq["Close"], d) for d in signal_days],
    }, index=signal_days)
    w.to_csv(HERE / "compare_weights.csv", float_format="%.6f")
    code_gap = (w["run_py_alpaca"] - w["project_code_alpaca_data"]).abs()
    helper_gap = (w["run_py_alpaca"] - w["live_helper_alpaca_data"]).abs()
    data_gap = (w["project_code_alpaca_data"] - w["project_code_project_data"]).abs()

    # 2) Performance over the common window.
    targets_p = irc.target_weights("V", p_qqq["Close"], irc.month_end_sessions(p_qqq.index))
    targets_a = irc.target_weights("V", a_qqq["Close"], irc.month_end_sessions(a_qqq.index))
    sim_pp = irc.simulate(p_qqq, p_bil, targets_p, start, end)
    sim_pa = irc.simulate(a_qqq, a_bil, targets_a, start, end)
    run_curve = run_equity.loc[:end] / alpaca_run.CONFIG["initial_cash"]
    perf = {
        "project_code_project_data": irc.metrics(sim_pp["equity"], p_bil["Close"], sim_pp["weights"]),
        "project_code_alpaca_data": irc.metrics(sim_pa["equity"], a_bil["Close"], sim_pa["weights"]),
        "run_py_alpaca_data": irc.metrics(run_curve, a_bil["Close"]),
    }
    yearly = {
        "project_code_project_data": irc.yearly_returns(sim_pp["equity"]),
        "project_code_alpaca_data": irc.yearly_returns(sim_pa["equity"]),
        "run_py_alpaca_data": irc.yearly_returns(run_curve),
    }
    # Day-by-day gap between project code and run.py on the SAME Alpaca data.
    daily_gap = (sim_pa["equity"].pct_change() - run_curve.pct_change()).abs().dropna()

    # 3) Live calendar vs Alpaca calendar for the next signal.
    cal = alpaca_run.load_calendar()
    alpaca_month_ends = alpaca_run.month_end_sessions(cal)
    live_sched = v_rule_signal.month_end_schedule(pd.Timestamp("2026-10-23"))
    nxt = live_sched["next"]
    alpaca_trade_day = cal[cal.searchsorted(nxt, side="right")]
    calendar_check = {
        "live_next_signal_day": str(nxt.date()),
        "alpaca_month_end_for_that_month": str(alpaca_month_ends[(alpaca_month_ends.year == nxt.year) & (alpaca_month_ends.month == nxt.month)][0].date()),
        "alpaca_next_session_after_signal": str(alpaca_trade_day.date()),
        # Skip the unfinished current month: its "last session" in the data
        # is just the newest bar, not a real month end.
        "month_ends_disagree_2016_2026": [str(d.date()) for d in irc.month_end_sessions(p_qqq.loc["2016":].index).difference(alpaca_month_ends)
                                          if d < end.replace(day=1)],
    }

    out = {
        "window": [str(start.date()), str(end.date())],
        "weights": {
            "signal_days": int(len(w)),
            "max_gap_run_py_vs_project_code_same_data": float(code_gap.max()),
            "max_gap_run_py_vs_live_helper_same_data": float(helper_gap.max()),
            "max_gap_project_data_vs_alpaca_data": float(data_gap.max()),
            "mean_gap_project_data_vs_alpaca_data": float(data_gap.mean()),
            "worst_data_gap_day": str(data_gap.idxmax().date()),
        },
        "performance": perf,
        "yearly_returns_pct": yearly,
        "daily_return_gap_same_data": {"max": float(daily_gap.max()), "mean": float(daily_gap.mean())},
        "calendar": calendar_check,
    }
    (HERE / "compare.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
