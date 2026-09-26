"""
edge_core_stop.py — pre-registered add-on: do the live 5% core ETF stops help?

PLAIN ENGLISH: the live paper account puts a 5% trailing stop on SPY and QQQ
(10% on TQQQ).  The backtest never had these stops.  This script replays the
backtest's core ETF holdings day by day with those stops, to see whether they
help (less drawdown without giving up much return) or hurt.

The rule was written down before running: see "Add-on H-edge-core-stop" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  It uses the same 20 on-time
runs on the point-in-time stock list as test S of edge_check.py.

Run from the project root (needs local data/ and signals/, and internet access
once for the S&P 500 membership history):
    python research_evidence/edge_check_20260926/edge_core_stop.py
Output: research_evidence/edge_check_20260926/edge_core_stop.json
Research only: nothing is published and no official report is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("edge_check", HERE / "edge_check.py")
edge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edge)

HYPOTHESIS = "H-edge-core-stop"
# The live trails (alpaca_protection.py defaults).
CORE_TRAIL_PCT = {"SPY": 0.05, "QQQ": 0.05, "TQQQ": 0.10}


def stop_exit_from_close(bars: pd.DataFrame, entry_date, exit_date, stop_pct: float) -> dict:
    """Like edge.stop_exit, but the position is bought at the CLOSE of entry_date.

    PLAIN ENGLISH: the engine buys the core ETFs at the day's closing price,
    so the stop can only start working the next day.  The high-water mark
    starts at that closing price.
    """
    entry_day = pd.Timestamp(entry_date)
    if entry_day not in bars.index:
        raise ValueError(f"no price bar on the entry day {entry_date}")
    entry = float(bars.loc[entry_day, "Close"])
    window = bars.loc[entry_day:pd.Timestamp(exit_date)].iloc[1:]
    plain_exit = float(window["Close"].iloc[-1]) if not window.empty else entry
    high_water = entry
    for day, bar in window.iterrows():
        level = high_water * (1.0 - stop_pct)
        if float(bar["Open"]) <= level:
            return {"stopped": True, "stop_date": day, "plain_return": plain_exit / entry - 1.0,
                    "stop_return": float(bar["Open"]) / entry - 1.0}
        if float(bar["Low"]) <= level:
            return {"stopped": True, "stop_date": day, "plain_return": plain_exit / entry - 1.0,
                    "stop_return": level / entry - 1.0}
        high_water = max(high_water, float(bar["High"]))
    return {"stopped": False, "stop_date": None, "plain_return": plain_exit / entry - 1.0,
            "stop_return": plain_exit / entry - 1.0}


def core_stop_adjusted_equity(equity: pd.Series, trades: pd.DataFrame, etf_bars: dict,
                              session_offset, stop_cost_pct: float, trails: dict = CORE_TRAIL_PCT) -> tuple[pd.Series, dict]:
    """Rebuild the period equity as if every core ETF holding had its live stop.

    Each ETF's weight is core_gross x its core weight, as in the engine.  A
    stopped ETF's money sits in cash for the rest of the period and pays one
    extra ETF trade.
    """
    values = [float(equity.iloc[0])]
    dates = [equity.index[0]]
    stops = 0
    holdings = 0
    for _, trade in trades.iterrows():
        core_gross = float(trade["core_gross"])
        weights = json.loads(trade["core_weights_json"] or "{}")
        delay = int(trade.get("entry_delay_days", 0) or 0)
        entry_day = session_offset(pd.Timestamp(trade["date"]), delay)
        change = 0.0
        for ticker, raw_weight in weights.items():
            weight = core_gross * float(raw_weight)
            ticker = str(ticker).upper()
            if weight == 0.0 or ticker not in trails:
                continue
            holdings += 1
            result = stop_exit_from_close(etf_bars[ticker], entry_day, trade["exit_date"], trails[ticker])
            if result["stopped"]:
                stops += 1
                change += weight * (result["stop_return"] - result["plain_return"])
                change -= abs(weight) * stop_cost_pct
        values.append(values[-1] * (1.0 + float(trade["period_return"]) + change))
        dates.append(pd.Timestamp(trade["label_end_date"]))
    rebuilt = pd.Series(values, index=pd.DatetimeIndex(dates))
    rebuilt = rebuilt[~rebuilt.index.duplicated(keep="last")]
    return rebuilt, {"core_holdings": holdings, "core_stop_exits": stops}


def judge(rows: list[dict]) -> dict:
    """Same keep/drop rule as the stock stop in edge_check.judge."""
    base = edge._median([r["decision_alpha_vs_qqq_pct"] for r in rows if r["test"] == "S"])
    with_stop = edge._median([r["decision_alpha_vs_qqq_pct"] for r in rows if r["test"] == "TC"])
    base_dd = edge._median([r["decision_period_max_drawdown_pct"] for r in rows if r["test"] == "S"])
    stop_dd = edge._median([r["decision_period_max_drawdown_pct"] for r in rows if r["test"] == "TC"])
    alpha_ok = (base is not None and with_stop is not None
                and with_stop >= base - edge.STOP_MAX_ALPHA_GIVEBACK * abs(base))
    dd_ok = base_dd is not None and stop_dd is not None and stop_dd >= base_dd + edge.STOP_MIN_DRAWDOWN_GAIN_PTS
    return {
        "numbers": {"S_median_alpha": base, "TC_median_alpha_with_core_stop": with_stop,
                    "S_median_period_drawdown": base_dd, "TC_median_period_drawdown": stop_dd,
                    "S_diagnostic_median": edge._median([r["diagnostic_alpha_vs_qqq_pct"] for r in rows if r["test"] == "S"]),
                    "TC_diagnostic_median": edge._median([r["diagnostic_alpha_vs_qqq_pct"] for r in rows if r["test"] == "TC"])},
        "gates": {"core_stop_alpha_ok": alpha_ok, "core_stop_drawdown_ok": dd_ok},
        "core_stop_verdict": "keep the core ETF stops" if (alpha_ok and dd_ok) else "drop the core ETF stops",
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered core ETF stop test (research only).")
    parser.add_argument("--offsets", type=int, default=edge.N_OFFSETS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--out", default=str(HERE / "edge_core_stop.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
    import core_satellite_alpha as core
    from settings import DATA_DIR
    from validation_bundle import load_approved_research_config

    membership = edge._membership_module()
    if args.membership_csv:
        raw = Path(args.membership_csv).read_bytes()
    else:
        import requests
        response = requests.get(membership.SOURCE_URL, timeout=60)
        response.raise_for_status()
        raw = response.content
    snapshots = membership.load_snapshots(raw.decode("utf-8"))

    config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
    specs = load_feature_specs(write_health_outputs=False)
    panel = core._ensure_robust_score_columns(attach_scores(
        load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
    panel_pit = panel.loc[edge.point_in_time_mask(panel, snapshots, membership.PREDECESSORS)].reset_index(drop=True)
    sessions = core._nyse_sessions(panel["date"].min(), panel["date"].max())
    end = sessions[-30]
    stop_cost_pct = float(core.ETF_TURNOVER_COST_PCT) * float(config.get("cost_stress", 1.0))
    etf_bars = {}
    for ticker in CORE_TRAIL_PCT:
        path = Path(DATA_DIR) / f"{ticker}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["Open", "High", "Low", "Close"])
        index = pd.to_datetime(bars.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        bars.index = index
        etf_bars[ticker] = bars.sort_index()

    rows: list[dict] = []
    notes = {"membership_source_sha256": hashlib.sha256(raw).hexdigest(), "etf_stop_cost_pct": stop_cost_pct,
             "trails": CORE_TRAIL_PCT}
    for offset in range(args.offsets):
        t0 = time.time()
        equity, trades, extra = core.run_core_satellite(
            panel_pit, {**config, "entry_delay_days": 0}, evaluation_start=sessions[offset], evaluation_end=end)
        rows.append(edge._summarise(core, "S", equity, extra, delay=0, offset=offset, t0=t0))
        stopped, info = core_stop_adjusted_equity(equity, trades, etf_bars, core._session_offset, stop_cost_pct)
        rows.append(edge._summarise(core, "TC", stopped, info, delay=0, offset=offset))
        print(json.dumps(rows[-1]), flush=True)
    result = judge(rows)
    Path(args.out).write_text(json.dumps({
        "hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
        "valid_full_test": args.offsets == edge.N_OFFSETS, "end_date": str(pd.Timestamp(end).date()),
        "notes": notes, "rows": rows, "result": result}, indent=1, default=str))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
