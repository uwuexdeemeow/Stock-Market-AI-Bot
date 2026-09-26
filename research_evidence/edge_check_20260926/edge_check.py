"""
edge_check.py — pre-registered "is the incumbent's edge real?" check.

PLAIN ENGLISH: the incumbent strategy looks very strong in backtests from
2013 to 2022, but two things could be inflating it:

  1. Hindsight in the stock list.  The strategy picks from a fixed list of
     today's big companies.  In 2012 nobody knew which companies would still
     be big in 2026.  Test S only lets the strategy pick a stock on dates
     when that stock was actually in the S&P 500.
  2. Luck.  Holding only 3 stocks, a strategy can look great by chance.
     Test M ("monkeys") replaces the strategy's scores with random numbers
     100 times.  If the real strategy is not clearly better than random
     picks from the same list, its edge is not shown.

It also answers a question the owner needs for fixing H1: does the 8%
trailing stop that the live account uses actually help?  Test T replays
every stock trade day by day and applies the stop.

The rules for passing were written down BEFORE any run: see "Hypothesis
H-edge" in Documentation/DELAY_STRESS_PAPER_ADVISORY.md.

Run from the project root (needs local data/ and signals/, and internet
access once to download the free S&P 500 membership history):
    python research_evidence/edge_check_20260926/edge_check.py
Quick smoke test (NOT a valid result):
    python research_evidence/edge_check_20260926/edge_check.py --offsets 2 --monkeys 4
Output: research_evidence/edge_check_20260926/edge_check.json
Research only: nothing is published and no official report is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))

OUT_DIR = Path(__file__).resolve().parent

# ── Fixed settings (pre-registered; do not change after results) ───────────
HYPOTHESIS = "H-edge"
N_OFFSETS = 20                        # one per possible calendar start day
N_MONKEYS = 100                       # random-pick runs
DELAYS = (0, 1)                       # on-time fills and one-day-late fills
DECISION_WINDOW = ("2013-01-01", "2022-12-31")    # judged here
DIAGNOSTIC_WINDOW = ("2023-01-01", "2026-12-31")  # printed only
S1_MIN_SHARE_OF_R0 = 0.5              # S must keep half of R0's late alpha
L1_PERCENTILE = 95.0                  # S must beat this monkey percentile
STOP_PCT = 0.08                       # the live account's trailing stop
STOP_MAX_ALPHA_GIVEBACK = 0.10        # stop may cost at most 10% of alpha
STOP_MIN_DRAWDOWN_GAIN_PTS = 1.0      # ... and must cut drawdown by 1 point
# The three score columns the incumbent (score_source = regime_adaptive)
# reads, one per market regime.  The monkeys randomise all three.
SCORE_COLS = ("factor_risk_on_score", "factor_defensive_score", "factor_walkforward_score")

MEMBERSHIP_SCRIPT = Path(__file__).resolve().parents[1] / "survivorship_check_20260926" / "watchlist_membership_check.py"


def _membership_module():
    """Reuse the loader and source URL from the earlier survivorship check."""
    spec = importlib.util.spec_from_file_location("watchlist_membership_check", MEMBERSHIP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Test S: point-in-time stock list (pure function, tested) ───────────────

def point_in_time_mask(panel: pd.DataFrame, snapshots: pd.DataFrame, predecessors: dict) -> pd.Series:
    """True for rows whose stock was in the S&P 500 on that row's date.

    PLAIN ENGLISH: `snapshots` lists the index members on many dates.  For
    each row we look up the latest snapshot on or before its date.  A stock
    also counts as a member if its known older ticker was (for example META
    was listed as FB).  Rows before the first snapshot are not allowed.
    """
    snap_dates = pd.DatetimeIndex(snapshots["date"])
    members = list(snapshots["members"])
    positions = snap_dates.searchsorted(pd.DatetimeIndex(panel["date"]), side="right") - 1
    tickers = panel["ticker"].astype(str).str.upper().to_numpy()
    allowed = np.zeros(len(panel), dtype=bool)
    for i, (pos, ticker) in enumerate(zip(positions, tickers)):
        if pos < 0:
            continue
        names = (ticker,) + tuple(predecessors.get(ticker, ()))
        allowed[i] = any(name in members[pos] for name in names)
    return pd.Series(allowed, index=panel.index)


# ── Test M: random picks (pure function, tested) ───────────────────────────

def randomized_scores(panel: pd.DataFrame, seed: int, cols=SCORE_COLS) -> pd.DataFrame:
    """Copy of `panel` whose score columns hold random numbers.

    A stock gets a random score only where it had a real score, so the
    monkeys can pick from exactly the same stocks on exactly the same days.
    All three regime columns get the SAME random number, so a regime change
    alone doesn't reshuffle the picks.
    """
    out = panel.copy()
    rng = np.random.default_rng(seed)
    noise = pd.Series(rng.random(len(out)), index=out.index)
    for col in cols:
        if col in out.columns:
            out[col] = noise.where(out[col].notna())
    return out


# ── Test T: simulated trailing stop (pure functions, tested) ───────────────

def stop_exit(bars: pd.DataFrame, entry_date, exit_date, stop_pct: float = STOP_PCT) -> dict:
    """Replay one stock trade day by day with a trailing stop.

    `bars` has Open/High/Low/Close by date.  Buy at the Open of `entry_date`;
    without a stop, sell at the Close of `exit_date`.  The stop level is
    (1 - stop_pct) x the highest price seen on EARLIER days (the entry price
    on day one).  If a day opens below the stop, sell at that Open (a gap);
    if its Low touches the stop, sell at the stop level.  Today's High only
    raises the stop from tomorrow on (we can't know the order of the high
    and the low within a day).
    """
    window = bars.loc[pd.Timestamp(entry_date):pd.Timestamp(exit_date)]
    if window.empty:
        raise ValueError(f"no price bars between {entry_date} and {exit_date}")
    entry = float(window["Open"].iloc[0])
    plain_exit = float(window["Close"].iloc[-1])
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


def stop_adjusted_equity(equity: pd.Series, trades: pd.DataFrame, bars_by_ticker: dict,
                         session_offset, stop_cost_pct: float, stop_pct: float = STOP_PCT) -> tuple[pd.Series, dict]:
    """Rebuild the engine's period equity as if every stock had the stop.

    For each 20-day period, each stock's return changes from its plain
    return to its stopped return (both from the same daily bars), weighted by
    its overlay weight.  A stopped stock's money sits in cash for the rest of
    the period.  Each stop exit pays one extra stock trade.  Everything else
    (ETF core, regime, costs) is the engine's own number.
    """
    values = [float(equity.iloc[0])]
    dates = [equity.index[0]]
    stops = 0
    positions = 0
    for _, trade in trades.iterrows():
        weights = json.loads(trade["overlay_weights_json"] or "{}")
        delay = int(trade.get("entry_delay_days", 0) or 0)
        stock_entry = session_offset(pd.Timestamp(trade["date"]), 1 + delay)
        change = 0.0
        for ticker, weight in weights.items():
            positions += 1
            result = stop_exit(bars_by_ticker[str(ticker).upper()], stock_entry, trade["exit_date"], stop_pct)
            if result["stopped"]:
                stops += 1
                change += float(weight) * (result["stop_return"] - result["plain_return"])
                change -= abs(float(weight)) * stop_cost_pct
        values.append(values[-1] * (1.0 + float(trade["period_return"]) + change))
        dates.append(pd.Timestamp(trade["label_end_date"]))
    rebuilt = pd.Series(values, index=pd.DatetimeIndex(dates))
    rebuilt = rebuilt[~rebuilt.index.duplicated(keep="last")]
    return rebuilt, {"stock_positions": positions, "stop_exits": stops}


# ── Judging (pure function, tested with fake rows) ─────────────────────────

def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _alphas(rows, test, delay, key="decision_alpha_vs_qqq_pct"):
    return [r[key] for r in rows if r["test"] == test and r["delay"] == delay and r.get(key) is not None]


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-edge rules."""
    r0_late = _median(_alphas(rows, "R0", 1))
    s_late = _median(_alphas(rows, "S", 1))
    s_on_time = _median(_alphas(rows, "S", 0))
    monkeys = _alphas(rows, "M", 0)
    monkey_p95 = float(np.percentile(monkeys, L1_PERCENTILE)) if monkeys else None
    t_alpha = _median(_alphas(rows, "T", 0))
    t_dd = _median(_alphas(rows, "T", 0, "decision_period_max_drawdown_pct"))
    s_dd = _median(_alphas(rows, "S", 0, "decision_period_max_drawdown_pct"))

    s1 = (s_late is not None and r0_late is not None and s_late > 0
          and s_late >= S1_MIN_SHARE_OF_R0 * r0_late)
    l1 = s_on_time is not None and monkey_p95 is not None and s_on_time > monkey_p95
    stop_alpha_ok = (t_alpha is not None and s_on_time is not None
                     and t_alpha >= s_on_time - STOP_MAX_ALPHA_GIVEBACK * abs(s_on_time))
    # Drawdowns are negative numbers: "shallower" means closer to zero.
    stop_dd_ok = t_dd is not None and s_dd is not None and t_dd >= s_dd + STOP_MIN_DRAWDOWN_GAIN_PTS
    monkey_rank = (float(np.mean([m < s_on_time for m in monkeys]) * 100.0)
                   if monkeys and s_on_time is not None else None)
    return {
        "numbers": {
            "R0_median_late_alpha": r0_late, "S_median_late_alpha": s_late,
            "S_median_on_time_alpha": s_on_time, "monkey_p95_alpha": monkey_p95,
            "monkey_median_alpha": _median(monkeys),
            "S_on_time_beats_pct_of_monkeys": monkey_rank,
            "T_median_alpha_with_stop": t_alpha,
            "T_median_period_drawdown_with_stop": t_dd, "S_median_period_drawdown": s_dd,
            "R0_diagnostic_median": _median(_alphas(rows, "R0", 0, "diagnostic_alpha_vs_qqq_pct")),
            "S_diagnostic_median": _median(_alphas(rows, "S", 0, "diagnostic_alpha_vs_qqq_pct")),
            "M_diagnostic_median": _median(_alphas(rows, "M", 0, "diagnostic_alpha_vs_qqq_pct")),
            "T_diagnostic_median": _median(_alphas(rows, "T", 0, "diagnostic_alpha_vs_qqq_pct")),
        },
        "gates": {"S1_survivorship": s1, "L1_beats_random_picks": l1,
                  "stop_alpha_ok": stop_alpha_ok, "stop_drawdown_ok": stop_dd_ok},
        "edge_verdict": "edge shown" if (s1 and l1) else "edge not shown",
        "stop_verdict": "keep the 8% stop" if (stop_alpha_ok and stop_dd_ok) else "drop the 8% stop",
    }


# ── Running the engine ─────────────────────────────────────────────────────

def _max_drawdown_pct(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min() * 100.0) if len(equity) else 0.0


def _window_alpha(core, equity: pd.Series, window: tuple[str, str]) -> float | None:
    bench = core.benchmark_equity(pd.DatetimeIndex(equity.index))
    return core._holdout_comparisons(equity, bench, start=window[0], end=window[1]).get("alpha_vs_qqq_pct")


def _summarise(core, test: str, equity: pd.Series, extra: dict, *, delay: int, offset: int, seed=None, t0=None) -> dict:
    in_window = equity[(equity.index >= DECISION_WINDOW[0]) & (equity.index <= DECISION_WINDOW[1])]
    row = {
        "test": test, "delay": delay, "offset": offset,
        "decision_alpha_vs_qqq_pct": _window_alpha(core, equity, DECISION_WINDOW),
        "diagnostic_alpha_vs_qqq_pct": _window_alpha(core, equity, DIAGNOSTIC_WINDOW),
        "decision_period_max_drawdown_pct": round(_max_drawdown_pct(in_window), 2),
    }
    if seed is not None:
        row["seed"] = seed
    row.update({k: extra.get(k) for k in ("turnover_pct", "avg_overlay_positions", "top_ticker_overlay_contributor",
                                          "stock_positions", "stop_exits", "core_holdings", "core_stop_exits")
                if k in extra})
    if t0 is not None:
        row["secs"] = round(time.time() - t0, 1)
    return row


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered edge check (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS, help="calendar start days (20 = full test)")
    parser.add_argument("--monkeys", type=int, default=N_MONKEYS, help="random-pick runs (100 = full test)")
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--out", default=str(OUT_DIR / "edge_check.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
    import core_satellite_alpha as core
    from settings import DATA_DIR
    from validation_bundle import load_approved_research_config

    membership = _membership_module()
    if args.membership_csv:
        raw = Path(args.membership_csv).read_bytes()
    else:
        import requests
        response = requests.get(membership.SOURCE_URL, timeout=60)
        response.raise_for_status()
        raw = response.content
    snapshots = membership.load_snapshots(raw.decode("utf-8"))

    config = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
    # write_health_outputs=False: research must not rewrite daily-workflow files.
    specs = load_feature_specs(write_health_outputs=False)
    panel = core._ensure_robust_score_columns(attach_scores(
        load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
    missing = [c for c in SCORE_COLS if c not in panel.columns]
    if missing:
        raise SystemExit(f"Score columns missing from the panel: {missing}")
    mask = point_in_time_mask(panel, snapshots, membership.PREDECESSORS)
    panel_pit = panel.loc[mask].reset_index(drop=True)

    sessions = core._nyse_sessions(panel["date"].min(), panel["date"].max())
    end = sessions[-30]   # same end for every run, so the last labels are complete
    stop_cost_pct = float(core.calibrated_turnover_cost_pct()) * float(config.get("cost_stress", 1.0))
    bars_by_ticker = {}
    for ticker in sorted(panel["ticker"].astype(str).str.upper().unique()):
        bars = pd.read_parquet(Path(DATA_DIR) / f"{ticker}.parquet", columns=["Open", "High", "Low", "Close"])
        index = pd.to_datetime(bars.index)
        if index.tz is not None:
            index = index.tz_localize(None)   # match the engine's plain dates
        bars.index = index
        bars_by_ticker[ticker] = bars.sort_index()

    notes = {
        "membership_source_url": membership.SOURCE_URL,
        "membership_source_sha256": hashlib.sha256(raw).hexdigest(),
        "panel_rows": int(len(panel)), "point_in_time_rows": int(len(panel_pit)),
        "rows_removed_share": round(1.0 - len(panel_pit) / max(len(panel), 1), 4),
        "stop_cost_pct": stop_cost_pct,
        "score_cols_used_by_monkey_runs": set(),
    }
    out_path = Path(args.out)
    rows: list[dict] = []

    def save(result: dict | None = None) -> None:
        payload = {
            "hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
            "valid_full_test": args.offsets == N_OFFSETS and args.monkeys == N_MONKEYS,
            "end_date": str(pd.Timestamp(end).date()), "offsets": args.offsets, "monkeys": args.monkeys,
            "notes": {**notes, "score_cols_used_by_monkey_runs": sorted(notes["score_cols_used_by_monkey_runs"])},
            "rows": rows,
        }
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    def record(row: dict) -> None:
        rows.append(row)
        print(json.dumps(row), flush=True)
        save()   # a crash keeps the finished rows

    for test, test_panel in (("R0", panel), ("S", panel_pit)):
        for delay in DELAYS:
            for offset in range(args.offsets):
                t0 = time.time()
                run_cfg = {**config, "entry_delay_days": delay}
                equity, trades, extra = core.run_core_satellite(
                    test_panel, run_cfg, evaluation_start=sessions[offset], evaluation_end=end)
                record(_summarise(core, test, equity, extra, delay=delay, offset=offset, t0=t0))
                if test == "S" and delay == 0:
                    # Test T: the same run, replayed with the 8% trailing stop.
                    t1 = time.time()
                    stopped, stop_info = stop_adjusted_equity(
                        equity, trades, bars_by_ticker, core._session_offset, stop_cost_pct)
                    record(_summarise(core, "T", stopped, stop_info, delay=0, offset=offset, t0=t1))

    for seed in range(args.monkeys):
        t0 = time.time()
        offset = seed % N_OFFSETS
        monkey_panel = randomized_scores(panel_pit, seed)
        equity, trades, extra = core.run_core_satellite(
            monkey_panel, {**config, "entry_delay_days": 0}, evaluation_start=sessions[offset], evaluation_end=end)
        if "score_col" in trades:
            notes["score_cols_used_by_monkey_runs"].update(str(c) for c in trades["score_col"].unique())
        # Blended columns are mixes of the randomised columns, so they are fine.
        unexpected = {c for c in notes["score_cols_used_by_monkey_runs"]
                      if c not in SCORE_COLS and not c.startswith("_blended_regime_score_")}
        if unexpected:
            # A column the monkeys didn't randomise would make them part-real.
            raise SystemExit(f"Monkey runs used non-randomised score columns: {sorted(unexpected)}")
        record(_summarise(core, "M", equity, extra, delay=0, offset=offset, seed=seed, t0=t0))

    result = judge(rows)
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "edge_verdict", "stop_verdict")}, indent=1))
    return result


if __name__ == "__main__":
    main()
