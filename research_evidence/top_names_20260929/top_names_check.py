"""
top_names_check.py — pre-registered "H-top-names" concentration test.

PLAIN ENGLISH: the incumbent strategy holds only 3 stocks at a time.  A great
backtest could come from real skill spread over many picks, or from a few
lucky names (NVDA alone was 20% of overlay profit) or a few giant winners.
This script asks two questions:

  N1 / N3: what if the 1 or 3 stocks that earned the most had never been
           on the list?  (The engine then picks the next-best stocks.)
  P1:      what if, in every 20-day period, the best of the 3 picks had only
           earned what a typical stock earned?  The same cut is applied to
           100 random-pick runs (MP), so the harsh cut is judged fairly.

The rules were written down BEFORE any run: see "Hypothesis H-top-names" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only: nothing is
published, and no gate, config or stock list changes.

Run from the project root (needs data/ and signals/, and internet once for
the S&P 500 membership history):
    python research_evidence/top_names_20260929/top_names_check.py
Quick smoke test (NOT a valid result):
    ... --offsets 2 --monkeys 4
Output: research_evidence/top_names_20260929/top_names_check.json
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
HYPOTHESIS = "H-top-names"
N_OFFSETS = 20
N_MONKEYS = 100
DELAYS = (0, 1)
DECISION_WINDOW = ("2013-01-01", "2022-12-31")
T1_MIN_SHARE_OF_S = 0.5
T2_PERCENTILE = 95.0
FALLBACK_SCORE_COL = "factor_walkforward_score"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edge_module():
    return _load_module("edge_check", OUT_DIR.parent / "edge_check_20260926" / "edge_check.py")


def membership_module():
    return _load_module("watchlist_membership_check",
                        OUT_DIR.parent / "survivorship_check_20260926" / "watchlist_membership_check.py")


# ── Pure helpers (tested) ──────────────────────────────────────────────────

def label_column(holding_days: int, horizon_days: int) -> str:
    """The panel's on-time holding-return column the engine itself uses."""
    return "forward_return" if holding_days == horizon_days else f"forward_return_{holding_days}d"


def _weights(trade) -> dict[str, float]:
    raw = trade.get("overlay_weights_json") if hasattr(trade, "get") else trade["overlay_weights_json"]
    try:
        return {str(k).upper(): float(v) for k, v in json.loads(raw or "{}").items()}
    except (TypeError, ValueError):
        return {}


def label_lookup(panel: pd.DataFrame, label: str) -> dict[tuple[pd.Timestamp, str], float]:
    """(decision date, ticker) -> that stock's holding return."""
    frame = panel[["date", "ticker", label]].dropna()
    return {(pd.Timestamp(d), str(t).upper()): float(r)
            for d, t, r in zip(frame["date"], frame["ticker"], frame[label])}


def ticker_contributions(trades: pd.DataFrame, labels: dict, window=DECISION_WINDOW) -> dict[str, float]:
    """Add up weight × holding return per ticker over periods in `window`."""
    out: dict[str, float] = {}
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    for _, trade in trades.iterrows():
        day = pd.Timestamp(trade["date"])
        if not (start <= day <= end):
            continue
        for ticker, weight in _weights(trade).items():
            ret = labels.get((day, ticker))
            if ret is not None:
                out[ticker] = out.get(ticker, 0.0) + weight * ret
    return out


def top_contributors(per_run: list[dict[str, float]], k: int) -> list[str]:
    """Average each ticker's contribution across runs and return the top k.

    A run where a ticker was never held counts as 0 for that ticker.
    Ties are broken by ticker name so the choice is repeatable.
    """
    tickers = sorted({t for run in per_run for t in run})
    n = max(len(per_run), 1)
    avg = {t: sum(run.get(t, 0.0) for run in per_run) / n for t in tickers}
    return [t for t, _ in sorted(avg.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def median_label_by_date(panel: pd.DataFrame, label: str, score_col: str = FALLBACK_SCORE_COL) -> dict:
    """Median holding return of all stocks that had a score on each date."""
    mask = panel[label].notna()
    if score_col in panel.columns:
        mask &= panel[score_col].notna()
    frame = panel.loc[mask, ["date", label]]
    return {pd.Timestamp(d): float(v) for d, v in frame.groupby("date")[label].median().items()}


def best_pick_swapped_equity(equity: pd.Series, trades: pd.DataFrame, labels: dict,
                             medians: dict) -> tuple[pd.Series, dict]:
    """Rebuild the period equity with each period's best pick made "typical".

    PLAIN ENGLISH: in each 20-day period, find the pick with the highest
    holding return and pretend it only earned the median stock's return.
    Its weight stays the same, so the period return changes by
    weight × (median − best).  Everything else is the engine's own number.
    """
    values = [float(equity.iloc[0])]
    dates = [equity.index[0]]
    swapped = 0
    for _, trade in trades.iterrows():
        day = pd.Timestamp(trade["date"])
        change = 0.0
        picks = [(labels.get((day, t)), t, w) for t, w in _weights(trade).items() if w > 0]
        picks = [p for p in picks if p[0] is not None]
        median = medians.get(day)
        if picks and median is not None:
            best_ret, _ticker, weight = max(picks, key=lambda p: (p[0], p[1]))
            change = weight * (median - best_ret)
            swapped += 1
        values.append(values[-1] * (1.0 + float(trade["period_return"]) + change))
        dates.append(pd.Timestamp(trade["label_end_date"]))
    rebuilt = pd.Series(values, index=pd.DatetimeIndex(dates))
    return rebuilt[~rebuilt.index.duplicated(keep="last")], {"periods_swapped": swapped}


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _alphas(rows, test, delay, key="decision_alpha_vs_qqq_pct"):
    return [r[key] for r in rows if r["test"] == test and r["delay"] == delay and r.get(key) is not None]


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-top-names rules."""
    s_late = _median(_alphas(rows, "S", 1))
    n3_late = _median(_alphas(rows, "N3", 1))
    p1 = _median(_alphas(rows, "P1", 0))
    monkeys = _alphas(rows, "MP", 0)
    mp_p95 = float(np.percentile(monkeys, T2_PERCENTILE)) if monkeys else None

    t1 = (n3_late is not None and s_late is not None and n3_late > 0
          and n3_late >= T1_MIN_SHARE_OF_S * s_late)
    t2 = p1 is not None and mp_p95 is not None and p1 > mp_p95
    labels = []
    if not t1:
        labels.append("edge rests on a few names")
    if not t2:
        labels.append("edge rests on each period's one big winner")
    on_time_s = _alphas(rows, "S", 0)
    return {
        "numbers": {
            "S_median_late_alpha": s_late, "N1_median_late_alpha": _median(_alphas(rows, "N1", 1)),
            "N3_median_late_alpha": n3_late, "N3_share_of_S": (n3_late / s_late) if (n3_late is not None and s_late) else None,
            "P1_median_alpha": p1, "MP_p95_alpha": mp_p95, "MP_median_alpha": _median(monkeys),
        },
        "diagnostics": {
            "S_on_time_spread": (max(on_time_s) - min(on_time_s)) if on_time_s else None,
            "S_2023_2026_median": _median(_alphas(rows, "S", 0, "diagnostic_alpha_vs_qqq_pct")),
            "N3_2023_2026_median": _median(_alphas(rows, "N3", 0, "diagnostic_alpha_vs_qqq_pct")),
            "P1_2023_2026_median": _median(_alphas(rows, "P1", 0, "diagnostic_alpha_vs_qqq_pct")),
        },
        "gates": {"T1_names": t1, "T2_big_winners": t2},
        "verdict": "edge is broad" if (t1 and t2) else "; ".join(labels),
    }


# ── Shared set-up with H-benchmark ─────────────────────────────────────────

def build_s_panel(membership_csv: str | None = None):
    """The incumbent's panel restricted to point-in-time S&P 500 members (H-edge test S)."""
    from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
    import core_satellite_alpha as core
    from validation_bundle import load_approved_research_config

    edge = edge_module()
    membership = membership_module()
    if membership_csv:
        raw = Path(membership_csv).read_bytes()
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
    mask = edge.point_in_time_mask(panel, snapshots, membership.PREDECESSORS)
    panel_s = panel.loc[mask].reset_index(drop=True)
    sessions = core._nyse_sessions(panel["date"].min(), panel["date"].max())
    notes = {"membership_source_url": membership.SOURCE_URL,
             "membership_source_sha256": hashlib.sha256(raw).hexdigest(),
             "panel_rows": int(len(panel)), "point_in_time_rows": int(len(panel_s))}
    return edge, core, config, panel_s, sessions, notes


# ── Main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-top-names check (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--monkeys", type=int, default=N_MONKEYS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--out", default=str(OUT_DIR / "top_names_check.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import HORIZON_DAYS

    edge, core, config, panel_s, sessions, notes = build_s_panel(args.membership_csv)
    end = sessions[-30]   # same end for every run, so the last labels are complete
    holding_days = int(config.get("holding_days", HORIZON_DAYS))
    label = label_column(holding_days, HORIZON_DAYS)
    labels = label_lookup(panel_s, label)
    medians = median_label_by_date(panel_s, label)
    notes.update({"holding_label": label, "chosen_top_tickers": None})
    rows: list[dict] = []
    out_path = Path(args.out)
    valid = args.offsets == N_OFFSETS and args.monkeys == N_MONKEYS

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": str(pd.Timestamp(end).date()),
                   "offsets": args.offsets, "monkeys": args.monkeys, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    def record(test, equity, extra, *, delay, offset, seed=None, t0=None):
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, seed=seed, t0=t0)
        row.update({k: v for k, v in extra.items() if k == "periods_swapped"})
        rows.append(row)
        print(json.dumps(row), flush=True)
        save()

    def run(panel, delay, offset):
        return core.run_core_satellite(panel, {**config, "entry_delay_days": delay},
                                       evaluation_start=sessions[offset], evaluation_end=end)

    # S (and P1 from its on-time runs); contributions pick the top names.
    contributions = []
    for delay in DELAYS:
        for offset in range(args.offsets):
            t0 = time.time()
            equity, trades, extra = run(panel_s, delay, offset)
            record("S", equity, extra, delay=delay, offset=offset, t0=t0)
            if delay == 0:
                contributions.append(ticker_contributions(trades, labels))
                swapped, info = best_pick_swapped_equity(equity, trades, labels, medians)
                record("P1", swapped, info, delay=0, offset=offset)

    top3 = top_contributors(contributions, 3)
    avg_total = {t: sum(c.get(t, 0.0) for c in contributions) / max(len(contributions), 1) for t in top3}
    all_total = sum(sum(c.values()) for c in contributions) / max(len(contributions), 1)
    notes["chosen_top_tickers"] = top3
    notes["top_ticker_avg_contribution"] = avg_total
    notes["top3_share_of_overlay_profit"] = (sum(avg_total.values()) / all_total) if all_total else None
    save()
    print(f"[top names] chosen before N1/N3: {top3}", flush=True)

    for test, removed in (("N1", top3[:1]), ("N3", top3)):
        panel = panel_s[~panel_s["ticker"].astype(str).str.upper().isin(removed)].reset_index(drop=True)
        for delay in DELAYS:
            for offset in range(args.offsets):
                t0 = time.time()
                equity, _trades, extra = run(panel, delay, offset)
                record(test, equity, extra, delay=delay, offset=offset, t0=t0)

    for seed in range(args.monkeys):
        t0 = time.time()
        offset = seed % N_OFFSETS
        equity, trades, _extra = core.run_core_satellite(
            edge.randomized_scores(panel_s, seed), {**config, "entry_delay_days": 0},
            evaluation_start=sessions[offset], evaluation_end=end)
        unexpected = {c for c in trades.get("score_col", pd.Series(dtype=str)).astype(str).unique()
                      if c not in edge.SCORE_COLS and not c.startswith("_blended_regime_score_")}
        if unexpected:
            raise SystemExit(f"Random-pick runs used non-randomised score columns: {sorted(unexpected)}")
        swapped, info = best_pick_swapped_equity(equity, trades, labels, medians)
        record("MP", swapped, info, delay=0, offset=offset, seed=seed, t0=t0)

    result = judge(rows)
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdict")}, indent=1))
    return result


if __name__ == "__main__":
    main()
