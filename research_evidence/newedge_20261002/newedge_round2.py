"""
newedge_round2.py — pre-registered "H-newedge-2": three changes to the stock score.

PLAIN ENGLISH: the part of the strategy that clearly works is the score that
ranks stocks.  Round 1 changed everything around it and nothing helped, so
this round changes the score itself, three ways:

  Q  low-volatility filter: each day the jumpiest third of stocks can't be
     picked (a price-only "quality" screen).
  M  smoothed score: each stock's score is averaged over its last 5 days, so
     one noisy day matters less.
  N  sector-neutral score: stocks are ranked against their own sector and at
     most 1 stock per sector is held, to avoid piling into one industry.

Only the three score columns the incumbent picks with are changed, after the
normal panel is built.  The engine file is never edited.

Rules (written BEFORE any run): "Hypothesis H-newedge-2" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only.

Run from the project root (needs data/ and signals/):
    python research_evidence/newedge_20261002/newedge_round2.py --membership-csv PATH
Smoke test (NOT a valid result):  ... --offsets 2 --no-exam
Output: research_evidence/newedge_20261002/newedge_round2.json
"""
from __future__ import annotations

import argparse
import contextlib
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
HYPOTHESIS = "H-newedge-2"
N_OFFSETS = 20
DELAYS = (0, 1)
SCORE_COLS = ("factor_risk_on_score", "factor_defensive_score", "factor_walkforward_score")
Q_VOL_COL = "factor_idio_vol_252_spy"
Q_CUT_RANK = 2.0 / 3.0
M_WINDOW = 5
N_MAX_PER_SECTOR = 1
G1_MIN_ALPHA_SHARE = 0.90
G2_MIN_SHARE_WITHOUT_TOP3 = 0.50
M_MAX_SPREAD_SHARE = 0.75


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Score changes (pure, tested) ───────────────────────────────────────────

def quality_filter(panel: pd.DataFrame) -> pd.DataFrame:
    """Q: the most volatile third of each day's stocks get no score."""
    out = panel.copy()
    vol = pd.to_numeric(out[Q_VOL_COL], errors="coerce")
    rank = vol.groupby(out["date"]).rank(pct=True)
    too_jumpy = rank > Q_CUT_RANK
    for col in SCORE_COLS:
        if col in out.columns:
            out.loc[too_jumpy, col] = np.nan
    return out


def smooth_scores(panel: pd.DataFrame, window: int = M_WINDOW) -> pd.DataFrame:
    """M: each score becomes its average over the stock's last `window` rows.

    Only today and earlier rows are used, so no future information leaks in.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    for col in SCORE_COLS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.groupby(out["ticker"]).transform(
                lambda s: s.rolling(window, min_periods=window).mean())
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def sector_neutral(panel: pd.DataFrame) -> pd.DataFrame:
    """N: each score becomes its percentile rank within its sector that day."""
    out = panel.copy()
    for col in SCORE_COLS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.groupby([out["date"], out["sector"]]).rank(pct=True)
    return out


TRANSFORMS = {"Q": quality_filter, "M": smooth_scores, "N": sector_neutral}


def idea_config(idea: str, config: dict) -> dict:
    return {**config, "max_per_sector": N_MAX_PER_SECTOR} if idea == "N" else dict(config)


# ── Judging (pure, tested) ─────────────────────────────────────────────────

def _median(values):
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _by_offset(rows, test, delay):
    return {r["offset"]: r.get("decision_alpha_vs_qqq_pct") for r in rows
            if r["test"] == test and r["delay"] == delay and r.get("decision_alpha_vs_qqq_pct") is not None}


def _delay_cost(rows, test):
    on, late = _by_offset(rows, test, 0), _by_offset(rows, test, 1)
    costs = [on[o] - late[o] for o in on if o in late]
    return float(np.mean(costs)) if costs else None


def _spread(rows, test):
    on = list(_by_offset(rows, test, 0).values())
    return (max(on) - min(on)) if on else None


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-newedge-2 decision gates."""
    s_med = _median(_by_offset(rows, "S", 1).values())
    s_spread, s_cost = _spread(rows, "S"), _delay_cost(rows, "S")
    numbers = {"S_median_late_alpha": s_med, "S_on_time_spread": s_spread, "S_mean_delay_cost": s_cost}
    gates, decision = {}, {}
    for idea in ("Q", "M", "N"):
        med = _median(_by_offset(rows, idea, 1).values())
        numbers[f"{idea}_median_late_alpha"] = med
        numbers[f"{idea}_on_time_spread"] = _spread(rows, idea)
        numbers[f"{idea}_mean_delay_cost"] = _delay_cost(rows, idea)
        g1 = med is not None and s_med is not None and med >= G1_MIN_ALPHA_SHARE * s_med
        if idea == "M":
            spread, cost = numbers["M_on_time_spread"], numbers["M_mean_delay_cost"]
            g2 = (spread is not None and s_spread is not None and cost is not None and s_cost is not None
                  and spread <= M_MAX_SPREAD_SHARE * s_spread and cost <= s_cost)
        else:
            n3 = _median(_by_offset(rows, f"{idea}_N3", 1).values())
            numbers[f"{idea}_N3_median_late_alpha"] = n3
            g2 = (n3 is not None and med is not None and med > 0 and n3 > 0
                  and n3 >= G2_MIN_SHARE_WITHOUT_TOP3 * med)
        gates[f"{idea}_G1"], gates[f"{idea}_G2"] = bool(g1), bool(g2)
        decision[idea] = bool(g1 and g2)
    return {"numbers": numbers, "gates": gates, "decision_pass": decision}


# ── Final exam ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def score_patch(core, idea: str):
    """Apply the idea's score change inside the stress test's own panel."""
    original = core._ensure_robust_score_columns

    def patched(panel):
        return TRANSFORMS[idea](original(panel))

    core._ensure_robust_score_columns = patched
    try:
        yield
    finally:
        core._ensure_robust_score_columns = original


def run_exam(core, idea: str, config: dict) -> dict:
    import core_satellite_execution_stress as stress
    from settings import LOG_DIR

    path = OUT_DIR / f"candidate_round2_{idea}.json"
    path.write_text(json.dumps(idea_config(idea, config), indent=1, default=str))
    name = f"newedge2_{idea}"
    with score_patch(core, idea):
        stress.main(["--candidate-json", str(path), "--candidate-name", name])
    report = json.loads((Path(LOG_DIR) / f"research_candidate_execution_stress_{name}.json").read_text())
    scenarios = [r for r in report["rows"] if not str(r["scenario"]).startswith("delta")]
    failed = {r["scenario"]: r["failed_gates"] for r in scenarios if r.get("failed_gates")}
    return {"pass": not failed, "failed": failed,
            "holdout_alpha_vs_qqq": {r["scenario"]: r.get("holdout_alpha_vs_qqq_pct") for r in scenarios}}


# ── Main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-newedge-2 round (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--no-exam", action="store_true", help="skip the final exam (smoke tests only)")
    parser.add_argument("--out", default=str(OUT_DIR / "newedge_round2.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import HORIZON_DAYS

    top = _load("top_names_check", OUT_DIR.parent / "top_names_20260929" / "top_names_check.py")
    edge, core, config, panel_s, sessions, notes = top.build_s_panel(args.membership_csv)
    end = sessions[-30]
    labels = top.label_lookup(panel_s, top.label_column(int(config.get("holding_days", HORIZON_DAYS)), HORIZON_DAYS))
    panels = {"S": panel_s, **{idea: fn(panel_s) for idea, fn in TRANSFORMS.items()}}
    rows: list[dict] = []
    out_path = Path(args.out)
    valid = args.offsets == N_OFFSETS and not args.no_exam

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": str(pd.Timestamp(end).date()),
                   "offsets": args.offsets, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    def run(test, cfg, panel, delay, offset):
        t0 = time.time()
        equity, trades, extra = core.run_core_satellite(
            panel, {**cfg, "entry_delay_days": delay}, evaluation_start=sessions[offset], evaluation_end=end)
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, t0=t0)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("test", "delay", "offset", "decision_alpha_vs_qqq_pct", "secs")}),
              flush=True)
        save()
        return trades

    contributions = {"Q": [], "N": []}
    for test in ("S", "Q", "M", "N"):
        cfg = idea_config(test, config)
        for delay in DELAYS:
            for offset in range(args.offsets):
                trades = run(test, cfg, panels[test], delay, offset)
                if test in contributions and delay == 0:
                    contributions[test].append(top.ticker_contributions(trades, labels))
    for idea in ("Q", "N"):
        top3 = top.top_contributors(contributions[idea], 3)
        notes[f"{idea}_top3_tickers"] = top3
        panel = panels[idea]
        panel = panel[~panel["ticker"].astype(str).str.upper().isin(top3)].reset_index(drop=True)
        for offset in range(args.offsets):
            run(f"{idea}_N3", idea_config(idea, config), panel, 1, offset)

    result = judge(rows)
    result["exam"] = {}
    if not args.no_exam:
        for idea in ("Q", "M", "N"):
            if result["decision_pass"][idea]:
                result["exam"][idea] = run_exam(core, idea, config)
    result["verdicts"] = {
        idea: ("fails decision gates" if not result["decision_pass"][idea]
               else ("passes" if result["exam"].get(idea, {}).get("pass") else "fails the exam"))
        for idea in ("Q", "M", "N")
    }
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdicts")}, indent=1))
    return result


if __name__ == "__main__":
    main()
