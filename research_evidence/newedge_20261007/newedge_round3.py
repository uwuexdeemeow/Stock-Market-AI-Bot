"""
newedge_round3.py — pre-registered "H-newedge-3": three ideas built on real earnings dates.

PLAIN ENGLISH: the nine earlier ideas all re-used the same price and volume
numbers.  This round adds one new kind of information: the days on which
companies really reported their results.  The project never had those dates
(the panel just says "60 days" everywhere), so two things could not be tested:

  P   earnings drift: buy the stocks whose price jumped most after their
      latest report (good news tends to keep working for some weeks).
  X   real earnings blackout: the current strategy already has a rule
      "don't buy within 5 days before earnings", but with no dates it never
      did anything.  Here it gets real dates.
  PS  blended score: half the current score, half the earnings score.

The dates come from the SEC's free EDGAR service: a company that publishes
results files a form "8-K" with item "2.02".  Only panel columns are changed,
after the normal panel is built.  The engine file is never edited.

Rules (written BEFORE any run): "Hypothesis H-newedge-3" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only.

Run from the project root (needs data/ and signals/):
    python research_evidence/newedge_20261007/newedge_round3.py --membership-csv PATH --coverage-only
    python research_evidence/newedge_20261007/newedge_round3.py --membership-csv PATH
Smoke test (NOT a valid result):  ... --offsets 2 --no-exam
Output: research_evidence/newedge_20261007/newedge_round3.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Let Python find the project modules when run from the project root.
sys.path.insert(0, str(Path.cwd()))

OUT_DIR = Path(__file__).resolve().parent
EVENTS_PATH = OUT_DIR / "earnings_events.json"

# ── Fixed settings (pre-registered; do not change after results) ───────────
HYPOTHESIS = "H-newedge-3"
N_OFFSETS = 20
DELAYS = (0, 1)
SCORE_COLS = ("factor_risk_on_score", "factor_defensive_score", "factor_walkforward_score")
IDEAS = ("P", "X", "PS")
DRIFT_WINDOW = 40            # sessions after the reaction day in which the reaction still counts
CLOSE_HOUR_NY = 16           # filings at or after 16:00 New York time move the next session
NEUTRAL_RANK = 0.5           # PS: earnings rank of a stock with no recent report
BLEND_WEIGHT = 0.5           # PS: weight of each half
NO_EVENT_DAYS = 60.0         # X: value when no next report is known (the panel's old placeholder)
MAX_EVENT_DAYS = 120.0       # X: cap, as in fundamental_features.py
COVERAGE_WINDOW = ("2012-01-01", "2022-12-31")
COVERAGE_MIN_ROWS = 40       # a ticker-quarter counts only with at least this many panel rows
COVERAGE_MIN_SHARE = 0.90    # gate C0
X_MIN_ALPHA_SHARE = 0.90     # gate X1
X_MAX_SPREAD_SHARE = 0.75    # gate X2
PS_MIN_ALPHA_SHARE = 0.75    # gate PS1
MIN_SHARE_WITHOUT_TOP3 = 0.50  # gates P3 and PS2

# ── SEC EDGAR ──────────────────────────────────────────────────────────────
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
DEFAULT_SEC_USER_AGENT = "StockBot personal research script"

# The same company under an older SEC number (mergers and re-registrations).
# PLAIN ENGLISH: when a company reorganises, the SEC gives it a new number
# and its old filings stay under the old one.  Without the old number the
# early years would have no earnings dates.  Fixed before any engine run.
PREDECESSOR_CIKS: dict[str, list[int]] = {
    "AVGO": [1649338, 1441634],   # Broadcom Ltd (2016-18) and Avago Technologies (before 2016)
    "DIS": [1001039],             # the old Walt Disney Co (before the 2019 re-registration)
    "GOOGL": [1288776],           # Google Inc (before Alphabet, 2015)
    "LIN": [884905],              # Praxair Inc (before the 2018 Linde merger)
    "XOM": [34088],               # Exxon Mobil Corp (before the 2026 re-registration)
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Earnings events: download ──────────────────────────────────────────────

def filing_times(block: dict) -> list[str]:
    """Acceptance times of the earnings filings in one SEC filing list.

    PLAIN ENGLISH: the SEC list has one row per filing.  We keep rows whose
    form is exactly "8-K" (not the amended "8-K/A") and whose items include
    "2.02", the item number for published results.
    """
    forms = block.get("form", [])
    items = block.get("items", [])
    times = block.get("acceptanceDateTime", [])
    out = []
    for form, item, stamp in zip(forms, items, times):
        codes = {code.strip() for code in str(item or "").split(",")}
        if form == "8-K" and "2.02" in codes and stamp:
            out.append(str(stamp))
    return out


def _sec_get(session, url: str):
    agent = os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT)
    time.sleep(0.2)   # the SEC allows 10 requests a second; stay well under
    response = session.get(url, headers={"User-Agent": agent}, timeout=60)
    response.raise_for_status()
    return response.json()


def download_events(tickers: list[str], known: dict | None = None) -> dict:
    """Fetch earnings filing times for `tickers`, re-using anything in `known`."""
    import requests

    store = known or {"events": {}, "ciks": {}, "names": {}}
    missing = [t for t in tickers if t not in store["events"]]
    if not missing:
        return store
    session = requests.Session()
    cik_by_ticker = {str(v["ticker"]).upper(): int(v["cik_str"])
                     for v in _sec_get(session, SEC_TICKERS_URL).values()}
    for ticker in missing:
        ciks = ([cik_by_ticker[ticker]] if ticker in cik_by_ticker else []) + PREDECESSOR_CIKS.get(ticker, [])
        stamps: set[str] = set()
        for cik in ciks:
            data = _sec_get(session, SEC_SUBMISSIONS_URL.format(cik=cik))
            store["names"][str(cik)] = data.get("name")
            stamps.update(filing_times(data["filings"]["recent"]))
            # Older filings sit in extra files listed next to the recent ones.
            for extra in data["filings"].get("files", []):
                stamps.update(filing_times(_sec_get(session, SEC_SUBMISSIONS_FILE_URL.format(name=extra["name"]))))
        store["events"][ticker] = sorted(stamps)
        store["ciks"][ticker] = ciks
        print(json.dumps({"ticker": ticker, "ciks": ciks, "events": len(stamps)}), flush=True)
    return store


def load_events(tickers: list[str], path: Path = EVENTS_PATH) -> dict:
    """Read the saved events file and download whatever is missing."""
    store = json.loads(path.read_text()) if path.exists() else None
    store = download_events(sorted(tickers), store)
    store["predecessor_ciks"] = PREDECESSOR_CIKS
    path.write_text(json.dumps(store, indent=1, sort_keys=True))
    return store


# ── Earnings events: turn filing times into panel numbers (pure, tested) ───

def reaction_index(stamp: str, days: pd.DatetimeIndex) -> int | None:
    """Position in `days` of the first session that could react to a filing.

    PLAIN ENGLISH: results published before the 16:00 close move the price
    the same day.  Results published after the close (most of them), or on a
    weekend or holiday, move it on the next trading day.
    """
    moment = pd.Timestamp(stamp)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    local = moment.tz_convert("America/New_York")
    day = local.normalize().tz_localize(None)
    if len(days) == 0 or day < days[0]:
        return None
    pos = int(days.searchsorted(day, side="left"))
    is_session = pos < len(days) and days[pos] == day
    if is_session and local.hour >= CLOSE_HOUR_NY:
        pos += 1
    return pos if pos < len(days) else None


def earnings_frames(panel: pd.DataFrame, events: dict[str, list[str]]):
    """Build the earnings numbers for every (day, stock).

    Returns three things:
      latest    — the stock's latest reaction R on days E+1 .. E+40, else empty
      days_to   — calendar days until the stock's next reaction day
      table     — one row per event (ticker, reaction day, R), for the record

    R is the stock's return from the close before the reaction day to the
    close after it, minus the median of the same two-day return over all
    panel stocks.  It is first used on E+1, when that close is known, so no
    future information leaks in.
    """
    dates = pd.to_datetime(panel["date"])
    days = pd.DatetimeIndex(sorted(dates.unique()))
    close = (panel.assign(date=dates).pivot_table(index="date", columns="ticker", values="Close", aggfunc="last")
             .reindex(days))
    two_day = close.shift(-1) / close.shift(1) - 1.0
    abnormal = two_day.sub(two_day.median(axis=1), axis=0)
    latest = pd.DataFrame(np.nan, index=days, columns=close.columns)
    days_to = pd.DataFrame(NO_EVENT_DAYS, index=days, columns=close.columns)
    table = []
    for ticker in close.columns:
        found = {reaction_index(stamp, days) for stamp in events.get(str(ticker), [])}
        positions = sorted(p for p in found if p is not None)
        col = latest.columns.get_loc(ticker)
        for pos in positions:   # oldest first, so a newer report replaces an older one
            reaction = abnormal.iat[pos, col]
            table.append({"ticker": str(ticker), "reaction_day": days[pos], "reaction": reaction})
            if pd.notna(reaction):
                latest.iloc[pos + 1: pos + 1 + DRIFT_WINDOW, col] = reaction
        if positions:
            event_days = days[positions]
            nxt = event_days.searchsorted(days, side="left")
            known = nxt < len(event_days)
            gap = (event_days[np.clip(nxt, 0, len(event_days) - 1)] - days).days.to_numpy(dtype=float)
            days_to.iloc[:, col] = np.where(known, np.minimum(gap, MAX_EVENT_DAYS), NO_EVENT_DAYS)
    return latest, days_to, pd.DataFrame(table, columns=["ticker", "reaction_day", "reaction"])


def _per_row(frame: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    """Look up a (day × stock) table for every row of the panel."""
    stacked = frame.stack(future_stack=True)
    index = pd.MultiIndex.from_arrays([pd.to_datetime(panel["date"]), panel["ticker"]])
    return stacked.reindex(index).to_numpy()


def earnings_score(panel: pd.DataFrame, events: dict) -> pd.DataFrame:
    """P: the three score columns become the rank of the latest good earnings reaction."""
    out = panel.copy()
    latest, _days_to, _table = earnings_frames(out, events)
    reaction = pd.Series(_per_row(latest, out), index=out.index)
    reaction = reaction.where(reaction > 0)          # only good news counts
    score = reaction.groupby(out["date"]).rank(pct=True)
    for col in SCORE_COLS:
        if col in out.columns:
            out[col] = score
    return out


def real_blackout(panel: pd.DataFrame, events: dict) -> pd.DataFrame:
    """X: fill `days_to_next_earnings` with real dates; the score is untouched."""
    out = panel.copy()
    _latest, days_to, _table = earnings_frames(out, events)
    out["days_to_next_earnings"] = _per_row(days_to, out)
    out["days_to_next_earnings"] = out["days_to_next_earnings"].fillna(NO_EVENT_DAYS)
    return out


def blended_score(panel: pd.DataFrame, events: dict) -> pd.DataFrame:
    """PS: half the current score's rank, half the earnings reaction's rank."""
    out = panel.copy()
    latest, _days_to, _table = earnings_frames(out, events)
    reaction = pd.Series(_per_row(latest, out), index=out.index)
    earnings_rank = reaction.groupby(out["date"]).rank(pct=True).fillna(NEUTRAL_RANK)
    for col in SCORE_COLS:
        if col in out.columns:
            own_rank = pd.to_numeric(out[col], errors="coerce").groupby(out["date"]).rank(pct=True)
            out[col] = BLEND_WEIGHT * own_rank + (1.0 - BLEND_WEIGHT) * earnings_rank
    return out


TRANSFORMS = {"P": earnings_score, "X": real_blackout, "PS": blended_score}


# ── Step 0: data coverage gate (pure, tested) ──────────────────────────────

def coverage(panel: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Gate C0: share of ticker-quarters (2012–2022) with at least one usable event."""
    dates = pd.to_datetime(panel["date"])
    inside = (dates >= COVERAGE_WINDOW[0]) & (dates <= COVERAGE_WINDOW[1])
    rows = panel.loc[inside, ["ticker"]].assign(quarter=dates[inside].dt.to_period("Q"))
    counts = rows.groupby(["ticker", "quarter"]).size()
    needed = set(counts[counts >= COVERAGE_MIN_ROWS].index)
    usable = table.dropna(subset=["reaction"])
    have = set(zip(usable["ticker"], pd.to_datetime(usable["reaction_day"]).dt.to_period("Q")))
    missing = sorted(needed - have)
    share = (len(needed) - len(missing)) / len(needed) if needed else 0.0
    gaps: dict[str, int] = {}
    for ticker, _quarter in missing:
        gaps[ticker] = gaps.get(ticker, 0) + 1
    return {"ticker_quarters": len(needed), "covered": len(needed) - len(missing), "share": round(share, 4),
            "C0_pass": bool(share >= COVERAGE_MIN_SHARE),
            "missing_quarters_by_ticker": dict(sorted(gaps.items(), key=lambda kv: -kv[1]))}


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


def _keeps_half_without_top3(rows, idea, numbers) -> bool:
    med = numbers[f"{idea}_median_late_alpha"]
    n3 = _median(_by_offset(rows, f"{idea}_N3", 1).values())
    numbers[f"{idea}_N3_median_late_alpha"] = n3
    return bool(n3 is not None and med is not None and med > 0 and n3 > 0
                and n3 >= MIN_SHARE_WITHOUT_TOP3 * med)


def judge(rows: list[dict], n_offsets: int = N_OFFSETS) -> dict:
    """Apply the pre-registered H-newedge-3 decision gates."""
    s_med = _median(_by_offset(rows, "S", 1).values())
    s_spread, s_cost = _spread(rows, "S"), _delay_cost(rows, "S")
    numbers = {"S_median_late_alpha": s_med, "S_on_time_spread": s_spread, "S_mean_delay_cost": s_cost}
    for idea in IDEAS:
        numbers[f"{idea}_median_late_alpha"] = _median(_by_offset(rows, idea, 1).values())
        numbers[f"{idea}_on_time_spread"] = _spread(rows, idea)
        numbers[f"{idea}_mean_delay_cost"] = _delay_cost(rows, idea)
    gates = {}

    # P: a new signal, judged on its own.
    p_all = list(_by_offset(rows, "P", 0).values()) + list(_by_offset(rows, "P", 1).values())
    numbers["P_runs_beating_qqq"] = sum(1 for v in p_all if v > 0)
    numbers["P_worst_run_alpha"] = min(p_all) if p_all else None
    gates["P1"] = len(p_all) == 2 * n_offsets and all(v > 0 for v in p_all)
    p_cost = numbers["P_mean_delay_cost"]
    gates["P2"] = bool(p_cost is not None and s_cost is not None and p_cost <= s_cost)
    gates["P3"] = _keeps_half_without_top3(rows, "P", numbers)

    # X: the same strategy, but it should be steadier.
    x_med, x_spread, x_cost = (numbers["X_median_late_alpha"], numbers["X_on_time_spread"],
                               numbers["X_mean_delay_cost"])
    gates["X1"] = bool(x_med is not None and s_med is not None and x_med >= X_MIN_ALPHA_SHARE * s_med)
    gates["X2"] = bool(x_spread is not None and s_spread is not None and x_cost is not None and s_cost is not None
                       and x_spread <= X_MAX_SPREAD_SHARE * s_spread and x_cost <= s_cost)

    # PS: the same strategy, but it should lean less on a few names.
    ps_med = numbers["PS_median_late_alpha"]
    gates["PS1"] = bool(ps_med is not None and s_med is not None and ps_med >= PS_MIN_ALPHA_SHARE * s_med)
    gates["PS2"] = _keeps_half_without_top3(rows, "PS", numbers)

    decision = {"P": gates["P1"] and gates["P2"] and gates["P3"],
                "X": gates["X1"] and gates["X2"],
                "PS": gates["PS1"] and gates["PS2"]}
    return {"numbers": numbers, "gates": {k: bool(v) for k, v in gates.items()},
            "decision_pass": {k: bool(v) for k, v in decision.items()}}


# ── Final exam ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def panel_patch(core, idea: str):
    """Apply the idea's panel change inside the stress test's own panel."""
    original = core._ensure_robust_score_columns

    def patched(panel):
        built = original(panel)
        store = load_events(sorted(built["ticker"].astype(str).unique()))
        return TRANSFORMS[idea](built, store["events"])

    core._ensure_robust_score_columns = patched
    try:
        yield
    finally:
        core._ensure_robust_score_columns = original


def run_exam(core, idea: str, config: dict) -> dict:
    import core_satellite_execution_stress as stress
    from settings import LOG_DIR

    path = OUT_DIR / f"candidate_round3_{idea}.json"
    path.write_text(json.dumps(dict(config), indent=1, default=str))
    name = f"newedge3_{idea}"
    with panel_patch(core, idea):
        stress.main(["--candidate-json", str(path), "--candidate-name", name])
    report = json.loads((Path(LOG_DIR) / f"research_candidate_execution_stress_{name}.json").read_text())
    scenarios = [r for r in report["rows"] if not str(r["scenario"]).startswith("delta")]
    failed = {r["scenario"]: r["failed_gates"] for r in scenarios if r.get("failed_gates")}
    return {"pass": not failed, "failed": failed,
            "holdout_alpha_vs_qqq": {r["scenario"]: r.get("holdout_alpha_vs_qqq_pct") for r in scenarios}}


# ── Main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-newedge-3 round (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--coverage-only", action="store_true", help="stop after the data coverage gate C0")
    parser.add_argument("--no-exam", action="store_true", help="skip the final exam (smoke tests only)")
    parser.add_argument("--out", default=str(OUT_DIR / "newedge_round3.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import HORIZON_DAYS

    top = _load("top_names_check", OUT_DIR.parent / "top_names_20260929" / "top_names_check.py")
    edge, core, config, panel_s, sessions, notes = top.build_s_panel(args.membership_csv)

    # Step 0: earnings dates and the coverage gate, before any engine run.
    store = load_events(sorted(panel_s["ticker"].astype(str).unique()))
    events = store["events"]
    _latest, _days_to, table = earnings_frames(panel_s, events)
    cover = coverage(panel_s, table)
    notes.update({"earnings_source": "SEC EDGAR submissions, form 8-K item 2.02",
                  "earnings_events": int(len(table)), "earnings_usable_events": int(table["reaction"].notna().sum()),
                  "predecessor_ciks": PREDECESSOR_CIKS, "coverage": cover})
    print(json.dumps({"coverage": cover}, indent=1), flush=True)
    out_path = Path(args.out)
    if args.coverage_only or not cover["C0_pass"]:
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": False, "notes": notes, "rows": [],
                   "result": {"coverage_only": bool(args.coverage_only), "C0_pass": cover["C0_pass"],
                              "verdicts": ({} if cover["C0_pass"]
                                           else {idea: "not testable with this source" for idea in IDEAS})}}
        if not args.coverage_only:
            out_path.write_text(json.dumps(payload, indent=1, default=str))
        return payload["result"]

    end = sessions[-30]
    labels = top.label_lookup(panel_s, top.label_column(int(config.get("holding_days", HORIZON_DAYS)), HORIZON_DAYS))
    panels = {"S": panel_s, **{idea: fn(panel_s, events) for idea, fn in TRANSFORMS.items()}}
    rows: list[dict] = []
    valid = args.offsets == N_OFFSETS and not args.no_exam

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": str(pd.Timestamp(end).date()),
                   "offsets": args.offsets, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    def run(test, panel, delay, offset):
        t0 = time.time()
        equity, trades, extra = core.run_core_satellite(
            panel, {**config, "entry_delay_days": delay}, evaluation_start=sessions[offset], evaluation_end=end)
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, t0=t0)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("test", "delay", "offset", "decision_alpha_vs_qqq_pct", "secs")}),
              flush=True)
        save()
        return trades

    contributions = {"P": [], "PS": []}
    for test in ("S",) + IDEAS:
        for delay in DELAYS:
            for offset in range(args.offsets):
                trades = run(test, panels[test], delay, offset)
                if test in contributions and delay == 0:
                    contributions[test].append(top.ticker_contributions(trades, labels))
    for idea in ("P", "PS"):
        top3 = top.top_contributors(contributions[idea], 3)
        notes[f"{idea}_top3_tickers"] = top3
        panel = panels[idea]
        panel = panel[~panel["ticker"].astype(str).str.upper().isin(top3)].reset_index(drop=True)
        for offset in range(args.offsets):
            run(f"{idea}_N3", panel, 1, offset)

    result = judge(rows, args.offsets)
    result["exam"] = {}
    if not args.no_exam:
        for idea in IDEAS:
            if result["decision_pass"][idea]:
                result["exam"][idea] = run_exam(core, idea, config)
    result["verdicts"] = {
        idea: ("fails decision gates" if not result["decision_pass"][idea]
               else ("passes" if result["exam"].get(idea, {}).get("pass") else "fails the exam"))
        for idea in IDEAS
    }
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdicts")}, indent=1))
    return result


if __name__ == "__main__":
    main()
