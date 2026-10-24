"""
forward_test_monitor.py — monthly numbers for Forward Test Rule F1.

PLAIN ENGLISH: Documentation/FORWARD_TEST_RULE.md says the paper strategy
is judged only on how it does FROM NOW ON, with three measures written down
in advance:

  M1  return vs a fair yardstick: the paper account's return minus what you
      would have made holding QQQ with the same invested share (the rest in
      BIL, a T-bill fund), chained day by day and annualised.
  M2  real trading cost: how far each fill price was from that day's Open,
      in basis points (1 bp = 0.01%).  Positive = worse than the Open.
  M3  can it trade? the share of scheduled rebalance dates that the safety
      gate blocked.

This script computes them from the daily run's own files (the
`signals/latest` branch), prints them, and with --append-log adds one line
to the log table in FORWARD_TEST_RULE.md.  It only reads trading data; it
never places orders or changes a gate.  Nothing is judged here: the rule's
checkpoints (12 and 24 periods) decide.

Rule V (from the 2026-10-30 signal) is judged by Forward Test Rule FV
(Documentation/V_RULE_FORWARD_TEST.md) instead: V1 worst fall and V2 Sharpe
vs buy-and-hold QQQ over the same days, V3 fill cost vs the Open, and V4 the
monthly gap to a simulation of the rule (information only).  V1, V2 and V4
use the exact functions of the tested H-index-risk script.

How to run (project root, needs git and internet):
    python forward_test_monitor.py                 # print the F1 numbers
    python forward_test_monitor.py --append-log    # also add a log line
    python forward_test_monitor.py --rule v --append-log   # rule V (FV)
Output: signals/forward_test_check.json (F1) or signals/forward_test_check_v.json (FV)
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RULE_DOC = Path("Documentation") / "FORWARD_TEST_RULE.md"
OUT_JSON = Path("signals") / "forward_test_check.json"
REF = "origin/signals/latest"         # where the daily run publishes its files
HOLDING_SESSIONS = 20                 # one rebalance period = 20 NYSE sessions
T_BLOCK_WINDOW, T_BLOCK_LIMIT = 6, 4  # T-block: 4+ blocked of any 6 in a row
MIN_DAYS_TO_ANNUALISE = 20            # fewer days than this: M1 is "n/a"

# Forward Test Rule FV (rule V on the paper account).
V_RULE_DOC = Path("Documentation") / "V_RULE_FORWARD_TEST.md"
V_OUT_JSON = Path("signals") / "forward_test_check_v.json"
V_FIRST_SIGNAL_DAY = "2026-10-30"     # first month-end signal traded under rule V
V_CHECKPOINT_REBALANCES = 12
INDEX_RISK_SCRIPT = Path("research_evidence") / "index_risk_20261016" / "index_risk_check.py"


def index_risk_module():
    """The tested H-index-risk script, loaded so FV uses its exact math."""
    spec = importlib.util.spec_from_file_location("index_risk_check", INDEX_RISK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Reading the daily run's files ──────────────────────────────────────────

def read_signal_file(name: str, *, ref: str = REF, signals_dir: str | None = None) -> str:
    """Text of signals/<name> from a local folder or from the git branch."""
    if signals_dir:
        return (Path(signals_dir) / name).read_text(encoding="utf-8")
    return subprocess.run(["git", "show", f"{ref}:signals/{name}"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def ny_date(stamp) -> pd.Timestamp:
    """A UTC timestamp's calendar date in New York."""
    return pd.Timestamp(stamp).tz_convert("America/New_York").tz_localize(None).normalize()


# ── Pure calculations (tested) ─────────────────────────────────────────────

def scheduled_dates(period: str, sessions: pd.DatetimeIndex, step: int = HOLDING_SESSIONS) -> pd.DatetimeIndex:
    """Every scheduled rebalance signal day on the 20-session grid through `period`.

    PLAIN ENGLISH: the account records the latest scheduled day (`period`).
    The others are every 20th NYSE session before and after it.
    """
    pos = int(sessions.get_loc(pd.Timestamp(period)))
    return sessions[pos % step::step]


def periods_done(grid: pd.DatetimeIndex, sessions: pd.DatetimeIndex, epoch_start, as_of,
                 step: int = HOLDING_SESSIONS) -> int:
    """Completed 20-day periods that started on or after the epoch start."""
    start, as_of = pd.Timestamp(epoch_start).normalize(), pd.Timestamp(as_of)
    done = 0
    for day in grid[grid >= start]:
        pos = int(sessions.get_loc(day))
        if pos + step < len(sessions) and sessions[pos + step] <= as_of:
            done += 1
    return done


def blocked_share(grid: pd.DatetimeIndex, sessions: pd.DatetimeIndex, outcomes: pd.DataFrame,
                  epoch_start, as_of) -> dict:
    """M3: how many scheduled rebalances were blocked by the safety gate.

    A rebalance's orders go in on the session after its signal day; the
    first daily run on that session decides.  Its status "blocked" counts as
    blocked; no run at all is reported separately as "missing".
    """
    start, as_of = pd.Timestamp(epoch_start).normalize(), pd.Timestamp(as_of)
    runs = outcomes.assign(day=outcomes["generated_at"].map(ny_date)).sort_values("generated_at")
    flags, missing = [], []
    for day in grid[grid >= start]:
        pos = int(sessions.get_loc(day))
        if pos + 1 >= len(sessions) or sessions[pos + 1] > as_of:
            continue
        first = runs[runs["day"] == sessions[pos + 1]].head(1)
        if first.empty:
            missing.append(str(day.date()))
            continue
        flags.append(bool(first["status"].iloc[0] == "blocked"))
    worst = max((sum(flags[i:i + T_BLOCK_WINDOW]) for i in range(len(flags))), default=0)
    return {"scheduled": len(flags), "blocked": int(sum(flags)), "missing_runs": missing,
            "t_block_triggered": worst >= T_BLOCK_LIMIT}


def relative_return(equity: pd.DataFrame, qqq: pd.Series, bil: pd.Series, epoch_start) -> dict:
    """M1: account return minus the exposure-matched QQQ/BIL yardstick.

    Each day the yardstick holds QQQ at the account's invested share from the
    day before (invested / equity) and BIL for the rest.  Both are chained
    and annualised (252 trading days a year); M1 is the difference.
    """
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    base = frame.index[frame.index <= pd.Timestamp(epoch_start).normalize()]
    frame = frame.loc[(base.max() if len(base) else frame.index.min()):]
    if len(frame) < 2:
        return {"days": 0, "m1_annualised": None}
    account = frame["equity"].pct_change().iloc[1:]
    share = (frame["invested"] / frame["equity"]).clip(0.0, 2.0).shift(1).iloc[1:]
    q = qqq.reindex(frame.index).ffill().pct_change().iloc[1:]
    b = bil.reindex(frame.index).ffill().pct_change().iloc[1:].fillna(0.0)
    bench = share * q + (1.0 - share) * b
    days = int(len(account))
    acc_total = float((1 + account).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    out = {"days": days, "first_date": str(frame.index[0].date()), "last_date": str(frame.index[-1].date()),
           "account_return": acc_total, "yardstick_return": bench_total,
           "average_invested_share": float(share.mean()), "m1_annualised": None}
    if days >= MIN_DAYS_TO_ANNUALISE:
        ann = lambda total: (1 + total) ** (252 / days) - 1  # noqa: E731
        out["m1_annualised"] = float(ann(acc_total) - ann(bench_total))
    return out


def fill_costs(fills: pd.DataFrame, opens: dict[tuple[str, pd.Timestamp], float]) -> dict:
    """M2: mean fill cost vs that day's Open in bps, with a 95% interval."""
    costs = []
    for _, row in fills.iterrows():
        day = ny_date(row["submitted_at"])
        ref = opens.get((str(row["ticker"]).upper(), day))
        price = pd.to_numeric(row.get("filled_avg_price"), errors="coerce")
        if ref is None or not np.isfinite(price) or ref <= 0:
            continue
        sign = -1.0 if str(row["side"]).lower() == "sell" else 1.0
        costs.append(sign * (price / ref - 1.0) * 1e4)
    n = len(costs)
    if not n:
        return {"fills": 0, "mean_bps": None, "ci95_bps": None}
    mean = float(np.mean(costs))
    half = 1.96 * float(np.std(costs, ddof=1)) / math.sqrt(n) if n > 1 else None
    return {"fills": n, "mean_bps": mean, "ci95_bps": [mean - half, mean + half] if half is not None else None}


def weighted_fill_cost(fills: pd.DataFrame, opens: dict[tuple[str, pd.Timestamp], float]) -> dict:
    """V3: fill cost vs that day's Open in bps, weighted by trade value."""
    costs, values = [], []
    for _, row in fills.iterrows():
        day = ny_date(row["submitted_at"])
        ref = opens.get((str(row["ticker"]).upper(), day))
        price = pd.to_numeric(row.get("filled_avg_price"), errors="coerce")
        qty = abs(float(pd.to_numeric(row.get("quantity"), errors="coerce")))
        if ref is None or not np.isfinite(price) or ref <= 0 or not np.isfinite(qty) or qty <= 0:
            continue
        sign = -1.0 if str(row["side"]).lower() == "sell" else 1.0
        costs.append(sign * (price / ref - 1.0) * 1e4)
        values.append(qty * float(price))
    if not costs:
        return {"fills": 0, "mean_bps": None}
    return {"fills": len(costs), "mean_bps": float(np.average(costs, weights=values))}


def month_end_signal_days(sessions: pd.DatetimeIndex, first: str = V_FIRST_SIGNAL_DAY) -> pd.DatetimeIndex:
    """Rule V signal days (last NYSE session of each month) from `first` on."""
    ends = pd.DatetimeIndex(sessions.to_series().groupby([sessions.year, sessions.month]).max().to_numpy())
    return ends[ends >= pd.Timestamp(first)]


def v_rebalances_done(signal_days: pd.DatetimeIndex, sessions: pd.DatetimeIndex, as_of) -> int:
    """Signal days whose trading session (the next one) is on or before `as_of`."""
    done = 0
    for day in signal_days:
        pos = int(sessions.get_loc(day))
        if pos + 1 < len(sessions) and sessions[pos + 1] <= pd.Timestamp(as_of):
            done += 1
    return done


def v_measures(equity: pd.DataFrame, qqq: pd.DataFrame, bil: pd.DataFrame, sessions: pd.DatetimeIndex,
               as_of, ir=None) -> dict:
    """V1, V2 and V4 for Forward Test Rule FV.

    PLAIN ENGLISH: the account curve and buy-and-hold QQQ both start at the
    close before the first V trade.  Drawdown and Sharpe use the tested
    script's `metrics`.  V4 compares each completed month's account return
    with the tested `simulate` of rule V over the same month.
    """
    ir = ir or index_risk_module()
    as_of = pd.Timestamp(as_of).normalize()
    first_trade = sessions[sessions > pd.Timestamp(V_FIRST_SIGNAL_DAY)][0]
    base = sessions[sessions < first_trade][-1]
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    curve = frame.drop_duplicates("date", keep="last").set_index("date")["equity"].astype(float).sort_index()
    curve = curve.loc[base:as_of]
    out = {"first_trade_day": str(first_trade.date()), "base_day": str(base.date()),
           "days": max(0, len(curve) - 1)}
    if len(curve) < 3:
        out["status"] = "collecting"
        return out
    if curve.index[0] != base:
        out["status"] = "account_equity_missing_on_base_day"
        return out
    qqq_curve = qqq["Close"].reindex(curve.index).ffill()
    bil_close = bil["Close"]
    out["account"] = ir.metrics(curve / curve.iloc[0], bil_close)
    out["qqq"] = ir.metrics(qqq_curve / qqq_curve.iloc[0], bil_close)

    # V4: monthly gap to the simulated rule (information only).
    shared = qqq.index.intersection(bil.index)
    qqq_s, bil_s = qqq.loc[shared], bil.loc[shared]
    all_month_ends = month_end_signal_days(sessions, "2000-01-01")
    signal_days = all_month_ends[all_month_ends.isin(shared) & (all_month_ends <= as_of)]
    targets = ir.target_weights("V", qqq_s["Close"], signal_days)
    sim = ir.simulate(qqq_s, bil_s, targets, first_trade, as_of)["equity"]
    gaps = []
    previous = base
    for day in signal_days[signal_days > base]:
        if day not in curve.index or day not in sim.index:
            continue
        account = float(curve.loc[day] / curve.loc[previous] - 1.0)
        start_value = 1.0 if previous == base else float(sim.loc[previous])
        simulated = float(sim.loc[day] / start_value - 1.0)
        gaps.append({"month": str(day.date())[:7], "account_pct": round(100 * account, 3),
                     "simulated_pct": round(100 * simulated, 3),
                     "gap_pts": round(100 * (account - simulated), 3)})
        previous = day
    out["v4_monthly"] = gaps
    out["status"] = "ok"
    return out


# ── Putting it together ────────────────────────────────────────────────────

def _prices(ticker: str, start, end) -> pd.DataFrame:
    from data_provider import download_single
    frame = download_single(ticker, start=str(pd.Timestamp(start).date()),
                            end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()))
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return frame


def compute(args) -> dict:
    from v_rule_signal import _nyse_sessions

    if not args.signals_dir:
        subprocess.run(["git", "fetch", "-q", "origin", "signals/latest"], check=True)
    read = lambda name: read_signal_file(name, signals_dir=args.signals_dir)  # noqa: E731
    epoch = json.loads(read("paper_validation_epoch.json"))
    period = json.loads(read("paper_rebalance_period.json"))
    equity = pd.read_csv(io.StringIO(read("alpaca_paper_equity.csv")))
    outcomes = pd.read_csv(io.StringIO(read("alpaca_submit_outcomes.csv")))
    fills = pd.read_csv(io.StringIO(read("alpaca_paper_log.csv")), low_memory=False)

    as_of = pd.Timestamp(args.as_of or date.today())
    epoch_start = pd.Timestamp(epoch["started_at"])
    start_day = ny_date(epoch_start)
    sessions = _nyse_sessions("2025-01-01", as_of + pd.Timedelta(days=120))
    grid = scheduled_dates(period["period"], sessions)

    fills = fills[(fills["fill_status"].astype(str) == "filled")
                  & (pd.to_datetime(fills["submitted_at"], utc=True) >= epoch_start)]
    opens: dict = {}
    for ticker in sorted(fills["ticker"].astype(str).str.upper().unique()):
        bars = _prices(ticker, start_day - pd.Timedelta(days=5), as_of)
        opens.update({(ticker, d): float(o) for d, o in bars["Open"].items()})
    qqq = _prices("QQQ", start_day - pd.Timedelta(days=10), as_of)["Close"]
    bil = _prices("BIL", start_day - pd.Timedelta(days=10), as_of)["Close"]

    return {
        "as_of": str(as_of.date()), "epoch_id": epoch.get("epoch_id"), "epoch_start": str(epoch_start),
        "periods_done": periods_done(grid, sessions, start_day, as_of),
        "next_scheduled_signal_day": period.get("next_scheduled_rebalance_date"),
        "M1": relative_return(equity, qqq, bil, start_day),
        "M2": fill_costs(fills, opens),
        "M3": blocked_share(grid, sessions, outcomes, start_day, as_of),
        "note": "Numbers only. Judged at checkpoints (12 and 24 periods) by FORWARD_TEST_RULE.md.",
    }


def compute_v(args) -> dict:
    """Forward Test Rule FV numbers from the daily run's files."""
    from v_rule_signal import _nyse_sessions

    if not args.signals_dir:
        subprocess.run(["git", "fetch", "-q", "origin", "signals/latest"], check=True)
    read = lambda name: read_signal_file(name, signals_dir=args.signals_dir)  # noqa: E731
    equity = pd.read_csv(io.StringIO(read("alpaca_paper_equity.csv")))
    fills = pd.read_csv(io.StringIO(read("alpaca_paper_log.csv")), low_memory=False)
    as_of = pd.Timestamp(args.as_of or date.today())
    sessions = _nyse_sessions("2025-01-01", as_of + pd.Timedelta(days=120))
    signal_days = month_end_signal_days(sessions)
    first_trade = sessions[sessions > pd.Timestamp(V_FIRST_SIGNAL_DAY)][0]

    fills = fills[(fills["fill_status"].astype(str) == "filled")
                  & (pd.to_datetime(fills["submitted_at"], utc=True).map(ny_date) >= first_trade)]
    opens: dict = {}
    for ticker in sorted(fills["ticker"].astype(str).str.upper().unique()):
        bars = _prices(ticker, first_trade - pd.Timedelta(days=5), as_of)
        opens.update({(ticker, d): float(o) for d, o in bars["Open"].items()})
    qqq = _prices("QQQ", "2025-01-01", as_of)[["Open", "Close"]]
    bil = _prices("BIL", "2025-01-01", as_of)[["Open", "Close"]]
    return {
        "rule": "FV", "as_of": str(as_of.date()),
        "rebalances_done": v_rebalances_done(signal_days, sessions, as_of),
        "checkpoint_at": V_CHECKPOINT_REBALANCES,
        "V1_V2_V4": v_measures(equity, qqq, bil, sessions, as_of),
        "V3": weighted_fill_cost(fills, opens),
        "note": "Numbers only. Judged after 12 rebalances by V_RULE_FORWARD_TEST.md.",
    }


def v_log_line(result: dict, note: str) -> str:
    """One row of the FV monthly log table."""
    m = result["V1_V2_V4"]
    acc, qqq = m.get("account", {}), m.get("qqq", {})
    fmt = lambda v, suffix="": "n/a" if v is None else f"{v}{suffix}"  # noqa: E731
    cost = result["V3"]["mean_bps"]
    cost_text = "n/a" if cost is None else f"{cost:.1f}"
    last_gap = m["v4_monthly"][-1]["gap_pts"] if m.get("v4_monthly") else None
    return (f"| {result['as_of'][:7]} | {result['rebalances_done']} | {fmt(acc.get('max_drawdown_pct'), '%')} | "
            f"{fmt(qqq.get('max_drawdown_pct'), '%')} | {fmt(acc.get('sharpe'))} | {fmt(qqq.get('sharpe'))} | "
            f"{cost_text} | {fmt(last_gap)} | {note} |")


def _signed(value: float, digits: int = 1, suffix: str = "") -> str:
    text = f"{abs(value):.{digits}f}{suffix}"
    return ("−" if value < 0 else "+") + text


def log_line(result: dict, note: str) -> str:
    m1, m2, m3 = result["M1"], result["M2"], result["M3"]
    m1_text = (f"{_signed(100 * m1['m1_annualised'], 1, '%')} a year ({m1['days']} days)"
               if m1.get("m1_annualised") is not None
               else f"n/a ({m1['days']} days; account {_signed(100 * m1.get('account_return', 0.0), 2, '%')}"
                    f" vs yardstick {_signed(100 * m1.get('yardstick_return', 0.0), 2, '%')})")
    if m2["mean_bps"] is None:
        m2_text = "n/a, 0 fills"
    elif m2["ci95_bps"]:
        lo, hi = m2["ci95_bps"]
        m2_text = f"{_signed(m2['mean_bps'], 0)} bps ({_signed(lo, 0)} to {_signed(hi, 0)}), {m2['fills']} fills"
    else:
        m2_text = f"{_signed(m2['mean_bps'], 0)} bps, {m2['fills']} fill"
    m3_text = f"{m3['blocked']} of {m3['scheduled']} scheduled dates"
    if m3["t_block_triggered"]:
        m3_text += " — **T-block triggered**"
    return f"| {result['as_of']} | {result['periods_done']} | {m1_text} | {m2_text} | {m3_text} | {note} |"


def append_log(line: str, doc: Path = RULE_DOC, heading: str = "## Log") -> None:
    """Add the line after the last row of the log table."""
    lines = doc.read_text(encoding="utf-8").splitlines()
    start = next(i for i, text in enumerate(lines) if text.strip() == heading)
    last = max(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    lines.insert(last + 1, line)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Forward Test Rule F1 monthly numbers (read-only).")
    parser.add_argument("--as-of", help="date to measure up to (default: today)")
    parser.add_argument("--signals-dir", help="read files from this folder instead of the signals/latest branch")
    parser.add_argument("--append-log", action="store_true", help="add a line to FORWARD_TEST_RULE.md")
    parser.add_argument("--note", default="Monthly check (forward_test_monitor.py).")
    parser.add_argument("--rule", choices=("f1", "v"), default="f1",
                        help="f1 = core-alpha rule F1; v = rule V (Forward Test Rule FV)")
    args = parser.parse_args(argv)
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # "−" signs on Windows consoles
    if args.rule == "v":
        result = compute_v(args)
        V_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        V_OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
        line = v_log_line(result, args.note)
        print(json.dumps(result, indent=1))
        print("\nLog line:\n" + line)
        if args.append_log:
            append_log(line, V_RULE_DOC, "## Monthly log")
            print(f"\nAdded to {V_RULE_DOC}")
        return result
    result = compute(args)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    line = log_line(result, args.note)
    print(json.dumps(result, indent=1))
    print("\nLog line:\n" + line)
    if result["M3"]["t_block_triggered"]:
        print("\nT-BLOCK: 4 or more of 6 scheduled rebalances in a row were blocked. See FORWARD_TEST_RULE.md.")
    if args.append_log:
        append_log(line)
        print(f"\nAdded to {RULE_DOC}")
    return result


if __name__ == "__main__":
    main()
