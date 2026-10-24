"""
multi_asset_check.py — pre-registered "H-multi-asset" test.

PLAIN ENGLISH: rule V holds QQQ (scaled down when QQQ gets jumpy) and puts
the rest in BIL (a T-bill fund, basically cash).  This script asks: would a
MIX of QQQ, long Treasuries (TLT) and gold (GLD) do better than V?  These
three often don't fall at the same time, so the mix should swing less.

  V   reference: QQQ weight = min(1, 20% / QQQ's 63-session volatility),
      rest in BIL.  (Exactly the rule the paper account trades.)
  M   mix, no borrowing: each asset sized by 1 / its own volatility, then
      the whole mix scaled to 20% volatility, never more than 100%
      invested; the rest in BIL.
  ML  mix, with borrowing: the same, but it may hold up to 2x its money.
      The borrowed part pays BIL's return + 2.0 points a year.
  B   buy-and-hold QQQ, recorded only.

All rules decide at a month-end Close and trade at the next session's Open,
paying 5 bps per unit of weight traded.  A stress version trades one session
later, pays 25 bps, and borrows at BIL + 4.0 points.  The rules and the
pass/fail gates were written down BEFORE this script was run: see
"Hypothesis H-multi-asset" in Documentation/DELAY_STRESS_PAPER_ADVISORY.md.
Research only: nothing here changes the paper bot, V, a gate or a locked file.

Run from the project root (reads data/QQQ, TLT, GLD, BIL .parquet):
    python research_evidence/multi_asset_20261025/multi_asset_check.py
Output: research_evidence/multi_asset_20261025/multi_asset_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
RESULT_JSON = OUT_DIR / "multi_asset_check.json"
INDEX_RISK_JSON = OUT_DIR.parent / "index_risk_20261016" / "index_risk_check.json"
HYPOTHESIS = "H-multi-asset"

RISKY = ("QQQ", "TLT", "GLD")          # the assets the mix holds
CASH = "BIL"
DECISION = ("2007-07-02", "2022-12-30")
SUBPERIODS = (("2007-07-02", "2012-12-31"), ("2013-01-01", "2017-12-31"), ("2018-01-01", "2022-12-30"))
EXAM_START = "2023-01-03"
VOL_SESSIONS = 63
VOL_TARGET = 0.20
GROSS_CAP = {"V": 1.0, "M": 1.0, "ML": 2.0}
COST_BPS, STRESS_COST_BPS = 5.0, 25.0
BORROW_SPREAD, STRESS_BORROW_SPREAD = 0.02, 0.04   # a year, on top of BIL's return
STRESS_EXTRA_DELAY = 1
TRADING_DAYS = 252

# Gates (fixed in the pre-registration).
M1_SHARPE_EDGE = 0.10               # M's Sharpe must beat V's by this much
L1_CAGR_EDGE = 1.0                  # ML's CAGR must beat V's by this many points
L2_DRAWDOWN_SLACK = 5.0             # ML's max drawdown may be this many points worse
EXAM_SHARPE_SLACK = 0.20
EXAM_DRAWDOWN_SLACK = 5.0
PLUMBING_TOLERANCE = 0.3            # V here vs V in H-index-risk, CAGR points
MAX_MISSING_SHARE = 0.01            # D1: share of sessions an asset may miss


# ── Data ───────────────────────────────────────────────────────────────────

def _load(symbol: str, data_dir: str) -> pd.DataFrame:
    """Adjusted daily Open/Close for one ETF, dates without time zone."""
    frame = pd.read_parquet(Path(data_dir) / f"{symbol}.parquet", columns=["Open", "Close"])
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def load_prices(data_dir: str = "data") -> dict:
    """All four ETFs.

    Returns:
      - "trade": one Open/Close table per ETF, on the dates where ALL FOUR
        have a bar (days missing any ETF are dropped, never filled);
      - "risky_close": Closes of QQQ/TLT/GLD on their own shared dates (from
        2004), used for the volatility signals so they have full history
        on the first month end;
      - "qqq_close": QQQ alone, for V's signal (as in H-index-risk);
      - "missing_share": per ETF, the share of decision-window sessions it
        lacks (the D1 gate).
    """
    raw = {s: _load(s, data_dir) for s in (*RISKY, CASH)}
    shared = raw[CASH].index
    for s in RISKY:
        shared = shared.intersection(raw[s].index)
    union = raw[CASH].index
    for s in RISKY:
        union = union.union(raw[s].index)
    window = union[(union >= DECISION[0]) & (union <= DECISION[1])]
    missing = {s: round(float(1 - window.isin(raw[s].index).mean()), 5) for s in raw}
    risky_dates = raw[RISKY[0]].index
    for s in RISKY[1:]:
        risky_dates = risky_dates.intersection(raw[s].index)
    risky_close = pd.DataFrame({s: raw[s]["Close"].reindex(risky_dates) for s in RISKY})
    return {
        "trade": {s: raw[s].reindex(shared) for s in raw},
        "risky_close": risky_close,
        "qqq_close": raw["QQQ"]["Close"],
        "missing_share": missing,
    }


# ── Signals (pure, tested) ─────────────────────────────────────────────────

def month_end_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading session of each month in `index`."""
    series = index.to_series()
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).max().to_numpy())


def annual_vol(returns: pd.Series | np.ndarray) -> float:
    """Standard deviation of daily returns (ddof=1) times sqrt(252)."""
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(TRADING_DAYS))


def v_target(qqq_close: pd.Series, day, sessions: int = VOL_SESSIONS, target: float = VOL_TARGET) -> dict:
    """Rule V: QQQ = min(1, 20% / its 63-session volatility), rest in BIL."""
    returns = qqq_close.loc[:day].pct_change().dropna().tail(sessions)
    if len(returns) < sessions:
        weight = 1.0
    else:
        vol = annual_vol(returns)
        weight = float(min(1.0, target / vol)) if vol > 0 else 1.0
    return {"QQQ": weight, "TLT": 0.0, "GLD": 0.0, CASH: 1.0 - weight}


def mix_target(risky_close: pd.DataFrame, day, cap: float, sessions: int = VOL_SESSIONS,
               target: float = VOL_TARGET) -> dict:
    """Rules M and ML: inverse-volatility mix scaled to the volatility target.

    PLAIN ENGLISH:
      1. each asset's volatility over the last 63 sessions;
      2. base weights proportional to 1 / volatility (calmer assets get
         more), adding up to 1;
      3. the volatility of that base mix, measured on the mix's own daily
         returns (so it counts how the assets move together);
      4. scale k = min(cap, 20% / mix volatility).  Weights = k x base.
         If k < 1 the rest is BIL.  If k > 1 the part above 1 is borrowed
         (no BIL is held then).
    """
    returns = risky_close.loc[:day].pct_change().dropna().tail(sessions)
    if len(returns) < sessions:
        raise ValueError(f"Not enough history for the mix signal on {day}")
    vols = returns.apply(annual_vol)
    inverse = 1.0 / vols
    base = inverse / inverse.sum()
    mix_vol = annual_vol(returns.to_numpy() @ base.to_numpy())
    k = float(min(cap, target / mix_vol)) if mix_vol > 0 else cap
    weights = {s: float(k * base[s]) for s in RISKY}
    weights[CASH] = max(0.0, 1.0 - k)
    return weights


def target_table(rule: str, prices: dict, signal_days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row of target weights (QQQ, TLT, GLD, BIL) per signal day."""
    rows = []
    for day in signal_days:
        if rule in ("V",):
            rows.append(v_target(prices["qqq_close"], day))
        elif rule == "B":
            rows.append({"QQQ": 1.0, "TLT": 0.0, "GLD": 0.0, CASH: 0.0})
        else:
            rows.append(mix_target(prices["risky_close"], day, GROSS_CAP[rule]))
    return pd.DataFrame(rows, index=signal_days, columns=[*RISKY, CASH])


# ── Simulation (pure, tested) ──────────────────────────────────────────────

def simulate(trade: dict, targets: pd.DataFrame, start, end, *, delay: int = 0,
             cost_bps: float = COST_BPS, borrow_spread: float = BORROW_SPREAD,
             hold_only: bool = False) -> dict:
    """Run one rule from `start` to `end` on daily Open/Close prices.

    PLAIN ENGLISH: the money sits in up to four ETFs, plus a loan when the
    rule holds more than 100%.  Each target (decided at a month-end Close)
    is traded at the Open `1 + delay` sessions later; between trades every
    holding just moves with its own price.  A trade pays `cost_bps` per unit
    of weight changed, summed over all ETFs.  The loan grows each day by
    BIL's daily return plus `borrow_spread` / 252.  The first trade happens
    at `start`'s Open with the latest target decided before it, and pays on
    the whole amount bought.
    """
    symbols = [*RISKY, CASH]
    cal = trade[CASH].index
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    opens = pd.DataFrame({s: trade[s]["Open"] for s in symbols}).reindex(days)
    closes = pd.DataFrame({s: trade[s]["Close"] for s in symbols}).reindex(days)
    trade_on: dict[pd.Timestamp, pd.Series] = {}
    for signal_day, row in targets.iterrows():
        pos = cal.searchsorted(signal_day, side="right") + delay
        if pos < len(cal):
            trade_on[cal[pos]] = row
    first = targets[targets.index < days[0]]
    if first.empty:
        raise ValueError("No target decided before the start date.")
    trade_on[days[0]] = first.iloc[-1]

    cost = cost_bps / 1e4
    hold = pd.Series(0.0, index=symbols)       # dollars in each ETF
    debt = 0.0                                 # dollars borrowed
    value = 1.0
    equity, gross_path, weight_rows = [], [], []
    turnover = interest_paid = 0.0
    prev_close = None
    for i, day in enumerate(days):
        o, c = opens.loc[day], closes.loc[day]
        if i > 0:                                         # overnight move
            hold = hold * (o / prev_close)
            value = float(hold.sum()) - debt
        if day in trade_on and (i == 0 or not hold_only):
            target = trade_on[day].astype(float)
            current = hold / value if i > 0 else pd.Series(0.0, index=symbols)
            change = float((target - current).abs().sum())
            value *= 1.0 - cost * change
            turnover += change
            hold = value * target
            debt = value * max(0.0, float(target[list(RISKY)].sum()) - 1.0)
        hold = hold * (c / o)                              # intraday move
        if i > 0 and debt > 0:                             # one day of interest
            bil_day = float(c[CASH] / prev_close[CASH] - 1.0)
            interest = debt * (bil_day + borrow_spread / TRADING_DAYS)
            debt += interest
            interest_paid += interest
        value = float(hold.sum()) - debt
        if value <= 0:
            raise ValueError(f"Portfolio wiped out on {day.date()}")
        equity.append(value)
        gross_path.append(float(hold[list(RISKY)].sum()) / value)
        weight_rows.append(hold / value)
        prev_close = c
    curve = pd.Series(equity, index=days)
    return {"equity": curve, "gross": pd.Series(gross_path, index=days),
            "weights": pd.DataFrame(weight_rows, index=days), "turnover": turnover,
            "interest_paid": interest_paid}


def metrics(curve: pd.Series, bil_close: pd.Series) -> dict:
    """CAGR, Sharpe over BIL, max drawdown (all in %, Sharpe plain)."""
    returns = curve.pct_change().dropna()
    cash = bil_close.reindex(curve.index).pct_change().dropna().reindex(returns.index).fillna(0.0)
    excess = returns - cash
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = float(curve.iloc[-1] / 1.0)
    peak = curve.cummax()
    return {
        "cagr_pct": round(100 * (total ** (1 / years) - 1), 3) if years > 0 else None,
        "sharpe": round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)), 3) if excess.std() > 0 else None,
        "max_drawdown_pct": round(100 * float((curve / peak - 1).min()), 2),
        "total_return_pct": round(100 * (total - 1), 2),
    }


def period_cagr(curve: pd.Series, start, end) -> float:
    """CAGR of one slice of an equity curve, starting from the prior close."""
    before = curve[curve.index < pd.Timestamp(start)]
    inside = curve[(curve.index >= pd.Timestamp(start)) & (curve.index <= pd.Timestamp(end))]
    base = float(before.iloc[-1]) if len(before) else 1.0
    base_day = before.index[-1] if len(before) else inside.index[0]
    years = (inside.index[-1] - base_day).days / 365.25
    return round(100 * ((float(inside.iloc[-1]) / base) ** (1 / years) - 1), 3)


# ── Gates (fixed in the pre-registration) ──────────────────────────────────

def judge(decision: dict, stress: dict, subperiods: dict) -> dict:
    v, sv = decision["V"], stress["V"]
    m, sm = decision["M"], stress["M"]
    m1 = m["sharpe"] >= v["sharpe"] + M1_SHARPE_EDGE
    m2 = m["max_drawdown_pct"] >= v["max_drawdown_pct"]
    m3 = sm["sharpe"] >= sv["sharpe"] + M1_SHARPE_EDGE and sm["max_drawdown_pct"] >= sv["max_drawdown_pct"]
    ml, sml = decision["ML"], stress["ML"]
    l1 = ml["cagr_pct"] >= v["cagr_pct"] + L1_CAGR_EDGE
    l2 = ml["max_drawdown_pct"] >= v["max_drawdown_pct"] - L2_DRAWDOWN_SLACK
    l3 = ml["sharpe"] >= v["sharpe"]
    wins = sum(1 for p in subperiods.values() if p["ML"] > p["V"])
    l4 = wins >= 2
    l5 = sml["cagr_pct"] >= sv["cagr_pct"] + L1_CAGR_EDGE and sml["sharpe"] >= sv["sharpe"]
    return {
        "M": {"M1_sharpe_edge": bool(m1), "M2_drawdown": bool(m2), "M3_robust": bool(m3),
              "passes_decision": bool(m1 and m2 and m3)},
        "ML": {"L1_more_return": bool(l1), "L2_drawdown": bool(l2), "L3_sharpe": bool(l3),
               "L4_subperiods": bool(l4), "subperiod_wins": wins, "L5_robust": bool(l5),
               "passes_decision": bool(l1 and l2 and l3 and l4 and l5)},
    }


def judge_exam(exam: dict, rule: str) -> dict:
    e, v = exam[rule], exam["V"]
    sharpe = e["sharpe"] >= v["sharpe"] - EXAM_SHARPE_SLACK
    dd = e["max_drawdown_pct"] >= v["max_drawdown_pct"] - EXAM_DRAWDOWN_SLACK
    return {"sharpe_within_0_20": bool(sharpe), "drawdown_within_5": bool(dd), "passes": bool(sharpe and dd)}


# ── Diagnostics (recorded, not judged) ─────────────────────────────────────

def yearly_returns(curve: pd.Series) -> dict:
    year_end = curve.groupby(curve.index.year).last()
    start = pd.concat([pd.Series([1.0], index=[year_end.index[0] - 1]), year_end]).shift(1).dropna()
    return {int(y): round(100 * float(year_end[y] / start[y] - 1), 2) for y in year_end.index}


def run(data_dir: str = "data") -> dict:
    prices = load_prices(data_dir)
    trade = prices["trade"]
    bil_close = trade[CASH]["Close"]
    result = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
              "decision_window": DECISION, "data_last_date": str(bil_close.index[-1].date()),
              "d1_missing_share": prices["missing_share"]}
    if max(prices["missing_share"].values()) > MAX_MISSING_SHARE:
        result["verdicts"] = {"M": "not testable", "ML": "not testable"}
        return result

    cal = bil_close.index
    signal_days = month_end_sessions(cal)
    signal_days = signal_days[signal_days < cal[-1]]
    rules = ("B", "V", "M", "ML")
    targets = {rule: target_table(rule, prices, signal_days) for rule in rules}

    def window(start, end, *, delay=0, cost=COST_BPS, spread=BORROW_SPREAD, only=rules):
        results, sims = {}, {}
        for rule in only:
            sim = simulate(trade, targets[rule], start, end, delay=delay, cost_bps=cost,
                           borrow_spread=spread, hold_only=(rule == "B"))
            years = (sim["equity"].index[-1] - sim["equity"].index[0]).days / 365.25
            results[rule] = {**metrics(sim["equity"], bil_close),
                             "average_gross": round(float(sim["gross"].mean()), 3),
                             "average_weights": {s: round(float(w), 3) for s, w in sim["weights"].mean().items()},
                             "turnover_per_year": round(sim["turnover"] / years, 2),
                             "interest_paid_per_dollar_start": round(sim["interest_paid"], 4)}
            sims[rule] = sim
        return results, sims

    # Plumbing check first: V here must match V in H-index-risk.
    v_only, _ = window(*DECISION, only=("V",))
    reference = json.loads(INDEX_RISK_JSON.read_text(encoding="utf-8"))["decision"]["V"]["cagr_pct"]
    gap = abs(v_only["V"]["cagr_pct"] - reference)
    result["plumbing_check"] = {"v_cagr_here": v_only["V"]["cagr_pct"], "v_cagr_h_index_risk": reference,
                                "gap": round(gap, 3), "ok": bool(gap <= PLUMBING_TOLERANCE)}
    if gap > PLUMBING_TOLERANCE:
        result["verdicts"] = {"M": "engine check failed", "ML": "engine check failed"}
        return result

    decision, sims = window(*DECISION)
    stress, _ = window(*DECISION, delay=STRESS_EXTRA_DELAY, cost=STRESS_COST_BPS, spread=STRESS_BORROW_SPREAD)
    subperiods = {f"{a[:4]}-{b[:4]}": {rule: period_cagr(sims[rule]["equity"], a, b) for rule in rules}
                  for a, b in SUBPERIODS}
    gates = judge(decision, stress, subperiods)
    ml_targets = targets["ML"].loc[(targets["ML"].index >= DECISION[0]) & (targets["ML"].index <= DECISION[1])]
    ml_gross = ml_targets[list(RISKY)].sum(axis=1)
    ml_final = float(sims["ML"]["equity"].iloc[-1])
    diagnostics = {
        "yearly_returns_pct": {rule: yearly_returns(sims[rule]["equity"]) for rule in rules},
        "crash_years_pct": {rule: {y: yearly_returns(sims[rule]["equity"]).get(y) for y in (2008, 2020, 2022)}
                            for rule in rules},
        "ml_share_of_months_at_2x_cap": round(float((ml_gross >= GROSS_CAP["ML"] - 1e-9).mean()), 3),
        "ml_average_target_gross": round(float(ml_gross.mean()), 3),
        "ml_interest_share_of_gain_before_interest": round(
            sims["ML"]["interest_paid"] / (ml_final - 1.0 + sims["ML"]["interest_paid"]), 3),
    }
    result.update({"decision": decision, "stress": stress, "subperiod_cagr_pct": subperiods,
                   "gates": gates, "diagnostics": diagnostics})

    passing = [rule for rule in ("M", "ML") if gates[rule]["passes_decision"]]
    if passing:
        exam, _ = window(EXAM_START, cal[-1], only=("B", "V", *passing))
        result["exam_window"] = (EXAM_START, str(cal[-1].date()))
        result["exam"] = exam
        result["exam_gates"] = {rule: judge_exam(exam, rule) for rule in passing}
    result["verdicts"] = {
        rule: ("passes" if rule in passing and result["exam_gates"][rule]["passes"]
               else "fails the exam" if rule in passing else "fails decision gates")
        for rule in ("M", "ML")
    }
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-multi-asset check (research only).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=str(RESULT_JSON))
    args = parser.parse_args(argv)
    result = run(args.data_dir)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    keys = ("d1_missing_share", "plumbing_check", "decision", "stress", "subperiod_cagr_pct", "gates", "verdicts")
    print(json.dumps({k: result[k] for k in keys if k in result}, indent=1, default=str))
    return result


if __name__ == "__main__":
    main()
