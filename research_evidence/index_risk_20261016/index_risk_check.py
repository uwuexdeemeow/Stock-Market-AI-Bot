"""
index_risk_check.py — pre-registered "H-index-risk" test.

PLAIN ENGLISH: stock picking didn't beat QQQ on a fair stock list
(H-wide-universe).  This script asks a different question: can a simple,
fixed risk rule on QQQ itself give a clearly smoother ride than just
holding QQQ, at a small and known cost?

Two rules, settings fixed from the literature (nothing is tuned here):

  T  trend:   at each month end, hold 100% QQQ if its Close is above its
              200-session average, otherwise 100% BIL (a T-bill fund).
  V  vol target: at each month end, QQQ weight = min(1, 20% / QQQ's
              annualised volatility over the last 63 sessions), rest in BIL.

  B  reference: buy QQQ once and hold it.

Trades happen at the next session's Open after the month-end signal and
cost 5 bps per unit of weight traded.  A stress version trades one session
later and pays 25 bps.  The rules and pass/fail gates were written down
BEFORE this script was run: see "Hypothesis H-index-risk" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only: nothing here
changes the paper bot, a gate or a locked file.

Run from the project root (uses data/QQQ.parquet and data/BIL.parquet):
    python research_evidence/index_risk_20261016/index_risk_check.py
Output: research_evidence/index_risk_20261016/index_risk_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
RESULT_JSON = OUT_DIR / "index_risk_check.json"
HYPOTHESIS = "H-index-risk"

DECISION = ("2007-07-02", "2022-12-30")
EXAM_START = "2023-01-03"
TREND_SESSIONS = 200
VOL_SESSIONS = 63
VOL_TARGET = 0.20
COST_BPS, STRESS_COST_BPS = 5.0, 25.0
STRESS_EXTRA_DELAY = 1
TRADING_DAYS = 252

# Gates (fixed in the pre-registration).
R1_DRAWDOWN_IMPROVEMENT = 10.0      # percentage points smaller than B's
R3_MAX_CAGR_GIVE_UP = 3.0           # percentage points a year
EXAM_SHARPE_SLACK = 0.20


def load_prices(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted daily Open/Close for QQQ and BIL, on their shared dates."""
    frames = []
    for symbol in ("QQQ", "BIL"):
        frame = pd.read_parquet(Path(data_dir) / f"{symbol}.parquet", columns=["Open", "Close"])
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
        frames.append(frame.sort_index())
    qqq, bil = frames
    qqq_full = qqq.copy()                     # QQQ alone, for signal history before BIL starts
    shared = qqq.index.intersection(bil.index)
    return qqq_full, bil.loc[shared]


# ── Signals (pure, tested) ─────────────────────────────────────────────────

def month_end_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading session of each month in `index`."""
    series = index.to_series()
    return pd.DatetimeIndex(series.groupby([index.year, index.month]).max().to_numpy())


def trend_weight(close: pd.Series, day: pd.Timestamp, sessions: int = TREND_SESSIONS) -> float:
    """1.0 if the Close on `day` is above its `sessions`-day average, else 0.0."""
    history = close.loc[:day].tail(sessions)
    if len(history) < sessions:
        return 1.0
    return 1.0 if float(close.loc[day]) > float(history.mean()) else 0.0


def vol_weight(close: pd.Series, day: pd.Timestamp, sessions: int = VOL_SESSIONS,
               target: float = VOL_TARGET) -> float:
    """min(1, target / annualised volatility of the last `sessions` daily returns)."""
    returns = close.loc[:day].pct_change().dropna().tail(sessions)
    if len(returns) < sessions:
        return 1.0
    vol = float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    return float(min(1.0, target / vol)) if vol > 0 else 1.0


def target_weights(rule: str, qqq_close: pd.Series, signal_days: pd.DatetimeIndex) -> pd.Series:
    """QQQ target weight decided at each signal day's Close."""
    if rule == "B":
        return pd.Series(1.0, index=signal_days)
    fn = trend_weight if rule == "T" else vol_weight
    return pd.Series([fn(qqq_close, d) for d in signal_days], index=signal_days)


# ── Simulation (pure, tested) ──────────────────────────────────────────────

def simulate(qqq: pd.DataFrame, bil: pd.DataFrame, targets: pd.Series, start, end, *,
             delay: int = 0, cost_bps: float = COST_BPS, hold_only: bool = False) -> dict:
    """Run one rule from `start` to `end` on daily Open/Close prices.

    PLAIN ENGLISH: the money is split between QQQ and BIL.  Each target
    (decided at a month-end Close) is traded at the Open `1 + delay`
    sessions later; between trades the two holdings just move with their
    prices.  Every trade pays `cost_bps` per unit of weight changed.  The
    first trade happens at `start`'s Open, using the latest target decided
    before it.  `hold_only` (rule B) buys once and never trades again.
    """
    days = bil.loc[start:end].index
    q, b = qqq.reindex(days), bil.reindex(days)
    all_days = bil.index
    trade_on: dict[pd.Timestamp, float] = {}
    for signal_day, weight in targets.items():
        pos = all_days.searchsorted(signal_day, side="right") + delay
        if pos < len(all_days):
            trade_on[all_days[pos]] = float(weight)
    first = targets[targets.index < days[0]]
    if first.empty:
        raise ValueError("No target decided before the start date.")
    trade_on[days[0]] = float(first.iloc[-1])

    cost = cost_bps / 1e4
    q_val = b_val = 0.0
    value = 1.0
    equity, weights, turnover = [], [], 0.0
    prev_q_close = prev_b_close = None
    for i, day in enumerate(days):
        q_open, q_close = float(q.at[day, "Open"]), float(q.at[day, "Close"])
        b_open, b_close = float(b.at[day, "Open"]), float(b.at[day, "Close"])
        if i > 0:                                         # overnight move
            q_val *= q_open / prev_q_close
            b_val *= b_open / prev_b_close
            value = q_val + b_val
        if day in trade_on and (i == 0 or not hold_only):
            target = trade_on[day]
            # The first purchase buys everything (QQQ and/or BIL), so it
            # pays on the whole amount; later trades pay on the change only.
            change = abs(target - q_val / value) if i > 0 else 1.0
            value *= 1.0 - cost * change
            turnover += change
            q_val, b_val = value * target, value * (1.0 - target)
        q_val *= q_close / q_open                         # intraday move
        b_val *= b_close / b_open
        value = q_val + b_val
        equity.append(value)
        weights.append(q_val / value if value > 0 else 0.0)
        prev_q_close, prev_b_close = q_close, b_close
    curve = pd.Series(equity, index=days)
    return {"equity": curve, "weights": pd.Series(weights, index=days), "turnover": turnover}


def metrics(curve: pd.Series, bil_close: pd.Series, weights: pd.Series | None = None) -> dict:
    """CAGR, Sharpe over BIL, max drawdown (all in %, Sharpe plain)."""
    returns = curve.pct_change().dropna()
    cash = bil_close.reindex(curve.index).pct_change().dropna().reindex(returns.index).fillna(0.0)
    excess = returns - cash
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = float(curve.iloc[-1] / 1.0)
    peak = curve.cummax()
    out = {
        "cagr_pct": round(100 * (total ** (1 / years) - 1), 3) if years > 0 else None,
        "sharpe": round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)), 3) if excess.std() > 0 else None,
        "max_drawdown_pct": round(100 * float((curve / peak - 1).min()), 2),
        "total_return_pct": round(100 * (total - 1), 2),
    }
    if weights is not None:
        out["average_qqq_weight"] = round(float(weights.mean()), 3)
    return out


def judge(decision: dict, stress: dict) -> dict:
    """Apply gates R1–R4 to T and V against B (decision window)."""
    b = decision["B"]
    out = {}
    for rule in ("T", "V"):
        d, s = decision[rule], stress[rule]
        sb = stress["B"]
        r1 = d["max_drawdown_pct"] >= b["max_drawdown_pct"] + R1_DRAWDOWN_IMPROVEMENT
        r2 = d["sharpe"] >= b["sharpe"]
        r3 = d["cagr_pct"] >= b["cagr_pct"] - R3_MAX_CAGR_GIVE_UP
        r4 = (s["max_drawdown_pct"] >= sb["max_drawdown_pct"] + R1_DRAWDOWN_IMPROVEMENT
              and s["sharpe"] >= sb["sharpe"])
        out[rule] = {"R1_smoother": bool(r1), "R2_sharpe": bool(r2), "R3_affordable": bool(r3),
                     "R4_robust": bool(r4), "passes_decision": bool(r1 and r2 and r3 and r4)}
    return out


def judge_exam(exam: dict, rule: str) -> dict:
    e, b = exam[rule], exam["B"]
    dd = e["max_drawdown_pct"] >= b["max_drawdown_pct"]
    sharpe = e["sharpe"] >= b["sharpe"] - EXAM_SHARPE_SLACK
    return {"drawdown_not_worse": bool(dd), "sharpe_within_0_20": bool(sharpe), "passes": bool(dd and sharpe)}


# ── Diagnostics (recorded, not judged) ─────────────────────────────────────

def yearly_returns(curve: pd.Series) -> dict:
    year_end = curve.groupby(curve.index.year).last()
    start = pd.concat([pd.Series([1.0], index=[year_end.index[0] - 1]), year_end]).shift(1).dropna()
    return {int(y): round(100 * float(year_end[y] / start[y] - 1), 2) for y in year_end.index}


def switches_per_year(weights: pd.Series, threshold: float = 0.01) -> float:
    changes = (weights.diff().abs() > threshold).sum()
    years = (weights.index[-1] - weights.index[0]).days / 365.25
    return round(float(changes / years), 2) if years > 0 else 0.0


def run(data_dir: str = "data") -> dict:
    qqq, bil = load_prices(data_dir)
    signal_days = month_end_sessions(bil.index)
    signal_days = signal_days[signal_days < bil.index[-1]]
    targets = {rule: target_weights(rule, qqq["Close"], signal_days) for rule in ("B", "T", "V")}
    last_day = bil.index[-1]

    def window(start, end, *, delay=0, cost=COST_BPS):
        results, curves = {}, {}
        for rule in ("B", "T", "V"):
            sim = simulate(qqq, bil, targets[rule], start, end, delay=delay, cost_bps=cost, hold_only=(rule == "B"))
            results[rule] = {**metrics(sim["equity"], bil["Close"], sim["weights"]),
                             "turnover": round(sim["turnover"], 2)}
            curves[rule] = sim
        return results, curves

    decision, curves = window(*DECISION)
    stress, _ = window(*DECISION, delay=STRESS_EXTRA_DELAY, cost=STRESS_COST_BPS)
    gates = judge(decision, stress)
    diagnostics = {
        rule: {"yearly_returns_pct": yearly_returns(curves[rule]["equity"]),
               "switches_per_year": switches_per_year(curves[rule]["weights"]),
               "crash_years": {y: yearly_returns(curves[rule]["equity"]).get(y) for y in (2008, 2020, 2022)}}
        for rule in ("B", "T", "V")
    }
    result = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
              "decision_window": DECISION, "data_last_date": str(last_day.date()),
              "decision": decision, "stress": stress, "gates": gates, "diagnostics": diagnostics}

    passing = [rule for rule in ("T", "V") if gates[rule]["passes_decision"]]
    if passing:
        exam, _ = window(EXAM_START, last_day)
        result["exam_window"] = (EXAM_START, str(last_day.date()))
        result["exam"] = {rule: exam[rule] for rule in ("B", *passing)}
        result["exam_gates"] = {rule: judge_exam(exam, rule) for rule in passing}
    result["verdicts"] = {
        rule: ("passes" if rule in passing and result["exam_gates"][rule]["passes"]
               else "fails the exam" if rule in passing else "fails decision gates")
        for rule in ("T", "V")
    }
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-index-risk check (research only).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=str(RESULT_JSON))
    args = parser.parse_args(argv)
    result = run(args.data_dir)
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("decision", "stress", "gates", "verdicts")}, indent=1, default=str))
    return result


if __name__ == "__main__":
    main()
