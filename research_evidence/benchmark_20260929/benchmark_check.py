"""
benchmark_check.py — pre-registered "H-benchmark" fair-yardstick test.

PLAIN ENGLISH: the strategy is usually only partly in the market (the rest
sits in cash or SPY), so "beat QQQ by X points" mixes two things: picking
good stocks, and how much money was invested.  This script measures the
strategy against two fairer yardsticks:

  B-exp:  each 20-day period, QQQ at the SAME investment level the strategy
          actually had (the rest in cash, BIL).
  B-beta: a daily regression of the strategy on QQQ.  What's left after
          removing "moves with QQQ" is the alpha; a Newey-West t-stat says
          how sure we can be it isn't noise.

The rules were written down BEFORE any run: see "Hypothesis H-benchmark" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only.

Run from the project root (needs data/ and signals/, and internet once for
the S&P 500 membership history):
    python research_evidence/benchmark_20260929/benchmark_check.py
Quick smoke test (NOT a valid result):
    ... --offsets 2
Output: research_evidence/benchmark_20260929/benchmark_check.json
"""
from __future__ import annotations

import argparse
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
HYPOTHESIS = "H-benchmark"
N_OFFSETS = 20
DELAYS = (0, 1)
DECISION_WINDOW = ("2013-01-01", "2022-12-31")
DIAGNOSTIC_WINDOW = ("2023-01-01", "2026-12-31")
F2_MIN_T = 2.0
TRADING_DAYS = 252


def _top_names_module():
    path = OUT_DIR.parent / "top_names_20260929" / "top_names_check.py"
    spec = importlib.util.spec_from_file_location("top_names_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Pure helpers (tested) ──────────────────────────────────────────────────

def portfolio_weights(trade) -> dict[str, float]:
    """Each holding's share of the whole portfolio for one period.

    PLAIN ENGLISH: the core ETFs' weights are shares OF THE CORE, so they are
    multiplied by the core's size (`core_gross`).  The stock weights are
    already shares of the whole portfolio.
    """
    weights: dict[str, float] = {}
    core_gross = float(trade.get("core_gross", 0.0) or 0.0)
    for ticker, w in json.loads(trade.get("core_weights_json") or "{}").items():
        if float(w):
            weights[str(ticker).upper()] = weights.get(str(ticker).upper(), 0.0) + core_gross * float(w)
    for ticker, w in json.loads(trade.get("overlay_weights_json") or "{}").items():
        if float(w):
            weights[str(ticker).upper()] = weights.get(str(ticker).upper(), 0.0) + float(w)
    return weights


def period_path(bars: dict[str, pd.DataFrame], weights: dict[str, float], cash_bars: pd.DataFrame,
                entry: pd.Timestamp, exit_day: pd.Timestamp, cost: float = 0.0) -> pd.Series:
    """Daily returns of one period's holdings, bought at the entry Open.

    Value on day t = Σ weight × Close_t / Open_entry  +  cash weight × BIL
    growth.  The first day's return runs from the entry Open to its Close.
    The period's trading cost is taken off the first day.
    """
    days = cash_bars.loc[(cash_bars.index >= entry) & (cash_bars.index <= exit_day)].index
    if len(days) == 0:
        return pd.Series(dtype=float)
    cash_weight = 1.0 - sum(weights.values())
    value = pd.Series(cash_weight * (cash_bars.loc[days, "Close"] / cash_bars.loc[entry, "Open"]), index=days)
    for ticker, w in weights.items():
        frame = bars[ticker]
        closes = frame["Close"].reindex(days).ffill()
        opens = frame.loc[frame.index >= entry, "Open"]
        if opens.empty:
            continue   # no price after entry: that weight earns nothing
        value = value + w * closes / float(opens.iloc[0])
    previous = value.shift(1)
    previous.iloc[0] = 1.0
    returns = value / previous - 1.0
    returns.iloc[0] -= cost
    return returns


def exposure_matched_equity(trades: pd.DataFrame, qqq: pd.DataFrame, cash: pd.DataFrame,
                            start_value: float, start_date, session_offset) -> pd.Series:
    """B-exp: per period, g × QQQ + (1 − g) × cash, on the engine's period dates."""
    values = [float(start_value)]
    dates = [pd.Timestamp(start_date)]
    for _, trade in trades.iterrows():
        delay = int(trade.get("entry_delay_days", 0) or 0)
        entry = session_offset(pd.Timestamp(trade["date"]), 1 + delay)
        exit_day = pd.Timestamp(trade["exit_date"])
        g = float(trade["gross_exposure"])
        q = float(qqq["Close"].asof(exit_day) / qqq.loc[entry, "Open"] - 1.0)
        c = float(cash["Close"].asof(exit_day) / cash.loc[entry, "Open"] - 1.0) if entry in cash.index else 0.0
        values.append(values[-1] * (1.0 + g * q + (1.0 - g) * c))
        dates.append(pd.Timestamp(trade["label_end_date"]))
    out = pd.Series(values, index=pd.DatetimeIndex(dates))
    return out[~out.index.duplicated(keep="last")]


def window_return(series: pd.Series, window) -> float | None:
    part = series[(series.index >= pd.Timestamp(window[0])) & (series.index <= pd.Timestamp(window[1]))]
    if len(part) < 3:
        return None
    return float(part.iloc[-1] / part.iloc[0] - 1.0)


def beta_adjusted(strategy: pd.Series, qqq: pd.Series, cash: pd.Series, nw_tstat) -> dict:
    """Regress daily excess returns of the strategy on QQQ's.

    alpha (annualised) = intercept × 252.  The t-stat is the Newey-West
    t-stat of the mean of the daily series (strategy − β × QQQ), all in
    excess of cash, which equals the intercept's t-stat up to β's own
    estimation error.
    """
    frame = pd.concat({"y": strategy - cash, "x": qqq - cash}, axis=1).dropna()
    if len(frame) < 30:
        return {"observations": int(len(frame))}
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    residual_series = frame["y"] - beta * frame["x"]
    intercept = float(residual_series.mean())
    resid = residual_series - intercept
    resid_vol = float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "observations": int(len(frame)), "beta": round(beta, 4),
        "alpha_annual_pct": round(intercept * TRADING_DAYS * 100.0, 3),
        "information_ratio": round(intercept * TRADING_DAYS / resid_vol, 4) if resid_vol else None,
        "nw_tstat": round(float(nw_tstat(residual_series)), 4),
    }


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _values(rows, test, delay, key):
    return [r[key] for r in rows if r["test"] == test and r["delay"] == delay and r.get(key) is not None]


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H-benchmark rules."""
    bexp_late = _median(_values(rows, "S", 1, "bexp_alpha_pct"))
    t_late = _median(_values(rows, "S", 1, "nw_tstat"))
    f1 = bexp_late is not None and bexp_late > 0
    f2 = t_late is not None and t_late >= F2_MIN_T
    if f1 and f2:
        verdict = "edge holds against a fair benchmark"
    elif f1:
        verdict = "beats a fair benchmark, but not reliably"
    else:
        verdict = "headline alpha is mostly market exposure and timing of the core, not stock-picking skill"
    return {
        "numbers": {"S_median_late_bexp_alpha": bexp_late, "S_median_late_nw_tstat": t_late},
        "diagnostics": {
            "S_median_late_beta": _median(_values(rows, "S", 1, "beta")),
            "S_median_late_information_ratio": _median(_values(rows, "S", 1, "information_ratio")),
            "S_median_late_alpha_annual_pct": _median(_values(rows, "S", 1, "alpha_annual_pct")),
            "S_median_on_time_bexp_alpha": _median(_values(rows, "S", 0, "bexp_alpha_pct")),
            "S_median_on_time_nw_tstat": _median(_values(rows, "S", 0, "nw_tstat")),
            "S_median_late_alpha_vs_qqq": _median(_values(rows, "S", 1, "decision_alpha_vs_qqq_pct")),
            "S_2023_2026_median_bexp_alpha": _median(_values(rows, "S", 0, "diagnostic_bexp_alpha_pct")),
            "E_median_bexp_alpha": _median(_values(rows, "E", 0, "bexp_alpha_pct")),
            "E_median_nw_tstat": _median(_values(rows, "E", 0, "nw_tstat")),
        },
        "gates": {"F1_beats_exposure_matched": f1, "F2_t_stat": f2},
        "verdict": verdict,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def _bars(ticker: str) -> pd.DataFrame:
    from settings import DATA_DIR
    frame = pd.read_parquet(Path(DATA_DIR) / f"{ticker.upper()}.parquet", columns=["Open", "Close"])
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    return frame[~frame.index.duplicated(keep="last")].sort_index().apply(pd.to_numeric, errors="coerce")


def daily_returns(trades: pd.DataFrame, bars: dict, cash: pd.DataFrame, session_offset,
                  weights_fn=portfolio_weights, with_cost: bool = True) -> pd.Series:
    parts = []
    for _, trade in trades.iterrows():
        delay = int(trade.get("entry_delay_days", 0) or 0)
        entry = session_offset(pd.Timestamp(trade["date"]), 1 + delay)
        weights = weights_fn(trade)
        for ticker in weights:
            if ticker not in bars:
                bars[ticker] = _bars(ticker)
        cost = float(trade.get("cost", 0.0) or 0.0) if with_cost else 0.0
        parts.append(period_path(bars, weights, cash, entry, pd.Timestamp(trade["exit_date"]), cost))
    out = pd.concat(parts) if parts else pd.Series(dtype=float)
    return out[~out.index.duplicated(keep="first")].sort_index()


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-benchmark check (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--out", default=str(OUT_DIR / "benchmark_check.json"))
    args = parser.parse_args(argv)

    from backtest import _newey_west_tstat

    top = _top_names_module()
    edge, core, config, panel_s, sessions, notes = top.build_s_panel(args.membership_csv)
    end = sessions[-30]
    qqq, cash = _bars("QQQ"), _bars("BIL")
    bars = {"QQQ": qqq, "BIL": cash}
    e_config = {**config, "overlay_gross": 0.0, "concentration_overlay_mode": "off"}
    if isinstance(config.get("regime_preset"), dict):
        e_config["regime_preset"] = core._regime_preset_with_overlay_gross(config["regime_preset"], 0.0)

    rows: list[dict] = []
    out_path = Path(args.out)
    valid = args.offsets == N_OFFSETS

    def save(result=None):
        payload = {"hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
                   "valid_full_test": valid, "end_date": str(pd.Timestamp(end).date()),
                   "offsets": args.offsets, "notes": notes, "rows": rows}
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    def measure(test, cfg, panel, delay, offset):
        t0 = time.time()
        equity, trades, extra = core.run_core_satellite(
            panel, {**cfg, "entry_delay_days": delay}, evaluation_start=sessions[offset], evaluation_end=end)
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, t0=t0)
        bexp = exposure_matched_equity(trades, qqq, cash, equity.iloc[0], equity.index[0], core._session_offset)
        for key, window in (("bexp_alpha_pct", DECISION_WINDOW), ("diagnostic_bexp_alpha_pct", DIAGNOSTIC_WINDOW)):
            s, b = window_return(equity, window), window_return(bexp, window)
            row[key] = round((s - b) * 100.0, 2) if (s is not None and b is not None) else None
        strat = daily_returns(trades, bars, cash, core._session_offset)
        qqq_daily = daily_returns(trades, bars, cash, core._session_offset,
                                  weights_fn=lambda _t: {"QQQ": 1.0}, with_cost=False)
        cash_daily = daily_returns(trades, bars, cash, core._session_offset,
                                   weights_fn=lambda _t: {}, with_cost=False)
        in_window = lambda s: s[(s.index >= DECISION_WINDOW[0]) & (s.index <= DECISION_WINDOW[1])]
        row.update(beta_adjusted(in_window(strat), in_window(qqq_daily), in_window(cash_daily), _newey_west_tstat))
        # How well the daily rebuild matches the engine's own period returns.
        period_of_day = np.searchsorted(pd.DatetimeIndex(trades["exit_date"]).values, strat.index.values, side="left")
        rebuilt = (1.0 + strat).groupby(period_of_day).prod() - 1.0
        rebuilt = rebuilt[rebuilt.index < len(trades)]
        gap = rebuilt.to_numpy() - trades["period_return"].to_numpy()[rebuilt.index.to_numpy()]
        row["daily_rebuild_mean_abs_gap_pct"] = round(float(np.mean(np.abs(gap))) * 100.0, 4) if len(gap) else None
        rows.append(row)
        print(json.dumps(row), flush=True)
        save()

    for delay in DELAYS:
        for offset in range(args.offsets):
            measure("S", config, panel_s, delay, offset)
    for offset in range(args.offsets):
        measure("E", e_config, panel_s, 0, offset)

    result = judge(rows)
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdict")}, indent=1))
    return result


if __name__ == "__main__":
    main()
