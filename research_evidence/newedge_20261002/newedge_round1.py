"""
newedge_round1.py — pre-registered "H-newedge-1": three new-edge ideas in one round.

PLAIN ENGLISH: the current strategy's edge is real but leans on a few big
winners and breaks down when trades are late or expensive.  This script tests
three ideas aimed at those weaknesses, on the point-in-time stock list:

  W  "let winners run": keep a held stock while its 12-month momentum stays
     in the top 30%, instead of re-judging it by score every 20 days.
  V  "volatility-scaled core": shrink the ETF part when QQQ has been wild
     lately (never above the normal size, never below 30% of it).
  T  "multi-asset trend core": spread the ETF part over SPY, QQQ, long bonds
     (TLT), medium bonds (IEF) and gold (GLD), holding each only while its
     1-year return beats cash (BIL).

Each idea is switched on by temporarily swapping ONE helper of the engine
while this script runs; the engine file itself is never edited.

Rules (written BEFORE any run): "Hypothesis H-newedge-1" in
Documentation/DELAY_STRESS_PAPER_ADVISORY.md.  Research only.

Run from the project root (needs data/ incl. TLT, signals/, and internet once
for the S&P 500 membership history unless --membership-csv is given):
    python research_evidence/newedge_20261002/newedge_round1.py
Smoke test (NOT a valid result):  ... --offsets 2 --no-exam
Output: research_evidence/newedge_20261002/newedge_round1.json
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
HYPOTHESIS = "H-newedge-1"
N_OFFSETS = 20
DELAYS = (0, 1)
DECISION_WINDOW = ("2013-01-01", "2022-12-31")
PERIODS_PER_YEAR = 252 / 20
W_KEEP_MOMENTUM_RANK = 0.70
W_MIN_ALPHA_SHARE = 0.90
W_MAX_DELAY_COST_SHARE = 0.50
W_MIN_SHARE_WITHOUT_TOP3 = 0.50
V_TARGET_VOL = 0.20
V_LOOKBACK = 20
V_MIN_SCALE = 0.3
T_ASSETS = ("SPY", "QQQ", "TLT", "IEF", "GLD")
T_CASH = "BIL"
T_LOOKBACK = 252
MIN_WINNING_START_DAYS = 15


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Idea helpers (pure, tested) ────────────────────────────────────────────

def momentum_keep_set(day: pd.DataFrame, held: set[str], score_col: str,
                      threshold: float = W_KEEP_MOMENTUM_RANK) -> set[str]:
    """W: held stocks whose 12-1 momentum ranks in the top (1 - threshold).

    Ranked among the day's stocks that have a score, the same group the
    engine picks from.
    """
    ranked = day.dropna(subset=[score_col])
    if ranked.empty or "factor_mom_12_1" not in ranked.columns:
        return set()
    mom_rank = pd.to_numeric(ranked["factor_mom_12_1"], errors="coerce").rank(pct=True)
    tickers = ranked["ticker"].astype(str)
    return {t for t, r in zip(tickers, mom_rank) if t in held and pd.notna(r) and r >= threshold}


def vol_scale(closes: pd.Series, day, target: float = V_TARGET_VOL,
              lookback: int = V_LOOKBACK, floor: float = V_MIN_SCALE) -> float:
    """V: min(1, target / annualised volatility of the last `lookback` daily returns)."""
    past = closes[closes.index <= pd.Timestamp(day)].tail(lookback + 1)
    returns = past.pct_change().dropna()
    if len(returns) < lookback:
        return 1.0
    vol = float(returns.std(ddof=1) * np.sqrt(252.0))
    if vol <= 1e-9:
        return 1.0
    return float(min(1.0, max(floor, target / vol)))


def trend_weights(closes: dict[str, pd.Series], day, assets=T_ASSETS, cash=T_CASH,
                  lookback: int = T_LOOKBACK) -> dict[str, float]:
    """T: equal fifths in assets whose 1-year return beats cash; the rest in cash."""
    def past_return(series):
        past = series[series.index <= pd.Timestamp(day)]
        if len(past) <= lookback:
            return None
        return float(past.iloc[-1] / past.iloc[-1 - lookback] - 1.0)

    cash_ret = past_return(closes[cash])
    share = 1.0 / len(assets)
    weights = {cash: 0.0}
    for asset in assets:
        r = past_return(closes[asset])
        if r is not None and cash_ret is not None and r > cash_ret:
            weights[asset] = weights.get(asset, 0.0) + share
        else:
            weights[cash] += share
    return {k: round(v, 10) for k, v in weights.items() if v > 0}


def window_stats(equity: pd.Series, window=DECISION_WINDOW) -> dict:
    """Sharpe (per 20-day period, annualised) and max drawdown inside the window."""
    part = equity[(equity.index >= pd.Timestamp(window[0])) & (equity.index <= pd.Timestamp(window[1]))]
    if len(part) < 3:
        return {"sharpe": None, "max_dd_pct": None}
    rets = part.pct_change().dropna()
    sd = float(rets.std(ddof=1))
    sharpe = float(rets.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else None
    dd = float((part / part.cummax() - 1.0).min() * 100.0)
    return {"sharpe": sharpe, "max_dd_pct": dd}


# ── Judging (pure, tested) ─────────────────────────────────────────────────

def _median(values):
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _by_offset(rows, test, delay, key):
    return {r["offset"]: r.get(key) for r in rows if r["test"] == test and r["delay"] == delay}


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered decision gates (2013–2022, one-day-late runs)."""
    def late(test, key="decision_alpha_vs_qqq_pct"):
        return _by_offset(rows, test, 1, key)

    def delay_cost(test):
        on, lt = _by_offset(rows, test, 0, "decision_alpha_vs_qqq_pct"), late(test)
        costs = [on[o] - lt[o] for o in on if o in lt and on[o] is not None and lt[o] is not None]
        return float(np.mean(costs)) if costs else None

    s_med, w_med = _median(late("S").values()), _median(late("W").values())
    w3_med = _median(late("W_N3").values())
    s_cost, w_cost = delay_cost("S"), delay_cost("W")
    w1 = s_med is not None and w_med is not None and w_med >= W_MIN_ALPHA_SHARE * s_med
    w2 = s_cost is not None and w_cost is not None and (w_cost <= W_MAX_DELAY_COST_SHARE * s_cost)
    w3 = w_med is not None and w3_med is not None and w_med > 0 and w3_med >= W_MIN_SHARE_WITHOUT_TOP3 * w_med

    s_sh, s_dd = late("S", "sharpe"), late("S", "max_dd_pct")
    v_sh, v_dd = late("V", "sharpe"), late("V", "max_dd_pct")
    v_wins = sum(1 for o in v_sh if o in s_sh and None not in (v_sh[o], s_sh[o], v_dd.get(o), s_dd.get(o))
                 and v_sh[o] > s_sh[o] and v_dd[o] > s_dd[o])

    t_sh, e_sh = late("T_core", "sharpe"), late("E_core", "sharpe")
    t_wins = sum(1 for o in t_sh if o in e_sh and None not in (t_sh[o], e_sh[o]) and t_sh[o] > e_sh[o])
    t_dd = _median(late("T_core", "max_dd_pct").values())
    qqq_dd = _median(late("T_core", "qqq_max_dd_pct").values())
    t2 = t_dd is not None and qqq_dd is not None and t_dd > qqq_dd

    return {
        "numbers": {
            "S_median_late_alpha": s_med, "W_median_late_alpha": w_med, "W_N3_median_late_alpha": w3_med,
            "S_mean_delay_cost": s_cost, "W_mean_delay_cost": w_cost,
            "V_start_days_better_sharpe_and_dd": v_wins,
            "S_median_late_sharpe": _median(s_sh.values()), "V_median_late_sharpe": _median(v_sh.values()),
            "S_median_late_max_dd": _median(s_dd.values()), "V_median_late_max_dd": _median(v_dd.values()),
            "T_core_start_days_better_sharpe_than_E": t_wins,
            "T_core_median_sharpe": _median(t_sh.values()), "E_core_median_sharpe": _median(e_sh.values()),
            "T_core_median_max_dd": t_dd, "QQQ_median_max_dd": qqq_dd,
            "T_median_late_alpha": _median(late("T").values()), "V_median_late_alpha": _median(late("V").values()),
        },
        "gates": {
            "W1_alpha_kept": w1, "W2_delay_cost_halved": w2, "W3_not_few_names": w3,
            "V1_sharpe_and_dd": v_wins >= MIN_WINNING_START_DAYS,
            "T1_sharpe_vs_E": t_wins >= MIN_WINNING_START_DAYS, "T2_dd_vs_QQQ": t2,
        },
        "decision_pass": {
            "W": bool(w1 and w2 and w3),
            "V": v_wins >= MIN_WINNING_START_DAYS,
            "T": bool(t_wins >= MIN_WINNING_START_DAYS and t2),
        },
    }


# ── Swapping engine helpers while an idea runs ─────────────────────────────

@contextlib.contextmanager
def idea_patch(core, idea: str, closes: dict[str, pd.Series]):
    """Switch one idea on inside the engine, then switch it off again."""
    original_select = core._select_sticky_holdings
    original_alloc = core._resolve_allocation

    def select_w(day, held, *, score_col, **kwargs):
        keep = momentum_keep_set(day, set(held), score_col)
        kwargs["exit_rank_floor"] = 0.0          # keep every name in `keep`
        return original_select(day, keep, score_col=score_col, **kwargs)

    def alloc_v(dt, config, indicators):
        regime, weights, core_gross, overlay_gross = original_alloc(dt, config, indicators)
        return regime, weights, core_gross * vol_scale(closes["QQQ"], dt), overlay_gross

    def alloc_t(dt, config, indicators):
        regime, _weights, core_gross, overlay_gross = original_alloc(dt, config, indicators)
        return regime, trend_weights(closes, dt), core_gross, overlay_gross

    try:
        if idea == "W":
            core._select_sticky_holdings = select_w
        elif idea == "V":
            core._resolve_allocation = alloc_v
        elif idea == "T":
            core._resolve_allocation = alloc_t
        yield
    finally:
        core._select_sticky_holdings = original_select
        core._resolve_allocation = original_alloc


def t_config(config: dict) -> dict:
    """T needs price files for all its ETFs loaded by the engine."""
    return {**config, "core_weights": {t: 1.0 / 6 for t in (*T_ASSETS, T_CASH)}}


def zero_overlay(core, config: dict) -> dict:
    out = {**config, "overlay_gross": 0.0, "concentration_overlay_mode": "off"}
    if isinstance(config.get("regime_preset"), dict):
        out["regime_preset"] = core._regime_preset_with_overlay_gross(config["regime_preset"], 0.0)
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def _closes(ticker: str) -> pd.Series:
    from settings import DATA_DIR
    frame = pd.read_parquet(Path(DATA_DIR) / f"{ticker}.parquet", columns=["Close"])
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    return pd.to_numeric(frame["Close"], errors="coerce").dropna().sort_index()


def run_exam(core, idea: str, config: dict, closes: dict) -> dict:
    """The existing execution stress test, research-candidate mode, idea switched on."""
    import core_satellite_execution_stress as stress
    from settings import LOG_DIR

    path = OUT_DIR / f"candidate_{idea}.json"
    path.write_text(json.dumps(config, indent=1, default=str))
    name = f"newedge_{idea}"
    with idea_patch(core, idea, closes):
        stress.main(["--candidate-json", str(path), "--candidate-name", name])
    report = json.loads((Path(LOG_DIR) / f"research_candidate_execution_stress_{name}.json").read_text())
    scenarios = [r for r in report["rows"] if not str(r["scenario"]).startswith("delta")]
    failed = {r["scenario"]: r["failed_gates"] for r in scenarios if r.get("failed_gates")}
    return {"pass": not failed, "failed": failed,
            "holdout_alpha_vs_qqq": {r["scenario"]: r.get("holdout_alpha_vs_qqq_pct") for r in scenarios}}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered H-newedge-1 round (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS)
    parser.add_argument("--membership-csv", help="saved copy of the membership CSV (default: download it)")
    parser.add_argument("--no-exam", action="store_true", help="skip the final exam (smoke tests only)")
    parser.add_argument("--out", default=str(OUT_DIR / "newedge_round1.json"))
    args = parser.parse_args(argv)

    from alpha_factor_backtest import HORIZON_DAYS

    top = _load("top_names_check", OUT_DIR.parent / "top_names_20260929" / "top_names_check.py")
    edge, core, config, panel_s, sessions, notes = top.build_s_panel(args.membership_csv)
    end = sessions[-30]
    closes = {t: _closes(t) for t in (*T_ASSETS, T_CASH)}
    labels = top.label_lookup(panel_s, top.label_column(int(config.get("holding_days", HORIZON_DAYS)), HORIZON_DAYS))
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

    def run(test, idea, cfg, panel, delay, offset):
        t0 = time.time()
        with idea_patch(core, idea, closes):
            equity, trades, extra = core.run_core_satellite(
                panel, {**cfg, "entry_delay_days": delay}, evaluation_start=sessions[offset], evaluation_end=end)
        row = edge._summarise(core, test, equity, extra, delay=delay, offset=offset, t0=t0)
        row.update(window_stats(equity))
        bench = core.benchmark_equity(pd.DatetimeIndex(equity.index))
        row["qqq_max_dd_pct"] = window_stats(bench["QQQ"])["max_dd_pct"]
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("test", "delay", "offset", "decision_alpha_vs_qqq_pct",
                                                  "sharpe", "max_dd_pct", "secs")}), flush=True)
        save()
        return trades

    contributions = []
    for test, idea, cfg in (("S", None, config), ("W", "W", config), ("V", "V", config), ("T", "T", t_config(config))):
        for delay in DELAYS:
            for offset in range(args.offsets):
                trades = run(test, idea, cfg, panel_s, delay, offset)
                if test == "W" and delay == 0:
                    contributions.append(top.ticker_contributions(trades, labels))
    top3 = top.top_contributors(contributions, 3)
    notes["W_top3_tickers"] = top3
    panel_w3 = panel_s[~panel_s["ticker"].astype(str).str.upper().isin(top3)].reset_index(drop=True)
    for offset in range(args.offsets):
        run("W_N3", "W", config, panel_w3, 1, offset)
    for offset in range(args.offsets):
        run("T_core", "T", zero_overlay(core, t_config(config)), panel_s, 1, offset)
        run("E_core", None, zero_overlay(core, config), panel_s, 1, offset)

    result = judge(rows)
    result["exam"] = {}
    if not args.no_exam:
        for idea, cfg in (("W", config), ("V", config), ("T", t_config(config))):
            if result["decision_pass"][idea]:
                result["exam"][idea] = run_exam(core, idea, cfg, closes)
    result["verdicts"] = {
        idea: ("fails decision gates" if not result["decision_pass"][idea]
               else ("passes" if result["exam"].get(idea, {}).get("pass") else "fails the exam"))
        for idea in ("W", "V", "T")
    }
    save(result)
    print(json.dumps({k: result[k] for k in ("gates", "verdicts")}, indent=1))
    return result


if __name__ == "__main__":
    main()
