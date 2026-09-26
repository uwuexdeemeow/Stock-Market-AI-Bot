"""
signal_bakeoff.py — pre-registered "which new signal is best?" comparison.

PLAIN ENGLISH: the owner asked which of the signal ideas in
Documentation/SIGNAL_IDEAS_2026-09-26.md is best.  This script answers that
with ONE fixed test, written down before any run (see "Hypothesis
H-bakeoff" in Documentation/DELAY_STRESS_PAPER_ADVISORY.md):

  * Every idea keeps the incumbent's ETF core and market-regime switch.
    Only the stock "overlay" (the extra picks on top of SPY/QQQ) changes.
  * Every idea is run 40 times: the 20-day calendar starts on each of the
    20 possible first days, each with on-time fills and one-day-late fills.
    That is the same "timing luck" test that sank the incumbent.
  * The ideas are judged on 2013-2022 only.  2023-2026 was already looked at
    many times, so it is printed as a diagnostic and never decides anything.

Ideas that can win:
  A  slow momentum: 12-1 month momentum + sector-relative 12-1 momentum,
     top 10 stocks, equal weight.
  B  horizon-matched refresh: each year, re-pick features using only earlier
     data, keeping the ones that predict 20-day returns even when bought one
     day late.  Top 10 stocks, equal weight.
  C  sector ETF rotation: 11 sector ETFs ranked by 6- and 12-month trend,
     only ETFs above their 200-day average, top 3, equal weight.

Controls (reported for context, can never win):
  R0 the incumbent exactly as it is (top 3, sticky score weights).
  R1 the incumbent's score with idea A/B's book (top 10, equal weight), to
     show how much comes from the basket alone.
  E  no stock overlay at all: only the ETF core and regime switch.

Idea D (earnings drift) is NOT run: earnings dates are switched off in
settings.py (USE_EARNINGS_DATA = False), so the local data has no real
earnings history to test.

Run from the project root (needs local data/ and signals/).  Idea C needs
the sector ETF files first:
    python refresh_etf_data.py --symbols XLK XLY XLF XLV XLE XLI XLP XLU XLRE XLB XLC --refresh
    python research_evidence/signal_bakeoff_20260926/signal_bakeoff.py
Quick smoke test (2 calendar start days only, NOT a valid result):
    python research_evidence/signal_bakeoff_20260926/signal_bakeoff.py --offsets 2
Output: research_evidence/signal_bakeoff_20260926/signal_bakeoff.json
Research only: nothing is published, no official report is written, and the
output says approves_trading = false.
"""
from __future__ import annotations

import argparse
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
HYPOTHESIS = "H-bakeoff"
CANDIDATES = ("A", "B", "C")          # ideas allowed to win
CONTROLS = ("R0", "R1", "E")          # context only
N_OFFSETS = 20                        # one per possible calendar start day
DELAYS = (0, 1)                       # on-time fills and one-day-late fills
DECISION_WINDOW = ("2013-01-01", "2022-12-31")    # judged here
DIAGNOSTIC_WINDOW = ("2023-01-01", "2026-12-31")  # printed only
MAX_ON_TIME_SPREAD_PTS = 73.0         # same limit as H-tranche
MAX_MEAN_DELAY_COST_PTS = 5.0         # on-time minus late, averaged
SURVIVOR_FREE_SHARE = 0.5             # C wins if it reaches half the leader

# Idea A: slow momentum columns (both built in pipeline_shared.py).
MOMENTUM_COLS = ("factor_mom_12_1", "factor_resid_mom_sector_12_1")

# Idea B: the fixed pool of features it may choose from.  These are the 24
# features in signals/feature_research_summary.csv (the incumbent's research
# pool) plus the four slow features the incumbent's risk-on score uses.
B_FEATURE_POOL = (
    "factor_liquidity_dollar_vol_20d", "xs_rank_market_factor_liquidity_dollar_vol_20d",
    "xs_rank_market_factor_illiquidity_amihud_20d", "factor_illiquidity_amihud_20d",
    "dist_ma5", "ret_3d", "dist_ma10", "ret_5d", "xs_rank_market_ret_5d",
    "ret_vs_spy_5d", "ret_vs_qqq_5d", "xs_rank_sector_factor_illiquidity_amihud_20d",
    "xs_rank_sector_ret_5d", "ret_vs_sector_5d", "sector_rs_slope_5d", "ret_1d",
    "xs_rank_sector_factor_liquidity_dollar_vol_20d", "ret_vs_sector_1d", "macd_hist",
    "vwap_dist", "vol_trend_20_60", "uptick_ratio", "dist_ma20", "bb_pos",
    "factor_mom_12_1", "factor_resid_mom_sector_12_1", "factor_52w_high_proximity",
    "factor_beta_252_spy",
)
B_LABEL_LATE = "forward_return_delay1_20d"   # bought one day late, held 20 days
B_LABEL_ON_TIME = "forward_return_20d"
B_SAMPLE_EVERY = 21          # sessions between IC samples (labels don't overlap)
B_MIN_TRAIN_SAMPLES = 36     # about 3 years of samples before a year is scored
B_MIN_ABS_T = 2.0            # late IC must be clearly different from zero
B_MIN_LATE_SHARE = 0.5       # late IC must keep half of the on-time IC
B_MAX_FEATURES = 8

# Idea C: sector ETFs and trend rules.
SECTOR_ETFS = ("XLK", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC")
C_TREND_DAYS = 200
C_MOM_SHORT = 126            # about 6 months
C_MOM_LONG, C_MOM_SKIP = 252, 21   # 12-1 months

# The score column the engine reads when score_source = "factor_walkforward".
ENGINE_SCORE_COL = "factor_walkforward_score"


# ── Score builders (pure functions, tested with fake data) ─────────────────

def _pct_rank_by_date(panel: pd.DataFrame, col: str) -> pd.Series:
    """Rank a column within each date: 0 = worst stock that day, 1 = best."""
    values = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.groupby(panel["date"]).rank(pct=True)


def momentum_score(panel: pd.DataFrame) -> pd.Series:
    """Idea A: average daily rank of 12-1 momentum and sector-relative 12-1 momentum.

    The feature files fill missing early values with exactly 0.0.  Real
    momentum is never exactly zero, so 0.0 is treated as "not known yet" and
    that stock gets no score (it cannot be picked) on that date.
    """
    ranks = []
    for col in MOMENTUM_COLS:
        values = pd.to_numeric(panel[col], errors="coerce").mask(lambda s: s == 0.0)
        ranks.append(values.groupby(panel["date"]).rank(pct=True))
    both = pd.concat(ranks, axis=1)
    # A stock needs BOTH momentum numbers to get a score.
    return both.mean(axis=1).where(both.notna().all(axis=1))


def _ic_per_date(panel: pd.DataFrame, feature: str, label: str) -> pd.Series:
    """Spearman rank correlation between a feature and later returns, per date."""
    frame = panel[["date", feature, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    ranks = frame.groupby("date")[[feature, label]].rank(pct=True)
    ranks["date"] = frame["date"]
    return ranks.groupby("date").apply(
        lambda g: g[feature].corr(g[label]) if g[feature].nunique() > 1 and g[label].nunique() > 1 else np.nan
    ).dropna()


def select_b_features(train: pd.DataFrame, pool: tuple[str, ...] = B_FEATURE_POOL) -> list[dict]:
    """Idea B: choose features from `train` only (rows already cut before the year).

    A feature is kept when its one-day-late 20-day IC has |t| >= 2, points the
    same way as its on-time IC, and keeps at least half of it.  At most
    B_MAX_FEATURES are kept, strongest first.
    """
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    sample_dates = dates[::B_SAMPLE_EVERY]
    sample = train[train["date"].isin(sample_dates)]
    chosen = []
    for feature in pool:
        if feature not in sample.columns:
            continue
        late = _ic_per_date(sample, feature, B_LABEL_LATE)
        on_time = _ic_per_date(sample, feature, B_LABEL_ON_TIME)
        if len(late) < B_MIN_TRAIN_SAMPLES or late.std(ddof=1) <= 0:
            continue
        late_ic, on_time_ic = float(late.mean()), float(on_time.mean()) if len(on_time) else 0.0
        t_stat = late_ic / float(late.std(ddof=1)) * np.sqrt(len(late))
        keeps_half = abs(late_ic) >= B_MIN_LATE_SHARE * abs(on_time_ic)
        same_sign = np.sign(late_ic) == np.sign(on_time_ic)
        if abs(t_stat) >= B_MIN_ABS_T and same_sign and keeps_half:
            chosen.append({"feature": feature, "late_ic": round(late_ic, 5),
                           "on_time_ic": round(on_time_ic, 5), "t_stat": round(float(t_stat), 2)})
    chosen.sort(key=lambda row: abs(row["t_stat"]), reverse=True)
    return chosen[:B_MAX_FEATURES]


def horizon_matched_score(panel: pd.DataFrame, years: list[int]) -> tuple[pd.Series, dict]:
    """Idea B: build each year's score from features chosen on earlier data only.

    Leakage guard: the training rows for year Y must have their LATE 20-day
    label finished before 1 January of Y.  So no return that happens in Y (or
    later) can influence which features Y uses.
    """
    score = pd.Series(np.nan, index=panel.index, dtype=float)
    chosen_by_year: dict[str, list[dict]] = {}
    end_col = f"{B_LABEL_LATE}_end_date"
    label_end = pd.to_datetime(panel[end_col], errors="coerce")
    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        train = panel[(panel["date"] < year_start) & (label_end < year_start)]
        chosen = select_b_features(train)
        chosen_by_year[str(year)] = chosen
        in_year = panel["date"].dt.year == year
        if not chosen or not in_year.any():
            continue
        day_rows = panel.loc[in_year]
        parts = []
        for row in chosen:
            rank = _pct_rank_by_date(day_rows, row["feature"])
            # Flip features whose high values predicted LOWER returns.
            parts.append(rank if row["late_ic"] > 0 else 1.0 - rank)
        score.loc[in_year] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return score, chosen_by_year


def etf_rotation_panel(etf_frames: dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    """Idea C: turn sector ETF prices into rows the engine can pick from.

    Each ETF acts like a "stock" in its own sector.  Labels (future returns)
    are built exactly like alpha_factor_backtest.load_factor_panel: buy at
    the next day's Open, sell at the Close 20 sessions later (or one day
    later for the late-fill version).  The score is the average rank of
    6-month and 12-1 month returns, and is left empty (cannot be picked)
    when the ETF is below its 200-day average or lacks enough history.
    """
    frames = []
    for ticker, raw in etf_frames.items():
        df = raw.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        if getattr(df.index, "tz", None) is not None:
            # Some downloads carry a time zone; the stock rows don't.
            df.index = df.index.tz_localize(None)
        df = df.loc[df.index.notna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        close = pd.to_numeric(df["Close"], errors="coerce")
        opens = pd.to_numeric(df["Open"], errors="coerce")
        sub = pd.DataFrame({"date": df.index, "ticker": ticker, "sector": ticker,
                            "Open": opens.to_numpy(), "Close": close.to_numpy()})
        sub["_mom_short"] = (close / close.shift(C_MOM_SHORT) - 1.0).to_numpy()
        sub["_mom_long"] = (close.shift(C_MOM_SKIP) / close.shift(C_MOM_LONG) - 1.0).to_numpy()
        sub["_above_trend"] = (close > close.rolling(C_TREND_DAYS, min_periods=C_TREND_DAYS).mean()).to_numpy()
        observed = pd.Series(df.index, index=df.index)
        for delay in (0, 1):
            label = "forward_return_20d" if delay == 0 else "forward_return_delay1_20d"
            entry = opens.shift(-(1 + delay))
            exit_ = close.shift(-(20 + delay))
            sub[label] = (exit_ / entry - 1.0).to_numpy()
            sub[f"{label}_entry_date"] = observed.shift(-(1 + delay)).to_numpy()
            sub[f"{label}_end_date"] = observed.shift(-(20 + delay)).to_numpy()
            sub[f"{label}_entry_price"] = entry.to_numpy()
            sub[f"{label}_exit_price"] = exit_.to_numpy()
        frames.append(sub)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
    ranks = pd.concat([_pct_rank_by_date(panel, "_mom_short"), _pct_rank_by_date(panel, "_mom_long")], axis=1)
    eligible = panel["_above_trend"].astype(bool) & ranks.notna().all(axis=1)
    panel[ENGINE_SCORE_COL] = ranks.mean(axis=1).where(eligible)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Configs (incumbent + only the pre-registered changes) ──────────────────

def idea_config(idea: str, incumbent: dict, overlay_zero_preset=None) -> dict:
    """Return the engine config for one idea.  Core and regime stay the incumbent's."""
    cfg = dict(incumbent)
    if idea == "R0":
        return cfg
    if idea == "R1":
        cfg.update(shape="top10", weighting="equal")
        return cfg
    if idea == "E":
        cfg.update(overlay_gross=0.0, concentration_overlay_mode="off")
        if overlay_zero_preset is not None:
            cfg["regime_preset"] = overlay_zero_preset
        return cfg
    if idea in ("A", "B"):
        cfg.update(score_source="factor_walkforward", shape="top10", weighting="equal", score_blend=False)
        return cfg
    if idea == "C":
        cfg.update(score_source="factor_walkforward", shape="top3", weighting="equal",
                   score_blend=False, max_per_sector=1, earnings_blackout_days=0)
        return cfg
    raise ValueError(f"unknown idea {idea}")


# ── Judging (pure function, tested with fake rows) ─────────────────────────

def _median(values):
    return float(statistics.median(values)) if values else float("nan")


def summarise_idea(rows: list[dict]) -> dict:
    """Numbers for one idea from its 40 runs (decision window unless named)."""
    # A run whose window had no data counts as missing (and fails gate G0).
    usable = [r for r in rows if r.get("decision_alpha_vs_qqq_pct") is not None]
    on_time = {r["offset"]: r["decision_alpha_vs_qqq_pct"] for r in usable if r["delay"] == 0}
    late = {r["offset"]: r["decision_alpha_vs_qqq_pct"] for r in usable if r["delay"] == 1}
    paired = sorted(set(on_time) & set(late))
    all_values = list(on_time.values()) + list(late.values())
    diag = [r["diagnostic_alpha_vs_qqq_pct"] for r in rows if r.get("diagnostic_alpha_vs_qqq_pct") is not None]
    return {
        "runs": len(usable),
        "min_alpha_pts": round(min(all_values), 2) if all_values else None,
        "median_on_time_alpha_pts": round(_median(list(on_time.values())), 2),
        "median_late_alpha_pts": round(_median(list(late.values())), 2),
        "on_time_spread_pts": round(max(on_time.values()) - min(on_time.values()), 2) if on_time else None,
        "mean_delay_cost_pts": round(float(np.mean([on_time[k] - late[k] for k in paired])), 2) if paired else None,
        "diagnostic_median_alpha_pts": round(_median(diag), 2) if diag else None,
        "diagnostic_negative_runs": int(sum(v < 0 for v in diag)),
        "median_max_drawdown_pct": round(_median([r["decision_max_drawdown_pct"] for r in rows]), 2),
        "median_turnover_pct": round(_median([r["turnover_pct"] for r in rows]), 1),
    }


def judge(rows: list[dict], expected_runs: int = N_OFFSETS * len(DELAYS)) -> dict:
    """Apply the pre-registered rule and name the best idea (or none)."""
    by_idea = {idea: [r for r in rows if r["idea"] == idea] for idea in CANDIDATES + CONTROLS}
    summary = {idea: summarise_idea(idea_rows) for idea, idea_rows in by_idea.items() if idea_rows}
    control_e = summary.get("E", {}).get("median_late_alpha_pts")
    eligible = []
    for idea in CANDIDATES:
        stats = summary.get(idea)
        if not stats:
            continue
        gates = {
            "G0_all_runs_present": stats["runs"] == expected_runs,
            "G1_every_run_beats_qqq": stats["min_alpha_pts"] is not None and stats["min_alpha_pts"] > 0,
            "G2_start_day_spread_under_limit": stats["on_time_spread_pts"] is not None
                and stats["on_time_spread_pts"] < MAX_ON_TIME_SPREAD_PTS,
            "G3_delay_cost_under_limit": stats["mean_delay_cost_pts"] is not None
                and stats["mean_delay_cost_pts"] <= MAX_MEAN_DELAY_COST_PTS,
            "G4_beats_no_overlay_control": control_e is not None
                and stats["median_late_alpha_pts"] > control_e,
        }
        stats["gates"] = gates
        stats["eligible"] = all(gates.values())
        if stats["eligible"]:
            eligible.append(idea)
    winner, reason = None, "no candidate passed every gate; the incumbent stays on the paper advisory"
    if eligible:
        leader = max(eligible, key=lambda idea: summary[idea]["median_late_alpha_pts"])
        winner, reason = leader, "eligible candidate with the highest median one-day-late alpha"
        best_stock = max((i for i in eligible if i != "C"), default=None,
                         key=lambda idea: summary[idea]["median_late_alpha_pts"])
        if (leader != "C" and "C" in eligible and best_stock is not None
                and summary["C"]["median_late_alpha_pts"]
                >= SURVIVOR_FREE_SHARE * summary[best_stock]["median_late_alpha_pts"]):
            winner = "C"
            reason = ("C is eligible and reaches at least half of the best stock idea; "
                      "its history has no survivorship bias, so it is preferred")
    return {"summary": summary, "eligible": eligible, "winner": winner, "reason": reason}


# ── Running the engine ─────────────────────────────────────────────────────

def _max_drawdown_pct(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min() * 100.0) if len(equity) else 0.0


def _window_alpha(core, equity: pd.Series, window: tuple[str, str]) -> float | None:
    bench = core.benchmark_equity(pd.DatetimeIndex(equity.index))
    result = core._holdout_comparisons(equity, bench, start=window[0], end=window[1])
    return result.get("alpha_vs_qqq_pct")


def run_one(core, panel: pd.DataFrame, cfg: dict, sessions, offset: int, delay: int, end) -> dict:
    """One engine run: one calendar start day, on time or one day late."""
    t0 = time.time()
    run_cfg = {**cfg, "entry_delay_days": delay}
    equity, _trades, extra = core.run_core_satellite(
        panel, run_cfg, evaluation_start=sessions[offset], evaluation_end=end)
    in_window = equity[(equity.index >= DECISION_WINDOW[0]) & (equity.index <= DECISION_WINDOW[1])]
    return {
        "delay": delay, "offset": offset, "start": str(pd.Timestamp(sessions[offset]).date()),
        "decision_alpha_vs_qqq_pct": _window_alpha(core, equity, DECISION_WINDOW),
        "diagnostic_alpha_vs_qqq_pct": _window_alpha(core, equity, DIAGNOSTIC_WINDOW),
        "decision_max_drawdown_pct": round(_max_drawdown_pct(in_window), 2),
        "turnover_pct": extra.get("turnover_pct"),
        "avg_overlay_positions": extra.get("avg_overlay_positions"),
        "top_ticker_overlay_contributor": extra.get("top_ticker_overlay_contributor"),
        "secs": round(time.time() - t0, 1),
    }


def build_panels(ideas: list[str]) -> tuple[dict, dict, dict]:
    """Load local data once and build the panel each idea trades from."""
    from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
    import core_satellite_alpha as core
    from settings import DATA_DIR

    notes: dict = {}
    # write_health_outputs=False: this research run must not rewrite the
    # feature-health files that the daily workflow reads.
    specs = load_feature_specs(write_health_outputs=False)
    incumbent_panel = core._ensure_robust_score_columns(attach_scores(
        load_factor_panel(specs, require_forward_returns=False), specs, load_prediction_scores()))
    date_min, date_max = incumbent_panel["date"].min(), incumbent_panel["date"].max()
    panels = {idea: incumbent_panel for idea in ("R0", "R1", "E") if idea in ideas}

    if "A" in ideas or "B" in ideas:
        # Plain stock rows (labels, sector, earnings columns) plus the extra
        # feature columns, WITHOUT the incumbent's scores or feature-health flag.
        extra = sorted(set(MOMENTUM_COLS) | set(B_FEATURE_POOL))
        stock_panel = load_factor_panel([{"feature": c} for c in extra], require_forward_returns=False)
        notes["missing_feature_columns"] = [c for c in extra if c not in stock_panel.columns]
        if "A" in ideas:
            missing = [c for c in MOMENTUM_COLS if c not in stock_panel.columns]
            if missing:
                raise SystemExit(f"Idea A needs {missing} in data/*.parquet")
            panel_a = stock_panel.copy()
            panel_a[ENGINE_SCORE_COL] = momentum_score(panel_a)
            panels["A"] = panel_a
        if "B" in ideas:
            panel_b = stock_panel.copy()
            years = sorted(int(y) for y in panel_b["date"].dt.year.unique())
            panel_b[ENGINE_SCORE_COL], notes["b_features_by_year"] = horizon_matched_score(panel_b, years)
            panels["B"] = panel_b

    if "C" in ideas:
        etf_frames = {}
        for ticker in SECTOR_ETFS:
            path = Path(DATA_DIR) / f"{ticker}.parquet"
            if path.exists():
                etf_frames[ticker] = pd.read_parquet(path)
        notes["sector_etfs_found"] = sorted(etf_frames)
        missing = sorted(set(SECTOR_ETFS) - set(etf_frames))
        if missing:
            raise SystemExit(f"Idea C needs sector ETF files {missing}; run refresh_etf_data.py first (see top of file)")
        panels["C"] = etf_rotation_panel(etf_frames, date_min, date_max)
    return panels, notes, {"date_min": date_min, "date_max": date_max}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Pre-registered signal bake-off (research only).")
    parser.add_argument("--offsets", type=int, default=N_OFFSETS, help="calendar start days to run (20 = full test)")
    parser.add_argument("--ideas", nargs="*", default=list(CONTROLS + CANDIDATES))
    parser.add_argument("--out", default=str(OUT_DIR / "signal_bakeoff.json"))
    args = parser.parse_args(argv)

    import core_satellite_alpha as core
    from validation_bundle import load_approved_research_config

    incumbent = load_approved_research_config(Path("signals") / "core_satellite_alpha_metrics.json")
    zero_preset = None
    if isinstance(incumbent.get("regime_preset"), dict):
        zero_preset = core._regime_preset_with_overlay_gross(incumbent["regime_preset"], 0.0)

    panels, notes, span = build_panels(args.ideas)
    sessions = core._nyse_sessions(span["date_min"], span["date_max"])
    end = sessions[-30]   # same end for every run, so the last labels are complete
    out_path = Path(args.out)
    rows: list[dict] = []

    def save(result: dict | None = None) -> None:
        payload = {
            "hypothesis": HYPOTHESIS, "research_only": True, "approves_trading": False,
            "valid_full_test": args.offsets == N_OFFSETS and set(args.ideas) >= set(CANDIDATES + CONTROLS),
            "end_date": str(pd.Timestamp(end).date()), "offsets": args.offsets,
            "notes": notes, "rows": rows,
        }
        if result is not None:
            payload["result"] = result
        out_path.write_text(json.dumps(payload, indent=1, default=str))

    for idea in args.ideas:
        cfg = idea_config(idea, incumbent, zero_preset)
        for delay in DELAYS:
            for offset in range(args.offsets):
                row = {"idea": idea, **run_one(core, panels[idea], cfg, sessions, offset, delay, end)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                save()   # a crash keeps the finished rows
    result = judge(rows, expected_runs=args.offsets * len(DELAYS))
    save(result)
    print(json.dumps({k: result[k] for k in ("eligible", "winner", "reason")}, indent=1))
    return result


if __name__ == "__main__":
    main()
