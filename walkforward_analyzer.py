"""
Walkforward Analyzer
====================

Plain English: After every nested walkforward run, this script reads the
results CSV and tells you whether the strategy is actually working — or
whether the inner validation is lying to you.

It catches three failure modes that simple "did it beat QQQ?" checks miss:

1. **Anti-predictive scoring**: If the inner score is negatively correlated
   with OOS Sharpe, the validation methodology is broken (configs that look
   best in training do worst in real out-of-sample tests).

2. **Calibration disaster**: If the inner folds predict the strategy beats
   QQQ 100% of the time but OOS reality is 30%, the model is overconfident.

3. **Concentration vulnerability**: If OOS losses cluster in years where
   mega-caps dominated (QQQ >> SPY), the strategy structurally can't
   compete with concentrated market momentum.

Usage:
    python3 walkforward_analyzer.py                            # uses default file
    python3 walkforward_analyzer.py --csv path/to/results.csv  # custom file

Output: a printed report with PASS/FAIL flags for each check, plus
recommendations on what to fix next.
"""

# ── Imports — pandas for data, argparse for CLI, os for file checks ────
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from robustness_scoring import DEFAULT_OBJECTIVE, robustness_score_components
from safe_io import configure_console_output

configure_console_output()

# Default paths — where the walkforward script saves its outputs.
DEFAULT_CSV = "signals/core_satellite_nested_walkforward.csv"
DEFAULT_QQQ_PARQUET = "data/QQQ.parquet"
DEFAULT_SPY_PARQUET = "data/SPY.parquet"


def normalize_walkforward_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy common research-output aliases into the canonical analyzer names."""
    out = df.copy()
    aliases = {
        "fold_year": "outer_year",
        "oos_return_pct": "oos_total_return_pct",
    }
    for target, source in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def _missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """List required columns that are absent before pandas dropna is called."""
    return [column for column in columns if column not in df.columns]


# ── Helper: compute yearly QQQ vs SPY return gap as concentration proxy ──
# When QQQ heavily beats SPY in a year, mega-cap tech dominated the market.
# Our diversified picks struggle in those years, so we want to know which
# OOS years were "high concentration" vs "broad market."
def yearly_market_concentration(
    qqq_path: str = DEFAULT_QQQ_PARQUET,
    spy_path: str = DEFAULT_SPY_PARQUET,
) -> pd.DataFrame:
    """Return a DataFrame indexed by year with QQQ return, SPY return,
    and the concentration proxy (QQQ_return - SPY_return)."""
    # Load close prices for both ETFs.  These are the standard cap-weighted
    # benchmarks — QQQ = Nasdaq 100 (tech-heavy), SPY = S&P 500 (broader).
    qqq = pd.read_parquet(qqq_path)[["Close"]].rename(columns={"Close": "qqq"})
    spy = pd.read_parquet(spy_path)[["Close"]].rename(columns={"Close": "spy"})

    # Align by date index and add a year column for grouping.
    df = qqq.join(spy, how="inner")
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year

    # First/last price per year → percent return over the full year.
    grouped = df.groupby("year").agg(
        qqq_start=("qqq", "first"),
        qqq_end=("qqq", "last"),
        spy_start=("spy", "first"),
        spy_end=("spy", "last"),
    )
    grouped["qqq_return"] = (grouped["qqq_end"] / grouped["qqq_start"] - 1) * 100
    grouped["spy_return"] = (grouped["spy_end"] / grouped["spy_start"] - 1) * 100
    grouped["concentration_proxy"] = grouped["qqq_return"] - grouped["spy_return"]
    return grouped[["qqq_return", "spy_return", "concentration_proxy"]]


# ── Check 1: is the inner score predictive of OOS performance? ──────────
# A working validation should produce HIGH inner score -> HIGH matching OOS
# objective score.  The analyzer still prints OOS Sharpe and QQQ-alpha
# correlations, but the PASS/WARN/FAIL verdict follows the active objective.
def _oos_objective_target(row: pd.Series, objective: str) -> float:
    """Compute the OOS score using the same objective as inner selection."""
    mapped = {
        "sharpe": row.get("oos_sharpe"),
        "alpha_vs_qqq_pct": row.get("oos_alpha_vs_qqq_pct"),
        "max_drawdown_pct": row.get("oos_max_drawdown_pct"),
        "turnover_pct": row.get("oos_turnover_pct"),
    }
    return float(robustness_score_components(mapped, objective=objective)["robustness_score"])


def check_score_predictiveness(df: pd.DataFrame, *, objective: str = DEFAULT_OBJECTIVE) -> dict:
    """Compute whether inner score predicts OOS quality.

    The analyzer still reports Sharpe and QQQ-alpha correlations, but the
    verdict uses the OOS score built from the active objective.  This avoids
    declaring victory when a QQQ-alpha selector only improves Sharpe by chance.

    Returns a dict with the correlation values and a pass/fail verdict.
    """
    df = normalize_walkforward_columns(df)
    required = [
        "inner_score",
        "inner_mean_score",
        "oos_sharpe",
        "oos_alpha_vs_qqq_pct",
        "oos_max_drawdown_pct",
        "oos_turnover_pct",
    ]
    missing = _missing_columns(df, required)
    if missing:
        return {"valid": False, "reason": f"missing columns: {', '.join(missing)}"}
    valid = df.dropna(subset=required).copy()
    if len(valid) < 4:
        return {"valid": False, "reason": "not enough valid folds"}

    valid["oos_objective_score"] = valid.apply(_oos_objective_target, axis=1, objective=objective)
    corr_inner = valid["inner_score"].corr(valid["oos_sharpe"])
    corr_mean = valid["inner_mean_score"].corr(valid["oos_sharpe"])
    corr_alpha = valid["inner_score"].corr(valid["oos_alpha_vs_qqq_pct"])
    corr_objective = valid["inner_score"].corr(valid["oos_objective_score"])
    corr_mean_objective = valid["inner_mean_score"].corr(valid["oos_objective_score"])
    calibration_direction = None
    oos_positive_rate = None
    if "inner_mean_alpha_vs_qqq_pct" in valid.columns:
        calibration_direction = (
            (valid["inner_mean_alpha_vs_qqq_pct"] > 0)
            == (valid["oos_alpha_vs_qqq_pct"] > 0)
        ).mean() * 100
        oos_positive_rate = (valid["oos_alpha_vs_qqq_pct"] > 0).mean() * 100

    # Verdict thresholds:
    #   > 0.3  = working (PASS)
    #   0 to 0.3 = weak signal (WARN)
    #   < 0     = anti-predictive — broken (FAIL)
    if corr_objective > 0.3:
        verdict = "PASS"
        interpretation = _interpret_score_corr(corr_objective)
    elif corr_objective >= 0:
        verdict = "WARN"
        interpretation = _interpret_score_corr(corr_objective)
    elif (
        calibration_direction is not None
        and calibration_direction >= 60
        and oos_positive_rate is not None
        and oos_positive_rate >= 60
    ):
        verdict = "WARN"
        interpretation = (
            "Cross-year inner score correlation is negative, but sign calibration "
            "is healthy; treat this as noisy and confirm with selector replay."
        )
    else:
        verdict = "FAIL"
        interpretation = _interpret_score_corr(corr_objective)

    return {
        "valid": True,
        "objective": objective,
        "corr_inner_score_vs_oos_objective": round(corr_objective, 3),
        "corr_inner_mean_vs_oos_objective": round(corr_mean_objective, 3),
        "corr_inner_score_vs_oos_sharpe": round(corr_inner, 3),
        "corr_inner_mean_vs_oos_sharpe": round(corr_mean, 3),
        "corr_inner_score_vs_oos_alpha_vs_qqq": round(corr_alpha, 3),
        "calibration_direction_accuracy_pct": round(calibration_direction, 1)
        if calibration_direction is not None
        else None,
        "oos_positive_alpha_rate_pct": round(oos_positive_rate, 1)
        if oos_positive_rate is not None
        else None,
        "verdict": verdict,
        "interpretation": interpretation,
    }


def _interpret_score_corr(corr: float) -> str:
    """Plain-English explanation of what a correlation value means."""
    if corr > 0.5:
        return "Inner scoring works — high-scoring configs do well OOS."
    if corr > 0.3:
        return "Inner scoring has some signal but is noisy."
    if corr > 0:
        return "Inner scoring is barely better than random — close to noise."
    return "Inner scoring is anti-predictive — high-scoring configs do WORSE OOS. Validation methodology is broken."


# ── Check 2: is the model calibrated? ───────────────────────────────────
# If inner folds predict beating QQQ 100% of the time but OOS reality is
# 30%, the model is overconfident.  Track:
#   - inner_pos_rate: % of folds where inner predicted positive alpha
#   - oos_pos_rate: % of folds where OOS actually delivered positive alpha
#   - direction_accuracy: how often the SIGN matched
def check_calibration(df: pd.DataFrame) -> dict:
    """Compare inner predictions vs OOS reality for alpha vs QQQ."""
    df = normalize_walkforward_columns(df)
    required = ["inner_mean_alpha_vs_qqq_pct", "oos_alpha_vs_qqq_pct"]
    missing = _missing_columns(df, required)
    if missing:
        return {"valid": False, "reason": f"missing columns: {', '.join(missing)}"}
    valid = df.dropna(subset=required)
    if len(valid) < 4:
        return {"valid": False, "reason": "not enough valid folds"}

    inner_pos = (valid["inner_mean_alpha_vs_qqq_pct"] > 0).mean() * 100
    oos_pos = (valid["oos_alpha_vs_qqq_pct"] > 0).mean() * 100

    # Direction accuracy: did the sign of inner alpha match OOS alpha?
    sign_match = (
        (valid["inner_mean_alpha_vs_qqq_pct"] > 0)
        == (valid["oos_alpha_vs_qqq_pct"] > 0)
    ).mean() * 100

    # Overconfidence gap: how much more optimistic is inner vs OOS?
    gap = inner_pos - oos_pos

    # Verdict: a healthy model has direction accuracy >= 60% and
    # overconfidence gap < 30 percentage points.
    if sign_match >= 60 and gap < 30:
        verdict = "PASS"
    elif sign_match >= 40 and gap < 50:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "valid": True,
        "inner_predicts_beat_qqq_pct": round(inner_pos, 1),
        "oos_actually_beat_qqq_pct": round(oos_pos, 1),
        "overconfidence_gap_pct": round(gap, 1),
        "direction_accuracy_pct": round(sign_match, 1),
        "verdict": verdict,
    }


# ── Check 3: is the strategy concentration-vulnerable? ──────────────────
# Compare OOS alpha vs QQQ against market concentration each year.
# A negative correlation means the strategy underperforms when QQQ runs.
def check_concentration_vulnerability(
    df: pd.DataFrame,
    concentration: pd.DataFrame,
) -> dict:
    """Check whether OOS losses cluster in high-concentration years."""
    df = normalize_walkforward_columns(df)
    missing = _missing_columns(df, ["fold_year", "oos_alpha_vs_qqq_pct"])
    if missing:
        return {"valid": False, "reason": f"missing columns: {', '.join(missing)}"}
    valid = df.dropna(subset=["oos_alpha_vs_qqq_pct"]).set_index("fold_year")
    merged = valid[["oos_alpha_vs_qqq_pct"]].join(concentration, how="inner")
    if len(merged) < 4:
        return {"valid": False, "reason": "not enough valid folds for merge"}

    corr = merged["concentration_proxy"].corr(merged["oos_alpha_vs_qqq_pct"])

    # Bucket high vs low concentration years and compare mean OOS alpha.
    # Threshold: 5% gap between QQQ and SPY = "high concentration."
    high = merged[merged["concentration_proxy"] > 5]
    low = merged[merged["concentration_proxy"] <= 5]
    high_mean = float(high["oos_alpha_vs_qqq_pct"].mean()) if len(high) else float("nan")
    low_mean = float(low["oos_alpha_vs_qqq_pct"].mean()) if len(low) else float("nan")
    delta = high_mean - low_mean if pd.notna(high_mean) and pd.notna(low_mean) else float("nan")

    # Verdict thresholds:
    #   corr > -0.3 and delta > -5%: strategy is concentration-robust
    #   corr <= -0.5 or delta < -10%: structurally vulnerable
    if corr > -0.3 and (pd.isna(delta) or delta > -5):
        verdict = "PASS"
    elif corr > -0.5 and (pd.isna(delta) or delta > -10):
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "valid": True,
        "correlation_conc_vs_oos_alpha": round(corr, 3),
        "high_concentration_years": [int(y) for y in high.index],
        "high_conc_mean_oos_alpha_pct": round(high_mean, 2) if pd.notna(high_mean) else None,
        "low_concentration_years": [int(y) for y in low.index],
        "low_conc_mean_oos_alpha_pct": round(low_mean, 2) if pd.notna(low_mean) else None,
        "delta_pct": round(delta, 2) if pd.notna(delta) else None,
        "verdict": verdict,
    }


# ── Check 4: config stability ───────────────────────────────────────────
# Are we picking the same config most of the time, or hopping randomly?
# Random hopping = pure noise in the selector.
def check_config_stability(df: pd.DataFrame) -> dict:
    """Count unique configs and frequency of the most common one."""
    df = normalize_walkforward_columns(df)
    stability_column = "stable_family_signature" if "stable_family_signature" in df.columns else "selected_config"
    missing = _missing_columns(df, [stability_column])
    if missing:
        return {"valid": False, "reason": f"missing columns: {', '.join(missing)}"}
    valid = df.dropna(subset=[stability_column])
    if len(valid) < 3:
        return {"valid": False, "reason": "not enough valid folds"}

    counts = valid[stability_column].value_counts()
    n_unique = len(counts)
    n_folds = len(valid)
    top_config = counts.index[0]
    top_freq = counts.iloc[0] / n_folds

    # Verdict: top config should be picked >= 30% of folds.
    # 100% unique (n_unique == n_folds) = pure noise.
    if top_freq >= 0.3 and n_unique / n_folds < 0.7:
        verdict = "PASS"
    elif top_freq >= 0.2:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "valid": True,
        "unique_configs": int(n_unique),
        "total_folds": int(n_folds),
        "uniqueness_ratio": round(n_unique / n_folds, 3),
        "top_config": str(top_config),
        "top_config_frequency": round(top_freq, 3),
        "stability_column": stability_column,
        "verdict": verdict,
    }


# ── Summary statistics: the simple numbers ──────────────────────────────
def summary_stats(df: pd.DataFrame) -> dict:
    """Compute compound return, CAGR, mean Sharpe, etc."""
    df = normalize_walkforward_columns(df)
    required = [
        "oos_return_pct",
        "oos_sharpe",
        "oos_alpha_vs_qqq_pct",
        "oos_alpha_vs_spy_pct",
        "oos_max_drawdown_pct",
    ]
    missing = _missing_columns(df, required)
    if missing:
        return {"missing_columns": missing}
    valid = df.dropna(subset=["oos_return_pct"])
    if len(valid) < 1:
        return {}

    compound = (1 + valid["oos_return_pct"] / 100).prod()
    n_years = len(valid)
    cagr = compound ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "valid_folds": int(n_years),
        "failed_folds": int(df["oos_return_pct"].isna().sum()),
        "compound_return_pct": round((compound - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "mean_oos_return_pct": round(valid["oos_return_pct"].mean(), 2),
        "mean_oos_sharpe": round(valid["oos_sharpe"].mean(), 3),
        "mean_alpha_vs_qqq_pct": round(valid["oos_alpha_vs_qqq_pct"].mean(), 2),
        "mean_alpha_vs_spy_pct": round(valid["oos_alpha_vs_spy_pct"].mean(), 2),
        "worst_drawdown_pct": round(valid["oos_max_drawdown_pct"].min(), 2),
        "beat_qqq_count": int((valid["oos_alpha_vs_qqq_pct"] > 0).sum()),
    }


# ── Main runner: orchestrate all checks and print a report ──────────────
def check_fold_completeness(df: pd.DataFrame) -> dict:
    """Fail a walkforward result when any requested fold has no OOS result."""
    df = normalize_walkforward_columns(df)
    if "oos_return_pct" not in df:
        return {"valid": False, "reason": "missing_oos_return_pct"}

    failed = df[df["oos_return_pct"].isna()].copy()
    failed_years = (
        pd.to_numeric(failed.get("fold_year"), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    reason_counts = {}
    if "reason" in failed:
        reason_counts = {
            str(reason): int(count)
            for reason, count in failed["reason"].fillna("missing_reason").value_counts().items()
        }
    return {
        "valid": True,
        "failed_folds": int(len(failed)),
        "failed_years": failed_years,
        "reason_counts": reason_counts,
        "verdict": "FAIL" if len(failed) else "PASS",
    }


def analyze(csv_path: str, qqq_path: str, spy_path: str, *, objective: str = DEFAULT_OBJECTIVE) -> dict:
    """Run all checks and return a results dict.  Also prints a report."""
    if not os.path.exists(csv_path):
        print(f"✗ File not found: {csv_path}")
        sys.exit(1)

    df = normalize_walkforward_columns(pd.read_csv(csv_path))
    print(f"\n{'='*70}")
    print(f" WALKFORWARD ANALYZER")
    print(f" Source: {csv_path}")
    print(f" Objective: {objective}")
    print(f" Folds: {len(df)} ({int(df['fold_year'].min())}-{int(df['fold_year'].max())})")
    print(f"{'='*70}\n")

    # Section 1: summary stats
    summary = summary_stats(df)
    print("── SUMMARY STATS ──")
    print(f"  Valid folds:       {summary.get('valid_folds')}/{summary.get('valid_folds', 0) + summary.get('failed_folds', 0)}")
    print(f"  Compound return:   {summary.get('compound_return_pct')}%")
    print(f"  CAGR:              {summary.get('cagr_pct')}%")
    print(f"  Mean OOS Sharpe:   {summary.get('mean_oos_sharpe')}")
    print(f"  Mean alpha vs QQQ: {summary.get('mean_alpha_vs_qqq_pct')}%")
    print(f"  Beat QQQ:          {summary.get('beat_qqq_count')}/{summary.get('valid_folds')} years")
    print(f"  Worst drawdown:    {summary.get('worst_drawdown_pct')}%")

    # Missing OOS folds can make the remaining averages look healthier than
    # the full test, so show that failure before interpreting score checks.
    print("\n-- CHECK 0: FOLD COMPLETENESS --")
    completeness = check_fold_completeness(df)
    if completeness.get("valid"):
        print(f"  Verdict:  [{completeness['verdict']}]")
        if completeness["failed_folds"]:
            print(f"  Missing OOS folds: {completeness['failed_years']}")
            print(f"  Reasons:           {completeness['reason_counts']}")
    else:
        print(f"  Skipped: {completeness.get('reason')}")

    # Section 2: predictiveness check
    print("\n── CHECK 1: SCORE PREDICTIVENESS ──")
    pred = check_score_predictiveness(df, objective=objective)
    if pred.get("valid"):
        print(f"  Verdict:  [{pred['verdict']}]")
        print(f"  Inner score vs OOS objective correlation: {pred['corr_inner_score_vs_oos_objective']}")
        print(f"  Inner mean vs OOS objective correlation:  {pred['corr_inner_mean_vs_oos_objective']}")
        print(f"  Inner score vs OOS Sharpe correlation: {pred['corr_inner_score_vs_oos_sharpe']}")
        print(f"  Inner score vs OOS QQQ alpha correlation: {pred['corr_inner_score_vs_oos_alpha_vs_qqq']}")
        print(f"  → {pred['interpretation']}")
    else:
        print(f"  Skipped: {pred.get('reason')}")

    # Section 3: calibration check
    print("\n── CHECK 2: MODEL CALIBRATION ──")
    calib = check_calibration(df)
    if calib.get("valid"):
        print(f"  Verdict:  [{calib['verdict']}]")
        print(f"  Inner predicts beat QQQ: {calib['inner_predicts_beat_qqq_pct']}%")
        print(f"  OOS actually beats QQQ:  {calib['oos_actually_beat_qqq_pct']}%")
        print(f"  Overconfidence gap:      {calib['overconfidence_gap_pct']} pp")
        print(f"  Direction accuracy:      {calib['direction_accuracy_pct']}% (50% = coin flip)")
    else:
        print(f"  Skipped: {calib.get('reason')}")

    # Section 4: concentration vulnerability (needs ETF data)
    print("\n── CHECK 3: CONCENTRATION VULNERABILITY ──")
    try:
        conc = yearly_market_concentration(qqq_path, spy_path)
        vuln = check_concentration_vulnerability(df, conc)
        if vuln.get("valid"):
            print(f"  Verdict:  [{vuln['verdict']}]")
            print(f"  Correlation (concentration vs OOS alpha): {vuln['correlation_conc_vs_oos_alpha']}")
            print(f"  High-concentration years: {vuln['high_concentration_years']}")
            print(f"    Mean OOS alpha vs QQQ: {vuln['high_conc_mean_oos_alpha_pct']}%")
            print(f"  Low-concentration years:  {vuln['low_concentration_years']}")
            print(f"    Mean OOS alpha vs QQQ: {vuln['low_conc_mean_oos_alpha_pct']}%")
            print(f"  Delta (high - low):       {vuln['delta_pct']}%")
        else:
            print(f"  Skipped: {vuln.get('reason')}")
    except FileNotFoundError as e:
        print(f"  Skipped: ETF data missing ({e})")
        vuln = {"valid": False, "reason": "etf_missing"}

    # Section 5: config stability
    print("\n── CHECK 4: CONFIG STABILITY ──")
    stab = check_config_stability(df)
    if stab.get("valid"):
        print(f"  Verdict:  [{stab['verdict']}]")
        print(f"  Unique configs: {stab['unique_configs']}/{stab['total_folds']} (uniqueness {stab['uniqueness_ratio']})")
        print(f"  Top config:     {stab['top_config'][:80]}")
        print(f"  Top frequency:  {stab['top_config_frequency']}")
    else:
        print(f"  Skipped: {stab.get('reason')}")

    # Section 6: overall recommendation
    print("\n── OVERALL RECOMMENDATION ──")
    verdicts = [
        completeness.get("verdict"),
        pred.get("verdict"),
        calib.get("verdict"),
        vuln.get("verdict"),
        stab.get("verdict"),
    ]
    fail_count = sum(1 for v in verdicts if v == "FAIL")
    warn_count = sum(1 for v in verdicts if v == "WARN")
    pass_count = sum(1 for v in verdicts if v == "PASS")
    print(f"  PASS={pass_count}  WARN={warn_count}  FAIL={fail_count}")

    if fail_count == 0 and warn_count <= 1:
        print("  → Strategy looks healthy.  Safe to deploy.")
    elif fail_count == 0:
        print("  → Mixed signal.  Worth deploying but monitor closely.")
    else:
        print("  → Critical failures detected.  Don't deploy yet — fix the FAIL checks first.")
        for label, result in [
            ("fold completeness", completeness),
            ("score predictiveness", pred),
            ("model calibration", calib),
            ("concentration vulnerability", vuln),
            ("config stability", stab),
        ]:
            if result.get("verdict") == "FAIL":
                print(f"    • Fix {label}")

    print()

    return {
        "summary": summary,
        "fold_completeness": completeness,
        "score_predictiveness": pred,
        "calibration": calib,
        "concentration_vulnerability": vuln,
        "config_stability": stab,
        "verdicts": verdicts,
    }


# ── CLI entrypoint ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"Walkforward results CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--qqq", default=DEFAULT_QQQ_PARQUET, help="QQQ price parquet")
    parser.add_argument("--spy", default=DEFAULT_SPY_PARQUET, help="SPY price parquet")
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        choices=("sharpe", "alpha_vs_qqq", "hybrid"),
        help=f"Objective used for score-predictiveness target (default: {DEFAULT_OBJECTIVE})",
    )
    parser.add_argument("--json", action="store_true", help="Also write results to JSON")
    args = parser.parse_args()

    results = analyze(args.csv, args.qqq, args.spy, objective=args.objective)

    if args.json:
        out_path = Path(args.csv).with_suffix(".analyzer.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved JSON report: {out_path}")


if __name__ == "__main__":
    main()
