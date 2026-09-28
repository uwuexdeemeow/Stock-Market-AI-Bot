"""
core_satellite_calendar_robustness.py — rerun the core-satellite backtest from
every possible start day of the rebalance calendar.

PLAIN ENGLISH: the strategy rebalances every 20 trading days.  WHICH 20 days
depends only on the first day of the data, which is an arbitrary choice.
Starting one day later gives the same stocks a slightly different set of
entry and exit days, and the headline numbers can change a lot.  This script
runs the backtest once per possible start day ("offset") and reports the
spread, so a strategy is judged on its typical and worst result instead of
on one lucky (or unlucky) calendar.

This is research-only evidence.  It never approves or blocks trading.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import core_satellite_alpha as core
from core_satellite_execution_stress import candidate_name, load_candidate_config, mark_research_candidate
from core_satellite_survivorship_audit import _build_panel, _load_selected_config
from safe_io import atomic_write_csv, atomic_write_json
from settings import LOG_DIR, WATCHLIST


OUT_JSON = Path(LOG_DIR) / "core_satellite_calendar_robustness.json"
OUT_CSV = Path(LOG_DIR) / "core_satellite_calendar_robustness.csv"
# PLAIN ENGLISH: research candidates write to their own files.
CANDIDATE_PREFIX = "research_candidate_calendar_robustness"


def _offset_row(offset: int, metrics: dict) -> dict:
    """Pick the numbers we compare across start days out of one backtest."""
    comps = metrics.get("benchmark_comparisons", {})
    holdout = metrics.get("holdout_2023_2026", {}) or {}
    return {
        "offset": offset,
        "status": "ok",
        "error": "",
        "total_return_pct": metrics.get("total_return_pct"),
        "cagr_pct": metrics.get("cagr_pct"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "alpha_vs_qqq_pct": comps.get("QQQ", {}).get("alpha_pct"),
        "alpha_vs_blend_pct": comps.get("BLEND", {}).get("alpha_pct"),
        "holdout_return_pct": holdout.get("strategy_return_pct"),
        "holdout_alpha_vs_qqq_pct": holdout.get("alpha_vs_qqq_pct"),
        "holdout_alpha_vs_blend_pct": holdout.get("alpha_vs_blend_pct"),
    }


def run_offsets(
    panel: pd.DataFrame,
    config: dict,
    offsets: list[int],
    evaluate: Callable[[pd.DataFrame, dict], tuple] = core.evaluate,
) -> pd.DataFrame:
    """Backtest once per start-day offset.

    PLAIN ENGLISH: the calendar starts on the panel's first date, so dropping
    the first ``k`` dates moves every rebalance day by ``k`` sessions.  Losing
    ``k`` (< 20) warm-up days out of 15+ years does not matter.  One offset
    that hits a data problem is recorded as an error; the others still run.
    """
    dates = sorted(pd.to_datetime(panel["date"]).unique())
    rows = []
    for k in offsets:
        if k >= len(dates):
            rows.append({"offset": k, "status": "error", "error": "offset beyond data"})
            continue
        trimmed = panel[pd.to_datetime(panel["date"]) >= dates[k]]
        try:
            metrics, _equity, _trades = evaluate(trimmed, config)
        except ValueError as exc:
            rows.append({"offset": k, "status": "error", "error": str(exc)[:200]})
            continue
        rows.append(_offset_row(k, metrics))
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> dict:
    """Median / worst / best for each metric, plus how the usual calendar ranks."""
    ok = rows[rows["status"] == "ok"] if "status" in rows else rows.iloc[0:0]
    summary: dict = {
        "offsets_run": int(len(rows)),
        "offsets_ok": int(len(ok)),
        "offsets_failed": int(len(rows) - len(ok)),
    }
    for col in ("holdout_alpha_vs_qqq_pct", "holdout_alpha_vs_blend_pct", "total_return_pct",
                "alpha_vs_qqq_pct", "sharpe", "max_drawdown_pct"):
        values = pd.to_numeric(ok.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if values.empty:
            continue
        summary[col] = {
            "median": round(float(values.median()), 4),
            "p25": round(float(values.quantile(0.25)), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }
    holdout = pd.to_numeric(ok.get("holdout_alpha_vs_qqq_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    if not holdout.empty:
        # PLAIN ENGLISH: how many calendars did NOT beat QQQ in 2023-2026?
        summary["holdout_alpha_vs_qqq_nonpositive_share"] = round(float((holdout <= 0).mean()), 4)
        base = ok.loc[ok["offset"] == 0, "holdout_alpha_vs_qqq_pct"]
        if not base.empty and pd.notna(base.iloc[0]):
            base_value = float(base.iloc[0])
            # Rank 1 = the best calendar.  A usual calendar near rank 1 means
            # the headline number is flattered by luck.
            summary["usual_calendar_holdout_rank"] = int((holdout > base_value).sum() + 1)
            summary["usual_calendar_holdout_alpha_vs_qqq_pct"] = round(base_value, 4)
    return summary


def telegram_message(summary: dict) -> str:
    """Short plain-text summary for the weekly Telegram message."""
    holdout = summary.get("holdout_alpha_vs_qqq_pct", {})
    drawdown = summary.get("max_drawdown_pct", {})
    lines = [
        "Weekly calendar robustness (advisory, no trading effect)",
        f"Start days run: {summary.get('offsets_ok', 0)} ok, {summary.get('offsets_failed', 0)} failed",
        f"2023-26 alpha vs QQQ: median {holdout.get('median')}%, worst {holdout.get('min')}%, best {holdout.get('max')}%",
        f"Live calendar: {summary.get('usual_calendar_holdout_alpha_vs_qqq_pct')}% "
        f"(rank {summary.get('usual_calendar_holdout_rank')} of {summary.get('offsets_ok', 0)})",
        f"Calendars not beating QQQ: {round(100 * float(summary.get('holdout_alpha_vs_qqq_nonpositive_share', 0) or 0))}%",
        f"Max drawdown: median {drawdown.get('median')}%, worst {drawdown.get('min')}%",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-json", default=None,
                        help="Test a research candidate config instead of the approved one.")
    parser.add_argument("--candidate-name", default=None,
                        help="Short name for the candidate output files (default: the JSON file name).")
    parser.add_argument("--offsets", type=int, default=None,
                        help="How many start days to try (default: the config's holding_days).")
    parser.add_argument("--telegram", action="store_true",
                        help="Also send the summary to Telegram (needs TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_csv, out_json, cand_name = OUT_CSV, OUT_JSON, None
    if args.candidate_json:
        config = load_candidate_config(args.candidate_json)
        cand_name = candidate_name(args.candidate_json, args.candidate_name)
        out_csv = Path(LOG_DIR) / f"{CANDIDATE_PREFIX}_{cand_name}.csv"
        out_json = Path(LOG_DIR) / f"{CANDIDATE_PREFIX}_{cand_name}.json"
    else:
        config = _load_selected_config()
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    holding_days = int(config.get("holding_days", 20))
    n_offsets = int(args.offsets or holding_days)
    panel = _build_panel(list(WATCHLIST))
    rows = run_offsets(panel, config, list(range(n_offsets)))
    summary = summarize(rows)
    atomic_write_csv(rows, out_csv, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "core_satellite_rebalance_calendar_robustness",
        # PLAIN ENGLISH: advisory only — no gate reads this file.
        "approves_trading": False,
        "selected_config": config,
        "holding_days": holding_days,
        "summary": summary,
        "rows": rows.replace({np.nan: None}).to_dict(orient="records"),
    }
    if cand_name is not None:
        payload = mark_research_candidate(payload, candidate_path=args.candidate_json, name=cand_name)
    atomic_write_json(payload, out_json)

    print(f"Calendar robustness written -> {out_csv}")
    print(f"Detailed report -> {out_json}")
    cols = ["offset", "status", "holdout_alpha_vs_qqq_pct", "total_return_pct", "sharpe", "max_drawdown_pct", "error"]
    print(rows[[c for c in cols if c in rows.columns]].to_string(index=False))
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.telegram:
        # PLAIN ENGLISH: a failed message must not fail the report itself.
        from notifications import send_telegram
        sent = send_telegram(telegram_message(summary), parse_mode="")
        print(f"Telegram summary sent: {sent}")


if __name__ == "__main__":
    main()
