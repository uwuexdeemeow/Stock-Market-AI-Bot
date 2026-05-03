"""
etf_rotation_grid.py — predeclared benchmark-relative ETF rotation sweep.

The grid reuses saved walk-forward prediction CSVs from signals/. It does not
retrain the stock model and does not enable single-name paper trading.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import run_etf_rotation_backtest
from settings import (
    ETF_ROTATION_DEFENSIVE_MIXES,
    ETF_ROTATION_PRESETS,
    ETF_ROTATION_VOL_TARGETS,
    SIGNAL_DIR,
    WATCHLIST,
)


THRESHOLDS = (0.50, 0.55, 0.60)


def load_saved_predictions() -> dict[str, pd.DataFrame]:
    signal_dir = Path(SIGNAL_DIR)
    out: dict[str, pd.DataFrame] = {}
    for ticker in WATCHLIST:
        path = signal_dir / f"{ticker}_walkforward_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "date" not in df.columns or "direction_vote" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date", drop=False).sort_index()
        if not df.empty:
            out[ticker] = df
    return out


def main() -> None:
    predictions = load_saved_predictions()
    if not predictions:
        raise SystemExit("No saved walk-forward predictions found. Run backtest.py first.")

    rows: list[dict] = []
    for preset_name, preset in ETF_ROTATION_PRESETS.items():
        for defensive_name, defensive_mix in ETF_ROTATION_DEFENSIVE_MIXES.items():
            for threshold in THRESHOLDS:
                for vol_target in ETF_ROTATION_VOL_TARGETS:
                    metrics = run_etf_rotation_backtest(
                        predictions,
                        preset_name=preset_name,
                        preset=preset,
                        enter_threshold=threshold,
                        exit_threshold=max(0.35, threshold - 0.10),
                        neutral_threshold=max(0.45, threshold - 0.05),
                        vol_target=vol_target,
                        defensive_mix_name=defensive_name,
                        defensive_mix=defensive_mix,
                        write_outputs=False,
                        quiet=True,
                    )
                    if not metrics:
                        continue
                    comps = metrics.get("benchmark_comparisons", {})
                    rows.append({
                        "preset": preset_name,
                        "defensive_mix": defensive_name,
                        "enter_threshold": threshold,
                        "exit_threshold": max(0.35, threshold - 0.10),
                        "neutral_threshold": max(0.45, threshold - 0.05),
                        "vol_target": vol_target,
                        "total_return_pct": metrics.get("total_return_pct"),
                        "cagr_pct": metrics.get("cagr_pct"),
                        "sharpe": metrics.get("sharpe"),
                        "sortino": metrics.get("sortino"),
                        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                        "alpha_vs_spy_pct": comps.get("SPY", {}).get("alpha_pct"),
                        "alpha_vs_qqq_pct": comps.get("QQQ", {}).get("alpha_pct"),
                        "alpha_vs_blend_pct": comps.get("BLEND", {}).get("alpha_pct"),
                        "nw_tstat_vs_blend": metrics.get("nw_tstat_vs_blend"),
                        "max_gross_exposure_used": metrics.get("max_gross_exposure_used"),
                        "avg_gross_exposure": metrics.get("avg_gross_exposure"),
                        "turnover_pct": metrics.get("turnover_pct"),
                        "estimated_cost_pct": metrics.get("estimated_cost_pct"),
                        "paper_ready": metrics.get("paper_ready"),
                    })

    if not rows:
        raise SystemExit("ETF rotation grid produced no rows.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["paper_ready", "alpha_vs_blend_pct", "sharpe", "max_drawdown_pct", "total_return_pct"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    out_path = Path(SIGNAL_DIR) / "etf_rotation_grid.csv"
    summary.to_csv(out_path, index=False)

    print(f"Saved -> {out_path}")
    print("\nTop ETF rotation grid rows")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
