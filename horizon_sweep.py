from __future__ import annotations

import argparse
import shutil
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from settings import SIGNAL_DIR


def _saved_paths(stdout: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    suffix_map = {
        "_predictions.csv": "predictions_path",
        "_trades.csv": "trades_path",
        "_equity.csv": "equity_path",
        "_metrics.json": "metrics_path",
        "_ticker_breakdown.csv": "breakdown_path",
        "_trade_attribution.csv": "attribution_path",
    }
    for line in stdout.splitlines():
        if not line.startswith("Saved ->"):
            continue
        path = line.split("Saved ->", 1)[1].strip()
        for suffix, key in suffix_map.items():
            if path.endswith(suffix):
                paths[key] = path
    return paths


def _copy_horizon_artifacts(paths: dict[str, str], horizon: int, stamp: str) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, path in paths.items():
        if not path or not os.path.exists(path):
            continue
        root, ext = os.path.splitext(path)
        dst = f"{root}_h{int(horizon)}_{stamp}{ext}"
        shutil.copy2(path, dst)
        copied[key] = dst
    return copied


def run_horizon(
    horizon: int,
    mode: str,
    extra_args: list[str],
    train_first: bool,
    full_tune: bool,
    stamp: str,
) -> dict:
    env = os.environ.copy()
    env["RETURN_HORIZON_DAYS"] = str(int(horizon))

    train_exit_code = None
    train_tail = ""
    if train_first:
        train_cmd = [sys.executable, "train.py"]
        if not full_tune:
            train_cmd.append("--no-tune")
        train_proc = subprocess.run(train_cmd, env=env, text=True, capture_output=True)
        train_exit_code = int(train_proc.returncode)
        train_tail = (train_proc.stderr or train_proc.stdout)[-1500:]
        if train_proc.returncode != 0:
            return {
                "horizon_days": int(horizon),
                "train_exit_code": train_exit_code,
                "exit_code": None,
                "metrics_path": None,
                "error_tail": train_tail,
            }

    cmd = [sys.executable, "backtest.py", "--mode", mode, *extra_args]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)

    paths = _saved_paths(proc.stdout)
    copied_paths = _copy_horizon_artifacts(paths, horizon, stamp)
    metrics_path = copied_paths.get("metrics_path") or paths.get("metrics_path")

    row = {
        "horizon_days": int(horizon),
        "train_exit_code": train_exit_code,
        "exit_code": int(proc.returncode),
        "metrics_path": metrics_path,
        "trades_path": copied_paths.get("trades_path"),
        "equity_path": copied_paths.get("equity_path"),
    }
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row.update({
            "total_return_pct": metrics.get("total_return_pct"),
            "alpha_pct": metrics.get("alpha_pct"),
            "benchmark_return_pct": metrics.get("benchmark_return_pct"),
            "sharpe_ratio_daily": metrics.get("sharpe_ratio_daily"),
            "nw_tstat_vs_cash": metrics.get("nw_tstat_vs_cash"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "cash_overlay_avg_alloc": metrics.get("cash_overlay_avg_alloc"),
            "kill_all_pass": bool(metrics.get("kill_criteria", {}).get("results", {}).get("all_pass", False)),
        })
    else:
        row["error_tail"] = (proc.stderr or proc.stdout)[-1000:]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest.py across multiple RETURN_HORIZON_DAYS values.")
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--mode", default="long_only", choices=["long_short", "long_only", "long_only_bear_cash"])
    parser.add_argument("--train-first", action="store_true", help="Run train.py for each horizon before backtest.py.")
    parser.add_argument("--full-tune", action="store_true", help="With --train-first, run full train.py instead of train.py --no-tune.")
    parser.add_argument("items", nargs="*", help="Optional positional horizons, followed by extra backtest.py args after --")
    args = parser.parse_args()

    items = list(args.items)
    if "--" in items:
        sep = items.index("--")
        leading = items[:sep]
        extra_args = items[sep + 1:]
    else:
        leading = items
        extra_args = []

    if args.horizons is not None:
        horizons = args.horizons
        if leading:
            extra_args = leading + extra_args
    else:
        horizons = []
        while leading:
            try:
                horizons.append(int(leading[0]))
                leading = leading[1:]
            except ValueError:
                break
        extra_args = leading + extra_args
        if not horizons:
            horizons = [10, 15, 20]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [
        run_horizon(
            h,
            args.mode,
            extra_args,
            train_first=bool(args.train_first),
            full_tune=bool(args.full_tune),
            stamp=stamp,
        )
        for h in horizons
    ]
    summary = pd.DataFrame(rows)
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    out_path = os.path.join(SIGNAL_DIR, f"horizon_sweep_{stamp}.csv")
    summary.to_csv(out_path, index=False)
    print("Saved ->", out_path)
    print(summary.to_string(index=False))

    rank_cols = ["kill_all_pass", "sharpe_ratio_daily", "total_return_pct"]
    if not summary.empty and all(c in summary.columns for c in rank_cols):
        ranked = summary.sort_values(rank_cols, ascending=[False, False, False])
        best = ranked.iloc[0]
        print(
            f"\nBest horizon: {int(best['horizon_days'])}d "
            f"(Sharpe={best.get('sharpe_ratio_daily')}, return={best.get('total_return_pct')}%)"
        )
    elif not summary.empty:
        print("\nNo successful backtests to rank. Check error_tail above.")


if __name__ == "__main__":
    main()
