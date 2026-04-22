"""
train_and_optimize.py

Runs train.py first, then post_train_optimizer.py.
Supports:
- all tickers (no args)
- one ticker (--ticker NVDA)
- pass-through extra args to train.py / post_train_optimizer.py if needed later
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train.py and then post_train_optimizer.py")
    parser.add_argument("--ticker", type=str, default=None, help="Optional single ticker, e.g. NVDA")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "train.py"
    optimizer_path = base_dir / "post_train_optimizer.py"

    if not train_path.exists():
        raise SystemExit(f"Missing file: {train_path}")
    if not optimizer_path.exists():
        raise SystemExit(f"Missing file: {optimizer_path}")

    python_exe = sys.executable

    train_cmd = [python_exe, str(train_path)]
    opt_cmd = [python_exe, str(optimizer_path)]

    if args.ticker:
        ticker = args.ticker.upper()
        train_cmd += ["--ticker", ticker]
        opt_cmd += ["--ticker", ticker]

    run_step("TRAIN", train_cmd)
    run_step("POST-TRAIN OPTIMIZER", opt_cmd)

    print("\nDone. Training and optimizer both completed successfully.")


if __name__ == "__main__":
    main()
