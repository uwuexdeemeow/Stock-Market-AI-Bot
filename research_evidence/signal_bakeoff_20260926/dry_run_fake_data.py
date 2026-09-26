"""
dry_run_fake_data.py — check the bake-off script runs end to end, on FAKE data.

PLAIN ENGLISH: the real bake-off needs the project computer's data/ folder,
and it takes about 20 minutes.  Before spending that time, this script makes
sure the whole pipeline works: loading data, building every idea's score,
running the real strategy engine, and judging.  It:

  1. copies the project (your current files, including unsaved-to-Git edits)
     into a brand-new temporary folder;
  2. writes made-up random prices and features into that copy's data/ folder
     (the real data/ folder is never touched);
  3. runs signal_bakeoff.py --offsets 2 there (24 quick engine runs);
  4. checks the output has every run and a judged result.

The numbers it produces are MEANINGLESS (the prices are random).  It only
answers "does it run?".  Works in a cloud session with no real data.

Run from the project root:
    python research_evidence/signal_bakeoff_20260926/dry_run_fake_data.py [--keep]
It takes a few minutes.  --keep leaves the temporary folder for inspection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path("research_evidence/signal_bakeoff_20260926/signal_bakeoff.py")

# Runs inside the temporary copy (so its settings.DATA_DIR is the copy's data/).
FAKE_DATA_CODE = r'''
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd()))
from settings import DATA_DIR, WATCHLIST
from alpha_factor_backtest import load_feature_specs
import core_satellite_alpha as core

spec = importlib.util.spec_from_file_location("bakeoff", "research_evidence/signal_bakeoff_20260926/signal_bakeoff.py")
bakeoff = importlib.util.module_from_spec(spec); spec.loader.exec_module(bakeoff)
features = ({s["feature"] for s in load_feature_specs(write_health_outputs=False)} | set(bakeoff.B_FEATURE_POOL)
            | {"factor_beta_252_spy", "factor_idio_vol_252_spy", "hvol_20d", "xs_rank_sector_factor_idio_vol_252_spy",
               "xs_rank_market_factor_idio_vol_252_spy", "xs_rank_sector_hvol_20d", "xs_rank_market_hvol_20d"})
# ETFs start in 2000 so the regime switch has its warm-up history; stocks
# start in 2010 like the real factor files.  Real trading days only.
sessions = core._nyse_sessions(pd.Timestamp("2000-01-03"), pd.Timestamp("2026-08-12"))
rng = np.random.default_rng(7)
out = Path(DATA_DIR); out.mkdir(exist_ok=True)

def prices(first_day, drift, vol):
    idx = sessions[sessions >= first_day]
    close = 50 * np.exp(np.cumsum(drift + vol * rng.normal(size=len(idx))))
    open_ = close * np.exp(0.003 * rng.normal(size=len(idx)))
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.005,
                         "Low": np.minimum(open_, close) * 0.995, "Close": close,
                         "Volume": 1e6 * (1 + rng.random(len(idx)))}, index=idx)

for ticker in WATCHLIST:
    df = prices("2010-01-04", 0.0004, 0.018)
    for feature in sorted(features):
        df[feature] = rng.normal(size=len(df))
    close = df["Close"]
    df["factor_mom_12_1"] = (close.shift(21) / close.shift(252) - 1).fillna(0.0)
    df["factor_resid_mom_sector_12_1"] = (df["factor_mom_12_1"] - df["factor_mom_12_1"].rolling(5).mean()).fillna(0.0)
    df["days_to_next_earnings"] = 60.0
    df["days_since_earnings"] = 60.0
    df.to_parquet(out / f"{ticker}.parquet")
for ticker in ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD") + tuple(bakeoff.SECTOR_ETFS):
    # XLRE and XLC really started later; keep that so the short-history path is exercised.
    first = {"XLRE": "2015-10-08", "XLC": "2018-06-19"}.get(ticker, "2000-01-03")
    prices(first, 0.0004, 0.012).to_parquet(out / f"{ticker}.parquet")
vix = prices("2000-01-03", 0.0, 0.05)
vix["Close"] = 15 + 5 * np.abs(rng.normal(size=len(vix)))
vix.to_parquet(out / "^VIX.parquet")
vix["Close"] = vix["Close"] + 1.0
vix.to_parquet(out / "^VIX3M.parquet")
print(f"fake data: {len(list(out.glob('*.parquet')))} files")
'''


def copy_project(dest: Path) -> None:
    """Copy tracked and new (not ignored) files, so local edits are tested too."""
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.splitlines()
    for rel in listed:
        src = ROOT / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
    if (dest / "data").exists():
        raise SystemExit("refusing to continue: the copy already has a data/ folder")


def check_output(payload: dict, offsets: int) -> list[str]:
    """Problems with a dry-run output (empty list = fine)."""
    problems = []
    ideas = {row["idea"] for row in payload.get("rows", [])}
    expected_rows = offsets * 2 * 6
    if len(payload.get("rows", [])) != expected_rows:
        problems.append(f"expected {expected_rows} runs, got {len(payload.get('rows', []))}")
    if ideas != {"R0", "R1", "E", "A", "B", "C"}:
        problems.append(f"ideas run: {sorted(ideas)}")
    if payload.get("valid_full_test") is not False:
        problems.append("a smoke run must be marked valid_full_test = false")
    if payload.get("approves_trading") is not False:
        problems.append("output must say approves_trading = false")
    result = payload.get("result", {})
    for idea in ("A", "B", "C"):
        if "gates" not in result.get("summary", {}).get(idea, {}):
            problems.append(f"idea {idea} was not judged")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bake-off on fake data in a temporary copy.")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = parser.parse_args(argv)
    offsets = 2
    work = Path(tempfile.mkdtemp(prefix="bakeoff_dry_run_"))
    try:
        copy_project(work)
        subprocess.run([sys.executable, "-c", FAKE_DATA_CODE], cwd=work, check=True)
        run = subprocess.run([sys.executable, str(SCRIPT), "--offsets", str(offsets)],
                             cwd=work, capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-3000:], run.stderr[-5000:], sep="\n")
            print("DRY RUN FAILED: the bake-off script crashed (see above).")
            return 1
        payload = json.loads((work / SCRIPT.parent / "signal_bakeoff.json").read_text())
        problems = check_output(payload, offsets)
        seconds = sum(float(r.get("secs", 0)) for r in payload["rows"])
        print(f"{len(payload['rows'])} engine runs, {seconds:.0f}s of engine time "
              f"(a full run is about {seconds * 10 / 60:.0f} min of engine time plus loading)")
        if problems:
            print("DRY RUN FAILED:", *problems, sep="\n  ")
            return 1
        print("DRY RUN PASSED (numbers are from random data and mean nothing).")
        return 0
    finally:
        if args.keep:
            print(f"kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
