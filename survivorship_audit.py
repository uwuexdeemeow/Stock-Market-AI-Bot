"""
survivorship_audit.py — Build and profile crashed/delisted ticker histories.

This is intentionally separate from settings.WATCHLIST. The goal is to audit
whether the model survives pre-crash patterns without poisoning the production
core universe by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

import pipeline_shared
from settings import (
    CORE_WATCHLIST,
    DATA_DIR,
    LOG_DIR,
    SURVIVORSHIP_AUDIT_TICKERS,
    TRAIN_END,
    TRAIN_START,
)


REPORT_PATH = os.path.join(LOG_DIR, "survivorship_audit.json")


def _clean_date(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _profile_frame(ticker: str, df: pd.DataFrame, alias_used: str | None = None) -> dict:
    close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
    if close.empty:
        return {
            "ticker": ticker,
            "alias_used": alias_used,
            "status": "no_close",
            "rows": int(len(df)),
        }

    returns = close.pct_change().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    peak_idx = close.loc[:close.index[-1]].idxmax()
    peak_to_final = close.iloc[-1] / max(float(close.loc[peak_idx]), 1e-9) - 1.0
    final_252d = close.iloc[-1] / close.iloc[-253] - 1.0 if len(close) > 252 else np.nan

    return {
        "ticker": ticker,
        "alias_used": alias_used,
        "status": "ok",
        "rows": int(len(df)),
        "start": _clean_date(df.index.min()),
        "end": _clean_date(df.index.max()),
        "final_close": round(float(close.iloc[-1]), 4),
        "max_drawdown_pct": round(float(drawdown.min()) * 100.0, 2),
        "peak_to_final_pct": round(float(peak_to_final) * 100.0, 2),
        "final_252d_return_pct": (
            round(float(final_252d) * 100.0, 2) if np.isfinite(final_252d) else None
        ),
    }


def _build_with_alias(canonical: str, alias: str, start: str, end: str) -> pd.DataFrame:
    original_fetch: Callable = pipeline_shared.fetch_price_data

    def fetch_alias(_ticker: str, fetch_start: str, fetch_end: str) -> pd.DataFrame:
        return original_fetch(alias, fetch_start, fetch_end)

    pipeline_shared.fetch_price_data = fetch_alias
    try:
        return pipeline_shared.build_research_feature_frame(canonical, start, end)
    finally:
        pipeline_shared.fetch_price_data = original_fetch


def build_audit_data(
    start: str,
    end: str,
    *,
    force: bool = False,
    min_rows: int = 500,
    with_sentiment: bool = False,
) -> list[dict]:
    if not with_sentiment:
        pipeline_shared.USE_NEWS_SENTIMENT = False
        pipeline_shared.SOCIAL_SENTIMENT_ENABLED = False

    os.makedirs(DATA_DIR, exist_ok=True)
    rows: list[dict] = []
    for canonical, aliases in SURVIVORSHIP_AUDIT_TICKERS.items():
        out_path = os.path.join(DATA_DIR, f"{canonical}.parquet")
        if os.path.exists(out_path) and not force:
            try:
                existing = pd.read_parquet(out_path)
                profile = _profile_frame(canonical, existing)
                profile["status"] = "exists"
                profile["path"] = out_path
                rows.append(profile)
                continue
            except Exception:
                pass

        built = None
        alias_used = None
        errors: list[str] = []
        for alias in aliases:
            try:
                df = _build_with_alias(canonical, alias, start, end)
                if df is not None and len(df) >= min_rows:
                    built = df
                    alias_used = alias
                    break
                errors.append(f"{alias}: rows={0 if df is None else len(df)}")
            except Exception as exc:
                errors.append(f"{alias}: {exc}")

        if built is None:
            rows.append({
                "ticker": canonical,
                "aliases": aliases,
                "status": "unavailable",
                "errors": errors[-5:],
            })
            continue

        built.to_parquet(out_path, index=True)
        profile = _profile_frame(canonical, built, alias_used=alias_used)
        profile["path"] = out_path
        rows.append(profile)

    return rows


def existing_audit_profiles(min_rows: int = 500) -> list[dict]:
    rows: list[dict] = []
    for canonical in SURVIVORSHIP_AUDIT_TICKERS:
        path = os.path.join(DATA_DIR, f"{canonical}.parquet")
        if not os.path.exists(path):
            rows.append({"ticker": canonical, "status": "missing", "path": path})
            continue
        try:
            df = pd.read_parquet(path)
            profile = _profile_frame(canonical, df)
            profile["status"] = "available" if len(df) >= min_rows else "too_short"
            profile["path"] = path
            rows.append(profile)
        except Exception as exc:
            rows.append({"ticker": canonical, "status": "error", "error": str(exc), "path": path})
    return rows


def available_audit_tickers(profiles: list[dict], min_rows: int = 500) -> list[str]:
    return [
        str(row["ticker"]).upper()
        for row in profiles
        if row.get("status") in {"ok", "exists", "available"}
        and int(row.get("rows", 0) or 0) >= min_rows
    ]


def write_report(build_rows: list[dict] | None, profiles: list[dict], tickers_for_backtest: list[str]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "purpose": "survivorship_bias_audit",
        "build_results": build_rows or [],
        "profiles": profiles,
        "available_audit_tickers": available_audit_tickers(profiles),
        "core_plus_available_audit_tickers": tickers_for_backtest,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def run_backtest(tickers: list[str], mode: str, ablate_blend: bool) -> int:
    cmd = [
        sys.executable,
        "backtest.py",
        "--tickers",
        *tickers,
        "--mode",
        mode,
        "--no-eligibility-filter",
    ]
    if ablate_blend:
        cmd.append("--ablate-blend")
    return subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/profile crashed or delisted audit tickers")
    parser.add_argument("--build", action="store_true", help="Attempt to build audit parquet files")
    parser.add_argument("--force", action="store_true", help="Rebuild audit parquets even if they exist")
    parser.add_argument("--report", action="store_true", help="Write logs/survivorship_audit.json")
    parser.add_argument("--run-backtest", action="store_true", help="Run core + available audit tickers")
    parser.add_argument("--ablate-blend", action="store_true", help="Pass --ablate-blend to audit backtest")
    parser.add_argument("--with-sentiment", action="store_true", help="Include slow news sentiment while building")
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end", default=TRAIN_END)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--mode", choices=["long_only", "long_short", "long_only_bear_cash"], default="long_only")
    args = parser.parse_args()

    build_rows = None
    if args.build:
        build_rows = build_audit_data(
            args.start,
            args.end,
            force=args.force,
            min_rows=args.min_rows,
            with_sentiment=args.with_sentiment,
        )

    profiles = existing_audit_profiles(min_rows=args.min_rows)
    audit_tickers = available_audit_tickers(profiles, min_rows=args.min_rows)
    tickers_for_backtest = CORE_WATCHLIST + [t for t in audit_tickers if t not in CORE_WATCHLIST]

    if args.report or args.build or args.run_backtest:
        write_report(build_rows, profiles, tickers_for_backtest)
        print(f"Saved -> {REPORT_PATH}")

    print("Available audit tickers:", audit_tickers)
    print("Core + audit count:", len(tickers_for_backtest))
    if tickers_for_backtest:
        print("Backtest tickers:", " ".join(tickers_for_backtest))

    if args.run_backtest:
        raise SystemExit(run_backtest(tickers_for_backtest, args.mode, args.ablate_blend))


if __name__ == "__main__":
    main()
