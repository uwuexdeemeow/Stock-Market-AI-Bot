"""Calibrate backtest turnover costs from actual Alpaca paper fills.

PLAIN ENGLISH: The backtest should not guess that every trade costs the same.
This script groups measured fill slippage by buy/sell side, order type, and the
ticker's recent dollar-volume bucket, then writes conservative cost estimates.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from safe_io import atomic_write_json
from settings import DATA_DIR, SIGNAL_DIR, SLIPPAGE_BASE_PCT


DEFAULT_INPUT = Path(SIGNAL_DIR) / "alpaca_slippage_reversal_report.json"
DEFAULT_OUTPUT = Path(SIGNAL_DIR) / "execution_cost_calibration.json"
MIN_SAMPLE_COUNT = 20


def _liquidity_bucket(symbol: str, data_dir: Path) -> tuple[str, float | None]:
    """Classify a ticker using its latest 20-day dollar-volume feature."""
    path = data_dir / f"{str(symbol).upper()}.parquet"
    try:
        frame = pd.read_parquet(path)
        candidates = [
            "factor_liquidity_dollar_vol_20d",
            "dollar_volume_20d",
        ]
        column = next(column for column in candidates if column in frame.columns)
        value = float(pd.to_numeric(frame[column], errors="coerce").dropna().iloc[-1])
    except Exception:
        return "unknown", None
    if value >= 1_000_000_000:
        return "high", value
    if value >= 100_000_000:
        return "medium", value
    return "low", value


def _segment(rows: pd.DataFrame) -> dict:
    """Return conservative statistics for one group of fills."""
    values = pd.to_numeric(rows["slippage_bps"], errors="coerce").dropna().clip(lower=0.0)
    if values.empty:
        return {"samples": 0, "median_slippage_bps": None, "p75_slippage_bps": None}
    return {
        "samples": int(len(values)),
        "median_slippage_bps": round(float(values.median()), 4),
        "p75_slippage_bps": round(float(values.quantile(0.75)), 4),
        "mean_slippage_bps": round(float(values.mean()), 4),
    }


def build_execution_cost_calibration(
    report: dict,
    *,
    data_dir: Path = Path(DATA_DIR),
) -> dict:
    """Build side/liquidity cost buckets and one backtest recommendation."""
    rows = pd.DataFrame(report.get("orders", []) or [])
    required = {"symbol", "side", "order_type", "slippage_bps"}
    if rows.empty or not required.issubset(rows.columns):
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "collecting",
            "sample_count": 0,
            "minimum_sample_count": MIN_SAMPLE_COUNT,
            "recommended_one_way_slippage_bps": round(float(SLIPPAGE_BASE_PCT) * 10_000, 4),
            "segments": [],
        }
    liquidity = {
        symbol: _liquidity_bucket(symbol, data_dir)
        for symbol in sorted(set(rows["symbol"].astype(str).str.upper()))
    }
    rows["symbol"] = rows["symbol"].astype(str).str.upper()
    rows["liquidity_bucket"] = rows["symbol"].map(lambda symbol: liquidity[symbol][0])
    rows["dollar_volume_20d"] = rows["symbol"].map(lambda symbol: liquidity[symbol][1])
    rows["notional"] = (
        pd.to_numeric(rows.get("filled_qty"), errors="coerce").fillna(0.0)
        * pd.to_numeric(rows.get("fill_price"), errors="coerce").fillna(0.0)
    )

    segments: list[dict] = []
    for keys, group in rows.groupby(["side", "order_type", "liquidity_bucket"], dropna=False):
        side, order_type, bucket = keys
        segments.append({
            "side": str(side),
            "order_type": str(order_type),
            "liquidity_bucket": str(bucket),
            "median_notional": round(float(group["notional"].median()), 2),
            **_segment(group),
        })

    # Use normal rebalance fills for strategy costs. Trailing stops are risk
    # exits and would otherwise overstate every planned rebalance.
    normal = rows[rows["order_type"].astype(str) != "trailing_stop"]
    if normal.empty:
        normal = rows
    normal_slippage = pd.to_numeric(normal["slippage_bps"], errors="coerce").dropna().clip(lower=0.0)
    measured = float(normal_slippage.quantile(0.75)) if len(normal_slippage) else 0.0
    configured = float(SLIPPAGE_BASE_PCT) * 10_000
    recommendation = max(configured, measured)
    sample_count = int(len(rows))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_generated_at": report.get("generated_at"),
        "status": "ready" if sample_count >= MIN_SAMPLE_COUNT else "collecting",
        "sample_count": sample_count,
        "minimum_sample_count": MIN_SAMPLE_COUNT,
        "recommended_one_way_slippage_bps": round(recommendation, 4),
        "configured_floor_bps": round(configured, 4),
        "segments": segments,
    }


def calibrated_turnover_cost_pct(path: Path = DEFAULT_OUTPUT) -> float:
    """Return the one-way turnover cost fraction used by the backtest."""
    if os.environ.get("BACKTEST_USE_LIVE_COST_CALIBRATION", "1").strip().lower() not in {"1", "true", "yes"}:
        return float(SLIPPAGE_BASE_PCT)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ready":
            return float(SLIPPAGE_BASE_PCT)
        bps = float(payload.get("recommended_one_way_slippage_bps"))
        return max(float(SLIPPAGE_BASE_PCT), bps / 10_000.0)
    except Exception:
        return float(SLIPPAGE_BASE_PCT)


def causal_cost_parameters(report: dict, cutoff) -> dict:
    """Estimate residual slippage only from fully attributed pre-cutoff fills."""
    cutoff = pd.to_datetime(cutoff, utc=True)
    values = []
    for row in report.get("orders", []):
        timestamp = pd.to_datetime(row.get("filled_at"), utc=True, errors="coerce")
        if pd.isna(timestamp) or timestamp > cutoff:
            continue
        parts = pd.to_numeric(pd.Series([row.get(key) for key in
                 ("arrival_shortfall_bps", "half_spread_bps", "modeled_impact_bps")]), errors="coerce").to_numpy(dtype=float)
        if not all(np.isfinite(x) for x in parts):
            continue
        # Spread and impact are already charged by the fill model. Charging
        # their measured cost again as slippage would count them twice.
        values.append(max(0., float(parts[0] - parts[1] - parts[2])))
    enough = len(values) >= MIN_SAMPLE_COUNT
    residual = max(float(SLIPPAGE_BASE_PCT), float(np.quantile(values, .75)) / 10000) if enough else float(SLIPPAGE_BASE_PCT)
    return {"base_slippage_pct": residual, "cutoff": cutoff.isoformat(), "samples": len(values),
            "source": "precutoff_residual_calibration" if enough else "frozen_settings_fallback"}


def main() -> int:
    """Read the current slippage report and write calibration JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    calibration = build_execution_cost_calibration(report)
    atomic_write_json(calibration, args.output)
    print(
        f"Execution cost calibration: {calibration['status']} "
        f"samples={calibration['sample_count']} "
        f"recommended={calibration['recommended_one_way_slippage_bps']:.2f} bps"
    )
    print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
