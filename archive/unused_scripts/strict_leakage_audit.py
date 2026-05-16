"""
strict_leakage_audit.py - strict temporal and feature-contract audit.

This extends the lightweight leakage audit with three checks:

1. Future perturbation: change future OHLCV after a cutoff; features at or
   before the cutoff must not change.
2. Truncate/recompute: rebuild the same ticker with only data through the
   cutoff; historical features must match the full-data build.
3. Feature contract: every generated feature must be covered by an allowed
   prefix/exact-name rule, and obviously forward-looking feature names fail.

The dynamic checks neutralize external data builders so the audit focuses on
the local feature pipeline and stays deterministic/offline by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from settings import DATA_DIR, LOG_DIR  # noqa: E402

try:
    import pipeline_shared  # type: ignore  # noqa: E402
except Exception:
    pipeline_shared = None


DEFAULT_TICKERS = ("AAPL", "MSFT", "SPY")
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2024-01-01"
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
BASE_COLUMNS = {"Open", "High", "Low", "Close", "Volume", "target"}

CONTRACT_PREFIXES = (
    "ret_",
    "hvol_",
    "rsi_",
    "dist_ma",
    "ma_cross_",
    "macd",
    "atr_",
    "bb_",
    "vol_ratio",
    "volume_chg_",
    "vol_zscore_",
    "vol_pct_vs_",
    "vol_trend_",
    "hl_range",
    "spread_proxy",
    "roc_",
    "drawdown_",
    "weekly_",
    "monthly_",
    "tf_alignment",
    "spy_",
    "qqq_",
    "vix_",
    "macro_",
    "regime",
    "ret_vs_spy",
    "ret_vs_qqq",
    "breadth_",
    "pct_above_",
    "fund_",
    "factor_",
    "xs_rank_",
    "sector_vs_",
    "sector_ratio_",
    "sector_rs_",
    "alt_",
    "sent_z_",
)
CONTRACT_EXACT = {
    "obv_slope",
    "vwap_dist",
    "uptick_ratio",
    "variance_ratio",
    "eps_surprise_pct",
    "days_since_earnings",
    "days_to_next_earnings",
}
FORBIDDEN_FEATURE_PATTERNS = (
    re.compile(r"(^|_)target($|_)"),
    re.compile(r"(^|_)label($|_)"),
    re.compile(r"(^|_)future($|_)"),
    re.compile(r"(^|_)forward($|_)"),
    re.compile(r"(^|_)fwd($|_)"),
    re.compile(r"(^|_)exit_price($|_)"),
    re.compile(r"(^|_)entry_price($|_)"),
)

STATIC_SCAN_FILES = (
    "pipeline_shared.py",
    "cross_sectional_features.py",
    "fundamental_features.py",
    "xgb_feature_engineering.py",
    "labels.py",
    "train.py",
    "alpha_factor_backtest.py",
)
STATIC_PATTERNS = (
    ("negative_shift", re.compile(r"\.shift\(\s*-\s*\d+")),
    ("backfill", re.compile(r"\.(?:bfill|backfill)\s*\(")),
    ("centered_rolling", re.compile(r"rolling\([^)]*center\s*=\s*True")),
    ("full_sample_fit_transform", re.compile(r"\.fit_transform\s*\(")),
)
STATIC_ALLOWED_CONTEXT = (
    "target",
    "label",
    "forward_return",
    "fwd",
    "exit_price",
    "entry_price",
    "benchmark",
    "bench",
    "delayed_entry",
    "delayed_exit",
)


@dataclass
class StaticFinding:
    file: str
    line: int
    function: str | None
    kind: str
    severity: str
    text: str


def _read_local_price_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    path = Path(DATA_DIR) / f"{ticker.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    cols = [c for c in PRICE_COLUMNS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df.loc[start:end, cols].copy()
    return out.dropna(subset=[c for c in ("Close",) if c in out.columns])


def _perturb_after_cutoff(raw: pd.DataFrame, cutoff_pos: int) -> pd.DataFrame:
    out = raw.copy()
    future_idx = out.index[cutoff_pos + 1 :]
    if len(future_idx) == 0:
        return out
    for col in [c for c in PRICE_COLUMNS if c in out.columns]:
        values = pd.to_numeric(out.loc[future_idx, col], errors="coerce").to_numpy(dtype=float)
        if col == "Volume":
            out.loc[future_idx, col] = values[::-1] * 3.0 + 123.0
        else:
            scale = np.linspace(1.50, 0.50, len(values))
            out.loc[future_idx, col] = values[::-1] * scale
    return out


def _neutral_frame_for_args(*args, **kwargs) -> pd.DataFrame:
    for arg in args:
        if isinstance(arg, pd.DatetimeIndex):
            return pd.DataFrame(index=arg)
    return pd.DataFrame()


def _neutral_pead_features(_ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        index=dates,
        data={
            "eps_surprise_pct": 0.0,
            "days_since_earnings": 60.0,
            "days_to_next_earnings": 60.0,
        },
    )


def _build_research_frame_from_raw(raw: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    if pipeline_shared is None:
        raise RuntimeError("pipeline_shared could not be imported")

    originals = {
        "fetch_price_data": pipeline_shared.fetch_price_data,
        "build_multi_market": pipeline_shared.build_multi_market,
        "build_vix_features": pipeline_shared.build_vix_features,
        "build_sentiment_features": pipeline_shared.build_sentiment_features,
        "build_social_sentiment_features": pipeline_shared.build_social_sentiment_features,
        "build_pead_features": pipeline_shared.build_pead_features,
        "build_market_breadth_features": pipeline_shared.build_market_breadth_features,
    }

    def fake_fetch_price_data(fetch_ticker: str, *_args, **_kwargs) -> pd.DataFrame:
        if fetch_ticker.upper() == ticker.upper():
            return raw.copy()
        return pd.DataFrame()

    pipeline_shared.fetch_price_data = fake_fetch_price_data
    pipeline_shared.build_multi_market = _neutral_frame_for_args
    pipeline_shared.build_vix_features = _neutral_frame_for_args
    pipeline_shared.build_sentiment_features = _neutral_frame_for_args
    pipeline_shared.build_social_sentiment_features = _neutral_frame_for_args
    pipeline_shared.build_pead_features = _neutral_pead_features
    pipeline_shared.build_market_breadth_features = _neutral_frame_for_args
    try:
        return pipeline_shared.build_research_feature_frame(ticker, start, end)
    finally:
        for name, fn in originals.items():
            setattr(pipeline_shared, name, fn)


def _numeric_delta_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    *,
    atol: float,
) -> list[str]:
    changed: list[str] = []
    for col in left.columns:
        if col in BASE_COLUMNS or col not in right.columns:
            continue
        try:
            a = pd.to_numeric(left.loc[common_index, col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right.loc[common_index, col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        if len(a) == 0 or len(b) == 0:
            continue
        if not np.allclose(a, b, equal_nan=True, atol=atol, rtol=1e-9):
            changed.append(col)
    return changed


def _feature_contract_violations(frame: pd.DataFrame) -> dict:
    unknown: list[str] = []
    forbidden: list[str] = []
    features = [c for c in frame.columns if c not in BASE_COLUMNS]
    for col in features:
        if any(pattern.search(col.lower()) for pattern in FORBIDDEN_FEATURE_PATTERNS):
            forbidden.append(col)
        if col in CONTRACT_EXACT or any(col.startswith(prefix) for prefix in CONTRACT_PREFIXES):
            continue
        unknown.append(col)
    return {
        "feature_count": len(features),
        "unknown_features": sorted(set(unknown)),
        "forbidden_name_features": sorted(set(forbidden)),
    }


def audit_ticker(ticker: str, start: str, end: str, *, cutoff_fraction: float, atol: float) -> dict:
    raw = _read_local_price_frame(ticker, start, end)
    if raw.empty:
        return {"ticker": ticker, "status": "skip", "reason": "no local parquet price data"}
    if len(raw) < 120:
        return {"ticker": ticker, "status": "skip", "reason": f"not enough rows: {len(raw)}"}

    cutoff_pos = int(len(raw) * cutoff_fraction)
    cutoff_pos = min(max(cutoff_pos, 60), len(raw) - 20)
    cutoff_date = pd.Timestamp(raw.index[cutoff_pos])

    try:
        full = _build_research_frame_from_raw(raw.copy(), ticker, start, end)
        perturbed = _build_research_frame_from_raw(_perturb_after_cutoff(raw, cutoff_pos), ticker, start, end)
        truncated = _build_research_frame_from_raw(raw.iloc[: cutoff_pos + 1].copy(), ticker, start, str(cutoff_date.date()))
    except Exception as exc:
        return {"ticker": ticker, "status": "error", "error": str(exc)}

    common_perturb = pd.DatetimeIndex(full.index).intersection(pd.DatetimeIndex(perturbed.index))
    common_perturb = common_perturb[common_perturb <= cutoff_date]
    common_trunc = pd.DatetimeIndex(full.index).intersection(pd.DatetimeIndex(truncated.index))
    common_trunc = common_trunc[common_trunc <= cutoff_date]

    perturb_leaks = _numeric_delta_columns(full, perturbed, common_perturb, atol=atol)
    truncate_leaks = _numeric_delta_columns(full, truncated, common_trunc, atol=atol)
    contract = _feature_contract_violations(full)
    failed = bool(
        perturb_leaks
        or truncate_leaks
        or contract["unknown_features"]
        or contract["forbidden_name_features"]
    )

    return {
        "ticker": ticker,
        "status": "FAIL" if failed else "pass",
        "cutoff_date": str(cutoff_date.date()),
        "rows_raw": int(len(raw)),
        "rows_full_features": int(len(full)),
        "perturbation_leaks": perturb_leaks,
        "truncate_recompute_leaks": truncate_leaks,
        "contract": contract,
    }


def static_source_scan() -> list[dict]:
    findings: list[StaticFinding] = []
    benign_functions = {
        "benchmark_equity",
        "compare_to_benchmarks",
        "subperiod_metrics",
        "load_factor_panel",
    }
    for rel_path in STATIC_SCAN_FILES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        current_function: str | None = None
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            stripped = line.strip()
            match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
            if match:
                current_function = match.group(1)
            for kind, pattern in STATIC_PATTERNS:
                if not pattern.search(stripped):
                    continue
                context = stripped.lower()
                is_training_fit = kind == "full_sample_fit_transform" and "x_train" in context
                is_known_nonfeature = current_function in benign_functions
                is_label_or_target = any(token in context for token in STATIC_ALLOWED_CONTEXT)
                severity = "warn" if (is_training_fit or is_known_nonfeature or is_label_or_target) else "review"
                findings.append(
                    StaticFinding(
                        file=rel_path,
                        line=line_no,
                        function=current_function,
                        kind=kind,
                        severity=severity,
                        text=stripped[:220],
                    )
                )
    return [asdict(f) for f in findings]


def write_inventory(results: list[dict], path: Path) -> None:
    rows: list[dict] = []
    seen: set[str] = set()
    for result in results:
        contract = result.get("contract") or {}
        for col in contract.get("unknown_features", []):
            key = f"unknown:{col}"
            if key not in seen:
                rows.append({"feature": col, "status": "unknown_contract"})
                seen.add(key)
        for col in contract.get("forbidden_name_features", []):
            key = f"forbidden:{col}"
            if key not in seen:
                rows.append({"feature": col, "status": "forbidden_name"})
                seen.add(key)
        for family, cols in (
            ("future_perturbation", result.get("perturbation_leaks", [])),
            ("truncate_recompute", result.get("truncate_recompute_leaks", [])),
        ):
            for col in cols:
                key = f"{family}:{col}"
                if key not in seen:
                    rows.append({"feature": col, "status": family})
                    seen.add(key)
    pd.DataFrame(rows, columns=["feature", "status"]).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict leakage and feature-contract audit")
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cutoff-fraction", type=float, default=0.70)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--strict-static", action="store_true", help="Fail on non-whitelisted static scan findings")
    args = parser.parse_args(argv)

    os.makedirs(LOG_DIR, exist_ok=True)
    if pipeline_shared is None or not hasattr(pipeline_shared, "build_research_feature_frame"):
        print("[strict_leakage_audit] pipeline_shared.build_research_feature_frame not found.", file=sys.stderr)
        return 2

    results = [
        audit_ticker(t.upper(), args.start, args.end, cutoff_fraction=args.cutoff_fraction, atol=args.atol)
        for t in args.tickers
    ]
    static_findings = static_source_scan()
    blocking_static = [f for f in static_findings if f["severity"] == "review"]

    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "start": args.start,
        "end": args.end,
        "tickers": [t.upper() for t in args.tickers],
        "results": results,
        "static_findings": static_findings,
        "strict_static": bool(args.strict_static),
    }
    json_path = Path(LOG_DIR) / "strict_leakage_audit.json"
    csv_path = Path(LOG_DIR) / "strict_feature_audit.csv"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    write_inventory(results, csv_path)

    failed = [r for r in results if r.get("status") == "FAIL"]
    errors = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") == "skip"]
    print(f"[strict_leakage_audit] wrote {json_path}")
    print(f"[strict_leakage_audit] wrote {csv_path}")
    print(
        "[strict_leakage_audit] "
        f"pass={sum(r.get('status') == 'pass' for r in results)} "
        f"fail={len(failed)} error={len(errors)} skip={len(skipped)} "
        f"static_findings={len(static_findings)}"
    )
    if args.strict_static and blocking_static:
        print(f"[strict_leakage_audit] strict static scan has {len(blocking_static)} review findings.")
        return 1
    if failed or errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
