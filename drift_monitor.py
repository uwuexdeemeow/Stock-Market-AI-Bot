"""Create feature baselines and monitor daily distribution drift with PSI/KS.

PLAIN ENGLISH: a model learns from one range of inputs.  If today's inputs look
very different, its predictions may be unreliable even when the code still
runs.  Training saves a compact baseline; the daily command compares recent
features with that baseline and writes ``logs/drift.jsonl`` for monitor.py.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from safe_io import atomic_write_json, atomic_write_text
from settings import DATA_DIR, DRIFT_KS_ALERT, DRIFT_PSI_ALERT, LOG_DIR, MODEL_DIR, WATCHLIST


DEFAULT_LOG_PATH = Path(LOG_DIR) / "drift.jsonl"
DEFAULT_BASELINE_PATH = Path(MODEL_DIR) / "pooled_drift_baseline.json"


def _finite(values: Iterable) -> np.ndarray:
    """Convert mixed values to a clean numeric array for stable statistics."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series[np.isfinite(series)].to_numpy(dtype=float)


def snapshot_baseline(
    frame: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_BASELINE_PATH,
    metadata: dict | None = None,
) -> dict:
    """Save compact decile bins and reference quantiles for numeric features."""
    features: dict[str, dict] = {}
    for column in frame.columns:
        values = _finite(frame[column])
        if len(values) < 20 or np.unique(values).size < 2:
            continue
        raw_edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
        edges = np.unique(raw_edges)
        if len(edges) < 3:
            continue
        histogram_edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        counts, _ = np.histogram(values, bins=histogram_edges)
        proportions = counts / max(int(counts.sum()), 1)
        features[str(column)] = {
            "sample_count": int(len(values)),
            "bin_edges": [float(value) for value in edges[1:-1]],
            "bin_proportions": [float(value) for value in proportions],
            # 101 quantiles approximate the baseline CDF without storing every
            # historical row or exposing bulky training data.
            "reference_quantiles": [
                float(value) for value in np.quantile(values, np.linspace(0.0, 1.0, 101))
            ],
        }
    if not features:
        raise ValueError("no usable numeric features for drift baseline")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_count": len(features),
        "metadata": dict(metadata or {}),
        "features": features,
    }
    atomic_write_json(payload, output_path)
    return payload


def _psi(current: np.ndarray, feature: dict) -> float:
    """Calculate Population Stability Index using training-time bins."""
    edges = np.array([-np.inf, *feature.get("bin_edges", []), np.inf], dtype=float)
    counts, _ = np.histogram(current, bins=edges)
    actual = counts / max(int(counts.sum()), 1)
    expected = np.asarray(feature.get("bin_proportions", []), dtype=float)
    if len(actual) != len(expected):
        return float("nan")
    epsilon = 1e-6
    return float(np.sum((actual - expected) * np.log((actual + epsilon) / (expected + epsilon))))


def _ks_stat(current: np.ndarray, reference_quantiles: list[float]) -> float:
    """Approximate the two-sample KS distance from stored baseline quantiles."""
    reference = _finite(reference_quantiles)
    if not len(current) or not len(reference):
        return float("nan")
    points = np.unique(np.concatenate((current, reference)))
    current_sorted = np.sort(current)
    reference_sorted = np.sort(reference)
    current_cdf = np.searchsorted(current_sorted, points, side="right") / len(current_sorted)
    reference_cdf = np.searchsorted(reference_sorted, points, side="right") / len(reference_sorted)
    return float(np.max(np.abs(current_cdf - reference_cdf)))


def check_drift(frame: pd.DataFrame, baseline: dict) -> dict:
    """Compare current features with a baseline and classify overall health."""
    rows: dict[str, dict] = {}
    missing: list[str] = []
    for name, feature in (baseline.get("features", {}) or {}).items():
        if name not in frame.columns:
            missing.append(str(name))
            continue
        current = _finite(frame[name])
        if len(current) < 10:
            missing.append(str(name))
            continue
        psi = _psi(current, feature)
        ks = _ks_stat(current, feature.get("reference_quantiles", []))
        rows[str(name)] = {
            "psi": psi,
            "ks_stat": ks,
            "sample_count": int(len(current)),
            "status": (
                "drift" if psi >= DRIFT_PSI_ALERT or ks >= DRIFT_KS_ALERT
                else "caution" if psi >= DRIFT_PSI_ALERT * 0.5 or ks >= DRIFT_KS_ALERT * 0.5
                else "ok"
            ),
        }
    statuses = {row["status"] for row in rows.values()}
    status = "drift" if "drift" in statuses else "caution" if "caution" in statuses else "ok"
    if not rows:
        status = "no_data"
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "feature_count": len(rows),
        "missing_features": missing,
        "features": rows,
        "thresholds": {"psi": DRIFT_PSI_ALERT, "ks": DRIFT_KS_ALERT},
    }


def build_feature_frame(
    features: Iterable[str],
    *,
    tickers: Iterable[str] = WATCHLIST,
    lookback_rows: int | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read requested feature columns from local ticker parquets."""
    requested = list(dict.fromkeys(str(name) for name in features))
    # Saved metadata uses ISO dates.  Converting both sides to UTC lets this
    # filter work with either timezone-aware or ordinary date indexes.
    cutoff = pd.to_datetime(end_date, utc=True) if end_date is not None else None
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = Path(DATA_DIR) / f"{str(ticker).upper()}.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path, columns=None)
        if cutoff is not None:
            row_dates = pd.to_datetime(data.index, errors="coerce", utc=True)
            data = data.loc[row_dates <= cutoff]
        present = [name for name in requested if name in data.columns]
        if not present:
            continue
        part = data[present]
        if lookback_rows is not None:
            part = part.tail(max(1, int(lookback_rows)))
        frames.append(part.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=requested)


def snapshot_from_metadata(
    run_name: str,
    metadata: dict,
    *,
    tickers: Iterable[str] = WATCHLIST,
    output_dir: str | Path = MODEL_DIR,
) -> Path:
    """Create the training-time baseline associated with one registered model."""
    features = metadata.get("feature_cols_raw") or metadata.get("feature_cols") or []
    training_data_end = metadata.get("training_data_end")
    frame = build_feature_frame(
        features,
        tickers=tickers,
        lookback_rows=None,
        end_date=training_data_end,
    )
    # Training may deliberately write a challenger into an isolated shadow
    # directory. Keep its baseline beside that model instead of accidentally
    # overwriting the production monitor baseline.
    output = Path(output_dir) / f"{run_name}_drift_baseline.json"
    snapshot_baseline(
        frame,
        output_path=output,
        metadata={
            "run_name": run_name,
            "model_version": metadata.get("model_version"),
            "prediction_target": metadata.get("prediction_target"),
            "tickers": [str(ticker) for ticker in tickers],
            # This proves the baseline excludes calibration, test, and newer
            # rows instead of silently learning what later data looks like.
            "training_data_end": training_data_end,
        },
    )
    return output


def _append_json_line(path: Path, payload: dict) -> None:
    """Atomically append a monitor result so a crash cannot leave half a line."""
    prior = path.read_text(encoding="utf-8") if path.exists() else ""
    text = prior + ("" if not prior or prior.endswith("\n") else "\n") + json.dumps(payload) + "\n"
    atomic_write_text(path, text)


def main() -> int:
    """Snapshot a baseline or compare the recent market with an existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="pooled")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--lookback-rows", type=int, default=60)
    args = parser.parse_args()
    baseline_path = args.baseline or Path(MODEL_DIR) / f"{args.run_name}_drift_baseline.json"
    if args.snapshot:
        raise SystemExit("Training creates baselines automatically from in-memory metadata")
    if not baseline_path.exists():
        result = {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "no_data",
            "reason": f"baseline_missing:{baseline_path}",
            "features": {},
        }
        _append_json_line(DEFAULT_LOG_PATH, result)
        print(result["reason"])
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    frame = build_feature_frame(baseline.get("features", {}).keys(), lookback_rows=args.lookback_rows)
    result = check_drift(frame, baseline)
    result["baseline_path"] = str(baseline_path)
    _append_json_line(DEFAULT_LOG_PATH, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
