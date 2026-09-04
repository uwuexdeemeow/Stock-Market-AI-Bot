"""
shadow_paper_journal.py - record a paper-only signal for a shadow config.

PLAIN ENGLISH:
This script lets us watch a candidate live config without sending orders.
It temporarily points the normal core-satellite signal generator at a shadow
walkforward payload, captures the generated target weights, appends one row to
``signals/shadow_paper_journal.csv``, then restores the real live signal files.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import core_satellite_alpha as csa
import core_satellite_nested_walkforward as nested
from safe_io import atomic_write_csv, atomic_write_json, configure_console_output
from settings import DATA_DIR, SIGNAL_DIR
from validation_bundle import (
    build_validation_bundle,
    strategy_config_fingerprint,
    validate_validation_bundle,
    write_validation_bundle,
)


configure_console_output()

SHADOW_NAME = "riskoff_guard"
SHADOW_JOURNAL_PATH = Path(SIGNAL_DIR) / "shadow_paper_journal.csv"
SHADOW_EQUITY_PATH = Path(SIGNAL_DIR) / "shadow_paper_equity.csv"
SHADOW_LIVE_CONFIG_PATH = Path(SIGNAL_DIR) / "shadow_core_satellite_live_configs.json"
SHADOW_VALIDATION_EVIDENCE_PATH = Path(SIGNAL_DIR) / "shadow_validation_evidence.json"
SHADOW_VALIDATION_BUNDLE_PATH = Path(SIGNAL_DIR) / "shadow_core_satellite_validation_bundle.json"
DEFAULT_SHADOW_INITIAL_EQUITY = 100_000.0

# This is the candidate we validated but did NOT promote to paper trading.
SHADOW_CONFIG_SIGNATURE = (
    "h=20,ov=0.5,ma=100,vol=percentile:0.3,"
    "score=regime_adaptive_riskoff_guard,shape=top3,weighting=sticky_score,tqqq=0.0,risk=off"
)

# Fresh validation source: signals/wf_autoresearch_riskguard_full_20260524.json
# Keep these numbers in the shadow payload so the normal live gates can inspect
# the same approval-style fields they inspect for the current paper config.
SHADOW_SOURCE_METRICS = {
    "fold_count": 14,
    "best_config_frequency": 1.0,
    "approved_family_fold_count": 14,
    "approved_family_frequency": 1.0,
    "approved_family_worst_oos_turnover_pct": 533.29,
    "approved_family_mean_oos_max_drawdown_pct": -10.71,
    "approved_family_mean_oos_sharpe": 1.667,
    "mean_oos_sharpe": 1.667,
    "mean_oos_cagr_pct": 38.93,
    "mean_oos_alpha_vs_spy_pct": 22.86,
    "mean_oos_alpha_vs_qqq_pct": 15.82,
    "oos_positive_alpha_hit_rate": 0.857,
    "cost_stress_approval_pass": True,
    "fixed_cost_stress_pass_ratio": 0.857,
    "required_cost_stresses": [2.0, 3.0, 5.0],
    "mean_oos_max_drawdown_pct": -10.71,
    "worst_oos_max_drawdown_pct": -27.56,
    "worst_oos_turnover_pct": 533.29,
    "worst_oos_return_pct": -11.13,
    "selection_bias_gap_sharpe": 0.0,
    "approved_config_fold_count": 14,
    "approved_config_frequency": 1.0,
    "medium_risk_review_pass": True,
}


def _utc_now_text() -> str:
    """Return a compact UTC timestamp for journal/audit fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _shadow_initial_equity() -> float:
    """Read the starting fake account value for the shadow equity curve."""
    raw = os.environ.get("SHADOW_PAPER_INITIAL_EQUITY", str(DEFAULT_SHADOW_INITIAL_EQUITY))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = DEFAULT_SHADOW_INITIAL_EQUITY
    return value if value > 0 else DEFAULT_SHADOW_INITIAL_EQUITY


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert CSV/JSON values to floats while treating blanks as zero."""
    try:
        if value is None or pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_weights(value: Any) -> dict[str, float]:
    """Parse a JSON weight map like {"MU": 0.2} into clean floats."""
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    weights: dict[str, float] = {}
    for ticker, weight in parsed.items():
        ticker_text = str(ticker).upper().strip()
        weight_value = _as_float(weight)
        if ticker_text and abs(weight_value) > 1e-12:
            weights[ticker_text] = weights.get(ticker_text, 0.0) + weight_value
    return weights


def _target_weights_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Return the portfolio weights the shadow config wants to hold next."""
    weights: dict[str, float] = {}
    for ticker, field in (
        ("SPY", "target_spy_weight"),
        ("QQQ", "target_qqq_weight"),
        ("TQQQ", "target_tqqq_weight"),
    ):
        weight = _as_float(row.get(field))
        if abs(weight) > 1e-12:
            weights[ticker] = weights.get(ticker, 0.0) + weight

    # Overlay weights are the individual stock picks.  They arrive as JSON in
    # the journal row, so beginners can inspect the CSV without extra files.
    for ticker, weight in _json_weights(row.get("overlay_weights_json")).items():
        weights[ticker] = weights.get(ticker, 0.0) + weight

    return {ticker: round(float(weight), 10) for ticker, weight in sorted(weights.items())}


def _row_price_date(row: dict[str, Any]) -> str:
    """Use the latest completed market-data date, falling back to run_date."""
    for field in ("latest_factor_date", "run_date"):
        value = row.get(field)
        if value not in (None, ""):
            ts = pd.Timestamp(value)
            if not pd.isna(ts):
                return ts.date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def _close_on_or_before(ticker: str, date_text: str, *, data_dir: Path) -> tuple[str, float] | None:
    """Find the latest available close at or before a date for one ticker."""
    path = Path(data_dir) / f"{ticker.upper()}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.empty or "Close" not in frame.columns:
        return None

    close_raw = frame["Close"]
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0] if close_raw.shape[1] else pd.Series(dtype=float)
    close = pd.to_numeric(close_raw, errors="coerce")
    close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None).normalize()
    close = close.dropna()
    close = close[close > 0]
    if close.empty:
        return None

    target_date = pd.Timestamp(date_text).normalize()
    eligible = close[close.index <= target_date]
    if eligible.empty:
        return None
    actual_date = pd.Timestamp(eligible.index[-1]).date().isoformat()
    return actual_date, float(eligible.iloc[-1])


def _portfolio_return(
    weights: dict[str, float],
    start_date: str,
    end_date: str,
    *,
    data_dir: Path,
) -> tuple[float, str]:
    """Estimate one period of shadow P&L from prior weights and close prices."""
    if not weights:
        return 0.0, "no_prior_weights"
    if pd.Timestamp(start_date).normalize() >= pd.Timestamp(end_date).normalize():
        return 0.0, "same_price_date"

    total_return = 0.0
    missing: list[str] = []
    for ticker, weight in weights.items():
        start_close = _close_on_or_before(ticker, start_date, data_dir=data_dir)
        end_close = _close_on_or_before(ticker, end_date, data_dir=data_dir)
        if start_close is None or end_close is None:
            missing.append(ticker)
            continue
        ticker_return = (end_close[1] / start_close[1]) - 1.0
        total_return += float(weight) * float(ticker_return)

    if missing:
        return total_return, "partial_missing:" + ",".join(sorted(missing))
    return total_return, "ok"


def _shadow_candidate_config() -> dict[str, Any]:
    """Build the exact candidate config through the production grid helper."""
    configs = nested.iter_candidate_configs(
        strategy="core-alpha",
        holding_days=(20,),
        overlay_gross=(0.50,),
        ma_windows=(100,),
        high_vol_values=(0.30,),
        high_vol_modes=("percentile",),
        score_sources=("regime_adaptive_riskoff_guard",),
        shapes=("top3",),
        weightings=("sticky_score",),
        tqqq_weights=(0.0,),
        risk_control_modes=("off",),
    )
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly one shadow config, got {len(configs)}")
    config = configs[0]
    signature = nested.config_signature(config)
    if signature != SHADOW_CONFIG_SIGNATURE:
        raise RuntimeError(f"Shadow config drifted: {signature} != {SHADOW_CONFIG_SIGNATURE}")
    return config


def _default_medium_risk_review(base_payload: dict[str, Any]) -> dict[str, Any]:
    """Reuse existing medium-risk review shape, falling back to pass for shadow tracking."""
    current = (
        ((base_payload.get("approved_live_configs", {}) or {}).get("core-alpha", {}) or {})
        .get("medium_risk_review")
    )
    if isinstance(current, dict) and current.get("pass") is True:
        review = deepcopy(current)
        review["note"] = "inherited current medium-risk review shape for shadow paper tracking"
        return review
    return {
        "pass": True,
        "reasons": [],
        "note": "shadow paper tracking only; fixed validation passed live approval gates",
    }


def build_shadow_live_payload(
    base_payload: dict[str, Any],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a temporary live-config JSON payload for the shadow candidate."""
    candidate = _shadow_candidate_config()
    family = nested.stable_family_signature(candidate)
    created = created_at or _utc_now_text()
    thresholds = (
        ((base_payload.get("approvals", {}) or {}).get("core-alpha", {}) or {})
        .get("thresholds")
        or nested._APPROVAL_THRESHOLDS["core-alpha"]
    )
    approval = {
        "approved": True,
        "reasons": [],
        "thresholds": deepcopy(thresholds),
        "warnings": [],
        "strategy": "core-alpha",
        "approved_config_family": family,
        "approved_family_signature": family,
        "approved_family_fold_count": 14,
        "approved_family_frequency": 1.0,
        "approved_exact_config": SHADOW_CONFIG_SIGNATURE,
        "approved_config_fold_count": 14,
        "approved_config_frequency": 1.0,
        "source": "shadow_fixed_config_validation",
        "created_at": created,
    }
    medium_review = _default_medium_risk_review(base_payload)
    approved_live = {
        "strategy": "core-alpha",
        "approved_config_family": family,
        "approved_family_signature": family,
        "approved_exact_config": SHADOW_CONFIG_SIGNATURE,
        "config": nested.live_signal_config(candidate),
        "source_metrics": deepcopy(SHADOW_SOURCE_METRICS),
        "medium_risk_review": medium_review,
    }
    return {
        "created_at": created,
        "source_json": "signals/wf_autoresearch_riskguard_full_20260524.json",
        "method": "shadow_fixed_config_outer_walkforward_validation",
        "shadow": True,
        "shadow_name": SHADOW_NAME,
        "approvals": {"core-alpha": approval},
        "approved_live_configs": {"core-alpha": approved_live},
        "medium_risk_reviews": {"core-alpha": medium_review},
    }


def write_shadow_validation_bundle(
    shadow_payload: dict[str, Any],
    *,
    evidence_path: Path = SHADOW_VALIDATION_EVIDENCE_PATH,
    bundle_path: Path = SHADOW_VALIDATION_BUNDLE_PATH,
) -> dict:
    """Write temporary paper-only evidence for the temporary shadow config.

    PLAIN ENGLISH: The normal signal builder refuses to use a live config
    unless it points to a matching evidence bundle. The shadow config is a
    different, deliberately unpromoted config, so it needs its own temporary
    bundle. This never changes the real config or approves real-money trading.
    """
    approved = (shadow_payload.get("approved_live_configs", {}) or {}).get("core-alpha", {})
    approval = (shadow_payload.get("approvals", {}) or {}).get("core-alpha", {})
    evidence = {
        "shadow": True,
        "shadow_name": SHADOW_NAME,
        "shadow_config_signature": SHADOW_CONFIG_SIGNATURE,
        "config": approved.get("config", {}),
        "source_metrics": approved.get("source_metrics", {}),
        "note": "Temporary paper-only evidence for shadow tracking; not real-capital approval.",
    }
    atomic_write_json(evidence, evidence_path)
    bundle = build_validation_bundle(
        {
            "strategy": "core-alpha",
            "folds": [{"kind": "shadow_fixed_config_validation"}],
            "live_config_approval": approval,
            "approved_live_config": approved,
        },
        source_json=str(evidence_path),
        # This shadow path records hypothetical output only. It cannot submit
        # orders, so it does not borrow the production strategy's reports.
        report_paths={},
        require_robustness_reports=False,
    )
    write_validation_bundle(bundle, bundle_path)
    return bundle


def _snapshot_paths(paths: list[Path]) -> dict[Path, bytes | None]:
    """Remember file bytes so the shadow run can put normal outputs back."""
    snapshots: dict[Path, bytes | None] = {}
    for path in paths:
        snapshots[path] = path.read_bytes() if path.exists() else None
    return snapshots


def _restore_paths(snapshots: dict[Path, bytes | None]) -> None:
    """Restore files that the normal signal generator overwrote for shadow use."""
    for path, content in snapshots.items():
        if content is None:
            path.unlink(missing_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _first_signal_row(signal_path: Path) -> dict[str, Any]:
    """Read the single-row signal CSV written by core_satellite_alpha.py."""
    if not signal_path.exists():
        raise FileNotFoundError(f"Shadow signal missing: {signal_path}")
    rows = pd.read_csv(signal_path).to_dict(orient="records")
    if not rows:
        raise RuntimeError(f"Shadow signal empty: {signal_path}")
    return rows[0]


def _journal_row(
    signal: dict[str, Any],
    metrics: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten the generated signal into one append-only journal row."""
    validation = validation or {}
    return {
        "journaled_at": _utc_now_text(),
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "shadow_name": SHADOW_NAME,
        "shadow_config_signature": SHADOW_CONFIG_SIGNATURE,
        "paper_ready": signal.get("paper_ready"),
        "gates_all_pass": signal.get("gates_all_pass"),
        "reason": signal.get("reason"),
        "current_regime": signal.get("current_regime"),
        "latest_factor_date": signal.get("latest_factor_date"),
        "predicted_at": signal.get("predicted_at"),
        "gross_exposure": signal.get("gross_exposure"),
        "core_gross": signal.get("core_gross"),
        "overlay_gross": signal.get("overlay_gross"),
        "raw_overlay_gross": signal.get("raw_overlay_gross"),
        "target_spy_weight": signal.get("target_spy_weight"),
        "target_qqq_weight": signal.get("target_qqq_weight"),
        "target_tqqq_weight": signal.get("target_tqqq_weight"),
        "target_cash_weight": signal.get("target_cash_weight"),
        "overlay_tickers": signal.get("overlay_tickers"),
        "overlay_weights_json": signal.get("overlay_weights_json"),
        "raw_overlay_weights_json": signal.get("raw_overlay_weights_json"),
        "sticky_holdings_source": signal.get("sticky_holdings_source"),
        "sticky_holdings_used": signal.get("sticky_holdings_used"),
        "sticky_held_tickers": signal.get("sticky_held_tickers"),
        "live_config_hash": signal.get("live_config_hash"),
        "walkforward_approval_pass": signal.get("walkforward_approval_pass"),
        "nested_cost_stress_approval_pass": signal.get("nested_cost_stress_approval_pass"),
        "medium_risk_review_pass": signal.get("medium_risk_review_pass"),
        "feature_health_gate_pass": signal.get("feature_health_gate_pass"),
        "source_mean_oos_alpha_vs_qqq_pct": SHADOW_SOURCE_METRICS["mean_oos_alpha_vs_qqq_pct"],
        "source_mean_oos_sharpe": SHADOW_SOURCE_METRICS["mean_oos_sharpe"],
        "source_worst_oos_drawdown_pct": SHADOW_SOURCE_METRICS["worst_oos_max_drawdown_pct"],
        "source_worst_oos_turnover_pct": SHADOW_SOURCE_METRICS["worst_oos_turnover_pct"],
        "backtest_sharpe": metrics.get("sharpe"),
        "backtest_max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "backtest_total_return_pct": metrics.get("total_return_pct"),
        "validation_bundle_valid": bool(validation.get("bundle_valid", False)),
        "validation_fingerprint_match": bool(validation.get("fingerprint_match", False)),
        "validation_config_fingerprint": validation.get("config_fingerprint", ""),
        "validation_issues": ",".join(str(item) for item in validation.get("issues", []) or []),
    }


def append_shadow_journal(
    row: dict[str, Any],
    *,
    journal_path: Path = SHADOW_JOURNAL_PATH,
    replace_today: bool = True,
) -> Path:
    """Append the row, replacing today's prior shadow row by default."""
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if journal_path.exists():
        # GitHub Actions can restore a zero-byte journal on first run.  Treat
        # that the same as "no history yet" instead of crashing pandas.
        if journal_path.stat().st_size == 0:
            existing = pd.DataFrame()
        else:
            try:
                existing = pd.read_csv(journal_path)
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
        if replace_today and {"run_date", "shadow_config_signature"}.issubset(existing.columns):
            mask = (
                (existing["run_date"].astype(str) == str(row["run_date"]))
                & (existing["shadow_config_signature"].astype(str) == str(row["shadow_config_signature"]))
            )
            existing = existing.loc[~mask].copy()
        out = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        out = new_row
    atomic_write_csv(out, journal_path, index=False)
    return journal_path


def update_shadow_equity(
    row: dict[str, Any],
    *,
    equity_path: Path = SHADOW_EQUITY_PATH,
    data_dir: Path = Path(DATA_DIR),
    replace_today: bool = True,
    initial_equity: float | None = None,
) -> Path:
    """Append one simulated equity row for the shadow paper portfolio.

    PLAIN ENGLISH: The shadow journal says what we *would* hold.  This function
    turns yesterday's shadow holdings into today's fake account value, so it can
    be compared with the real Alpaca paper equity file.
    """
    equity_path = Path(equity_path)
    equity_path.parent.mkdir(parents=True, exist_ok=True)
    current_date = str(row.get("run_date") or datetime.now(timezone.utc).date().isoformat())
    signature = str(row.get("shadow_config_signature") or SHADOW_CONFIG_SIGNATURE)
    target_weights = _target_weights_from_row(row)
    price_date = _row_price_date(row)
    starting_equity = float(initial_equity or _shadow_initial_equity())

    existing = pd.DataFrame()
    if equity_path.exists() and equity_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(equity_path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()

    if not existing.empty:
        existing["date"] = existing["date"].astype(str)
        if "timestamp" not in existing.columns:
            existing["timestamp"] = existing["date"]
        if "shadow_config_signature" not in existing.columns:
            existing["shadow_config_signature"] = signature
        if replace_today:
            same_row = (
                (existing["date"] == current_date)
                & (existing["shadow_config_signature"].astype(str) == signature)
            )
            existing = existing.loc[~same_row].copy()

    prior = pd.DataFrame()
    if not existing.empty:
        same_shadow = existing["shadow_config_signature"].astype(str) == signature
        prior = existing.loc[same_shadow & (existing["date"].astype(str) < current_date)].copy()
        if not prior.empty:
            prior = prior.sort_values(["date", "timestamp"], kind="stable")

    if prior.empty:
        base_equity = starting_equity
        applied_weights: dict[str, float] = {}
        price_from_date = ""
        period_return, price_status = 0.0, "initialized"
        initial = starting_equity
    else:
        last = prior.iloc[-1].to_dict()
        base_equity = _as_float(last.get("equity"), starting_equity)
        initial = _as_float(last.get("initial_equity"), starting_equity)
        price_from_date = str(last.get("price_to_date") or last.get("price_date") or "")
        applied_weights = _json_weights(last.get("target_weights_json"))
        period_return, price_status = _portfolio_return(
            applied_weights,
            price_from_date,
            price_date,
            data_dir=Path(data_dir),
        )

    equity = base_equity * (1.0 + period_return)
    out_row = {
        "date": current_date,
        "timestamp": row.get("journaled_at") or _utc_now_text(),
        "shadow_name": row.get("shadow_name") or SHADOW_NAME,
        "shadow_config_signature": signature,
        "equity": round(equity, 2),
        "initial_equity": round(initial, 2),
        "period_return_pct": round(period_return * 100.0, 4),
        "total_return_pct": round(((equity / initial) - 1.0) * 100.0, 4) if initial else 0.0,
        "price_from_date": price_from_date,
        "price_to_date": price_date,
        "price_status": price_status,
        "applied_weights_json": json.dumps(applied_weights, sort_keys=True),
        "target_weights_json": json.dumps(target_weights, sort_keys=True),
        "target_cash_weight": round(_as_float(row.get("target_cash_weight")), 10),
        "gross_exposure": row.get("gross_exposure"),
        "paper_ready": row.get("paper_ready"),
        "gates_all_pass": row.get("gates_all_pass"),
        "reason": row.get("reason"),
    }

    out = pd.concat([existing, pd.DataFrame([out_row])], ignore_index=True, sort=False)
    out = out.sort_values(["date", "timestamp"], kind="stable")
    atomic_write_csv(out, equity_path, index=False)
    return equity_path


def run_shadow_journal(
    *,
    journal_path: Path = SHADOW_JOURNAL_PATH,
    equity_path: Path = SHADOW_EQUITY_PATH,
    restore_signal_artifacts: bool = True,
    replace_today: bool = True,
    ignore_stale: bool = False,
) -> Path:
    """Generate the shadow signal, journal it, and keep live files untouched."""
    signal_dir = Path(SIGNAL_DIR)
    live_config_path = signal_dir / "core_satellite_live_configs.json"
    base_payload = json.loads(live_config_path.read_text(encoding="utf-8"))
    shadow_payload = build_shadow_live_payload(base_payload)
    shadow_evidence_path = signal_dir / SHADOW_VALIDATION_EVIDENCE_PATH.name
    shadow_bundle_path = signal_dir / SHADOW_VALIDATION_BUNDLE_PATH.name
    shadow_bundle = write_shadow_validation_bundle(
        shadow_payload,
        evidence_path=shadow_evidence_path,
        bundle_path=shadow_bundle_path,
    )
    bundle_valid, bundle_issues = validate_validation_bundle(shadow_bundle)
    expected_fingerprint = strategy_config_fingerprint(
        (shadow_payload.get("approved_live_configs", {}).get("core-alpha", {}) or {}).get("config", {})
    )
    observed_fingerprint = str(shadow_bundle.get("config_fingerprint") or "")
    fingerprint_match = bool(observed_fingerprint and observed_fingerprint == expected_fingerprint)
    if not bundle_valid or not fingerprint_match:
        raise RuntimeError(
            "Shadow validation bundle failed: "
            + ",".join([*bundle_issues, "config_fingerprint_mismatch" if not fingerprint_match else ""])
        )
    validation_context = {
        "bundle_valid": bundle_valid,
        "fingerprint_match": fingerprint_match,
        "config_fingerprint": observed_fingerprint,
        "issues": bundle_issues,
    }
    shadow_payload["validation_bundle_path"] = str(shadow_bundle_path)
    shadow_payload["validation_bundle_hash"] = shadow_bundle["validation_bundle_hash"]
    shadow_payload["deployment_status"] = shadow_bundle["deployment"]["status"]
    shadow_payload["paper_approved"] = bool(shadow_bundle["deployment"]["paper_approved"])
    shadow_payload["real_capital_approved"] = False
    atomic_write_json(shadow_payload, SHADOW_LIVE_CONFIG_PATH)

    touched = [
        signal_dir / "core_satellite_alpha_signal.csv",
        signal_dir / "core_satellite_alpha_metrics.json",
        signal_dir / "core_satellite_alpha_equity.csv",
        signal_dir / "core_satellite_alpha_trades.csv",
    ]
    snapshots = _snapshot_paths(touched)
    original_live_path = csa.LIVE_CONFIG_PATH

    try:
        csa.LIVE_CONFIG_PATH = SHADOW_LIVE_CONFIG_PATH

        # The next block mirrors core_satellite_alpha.main() but swaps only the
        # live-config payload.  That means the shadow signal uses the same
        # features, freshness gates, sticky holdings, and signal writer.
        csa.validate_sector_map_coverage()
        quality_filter = csa._load_feature_quality_filter(strict=True)
        specs = csa._apply_live_feature_quality_filter(csa.load_feature_specs(), quality_filter)
        ml_scores = csa.load_prediction_scores()
        panel = csa._ensure_robust_score_columns(
            csa.attach_scores(csa.load_factor_panel(specs), specs, ml_scores)
        )
        signal_panel = csa._ensure_robust_score_columns(
            csa.attach_scores(csa.load_factor_panel(specs, require_forward_returns=False), specs, ml_scores)
        )
        csa._validate_live_feature_inputs(specs, signal_panel)
        freshness = csa.check_factor_freshness(signal_panel, ignore_stale=ignore_stale)
        if freshness["blocked"]:
            raise SystemExit(f"Aborting shadow journal: {freshness['message']}")

        _summary, metrics, signal_path = csa._generate_signal_from_approved_config(
            panel=panel,
            signal_panel=signal_panel,
            specs=specs,
            freshness=freshness,
            # PLAIN ENGLISH: the temporary shadow bundle proves its own config
            # identity above, but deliberately has no production robustness
            # reports because this script cannot submit orders. Use the
            # existing research-only path so the normal broker gate stays strict.
            allow_provisional_bundle=True,
        )
        signal = _first_signal_row(signal_path)
        journal_row = _journal_row(signal, metrics, validation_context)
        out_path = append_shadow_journal(
            journal_row,
            journal_path=journal_path,
            replace_today=replace_today,
        )
        update_shadow_equity(
            journal_row,
            equity_path=equity_path,
            replace_today=replace_today,
        )
        return out_path
    finally:
        csa.LIVE_CONFIG_PATH = original_live_path
        SHADOW_LIVE_CONFIG_PATH.unlink(missing_ok=True)
        shadow_evidence_path.unlink(missing_ok=True)
        shadow_bundle_path.unlink(missing_ok=True)
        if restore_signal_artifacts:
            _restore_paths(snapshots)


def main() -> None:
    """CLI entry point for GitHub Actions and manual local shadow runs."""
    parser = argparse.ArgumentParser(description="Append today's shadow config signal to a paper journal.")
    parser.add_argument("--journal-path", default=str(SHADOW_JOURNAL_PATH))
    parser.add_argument("--equity-path", default=str(SHADOW_EQUITY_PATH))
    parser.add_argument("--ignore-stale", action="store_true", help="Allow stale factor data for manual debugging.")
    parser.add_argument("--append-duplicate", action="store_true", help="Keep duplicate same-day rows.")
    parser.add_argument(
        "--no-restore-signal-artifacts",
        action="store_true",
        help="Leave generated shadow signal files in signals/ for debugging.",
    )
    args = parser.parse_args()

    path = run_shadow_journal(
        journal_path=Path(args.journal_path),
        equity_path=Path(args.equity_path),
        restore_signal_artifacts=not args.no_restore_signal_artifacts,
        replace_today=not args.append_duplicate,
        ignore_stale=bool(args.ignore_stale),
    )
    print(f"Shadow paper journal updated: {path}")
    print(f"Shadow paper equity updated: {Path(args.equity_path)}")
    print(f"Shadow config: {SHADOW_CONFIG_SIGNATURE}")


if __name__ == "__main__":
    main()
