"""Build and verify the single validation record used by paper trading.

PLAIN ENGLISH: A strategy is not proven by one JSON file. It also depends on
the exact configuration, price data, source code, analyzer results, and stress
reports. This module packs those facts into one checksummed bundle so the live
bot cannot accidentally combine results from different research runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json, atomic_write_text
from universe_membership import membership_status
from robustness_review import (
    DEFAULT_ROBUSTNESS_REPORT_PATHS,
    evaluate_medium_risk_review,
    read_report,
)


DEFAULT_BUNDLE_PATH = Path("signals/core_satellite_validation_bundle.json")
DEFAULT_LIVE_CONFIG_PATH = Path("signals/core_satellite_live_configs.json")
DEFAULT_RESEARCH_MANIFEST_PATH = Path("signals/research_run_manifest.json")
DEFAULT_CANONICAL_WALKFORWARD_PATH = Path("signals/core_satellite_nested_walkforward.json")
DEFAULT_REPORT_PATHS = {
    **DEFAULT_ROBUSTNESS_REPORT_PATHS,
}

# Factor decay describes a changing market edge, so it must be refreshed more
# often than the expensive structural stress tests.
REPORT_MAX_AGE_DAYS = {
    "survivorship": 60,
    "execution_stress": 60,
    "factor_decay": 7,
}

# Only fields that change trading behavior belong in the strategy identity.
# Extra display fields must not make equivalent configurations hash differently.
CONFIG_IDENTITY_FIELDS = (
    "score_source",
    "shape",
    "weighting",
    "holding_days",
    "overlay_gross",
    "regime_ma_window",
    "regime_high_vol",
    "high_vol_mode",
    "tqqq_weight",
    "risk_control_mode",
    "deployment_max_gross_exposure",
)


def _canonical_json(value: Any) -> str:
    """Return stable JSON text so equal values always have the same checksum."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_value(value: Any) -> str:
    """Return the full SHA-256 fingerprint for a JSON-compatible value."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strategy_config_identity(config: dict | None) -> dict:
    """Extract the behavior-changing fields used to compare two strategies."""
    config = dict(config or {})
    defaults = {
        "tqqq_weight": 0.0,
        "risk_control_mode": "off",
        "high_vol_mode": "fixed",
        # Older research configs did not distinguish a deployment ceiling.
        # None means no additional broker-style scaling was simulated.
        "deployment_max_gross_exposure": None,
    }
    return {
        field: config.get(field, defaults.get(field))
        for field in CONFIG_IDENTITY_FIELDS
    }


def strategy_config_fingerprint(config: dict | None) -> str:
    """Return a short readable checksum for the trading configuration."""
    return sha256_value(strategy_config_identity(config))[:16]


def _git_commit() -> str:
    """Return the checked-out commit, or an empty string outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def load_dataset_context(path: Path = DEFAULT_RESEARCH_MANIFEST_PATH) -> dict:
    """Read the research manifest that identifies the parquet input snapshot."""
    if not path.exists():
        return {
            "manifest_path": str(path),
            "manifest_exists": False,
            "dataset_fingerprint": "",
            "reason": "research_manifest_missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest_path": str(path),
            "manifest_exists": True,
            "dataset_fingerprint": "",
            "reason": f"research_manifest_invalid:{exc.__class__.__name__}",
        }
    input_data = payload.get("input_data", {}) or {}
    return {
        "manifest_path": str(path),
        "manifest_exists": True,
        "manifest_sha256": file_sha256(path),
        "dataset_fingerprint": str(input_data.get("combined_sha256", "")),
        "file_count": int(input_data.get("file_count", 0) or 0),
        "fingerprinted_count": int(input_data.get("fingerprinted_count", 0) or 0),
        "generated_at": payload.get("generated_at_utc"),
        "reason": "" if input_data.get("combined_sha256") else "dataset_fingerprint_missing",
    }


def report_validation_record(
    name: str,
    path: Path,
    *,
    expected_config_fingerprint: str,
    expected_dataset_fingerprint: str,
    now: datetime | None = None,
) -> dict:
    """Explain whether one robustness report belongs to this validation run."""
    record = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "match": False,
        "reasons": [],
    }
    if not path.exists():
        record["reasons"].append("missing_report")
        return record
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record["reasons"].append(f"invalid_report:{exc.__class__.__name__}")
        return record

    context = payload.get("validation_context", {}) or {}
    selected_config = payload.get("selected_config")
    observed_config = str(context.get("config_fingerprint", ""))
    if not observed_config and isinstance(selected_config, dict):
        observed_config = strategy_config_fingerprint(selected_config)
    observed_dataset = str(context.get("dataset_fingerprint", ""))

    generated_at = payload.get("generated_at") or payload.get("generated_at_utc")
    record.update({
        "sha256": file_sha256(path),
        "generated_at": generated_at,
        "observed_config_fingerprint": observed_config,
        "observed_dataset_fingerprint": observed_dataset,
    })
    if not observed_config:
        record["reasons"].append("config_fingerprint_missing")
    elif observed_config != expected_config_fingerprint:
        record["reasons"].append("config_fingerprint_mismatch")
    if not observed_dataset:
        record["reasons"].append("dataset_fingerprint_missing")
    elif observed_dataset != expected_dataset_fingerprint:
        record["reasons"].append("dataset_fingerprint_mismatch")
    # PLAIN ENGLISH: matching fingerprints say "same experiment", while this
    # age check says the experiment is still current enough to trade from.
    generated_ts = pd.to_datetime(generated_at, errors="coerce", utc=True)
    if pd.isna(generated_ts):
        record["reasons"].append("generated_at_missing_or_invalid")
    else:
        clock = pd.Timestamp(now or datetime.now(timezone.utc))
        max_age_days = int(REPORT_MAX_AGE_DAYS.get(name, 60))
        age_days = max(0.0, float((clock - generated_ts).total_seconds()) / 86_400.0)
        record["age_days"] = round(age_days, 3)
        record["max_age_days"] = max_age_days
        if age_days > max_age_days:
            record["reasons"].append("report_stale")
    record["match"] = not record["reasons"]
    return record


def current_robustness_evidence(
    *,
    expected_config_fingerprint: str,
    expected_dataset_fingerprint: str,
    report_paths: dict[str, Path] | None = None,
    require_reports: bool = True,
    now: datetime | None = None,
) -> dict:
    """Load, identify, and health-check the reports that exist right now."""
    paths = DEFAULT_REPORT_PATHS if report_paths is None else report_paths
    if not paths and not require_reports:
        return {
            "pass": True,
            "reasons": [],
            "reports": {},
            "medium_risk_review": {"pass": True, "reasons": [], "not_required": True},
        }

    records = {
        name: report_validation_record(
            name,
            Path(path),
            expected_config_fingerprint=expected_config_fingerprint,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            now=now,
        )
        for name, path in paths.items()
    }
    payloads = {name: read_report(Path(path)) for name, path in paths.items()}
    review = evaluate_medium_risk_review(
        survivorship=payloads.get("survivorship"),
        execution=payloads.get("execution_stress"),
        factor_decay=payloads.get("factor_decay"),
    )
    health_keys = {
        "survivorship": "survivorship_review",
        "execution_stress": "execution_stress_review",
        "factor_decay": "factor_decay_review",
    }
    for name, record in records.items():
        record["health"] = dict(review.get(health_keys.get(name, ""), {}) or {})

    reasons = [
        f"{name}:{reason}"
        for name, record in records.items()
        for reason in record.get("reasons", [])
    ]
    reasons.extend(str(reason) for reason in review.get("reasons", []))
    if require_reports:
        for required in DEFAULT_REPORT_PATHS:
            if required not in records:
                reasons.append(f"{required}:missing_report_record")
    return {
        "pass": not reasons and bool(review.get("pass", False)),
        "reasons": sorted(set(reasons)),
        "reports": records,
        "medium_risk_review": review,
    }


def add_validation_context(
    payload: dict,
    *,
    config: dict | None,
    dataset_context: dict | None = None,
) -> dict:
    """Stamp a robustness report with the strategy and dataset it evaluated."""
    out = dict(payload)
    dataset_context = dataset_context or load_dataset_context()
    out["validation_context"] = {
        "config_fingerprint": strategy_config_fingerprint(config),
        "config_identity": strategy_config_identity(config),
        "dataset_fingerprint": dataset_context.get("dataset_fingerprint", ""),
        "research_manifest_path": dataset_context.get("manifest_path"),
        "git_commit": _git_commit(),
    }
    return out


def build_validation_bundle(
    result: dict,
    *,
    source_json: str,
    analyzer: dict | None = None,
    report_paths: dict[str, Path] | None = None,
    dataset_context: dict | None = None,
    require_robustness_reports: bool = True,
) -> dict:
    """Build the complete, checksummed strategy-validation bundle."""
    approved = result.get("approved_live_config", {}) or {}
    config = approved.get("config", {}) or {}
    config_fingerprint = strategy_config_fingerprint(config)
    dataset_context = dataset_context or load_dataset_context()
    dataset_fingerprint = str(dataset_context.get("dataset_fingerprint", ""))
    # An empty mapping is meaningful for an explicitly non-trading shadow.
    # Only ``None`` asks for the normal production report set.
    report_paths = DEFAULT_REPORT_PATHS if report_paths is None else report_paths
    robustness = current_robustness_evidence(
        expected_config_fingerprint=config_fingerprint,
        expected_dataset_fingerprint=dataset_fingerprint,
        report_paths=report_paths,
        require_reports=require_robustness_reports,
    )
    reports = robustness["reports"]
    # Normal paper trading must carry all matching robustness evidence. A
    # temporary shadow experiment can explicitly opt out because it never
    # submits orders and cannot authorize real capital.
    report_matches = bool(robustness.get("pass", False))
    base_approval = dict(result.get("live_config_approval", {}) or {})

    provisional_reasons: list[str] = []
    source_path = Path(source_json) if source_json else None
    source_is_file = bool(source_path and source_path.is_file())
    if not source_is_file:
        provisional_reasons.append("walkforward_source_missing")
    if not result.get("folds"):
        provisional_reasons.append("walkforward_folds_missing")
    if not dataset_fingerprint:
        provisional_reasons.append("dataset_fingerprint_missing")
    if require_robustness_reports:
        provisional_reasons.extend(robustness.get("reasons", []))
    universe = membership_status()
    if not universe.get("complete", False):
        provisional_reasons.append("point_in_time_universe_incomplete")
    survivorship_capital_pass = bool(
        ((robustness.get("medium_risk_review", {}) or {}).get("survivorship_review", {}) or {}).get(
            "capital_approval_pass", False
        )
    )
    if not survivorship_capital_pass:
        provisional_reasons.append("survivorship_capital_evidence_incomplete")

    # PLAIN ENGLISH: A profitable fold summary is not enough to authorize
    # paper orders.  The exact walk-forward file, dataset fingerprint, folds,
    # and all three matching robustness reports must exist.  Point-in-time
    # universe completeness remains a real-money blocker, but does not stop the
    # deliberately provisional paper experiment.
    paper_approved = bool(
        base_approval.get("approved", False)
        and config
        and source_is_file
        and result.get("folds")
        and dataset_fingerprint
        and report_matches
    )

    bundle = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": result.get("strategy", "core-alpha"),
        "source_json": str(source_json),
        "source_json_sha256": file_sha256(source_path) if source_is_file else "",
        "git_commit": _git_commit(),
        "config": config,
        "config_identity": strategy_config_identity(config),
        "config_fingerprint": config_fingerprint,
        "dataset": dataset_context,
        "universe_membership": universe,
        "folds": result.get("folds", []),
        "summary": {
            key: result.get(key)
            for key in (
                "fold_count", "mean_oos_cagr_pct", "mean_oos_sharpe",
                "mean_oos_alpha_vs_spy_pct", "mean_oos_alpha_vs_qqq_pct",
                "oos_positive_alpha_hit_rate", "mean_oos_turnover_pct",
                "worst_oos_turnover_pct", "cumulative_oos_turnover_pct",
                "selection_bias_gap_sharpe", "fallback_fold_count",
                "fallback_rate", "fallback_years",
            )
        },
        "analyzer": analyzer or result.get("walkforward_analyzer", {}),
        "robustness_reports": reports,
        "robustness_review": robustness,
        "approval": base_approval,
        "deployment": {
            "status": "paper_provisional" if paper_approved else "rejected",
            "paper_approved": paper_approved,
            "real_capital_approved": False,
            "capital_approval_eligible": bool(
                paper_approved and survivorship_capital_pass and universe.get("complete", False)
            ),
            "integrity_status": (
                "verified"
                if report_matches and dataset_fingerprint and source_is_file and result.get("folds")
                else "provisional"
            ),
            "reasons": sorted(set(provisional_reasons)),
        },
    }
    bundle["validation_bundle_hash"] = sha256_value(bundle)
    return bundle


def validate_validation_bundle(bundle: dict) -> tuple[bool, list[str]]:
    """Verify the bundle checksum and minimum paper-trading fields."""
    issues: list[str] = []
    expected = str(bundle.get("validation_bundle_hash", ""))
    unsigned = dict(bundle)
    unsigned.pop("validation_bundle_hash", None)
    actual = sha256_value(unsigned)
    if not expected:
        issues.append("validation_bundle_hash_missing")
    elif expected != actual:
        issues.append("validation_bundle_hash_mismatch")
    if not bundle.get("config_fingerprint"):
        issues.append("config_fingerprint_missing")
    if int(bundle.get("schema_version", 0) or 0) < 2:
        issues.append("validation_bundle_schema_outdated")
    if not isinstance(bundle.get("deployment"), dict):
        issues.append("deployment_state_missing")
    robustness = bundle.get("robustness_review")
    if not isinstance(robustness, dict):
        issues.append("robustness_review_missing")
    elif bool((bundle.get("deployment", {}) or {}).get("paper_approved", False)) and not bool(robustness.get("pass", False)):
        issues.append("paper_approval_without_robustness_pass")
    return not issues, issues


def write_validation_bundle(bundle: dict, path: Path = DEFAULT_BUNDLE_PATH) -> Path:
    """Atomically write a completed validation bundle."""
    atomic_write_json(bundle, path)
    return path


def _matching_approved_config(result: dict, live: dict) -> tuple[bool, str]:
    """Confirm that walk-forward evidence describes the config used by paper trading."""
    evidence_config = ((result.get("approved_live_config", {}) or {}).get("config", {}) or {})
    live_config = (
        ((live.get("approved_live_configs", {}) or {}).get("core-alpha", {}) or {}).get("config", {})
        or {}
    )
    if not evidence_config:
        return False, "walkforward_approved_config_missing"
    if not live_config:
        return False, "live_approved_config_missing"
    if strategy_config_fingerprint(evidence_config) != strategy_config_fingerprint(live_config):
        return False, "walkforward_live_config_mismatch"
    return True, ""


def rebuild_from_walkforward(
    source_path: Path,
    *,
    live_config_path: Path = DEFAULT_LIVE_CONFIG_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    canonical_source_path: Path = DEFAULT_CANONICAL_WALKFORWARD_PATH,
    run_robustness: bool = False,
) -> tuple[Path, Path]:
    """Rebuild canonical evidence without ever enabling real-capital trading.

    PLAIN ENGLISH: old migrations sometimes kept the approved paper config but
    dropped its folds.  This repair accepts a complete walk-forward JSON only
    when its approved strategy exactly matches the strategy already used by the
    paper account.  That prevents an unrelated backtest from being attached to
    today's trading configuration.
    """
    result = json.loads(source_path.read_text(encoding="utf-8"))
    live = json.loads(live_config_path.read_text(encoding="utf-8"))
    if not bool((result.get("live_config_approval", {}) or {}).get("approved", False)):
        raise ValueError("walkforward_evidence_not_approved")
    if not result.get("folds"):
        raise ValueError("walkforward_folds_missing")
    matches, reason = _matching_approved_config(result, live)
    if not matches:
        raise ValueError(reason)

    # Store the accepted evidence at the tracked canonical path.  GitHub runners
    # cannot restore a developer's absolute or gitignored scratch filename.
    canonical_source_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(canonical_source_path, json.dumps(result, indent=2, default=str))
    atomic_write_csv(
        pd.DataFrame(result.get("folds", [])),
        canonical_source_path.with_suffix(".csv"),
        index=False,
    )

    if run_robustness:
        # Each command performs research calculations and writes a report; none
        # connects to the broker or submits an order.  The signal refresh makes
        # sure those reports read metrics from the same approved live config.
        env = dict(os.environ)
        env["STOCKBOT_SCRIPT_TELEGRAM_ENABLED"] = "0"
        for script in (
            "core_satellite_alpha.py",
            "core_satellite_execution_stress.py",
            "core_satellite_survivorship_audit.py",
            "factor_decay_monitor.py",
        ):
            completed = subprocess.run(
                [sys.executable, script],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"validation_report_failed:{script}:exit_{completed.returncode}")

    # Build after optional report refresh so fingerprints are compared against
    # the newest evidence, not reports that existed before the repair began.
    bundle = build_validation_bundle(result, source_json=str(canonical_source_path))
    write_validation_bundle(bundle, bundle_path)
    live.update({
        "source_json": str(canonical_source_path),
        "source_json_sha256": bundle.get("source_json_sha256", ""),
        "validation_bundle_path": str(bundle_path),
        "validation_bundle_hash": bundle.get("validation_bundle_hash", ""),
        "deployment_status": bundle.get("deployment", {}).get("status", "rejected"),
        "paper_approved": bool(bundle.get("deployment", {}).get("paper_approved", False)),
        # This repair is evidence plumbing, never permission to use real money.
        "real_capital_approved": False,
    })
    atomic_write_json(live, live_config_path)
    return live_config_path, bundle_path


def migrate_existing_live_config(
    live_config_path: Path = DEFAULT_LIVE_CONFIG_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
) -> tuple[Path, Path]:
    """Wrap the current paper config in a provisional bundle without promoting it."""
    live = json.loads(live_config_path.read_text(encoding="utf-8"))
    strategy = "core-alpha"
    approved = (live.get("approved_live_configs", {}) or {}).get(strategy, {}) or {}
    approval = (live.get("approvals", {}) or {}).get(strategy, {}) or {}
    result = {
        "strategy": strategy,
        "folds": [],
        "live_config_approval": approval,
        "approved_live_config": approved,
    }
    # PLAIN ENGLISH: The present paper config came from an ignored historical
    # research file, while the tracked walk-forward describes another config.
    # Do not falsely attach those folds to this config. A new Colab run will
    # replace this honest provisional record with matching evidence.
    source_json = ""
    bundle = build_validation_bundle(result, source_json=source_json)
    write_validation_bundle(bundle, bundle_path)
    live["validation_bundle_path"] = str(bundle_path)
    live["validation_bundle_hash"] = bundle["validation_bundle_hash"]
    live["deployment_status"] = "paper_provisional"
    live["paper_approved"] = bool(bundle["deployment"]["paper_approved"])
    live["real_capital_approved"] = False
    atomic_write_json(live, live_config_path)
    return live_config_path, bundle_path


def main() -> int:
    """Create a provisional bundle for the currently configured paper strategy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-config", type=Path, default=DEFAULT_LIVE_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--source-walkforward",
        type=Path,
        help="Repair the bundle from matching approved walk-forward JSON instead of creating an empty migration.",
    )
    parser.add_argument(
        "--run-robustness",
        action="store_true",
        help="Refresh signal, execution stress, survivorship, and factor-decay reports before rebuilding.",
    )
    args = parser.parse_args()
    if args.source_walkforward:
        live_path, bundle_path = rebuild_from_walkforward(
            args.source_walkforward,
            live_config_path=args.live_config,
            bundle_path=args.output,
            run_robustness=bool(args.run_robustness),
        )
        print(f"Updated paper config: {live_path}")
        print(f"Wrote validation bundle: {bundle_path}")
        print("Matching folds restored; real capital remains blocked")
        return 0
    live_path, bundle_path = migrate_existing_live_config(args.live_config, args.output)
    print(f"Updated paper config: {live_path}")
    print(f"Wrote validation bundle: {bundle_path}")
    print("Deployment status: paper_provisional; real capital remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
