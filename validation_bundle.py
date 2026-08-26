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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safe_io import atomic_write_json
from universe_membership import membership_status


DEFAULT_BUNDLE_PATH = Path("signals/core_satellite_validation_bundle.json")
DEFAULT_LIVE_CONFIG_PATH = Path("signals/core_satellite_live_configs.json")
DEFAULT_RESEARCH_MANIFEST_PATH = Path("signals/research_run_manifest.json")
DEFAULT_REPORT_PATHS = {
    "survivorship": Path("logs/core_satellite_survivorship_audit.json"),
    "execution_stress": Path("logs/core_satellite_execution_stress.json"),
    "factor_decay": Path("logs/factor_decay_monitor.json"),
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

    record.update({
        "sha256": file_sha256(path),
        "generated_at": payload.get("generated_at"),
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
    record["match"] = not record["reasons"]
    return record


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
) -> dict:
    """Build the complete, checksummed strategy-validation bundle."""
    approved = result.get("approved_live_config", {}) or {}
    config = approved.get("config", {}) or {}
    config_fingerprint = strategy_config_fingerprint(config)
    dataset_context = dataset_context or load_dataset_context()
    dataset_fingerprint = str(dataset_context.get("dataset_fingerprint", ""))
    report_paths = report_paths or DEFAULT_REPORT_PATHS
    reports = {
        name: report_validation_record(
            name,
            Path(path),
            expected_config_fingerprint=config_fingerprint,
            expected_dataset_fingerprint=dataset_fingerprint,
        )
        for name, path in report_paths.items()
    }
    report_matches = bool(reports) and all(row.get("match") for row in reports.values())
    base_approval = dict(result.get("live_config_approval", {}) or {})
    paper_approved = bool(base_approval.get("approved", False) and config)

    provisional_reasons: list[str] = []
    source_path = Path(source_json) if source_json else None
    source_is_file = bool(source_path and source_path.is_file())
    if not source_is_file:
        provisional_reasons.append("walkforward_source_missing")
    if not result.get("folds"):
        provisional_reasons.append("walkforward_folds_missing")
    if not dataset_fingerprint:
        provisional_reasons.append("dataset_fingerprint_missing")
    for name, row in reports.items():
        for reason in row.get("reasons", []):
            provisional_reasons.append(f"{name}:{reason}")
    universe = membership_status()
    if not universe.get("complete", False):
        provisional_reasons.append("point_in_time_universe_incomplete")

    bundle = {
        "schema_version": 1,
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
        "approval": base_approval,
        "deployment": {
            "status": "paper_provisional" if paper_approved else "rejected",
            "paper_approved": paper_approved,
            "real_capital_approved": False,
            "integrity_status": "verified" if report_matches and dataset_fingerprint else "provisional",
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
    if not isinstance(bundle.get("deployment"), dict):
        issues.append("deployment_state_missing")
    return not issues, issues


def write_validation_bundle(bundle: dict, path: Path = DEFAULT_BUNDLE_PATH) -> Path:
    """Atomically write a completed validation bundle."""
    atomic_write_json(bundle, path)
    return path


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
    args = parser.parse_args()
    live_path, bundle_path = migrate_existing_live_config(args.live_config, args.output)
    print(f"Updated paper config: {live_path}")
    print(f"Wrote validation bundle: {bundle_path}")
    print("Deployment status: paper_provisional; real capital remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
