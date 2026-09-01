"""Archive old paper evidence and track a clean reliability-validation epoch.

PLAIN ENGLISH: Bot behavior changed, so old and new paper results should not be
judged as one experiment. This script copies the current evidence into a dated
archive and creates a new epoch marker. Live operational files stay in place so
reconciliation and duplicate protection keep working.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_json
from run_evidence import enrich_payload
from settings import LOG_DIR, SIGNAL_DIR
from order_accounting import classify_logical_orders


EPOCH_FILE = Path(SIGNAL_DIR) / "paper_validation_epoch.json"
PAPER_VERSION_LOCK_FILE = Path(__file__).resolve().parent / "paper_version_lock.json"
ARCHIVE_ROOT = Path("archive/paper_epochs")
# These files can change what gets selected, sized, submitted, protected, or
# graded during the paper epoch.  Documentation and dashboard-only files are
# intentionally excluded so harmless wording/layout work does not halt paper
# trading.
PAPER_LOGIC_FILES = (
    ".github/workflows/daily_paper_trading.yml",
    ".github/workflows/factor_data_refresh.yml",
    ".github/workflows/independent_workflow_watchdog.yml",
    ".github/workflows/post_market_execution_quality.yml",
    ".github/workflows/shadow_paper_journal.yml",
    "alpaca_paper_trading.py",
    "alpaca_protection.py",
    "alpaca_sdk_adapter.py",
    "broker_interface.py",
    "broker_truth.py",
    "core_satellite_alpha.py",
    "daily_run.py",
    "data_manifest.py",
    "data_provider.py",
    "execution_guard.py",
    "execution_model.py",
    "execution_scorecard.py",
    "factor_data_health.py",
    "fill_monitor.py",
    "monitor_heartbeat.py",
    "order_accounting.py",
    "logs/core_satellite_execution_stress.json",
    "logs/core_satellite_survivorship_audit.json",
    "logs/factor_decay_monitor.json",
    "paper_health.py",
    "paper_shadow_compare.py",
    "paper_validation_epoch.py",
    "pipeline_shared.py",
    "portfolio_manager.py",
    "risk_sizing.py",
    "robustness_review.py",
    "run_evidence.py",
    # Dependency changes can alter data, sentiment, and broker behavior even
    # when the Python source stays unchanged, so freeze them with the release.
    "requirements-ci.txt",
    "requirements.txt",
    "settings.py",
    "signal_freshness.py",
    "trade_rules.py",
    "validation_bundle.py",
    "workflow_watchdog.py",
    "signals/core_satellite_live_configs.json",
    "signals/core_satellite_validation_bundle.json",
)
EVIDENCE_FILES = (
    Path(SIGNAL_DIR) / "alpaca_paper_log.csv",
    Path(SIGNAL_DIR) / "alpaca_paper_equity.csv",
    Path(SIGNAL_DIR) / "alpaca_slippage_reversal_report.json",
    Path(SIGNAL_DIR) / "alpaca_execution_scorecard.json",
    Path(SIGNAL_DIR) / "broker_truth.json",
    Path(SIGNAL_DIR) / "alpaca_paper_health.json",
    Path(SIGNAL_DIR) / "alpaca_submit_outcomes.csv",
)


def start_epoch(*, now: datetime | None = None) -> dict:
    """Copy current evidence and begin a new dated paper-validation period."""
    started = now or datetime.now(timezone.utc)
    epoch_id = started.strftime("paper-v%Y%m%dT%H%M%SZ")
    archive_dir = ARCHIVE_ROOT / epoch_id
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for source in EVIDENCE_FILES:
        if not source.exists():
            continue
        destination = archive_dir / source.name
        shutil.copy2(source, destination)
        archived.append(str(destination))
    payload = {
        "schema_version": 2,
        "epoch_id": epoch_id,
        "started_at": started.isoformat(timespec="seconds"),
        "status": "collecting",
        "archive_dir": str(archive_dir),
        "archived_files": archived,
        "requirements": {
            "minimum_trading_days": 30,
            "minimum_rebalance_events": 3,
            "minimum_accepted_orders": 20,
            "minimum_stage_comparison_fills": 20,
            "minimum_consecutive_classified_sessions": 10,
            "maximum_duplicate_orders": 0,
            "maximum_unexplained_orders": 0,
            "maximum_target_weight_gap": 0.02,
            "maximum_gross_exposure_gap": 0.05,
            "maximum_open_critical_incidents": 0,
            "minimum_fill_rate": 0.95,
            "maximum_average_slippage_bps": 5.0,
            "bad_slippage_threshold_bps": 2.0,
            "maximum_bad_slippage_rate": 0.40,
        },
        "real_capital_approved": False,
    }
    atomic_write_json(payload, EPOCH_FILE)
    return payload


def invalidate_epoch(
    *,
    reasons: list[str],
    epoch_path: Path = EPOCH_FILE,
    now: datetime | None = None,
) -> dict:
    """Mark old evidence unusable without starting an unapproved new epoch."""
    payload = json.loads(epoch_path.read_text(encoding="utf-8"))
    payload.update({
        "schema_version": 2,
        "status": "invalidated",
        "invalidated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "invalidation_reasons": sorted(set(str(reason) for reason in reasons if str(reason))),
        "real_capital_approved": False,
    })
    atomic_write_json(payload, epoch_path)
    return payload


def _sha256_file(path: Path) -> str:
    """Return one file checksum so later edits are detectable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_git_commit(project_root: Path) -> str:
    """Return the current saved Git version for human audit context."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def assert_clean_release_worktree(project_root: Path | None = None) -> None:
    """Refuse an epoch freeze when tracked or untracked code is not committed."""
    root = (project_root or Path(__file__).resolve().parent).resolve()
    if not (root / ".git").exists():
        return
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Paper epoch requires a clean committed release; commit/push and pass CI before freezing"
        )


def build_paper_version_lock(
    epoch: dict,
    *,
    project_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Build a checksum lock without changing the active epoch start.

    PLAIN ENGLISH: This takes a fingerprint of every file that can change paper
    decisions.  Future submission stops if one of those files moves.  The epoch
    ID and August 26 start time are copied exactly; no evidence is archived or
    reset.
    """
    root = (project_root or Path(__file__).resolve().parent).resolve()
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative_name in PAPER_LOGIC_FILES:
        path = root / relative_name
        if path.is_file():
            files[relative_name] = _sha256_file(path)
        else:
            missing.append(relative_name)
    if missing:
        raise RuntimeError(f"Cannot freeze paper version; missing files: {', '.join(missing)}")
    fingerprint_input = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "frozen_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "epoch_id": str(epoch.get("epoch_id", "")),
        "epoch_started_at": str(epoch.get("started_at", "")),
        "git_commit_at_freeze": _current_git_commit(root),
        "logic_fingerprint": hashlib.sha256(fingerprint_input).hexdigest(),
        "files": files,
        "policy": "block_paper_submission_on_locked_file_change",
        "real_capital_approved": False,
    }


def freeze_current_paper_version(
    *,
    epoch_path: Path = EPOCH_FILE,
    lock_path: Path = PAPER_VERSION_LOCK_FILE,
    project_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Write the current paper lock while preserving the existing epoch."""
    if not epoch_path.exists():
        raise RuntimeError(f"Active paper epoch is missing: {epoch_path}")
    epoch = json.loads(epoch_path.read_text(encoding="utf-8"))
    lock = build_paper_version_lock(epoch, project_root=project_root, now=now)
    atomic_write_json(lock, lock_path)
    return lock


def validate_paper_version_lock(
    *,
    epoch_path: Path = EPOCH_FILE,
    lock_path: Path = PAPER_VERSION_LOCK_FILE,
    project_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Check that the active epoch and all frozen paper files still match."""
    issues: list[str] = []
    if not lock_path.exists():
        return False, ["paper_version_lock_missing"]
    if not epoch_path.exists():
        return False, ["paper_epoch_missing"]
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        epoch = json.loads(epoch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"paper_version_lock_unreadable:{type(exc).__name__}"]

    if str(lock.get("epoch_id", "")) != str(epoch.get("epoch_id", "")):
        issues.append("paper_epoch_id_changed")
    if str(lock.get("epoch_started_at", "")) != str(epoch.get("started_at", "")):
        issues.append("paper_epoch_start_changed")

    root = (project_root or Path(__file__).resolve().parent).resolve()
    observed: dict[str, str] = {}
    expected = lock.get("files", {}) or {}
    for relative_name, expected_hash in expected.items():
        path = root / str(relative_name)
        if not path.is_file():
            issues.append(f"locked_file_missing:{relative_name}")
            continue
        observed_hash = _sha256_file(path)
        observed[str(relative_name)] = observed_hash
        if observed_hash != str(expected_hash):
            issues.append(f"locked_file_changed:{relative_name}")
    if not expected:
        issues.append("paper_version_file_manifest_empty")
    observed_input = json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_fingerprint = hashlib.sha256(observed_input).hexdigest()
    if len(observed) == len(expected) and observed_fingerprint != str(lock.get("logic_fingerprint", "")):
        issues.append("paper_logic_fingerprint_mismatch")
    return not issues, issues


def assert_paper_version_frozen() -> None:
    """Raise before submission when the frozen paper logic no longer matches."""
    valid, issues = validate_paper_version_lock()
    if not valid:
        raise RuntimeError(", ".join(issues))


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an optional CSV without making the epoch evaluator crash."""
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    """Read an optional JSON report."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _consecutive_session_count(values: pd.Series) -> int:
    """Count consecutive NYSE sessions working backward from the latest run."""
    dates = sorted(set(pd.to_datetime(values, errors="coerce", utc=True).dropna().dt.tz_convert("America/New_York").dt.normalize()))
    if not dates:
        return 0
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        sessions = set(pd.DatetimeIndex(calendar.sessions_in_range(dates[0].tz_localize(None), dates[-1].tz_localize(None))).tz_localize(None))
        observed = {date.tz_localize(None) for date in dates}
        count = 0
        for session in sorted(sessions, reverse=True):
            if session in observed:
                count += 1
            else:
                break
        return count
    except Exception:
        return len(dates)


def evaluate_epoch(epoch: dict) -> dict:
    """Measure progress against the fixed operational acceptance rules."""
    if str(epoch.get("status", "")).lower() == "invalidated":
        # PLAIN ENGLISH: changed rules make the old sample incomparable. Never
        # let later file changes revive that experiment.
        return {
            "schema_version": 2,
            "epoch_id": epoch.get("epoch_id"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "invalidated",
            "invalidation_reasons": list(epoch.get("invalidation_reasons", [])),
            "manual_real_capital_review_eligible": False,
            "real_capital_approved": False,
        }
    start = pd.to_datetime(epoch["started_at"], utc=True)
    journal = _read_csv(Path(SIGNAL_DIR) / "alpaca_submit_outcomes.csv")
    paper_log = _read_csv(Path(SIGNAL_DIR) / "alpaca_paper_log.csv")
    equity = _read_csv(Path(SIGNAL_DIR) / "alpaca_paper_equity.csv")

    if not journal.empty and "generated_at" in journal:
        journal["_time"] = pd.to_datetime(journal["generated_at"], errors="coerce", utc=True)
        journal = journal[journal["_time"] >= start]
    if not paper_log.empty:
        timestamp_col = next((column for column in ("submitted_at", "date") if column in paper_log), None)
        if timestamp_col:
            paper_log["_time"] = pd.to_datetime(paper_log[timestamp_col], errors="coerce", utc=True)
            paper_log = paper_log[paper_log["_time"] >= start]
    # The equity file stores both a calendar ``date`` and an exact ``timestamp``.
    # PLAIN ENGLISH: an epoch can start in the middle of a day.  Reading only
    # ``date`` turns that day's observation into midnight, which can make a real
    # post-start snapshot look older than the epoch.  Prefer the exact clock time
    # and use ``date`` only for older files that do not have it yet.
    if not equity.empty:
        equity_time_col = next((column for column in ("timestamp", "date") if column in equity), None)
        if equity_time_col:
            equity["_time"] = pd.to_datetime(equity[equity_time_col], errors="coerce", utc=True)
    if "_time" in equity:
        equity = equity[equity["_time"] >= start]

    accepted = int(pd.to_numeric(journal.get("accepted_orders", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    classified_sessions = _consecutive_session_count(journal["_time"]) if "_time" in journal else 0
    executed = journal[journal["status"].astype(str) == "executed"] if not journal.empty and "status" in journal else journal.iloc[0:0]
    rebalances = int(executed["_time"].dt.date.nunique()) if "_time" in executed else 0
    trading_days = int(equity["_time"].dt.date.nunique()) if "_time" in equity else 0
    # PLAIN ENGLISH: Stage 1 and Stage 2 are attempts at one requested trade.
    # Count the parent once, include canceled/pending broker-accepted orders in
    # the denominator, and require a complete fill for the promotion gate.
    order_accounting = classify_logical_orders(paper_log)
    accepted = int(order_accounting["accepted_logical_orders"])
    duplicates = int(order_accounting["duplicate_logical_orders"])
    fill_rate = order_accounting["complete_fill_rate"]
    any_fill_rate = order_accounting["any_fill_rate"]

    # PLAIN ENGLISH: the scorecard decides whether enough observations exist,
    # while the epoch computes its own rates from post-start rebalance fills.
    # This keeps old fills and protective stops out of the August experiment.
    execution_scorecard = _read_json(Path(SIGNAL_DIR) / "alpaca_execution_scorecard.json")
    execution_decision_eligible = bool(execution_scorecard.get("decision_eligible", False))
    slippage = _read_json(Path(SIGNAL_DIR) / "alpaca_slippage_reversal_report.json")
    # PLAIN ENGLISH: the operational report keeps older fills for context.
    # A clean epoch must ignore those rows and must not mix protective stops
    # with normal rebalance execution.
    epoch_slippage_values: list[float] = []
    for row in slippage.get("orders", []) or []:
        filled_at = pd.to_datetime((row or {}).get("filled_at"), errors="coerce", utc=True)
        order_type = str((row or {}).get("order_type", "")).lower()
        value = pd.to_numeric((row or {}).get("slippage_bps"), errors="coerce")
        if pd.isna(filled_at) or filled_at < start:
            continue
        if order_type in {"trailing_stop", "stop", "stop_limit"} or pd.isna(value):
            continue
        epoch_slippage_values.append(float(value))
    avg_slippage = float(pd.Series(epoch_slippage_values).mean()) if epoch_slippage_values else None
    bad_slippage_threshold = float((epoch.get("requirements", {}) or {}).get("bad_slippage_threshold_bps", 2.0))
    bad_count = int(sum(value > bad_slippage_threshold for value in epoch_slippage_values))
    analyzed = len(epoch_slippage_values)
    bad_rate = float(bad_count / analyzed) if analyzed else None

    # Compare passive Stage 1 with marketable-capped Stage 2 only inside this
    # fresh epoch. The client ID is broker evidence of which child filled. We
    # intentionally ignore the paper log and even the derived execution_stage
    # label because a logical row can blend prices from both attempts.
    stage_slippage: dict[str, list[float]] = {"stage1": [], "stage2": []}
    for row in slippage.get("orders", []) or []:
        filled_at = pd.to_datetime((row or {}).get("filled_at"), errors="coerce", utc=True)
        client_order_id = str((row or {}).get("client_order_id", "")).strip().lower()
        if client_order_id.endswith("-a1"):
            stage = "stage1"
        elif client_order_id.endswith("-a2"):
            stage = "stage2"
        else:
            stage = ""
        order_type = str((row or {}).get("order_type", "")).lower()
        value = pd.to_numeric((row or {}).get("slippage_bps"), errors="coerce")
        if pd.isna(filled_at) or filled_at < start or stage not in stage_slippage:
            continue
        if order_type in {"trailing_stop", "stop", "stop_limit"} or pd.isna(value):
            continue
        stage_slippage[stage].append(float(value))
    stage_comparison = {
        stage: {
            "measured_fills": len(values),
            "average_slippage_bps": float(pd.Series(values).mean()) if values else None,
            "median_slippage_bps": float(pd.Series(values).median()) if values else None,
        }
        for stage, values in stage_slippage.items()
    }

    broker_truth = _read_json(Path(SIGNAL_DIR) / "broker_truth.json")
    truth_summary = broker_truth.get("summary", {}) or {}
    unexplained = int(truth_summary.get("fail_count", 0) or 0)
    canonical_alignment = truth_summary.get("alignment", {}) or {}
    alignment_status = str(canonical_alignment.get("status", ""))
    max_weight_gap = pd.to_numeric(
        canonical_alignment.get("maximum_target_weight_gap"), errors="coerce"
    )
    gross_exposure_gap = pd.to_numeric(
        canonical_alignment.get("gross_exposure_gap"), errors="coerce"
    )
    if pd.isna(max_weight_gap):
        # Compatibility for reports written before canonical alignment existed.
        weight_gaps = []
        for row in broker_truth.get("rows", []) or []:
            target = pd.to_numeric(row.get("target_weight"), errors="coerce")
            actual = pd.to_numeric(row.get("broker_weight"), errors="coerce")
            if pd.notna(target) and pd.notna(actual):
                weight_gaps.append(abs(float(target) - float(actual)))
        max_weight_gap = max(weight_gaps, default=None)
    else:
        max_weight_gap = float(max_weight_gap)
    gross_exposure_gap = None if pd.isna(gross_exposure_gap) else float(gross_exposure_gap)
    incident_summary = truth_summary.get("alignment_incident_ledger", {}) or {}
    open_critical_incidents = int(incident_summary.get("open_incidents", 0) or 0)
    operational_ledger = _read_csv(Path(SIGNAL_DIR) / "operational_incident_ledger.csv")
    if not operational_ledger.empty and {"status", "severity"}.issubset(operational_ledger.columns):
        open_critical_incidents += int(
            (
                operational_ledger["status"].astype(str).str.lower().eq("open")
                & operational_ledger["severity"].astype(str).str.lower().eq("critical")
            ).sum()
        )

    requirements = epoch.get("requirements", {}) or {}
    minimum_stage_fills = int(requirements.get("minimum_stage_comparison_fills", 20))
    stage_review_ready = bool(
        analyzed >= minimum_stage_fills
        and stage_comparison["stage1"]["measured_fills"] > 0
        and stage_comparison["stage2"]["measured_fills"] > 0
    )
    stage1_avg = stage_comparison["stage1"]["average_slippage_bps"]
    stage2_avg = stage_comparison["stage2"]["average_slippage_bps"]
    stage_design_improves_slippage = bool(
        stage_review_ready
        and stage1_avg is not None
        and stage2_avg is not None
        and float(stage1_avg) <= float(stage2_avg)
    )
    checks = {
        "trading_days": trading_days >= int(requirements.get("minimum_trading_days", 30)),
        "rebalance_events": rebalances >= int(requirements.get("minimum_rebalance_events", 3)),
        "accepted_orders": accepted >= int(requirements.get("minimum_accepted_orders", 20)),
        "classified_sessions": classified_sessions >= int(requirements.get("minimum_consecutive_classified_sessions", 10)),
        "duplicate_orders": duplicates <= int(requirements.get("maximum_duplicate_orders", 0)),
        "classifiable_orders": (
            int(order_accounting["unclassifiable_rows"]) == 0
            and int(order_accounting["unclassifiable_logical_orders"]) == 0
        ),
        "unexplained_orders": unexplained <= int(requirements.get("maximum_unexplained_orders", 0)),
        "target_weight_gap": alignment_status == "pass" and max_weight_gap is not None and max_weight_gap <= float(requirements.get("maximum_target_weight_gap", 0.02)),
        "gross_exposure_gap": alignment_status == "pass" and gross_exposure_gap is not None and gross_exposure_gap <= float(requirements.get("maximum_gross_exposure_gap", 0.05)),
        "critical_incidents": open_critical_incidents <= int(requirements.get("maximum_open_critical_incidents", 0)),
        "fill_rate": fill_rate is not None and fill_rate >= float(requirements.get("minimum_fill_rate", 0.80)),
        "average_slippage": execution_decision_eligible and avg_slippage is not None and float(avg_slippage) <= float(requirements.get("maximum_average_slippage_bps", 10.0)),
        "bad_slippage_rate": execution_decision_eligible and bad_rate is not None and float(bad_rate) <= float(requirements.get("maximum_bad_slippage_rate", 0.60)),
        "stage_comparison_ready": stage_review_ready,
        "two_stage_design": stage_design_improves_slippage and fill_rate is not None and fill_rate >= float(requirements.get("minimum_fill_rate", 0.95)),
    }
    version_lock_valid, version_lock_issues = validate_paper_version_lock()
    checks["paper_version_lock"] = version_lock_valid
    operational_pass = all(checks.values())
    report = {
        "schema_version": 2,
        "epoch_id": epoch.get("epoch_id"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "operational_pass" if operational_pass else "collecting",
        "trading_days": trading_days,
        "rebalance_events": rebalances,
        "accepted_orders": accepted,
        "classified_sessions": classified_sessions,
        "duplicate_orders": duplicates,
        "unexplained_orders": unexplained,
        "maximum_target_weight_gap": max_weight_gap,
        "gross_exposure_gap": gross_exposure_gap,
        "alignment_status": alignment_status or "collecting",
        "open_critical_incidents": open_critical_incidents,
        "fill_rate": fill_rate,
        "complete_fill_rate": fill_rate,
        "any_fill_rate": any_fill_rate,
        "accepted_logical_orders": int(order_accounting["accepted_logical_orders"]),
        "fully_filled_logical_orders": int(order_accounting["fully_filled_logical_orders"]),
        "any_filled_logical_orders": int(order_accounting["any_filled_logical_orders"]),
        "partially_filled_logical_orders": int(order_accounting["partially_filled_logical_orders"]),
        "open_logical_orders": int(order_accounting["open_logical_orders"]),
        "canceled_unfilled_logical_orders": int(order_accounting["canceled_unfilled_logical_orders"]),
        "duplicate_child_attempts": int(order_accounting["duplicate_child_attempts"]),
        "child_attempts": int(order_accounting["child_attempts"]),
        "unclassifiable_order_rows": int(order_accounting["unclassifiable_rows"]),
        "unclassifiable_logical_orders": int(order_accounting["unclassifiable_logical_orders"]),
        "average_slippage_bps": avg_slippage,
        "bad_slippage_rate": bad_rate,
        "stage_comparison": stage_comparison,
        "stage_comparison_measured_fills": analyzed,
        "stage_comparison_minimum_fills": minimum_stage_fills,
        "stage_comparison_review_ready": stage_review_ready,
        "two_stage_design_improves_slippage": stage_design_improves_slippage,
        "execution_scorecard_decision_eligible": execution_decision_eligible,
        "execution_scorecard_schema_version": execution_scorecard.get("schema_version"),
        "paper_version_lock_valid": version_lock_valid,
        "paper_version_lock_issues": version_lock_issues,
        "checks": checks,
        # Passing creates evidence for a human approval decision. It never
        # enables live money automatically.
        "manual_real_capital_review_eligible": operational_pass,
        "real_capital_approved": False,
    }
    signal_as_of = str(broker_truth.get("inputs", {}).get("signal", {}).get("as_of", "") or "")
    return enrich_payload(report, signal_as_of=signal_as_of)


def main() -> int:
    """Start a new epoch or print progress for the active one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Evaluate the current epoch without starting a new one.")
    parser.add_argument(
        "--freeze-current",
        action="store_true",
        help="Freeze current paper logic without changing the active epoch start.",
    )
    parser.add_argument(
        "--invalidate-current",
        action="store_true",
        help="Mark the current epoch invalid after research or scoring logic changes.",
    )
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        help="Reason stored with --invalidate-current (may be repeated).",
    )
    args = parser.parse_args()
    if args.invalidate_current:
        reasons = args.reason or ["paper_evidence_rules_changed"]
        epoch = invalidate_epoch(reasons=reasons)
        print(f"Invalidated paper validation epoch: {epoch['epoch_id']}")
        print("Reasons: " + ", ".join(epoch["invalidation_reasons"]))
        return 0
    if args.freeze_current:
        assert_clean_release_worktree()
        lock = freeze_current_paper_version()
        print(f"Frozen paper version for existing epoch: {lock['epoch_id']}")
        print(f"Epoch start preserved: {lock['epoch_started_at']}")
        print(f"Logic fingerprint: {lock['logic_fingerprint']}")
        return 0
    if args.status:
        epoch = json.loads(EPOCH_FILE.read_text(encoding="utf-8"))
        report = evaluate_epoch(epoch)
        atomic_write_json(report, Path(SIGNAL_DIR) / "paper_validation_epoch_status.json")
        print(json.dumps(report, indent=2))
        return 0
    assert_clean_release_worktree()
    epoch = start_epoch()
    print(f"Started paper validation epoch: {epoch['epoch_id']}")
    print(f"Archived evidence -> {epoch['archive_dir']}")
    print("Operational files remain active; real capital remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
