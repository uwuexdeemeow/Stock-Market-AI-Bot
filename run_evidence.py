"""Build reproducible identity and evidence for one paper-trading run.

PLAIN ENGLISH: A report is useful only when we know which signal, code version,
and daily run created it.  This module gives every child script the same run ID,
adds safe fingerprints, tracks the rebalance lifecycle, and writes a manifest
only after the expected evidence files are complete.  It never contacts Alpaca
and never submits an order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from safe_io import atomic_write_json
from settings import SIGNAL_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS = Path(SIGNAL_DIR)
MANIFEST_FILE = SIGNALS / "paper_run_manifest.json"
REBALANCE_STATE_FILE = SIGNALS / "rebalance_state.json"

# These are evidence outputs, not strategy inputs.  The manifest becomes
# complete only when all four can be read and identify the same daily run.
REQUIRED_EVIDENCE_FILES = (
    SIGNALS / "broker_truth.json",
    SIGNALS / "alpaca_execution_scorecard.json",
    SIGNALS / "paper_validation_epoch_status.json",
    SIGNALS / "alpaca_paper_health.json",
)
OPTIONAL_EVIDENCE_FILES = (
    SIGNALS / "core_satellite_alpha_signal.csv",
    SIGNALS / "core_satellite_alpha_input_snapshot.csv",
    SIGNALS / "core_satellite_alpha_orders.csv",
    SIGNALS / "alpaca_paper_log.csv",
    SIGNALS / "alpaca_paper_equity.csv",
    SIGNALS / "alpaca_submit_outcome.json",
    SIGNALS / "alignment_incident_ledger.csv",
    SIGNALS / "operational_incident_ledger.csv",
    SIGNALS / "rebalance_state.json",
)
MULTI_RUN_HISTORY_FILES = {
    "alpaca_paper_log.csv",
    "alpaca_paper_equity.csv",
    "alignment_incident_ledger.csv",
    "operational_incident_ledger.csv",
}

# Hash only behavior-bearing, non-secret files.  Environment secrets and raw
# account identifiers are deliberately excluded from the fingerprint.
FINGERPRINT_FILES = (
    "alpaca_paper_trading.py",
    "broker_truth.py",
    "daily_run.py",
    "execution_guard.py",
    "execution_model.py",
    "paper_health.py",
    "risk_sizing.py",
    "settings.py",
    "trade_rules.py",
    "signals/core_satellite_live_configs.json",
)


def current_run_id(now: datetime | None = None) -> str:
    """Return the shared run ID, creating one when a script runs alone."""
    configured = os.environ.get("STOCKBOT_RUN_ID", "").strip()
    if configured:
        return configured
    clock = now or datetime.now(timezone.utc)
    return f"paper-{clock.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    """Hash one file without loading a large file fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    """Read the saved Git version; return unknown outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def configuration_fingerprint() -> str:
    """Hash the files that can materially affect a paper execution."""
    entries: dict[str, str] = {}
    for relative in FINGERPRINT_FILES:
        path = PROJECT_ROOT / relative
        entries[relative] = _sha256(path) if path.is_file() else "missing"
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _paper_version_fingerprint() -> str:
    """Read the existing frozen-paper fingerprint without changing the lock."""
    path = PROJECT_ROOT / "paper_version_lock.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("logic_fingerprint", ""))
    except Exception:
        return ""


def _account_hash() -> str:
    """Return a short irreversible account marker, never the raw account ID."""
    raw = os.environ.get("ALPACA_ACCOUNT_ID", "").strip()
    if not raw:
        status_path = SIGNALS / "alpaca_daily_status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            raw = str(status.get("account_id") or status.get("id") or "").strip()
        except Exception:
            raw = ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] if raw else "unavailable"


def build_run_context(*, signal_as_of: str = "", now: datetime | None = None) -> dict[str, Any]:
    """Build safe metadata shared by JSON reports and CSV audit rows."""
    clock = now or datetime.now(timezone.utc)
    return {
        "run_id": current_run_id(clock),
        "generated_at": clock.isoformat(timespec="seconds"),
        "signal_as_of": str(signal_as_of or ""),
        "git_commit": _git_commit(),
        "configuration_fingerprint": configuration_fingerprint(),
        "paper_version_fingerprint": _paper_version_fingerprint(),
        "paper_account_hash": _account_hash(),
    }


def enrich_payload(payload: dict[str, Any], *, signal_as_of: str = "", now: datetime | None = None) -> dict[str, Any]:
    """Attach reproducibility metadata while preserving every existing field."""
    output = dict(payload)
    output["run_context"] = build_run_context(signal_as_of=signal_as_of, now=now)
    output.setdefault("run_id", output["run_context"]["run_id"])
    return output


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and safely return an empty object on bad evidence."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def update_rebalance_state(
    status: str,
    *,
    run_id: str | None = None,
    signal_as_of: str = "",
    details: dict[str, Any] | None = None,
    path: Path = REBALANCE_STATE_FILE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist one idempotent rebalance lifecycle transition."""
    clock = now or datetime.now(timezone.utc)
    active_run = run_id or current_run_id(clock)
    prior = read_json(path)
    # A new run gets a fresh history. Repeating a transition in the same run
    # updates its details instead of creating duplicate history entries.
    history = list(prior.get("history", [])) if prior.get("run_id") == active_run else []
    transition = {
        "status": str(status),
        "at": clock.isoformat(timespec="seconds"),
        "details": details or {},
    }
    if history and history[-1].get("status") == status:
        history[-1] = transition
    else:
        history.append(transition)
    terminal = status in {"aligned", "rejected", "timed_out"}
    payload = {
        "schema_version": 1,
        "run_id": active_run,
        "signal_as_of": str(signal_as_of or prior.get("signal_as_of", "")),
        "status": str(status),
        "updated_at": transition["at"],
        "terminal": terminal,
        "history": history,
        "real_capital_approved": False,
    }
    atomic_write_json(payload, path)
    return payload


def build_evidence_manifest(
    *,
    required_files: tuple[Path, ...] = REQUIRED_EVIDENCE_FILES,
    output_path: Path = MANIFEST_FILE,
    now: datetime | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Validate a complete same-run evidence bundle and optionally publish it."""
    clock = now or datetime.now(timezone.utc)
    run_id = current_run_id(clock)
    files: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    required_set = {Path(path).name for path in required_files}
    all_files = tuple(required_files) + tuple(
        path for path in OPTIONAL_EVIDENCE_FILES if path.name not in required_set
    )
    for path in all_files:
        payload = read_json(path)
        observed_run = str(payload.get("run_id") or (payload.get("run_context", {}) or {}).get("run_id") or "")
        if not observed_run and path.suffix.lower() == ".csv" and path.is_file():
            try:
                csv = pd.read_csv(path, usecols=lambda column: column == "run_id")
                if "run_id" in csv and not csv["run_id"].dropna().empty:
                    observed_run = str(csv["run_id"].dropna().iloc[-1])
            except Exception:
                observed_run = ""
        try:
            relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            # Custom paths are useful in tests and offline validation. Keep a
            # safe evidence-style name rather than leaking an absolute path.
            relative_path = Path("signals") / path.name
        entry = {
            "path": str(path),
            "relative_path": str(relative_path).replace("\\", "/"),
            "required": path.name in required_set,
            "exists": path.is_file(),
            "readable": bool(payload),
            "run_id": observed_run,
            "sha256": _sha256(path) if path.is_file() else "",
        }
        if not path.is_file() and path.name in required_set:
            problems.append(f"missing:{path.name}")
        elif path.suffix.lower() == ".json" and path.is_file() and not payload:
            problems.append(f"unreadable:{path.name}")
        elif observed_run and observed_run != run_id and path.name not in MULTI_RUN_HISTORY_FILES:
            problems.append(f"mixed_run:{path.name}:{observed_run or 'missing'}")
        elif path.name in required_set and not observed_run:
            problems.append(f"mixed_run:{path.name}:missing")
        files[path.name] = entry
    manifest = {
        "schema_version": 1,
        "generated_at": clock.isoformat(timespec="seconds"),
        "run_id": run_id,
        "status": "complete" if not problems else "incomplete",
        "problems": problems,
        "files": files,
        "run_context": build_run_context(now=clock),
        "real_capital_approved": False,
    }
    if write:
        atomic_write_json(manifest, output_path)
    return manifest


def main() -> int:
    """Command-line entry point for the final evidence publication gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate evidence without writing the manifest.")
    parser.add_argument("--json", action="store_true", help="Print the complete manifest JSON.")
    args = parser.parse_args()
    payload = build_evidence_manifest(write=not args.check)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Paper evidence: {payload['status']} run_id={payload['run_id']}")
        for problem in payload["problems"]:
            print(f"  - {problem}")
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
