"""Classify a refreshed factor snapshot without weakening its trading gates.

PLAIN ENGLISH: an operational workflow error and an honest "do not trade"
safety result are not the same thing.  This script checks both, saves one clear
status file, and returns an error only when the evidence itself is incomplete
or mismatched.  A coherent snapshot that fails a safety test remains usable for
diagnosis, but ``trading_allowed`` stays false and the signal is made inert.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json
from validation_bundle import (
    current_robustness_evidence,
    load_dataset_context,
    validate_validation_bundle,
)

DEFAULT_BUNDLE_PATH = Path("signals/core_satellite_validation_bundle.json")
DEFAULT_STATUS_PATH = Path("signals/robustness_snapshot_status.json")
DEFAULT_SIGNAL_PATH = Path("signals/core_satellite_alpha_signal.csv")


def classify_snapshot(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> dict:
    """Return separate evidence-integrity and trading-safety decisions."""
    try:
        approved = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "snapshot_usable": False,
            "trading_allowed": False,
            "reasons": [f"validation_bundle_unreadable:{exc.__class__.__name__}"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    bundle_ok, bundle_issues = validate_validation_bundle(approved)
    if not bundle_ok:
        return {
            "status": "invalid",
            "snapshot_usable": False,
            "trading_allowed": False,
            "reasons": [f"validation_bundle_invalid:{issue}" for issue in bundle_issues],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    dataset = load_dataset_context()
    review = current_robustness_evidence(
        expected_config_fingerprint=str(approved.get("config_fingerprint", "")),
        expected_dataset_fingerprint=str(dataset.get("dataset_fingerprint", "")),
        expected_config=approved.get("config", {}),
    )
    dataset_ready = bool(dataset.get("dataset_fingerprint"))
    snapshot_usable = bool(dataset_ready and review.get("identity_pass", False))
    trading_allowed = bool(snapshot_usable and review.get("health_pass", False))
    if trading_allowed:
        status = "approved"
    elif snapshot_usable:
        status = "trading_blocked"
    else:
        status = "invalid"

    reasons = list(review.get("reasons", []))
    if not dataset_ready:
        reasons.append(str(dataset.get("reason") or "dataset_fingerprint_missing"))
    return {
        "status": status,
        "snapshot_usable": snapshot_usable,
        "trading_allowed": trading_allowed,
        "reasons": sorted({str(reason) for reason in reasons if reason}),
        "dataset_fingerprint": str(dataset.get("dataset_fingerprint", "")),
        "config_fingerprint": str(approved.get("config_fingerprint", "")),
        "identity_pass": bool(review.get("identity_pass", False)),
        "identity_reasons": list(review.get("identity_reasons", [])),
        "health_pass": bool(review.get("health_pass", False)),
        "health_reasons": list(review.get("health_reasons", [])),
        "medium_risk_review": review.get("medium_risk_review", {}),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def block_signal(status: dict, signal_path: Path = DEFAULT_SIGNAL_PATH) -> None:
    """Make any existing paper signal visibly non-tradable and zero-sized."""
    reason = "robustness_snapshot_blocked:" + ",".join(status.get("reasons", []) or ["unknown"])
    if signal_path.exists():
        try:
            frame = pd.read_csv(signal_path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()
    if frame.empty:
        frame = pd.DataFrame([{"paper_signal_type": "core_satellite_alpha"}])

    # PLAIN ENGLISH: false approval flags stop the broker.  Zero target weights
    # add a second visible safeguard so a human cannot mistake old allocations
    # in a blocked row for today's authorized targets.
    frame["paper_ready"] = False
    frame["gates_all_pass"] = False
    frame["reason"] = reason
    frame["target_spy_weight"] = 0.0
    frame["target_qqq_weight"] = 0.0
    frame["target_tqqq_weight"] = 0.0
    frame["target_cash_weight"] = 1.0
    frame["gross_exposure"] = 0.0
    frame["overlay_tickers"] = ""
    frame["overlay_weights_json"] = "{}"
    atomic_write_csv(frame, signal_path, index=False)


def write_github_outputs(status: dict) -> None:
    """Expose small booleans to later GitHub Actions steps when requested."""
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if not output_path:
        raise RuntimeError("GITHUB_OUTPUT is missing")
    with Path(output_path).open("a", encoding="utf-8") as handle:
        handle.write(f"snapshot_usable={str(bool(status['snapshot_usable'])).lower()}\n")
        handle.write(f"trading_allowed={str(bool(status['trading_allowed'])).lower()}\n")
        handle.write(f"status={status['status']}\n")


def main() -> int:
    """Write the status, optionally publish outputs, and fail only on bad evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--github-output", action="store_true")
    parser.add_argument("--require-trading-pass", action="store_true")
    args = parser.parse_args()

    status = classify_snapshot(args.bundle)
    atomic_write_json(status, args.status_output)
    if not status["trading_allowed"]:
        block_signal(status)
    if args.github_output:
        write_github_outputs(status)
    print(json.dumps(status, indent=2))

    if not status["snapshot_usable"]:
        return 1
    if args.require_trading_pass and not status["trading_allowed"]:
        return 1
    if not status["trading_allowed"]:
        print("::warning::Snapshot evidence is coherent, but safety review blocked trading. No orders are allowed.")
    # PLAIN ENGLISH: paper trading may continue past the known one-day-delay
    # weakness, but every run must say so loudly.  It is never capital approval.
    advisories = (
        (status.get("medium_risk_review", {}) or {}).get("execution_stress_review", {}) or {}
    ).get("paper_advisory_scenarios", []) or []
    if advisories:
        names = ", ".join(
            f"{row.get('scenario')} (2023-2026 vs QQQ {row.get('holdout_alpha_vs_qqq_pct')}%)"
            for row in advisories
        )
        print(
            "::warning::PAPER-ONLY advisory: the strategy fails one-day-delay stress "
            f"[{names}]. Paper orders continue; real capital stays blocked."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
