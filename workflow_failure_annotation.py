"""Publish one useful GitHub Actions annotation after a workflow failure.

PLAIN ENGLISH: GitHub normally makes you open several log sections to find the
failed step. Workflows pass their named step results to this helper, which puts
the failed step names and run link directly in the Actions summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def failed_step_ids(raw_steps: str) -> list[str]:
    """Return IDs for steps whose final outcome is failure or cancellation."""
    try:
        steps = json.loads(raw_steps or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(steps, dict):
        return []
    return sorted(
        str(step_id)
        for step_id, details in steps.items()
        if isinstance(details, dict)
        and details.get("outcome") in {"failure", "cancelled"}
    )


def failure_details(log_glob: str) -> list[str]:
    """Return useful root-cause details from the newest pipeline JSON log."""
    if not log_glob:
        return []
    matches = [Path(path) for path in glob.glob(log_glob)]
    if not matches:
        return []
    # The date is part of pipeline log names. Use it as a deterministic
    # tiebreaker because fast filesystems can assign equal modification times.
    newest = max(matches, key=lambda path: (path.stat().st_mtime, path.name))
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    results = payload.get("results", []) if isinstance(payload, dict) else []
    details: list[str] = []
    for result in results:
        if not isinstance(result, dict) or result.get("status") not in {
            "failed", "error", "timeout"
        }:
            continue
        name = str(result.get("name", "unknown"))
        reason = str(result.get("error", "")).strip()
        outcome = result.get("execution_outcome", {})
        if not reason and isinstance(outcome, dict):
            reason = str(outcome.get("reason_code", "")).strip()
        if not reason:
            reason = str(result.get("alignment_error", "")).strip()
        if not reason:
            stderr = str(result.get("stderr_tail", "")).strip().splitlines()
            reason = stderr[-1].strip() if stderr else ""
        detail = f"{name}: {result.get('status')}"
        if reason:
            detail += f" ({reason})"
        details.append(detail)
    return details


def main() -> int:
    """Read workflow metadata and print GitHub's machine-readable annotation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--detail-log-glob", default="")
    args = parser.parse_args()
    failed = failed_step_ids(os.environ.get("WORKFLOW_STEPS_JSON", "{}"))
    pipeline_details = failure_details(args.detail_log_glob)
    run_url = os.environ.get("WORKFLOW_RUN_URL", "").strip()
    detail = ", ".join(failed) if failed else "unknown step; inspect the run log"
    message = f"Failed step(s): {detail}."
    if pipeline_details:
        pipeline_summary = "; ".join(pipeline_details[:4])[:1200]
        message += " Pipeline: " + pipeline_summary + "."
    message += f" Run: {run_url or 'URL unavailable'}"
    # Percent and newlines have special meaning in the Actions command format.
    safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={args.workflow_file},line=1,title={args.workflow_name} failed::{safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
