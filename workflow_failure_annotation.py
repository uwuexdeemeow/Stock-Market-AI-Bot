"""Publish one useful GitHub Actions annotation after a workflow failure.

PLAIN ENGLISH: GitHub normally makes you open several log sections to find the
failed step. Workflows pass their named step results to this helper, which puts
the failed step names and run link directly in the Actions summary.
"""
from __future__ import annotations

import argparse
import json
import os


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


def main() -> int:
    """Read workflow metadata and print GitHub's machine-readable annotation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--workflow-name", required=True)
    args = parser.parse_args()
    failed = failed_step_ids(os.environ.get("WORKFLOW_STEPS_JSON", "{}"))
    run_url = os.environ.get("WORKFLOW_RUN_URL", "").strip()
    detail = ", ".join(failed) if failed else "unknown step; inspect the run log"
    message = f"Failed step(s): {detail}. Run: {run_url or 'URL unavailable'}"
    # Percent and newlines have special meaning in the Actions command format.
    safe = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={args.workflow_file},line=1,title={args.workflow_name} failed::{safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
