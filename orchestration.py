"""
orchestration.py — Single entry point the scheduler hits.

PLAIN ENGLISH:
Production systems don't run scripts by hand. A scheduler (cron, Airflow,
Prefect, etc.) wakes a daemon on a clock and calls ONE well-known
function. This file is that function. It sequences:
  scanner  →  research  →  predict  →  portfolio approve  →  paper trade
and writes a structured run log so you can tell, one row per day, whether
the pipeline finished cleanly.

CRON (simplest) — add to `crontab -e`:
    30 16 * * 1-5  cd "/path/to/Stock Market AI Bot" && \
                   /usr/bin/python3 orchestration.py daily >> logs/cron.log 2>&1

PREFECT / AIRFLOW:
Wrap `run_daily()` in a Task/DAG. The function is idempotent (safe to
retry) and returns a dict the orchestrator can surface in its UI.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

from settings import BASE_DIR, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "orchestration.log")),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orchestration")

STEPS = [
    ("scanner",  [sys.executable, "scanner.py"]),
    ("research", [sys.executable, "research.py"]),
    ("predict",  [sys.executable, "predict.py"]),
    ("paper",    [sys.executable, "moomoo_paper_trading.py"]),
]


def _run_step(name: str, cmd: list[str], timeout: int = 1800) -> dict:
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=BASE_DIR, check=False, timeout=timeout,
                            capture_output=True, text=True)
        return {"step": name, "rc": cp.returncode,
                "seconds": round(time.time() - t0, 1),
                "stderr_tail": cp.stderr[-500:] if cp.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"step": name, "rc": 124, "seconds": timeout, "stderr_tail": "timeout"}


def run_daily() -> dict:
    """Run every step, short-circuit on first failure."""
    run = {"started_at": datetime.utcnow().isoformat() + "Z", "steps": []}
    for name, cmd in STEPS:
        log.info("starting %s", name)
        result = _run_step(name, cmd)
        run["steps"].append(result)
        log.info("finished %s rc=%s", name, result["rc"])
        if result["rc"] != 0:
            log.error("step %s failed; aborting pipeline", name)
            break
    run["finished_at"] = datetime.utcnow().isoformat() + "Z"

    # append to a daily JSONL for easy grep / dashboarding
    with open(os.path.join(LOG_DIR, "pipeline_runs.jsonl"), "a") as f:
        f.write(json.dumps(run) + "\n")
    return run


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "daily":
        result = run_daily()
        ok = all(s["rc"] == 0 for s in result["steps"])
        sys.exit(0 if ok else 1)
    print(f"unknown mode: {mode}")
    sys.exit(2)
