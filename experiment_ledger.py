from __future__ import annotations

"""
experiment_ledger.py — append-only research ledger.

Every research/backtest run should leave a compact, machine-readable record so
strategy selection is based on evidence rather than memory. The JSONL file is
the source of truth; the CSV mirror is for quick spreadsheet inspection.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from settings import LOG_DIR

LEDGER_JSONL = os.path.join(LOG_DIR, "experiment_ledger.jsonl")
LEDGER_CSV = os.path.join(LOG_DIR, "experiment_ledger.csv")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def append_experiment(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    os.makedirs(LOG_DIR, exist_ok=True)
    row = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "git": _git_sha(),
        "params": _json_safe(params),
        "metrics": _json_safe(metrics),
        "artifacts": _json_safe(artifacts or {}),
        "notes": notes or "",
    }
    with open(LEDGER_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    flat = {
        "run_at": row["run_at"],
        "name": row["name"],
        "git": row["git"],
        "notes": row["notes"],
    }
    for prefix, data in (("param", row["params"]), ("metric", row["metrics"]), ("artifact", row["artifacts"])):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            flat[f"{prefix}_{key}"] = value

    old = pd.read_csv(LEDGER_CSV) if os.path.exists(LEDGER_CSV) else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame([flat])], ignore_index=True, sort=False)
    out.to_csv(LEDGER_CSV, index=False)
    return row
