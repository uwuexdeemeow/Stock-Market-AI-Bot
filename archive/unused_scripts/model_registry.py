"""
model_registry.py — MLflow-backed (optional) model registry with JSON fallback.

PLAIN ENGLISH:
Every training run should be reproducible. A "model registry" stores:
  * the binary model artifact (the .pkl/.json the code loads)
  * metadata: training window, features used, metrics, git commit, seed
  * a version tag so you can point prod at a specific version

If MLflow is installed, we use it. If not, we fall back to a plain JSON
ledger at `models/registry.json` so we still get provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from settings import MODEL_DIR

REGISTRY_JSON = os.path.join(MODEL_DIR, "registry.json")

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=MODEL_DIR, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def register(model_path: str, metrics: dict[str, Any], params: dict[str, Any], tag: str = "latest") -> dict:
    """Record a trained model. Copies artifact into a versioned path."""
    now_utc = datetime.now(timezone.utc)
    version = now_utc.strftime("%Y%m%d_%H%M%S")
    dst_dir = os.path.join(MODEL_DIR, "versions", version)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(model_path))
    shutil.copy2(model_path, dst_path)

    entry = {
        "version": version,
        "tag": tag,
        "path": dst_path,
        "sha256": _sha256(dst_path),
        "git": _git_sha(),
        "metrics": metrics,
        "params": params,
        "created_at": now_utc.isoformat(),
    }

    ledger = []
    if os.path.exists(REGISTRY_JSON):
        with open(REGISTRY_JSON) as f:
            ledger = json.load(f)
    ledger.append(entry)
    with open(REGISTRY_JSON, "w") as f:
        json.dump(ledger, f, indent=2)

    if _HAS_MLFLOW:
        mlflow.set_experiment("stock-ai-bot")
        with mlflow.start_run(run_name=version):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(dst_path)
    return entry


def latest(tag: str = "production") -> dict | None:
    if not os.path.exists(REGISTRY_JSON):
        return None
    with open(REGISTRY_JSON) as f:
        ledger = json.load(f)
    matches = [e for e in ledger if e.get("tag") == tag]
    return matches[-1] if matches else None
