"""
diagnostics.py — deeper environment/runtime checks.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg): print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {BLUE}→{RESET}  {msg}")
def hdr(msg): print(f"\n{BOLD}{msg}{RESET}\n{'─'*60}")


def detect_acceleration_backend():
    result = {
        "device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "cuda_device_name": None,
        "nvidia_hardware_present": False,
        "nvidia_hardware_name": None,
        "reason": "No CUDA or MPS backend detected",
    }
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            smi = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            if smi.returncode == 0:
                names = [x.strip() for x in smi.stdout.splitlines() if x.strip()]
                if names:
                    result["nvidia_hardware_present"] = True
                    result["nvidia_hardware_name"] = names[0]
    except Exception:
        pass

    try:
        import torch
    except Exception as e:
        result["reason"] = f"torch import failed: {e}"
        return result

    try:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["device"] = "cuda"
            result["reason"] = "CUDA-capable GPU detected and usable by PyTorch"
            return result
    except Exception as e:
        result["reason"] = f"torch.cuda check failed: {e}"

    try:
        result["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if result["mps_available"]:
            result["device"] = "mps"
            result["reason"] = "Apple Silicon MPS detected and usable by PyTorch"
            return result
    except Exception as e:
        result["reason"] = f"torch MPS check failed: {e}"

    if result["nvidia_hardware_present"]:
        result["reason"] = (
            f"NVIDIA GPU detected ({result['nvidia_hardware_name']}) but PyTorch CUDA is unavailable. "
            "This usually means a CPU-only torch build, missing/unsupported CUDA runtime, or driver mismatch."
        )
    return result


def check_env():
    hdr("SYSTEM / PYTHON")
    ok(f"Python: {sys.version.split()[0]}")
    ok(f"Platform: {platform.platform()}")
    ok(f"Executable: {sys.executable}")


def check_acceleration():
    hdr("ACCELERATION BACKEND")
    backend = detect_acceleration_backend()
    print(json.dumps(backend, indent=2))
    if backend["device"] == "cuda":
        ok(f"CUDA usable: {backend['cuda_device_name']}")
    elif backend["device"] == "mps":
        ok("MPS usable")
    else:
        warn(backend["reason"])


def check_imports():
    hdr("IMPORTS")
    modules = ["torch", "xgboost", "shap", "numpy", "pandas", "sklearn", "pyarrow", "moomoo"]
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            ok(f"{mod}: {ver}")
        except Exception as e:
            warn(f"{mod}: {e}")


def check_xgboost_runtime():
    hdr("XGBOOST RUNTIME")
    try:
        import numpy as np
        import xgboost as xgb

        X = np.random.randn(128, 8).astype("float32")
        y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype("int32")
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X, y)
        proba = model.predict_proba(X[:4])
        ok(f"XGBoost fit/predict passed (shape={proba.shape})")
    except Exception as e:
        err(f"XGBoost runtime failed: {e}")
        msg = str(e).lower()
        if "omp" in msg or "openmp" in msg or "libiomp" in msg:
            info("Likely OpenMP runtime conflict. On macOS remove duplicate OpenMP installs or use a consistent conda/pip stack.")


def check_shap():
    hdr("SHAP")
    try:
        import shap
        ok(f"SHAP import passed ({getattr(shap, '__version__', 'unknown')})")
    except Exception as e:
        err(f"SHAP failed: {e}")


def check_model_artifacts():
    hdr("MODEL ARTIFACTS")
    base = Path(__file__).resolve().parent / "models"
    if not base.exists():
        warn("models/ folder not found")
        return
    files = list(base.glob("*"))
    if not files:
        warn("models/ exists but is empty")
        return
    ok(f"Found {len(files)} files in models/")
    for suffix in [".pt", ".pkl", ".json"]:
        count = sum(1 for p in files if p.suffix == suffix)
        info(f"  {suffix}: {count}")


def main():
    print(f"\n{'='*60}")
    print("  DEEP DIAGNOSTICS")
    print(f"{'='*60}")
    check_env()
    check_acceleration()
    check_imports()
    check_xgboost_runtime()
    check_shap()
    check_model_artifacts()


if __name__ == "__main__":
    main()
