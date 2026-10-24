"""
Guard: live code must not load the legacy ML layer again.

PLAIN ENGLISH: the old XGBoost scripts (train.py, backtest.py, ...) were
moved to legacy/ on 2026-10-25 (Documentation/LEGACY_ML_CLEANUP_PLAN.md).
The daily trading install no longer has xgboost, scikit-learn or matplotlib.
If a live script imported a legacy module again, the daily run would crash
with "No module named ...".  These tests catch that before it reaches main.

Two kinds of check:
- static: read every live .py file's import lines (including imports that
  sit inside functions) without running them;
- runtime: start a fresh Python, import the main entry points, and look at
  which modules got loaded.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every module that moved to legacy/.
LEGACY_MODULES = {
    "train", "predict", "labels", "model_quality", "model_self_check",
    "model_registry", "confidence_calibration", "calibration_stability",
    "nested_cv", "xgb_feature_engineering", "shap_feature_reducer",
    "drift_monitor", "leakage_audit", "alternative_data_features", "backtest",
}
# Heavy ML packages that are only in requirements-legacy.txt now.
LEGACY_PACKAGES = {"xgboost", "sklearn", "matplotlib", "shap", "seaborn"}

# The scripts the daily run, the monthly V monitor and core-alpha start.
ENTRY_POINTS = [
    "daily_run", "alpaca_paper_trading", "v_rule_signal", "v_rule_helper",
    "broker_truth", "broker_health", "paper_health", "run_evidence",
    "execution_scorecard", "execution_guard", "alpaca_protection",
    "fill_monitor", "forward_test_monitor", "refresh_etf_data",
    "core_satellite_alpha", "alpha_factor_backtest", "factor_decay_monitor",
    "quant_performance_audit", "research", "quant_common",
]


def _live_python_files() -> list[Path]:
    """Root scripts plus the dashboard (everything except legacy/ and tests)."""
    files = list(ROOT.glob("*.py"))
    files += list((ROOT / "dashboard").glob("*.py"))
    files += list((ROOT / "pages").glob("*.py"))
    return sorted(files)


def _imported_top_names(path: Path) -> set[str]:
    """First part of every module name the file imports (e.g. 'sklearn')."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_legacy_scripts_are_not_back_in_the_project_root():
    back = sorted(name for name in LEGACY_MODULES if (ROOT / f"{name}.py").exists())
    assert back == [], f"Legacy scripts belong in legacy/: {back}"


@pytest.mark.parametrize("path", _live_python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_live_file_does_not_import_legacy_code_or_ml_packages(path):
    bad = sorted(_imported_top_names(path) & (LEGACY_MODULES | LEGACY_PACKAGES))
    assert bad == [], f"{path.name} imports legacy code/packages: {bad}"


def test_live_files_do_not_start_legacy_scripts():
    """A step like [python, 'drift_monitor.py'] would also fail on the runner.

    An explicit 'legacy/<name>.py' path is allowed: that is a deliberate,
    opt-in call to the old code (it needs requirements-legacy.txt).
    """
    hits = []
    for path in _live_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("\\", "/")
                if value.endswith(".py") and Path(value).stem in LEGACY_MODULES and not value.startswith("legacy/"):
                    hits.append(f"{path.name}:{node.lineno} {node.value}")
    assert hits == []


def test_entry_points_load_no_legacy_module_at_runtime():
    code = (
        "import importlib, json, sys\n"
        f"for name in {ENTRY_POINTS!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    loaded = set(json.loads(done.stdout.strip().splitlines()[-1]))
    assert sorted(loaded & (LEGACY_MODULES | LEGACY_PACKAGES)) == []
