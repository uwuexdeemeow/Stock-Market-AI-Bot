"""Load legacy top-level modules from the reorganized script folder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent
_SCRIPT_DIR = _ROOT / "Stock picking scripts"


def load_reorganized_module(public_name: str, filename: str) -> ModuleType:
    source = _SCRIPT_DIR / filename
    spec_name = f"_stock_picking_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(spec_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {public_name} implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = module
    spec.loader.exec_module(module)
    return module


def export_public(module: ModuleType, namespace: dict) -> None:
    public = getattr(module, "__all__", None)
    if public is None:
        public = [name for name in vars(module) if not name.startswith("_")]
    namespace["__all__"] = list(public)
    namespace.update({name: getattr(module, name) for name in public})
