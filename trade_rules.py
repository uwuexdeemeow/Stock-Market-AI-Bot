"""Compatibility import for the reorganized stock-picking scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "Stock picking scripts" / "trade_rules.py"
_SPEC = importlib.util.spec_from_file_location("_stock_picking_trade_rules", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load trade_rules implementation from {_SOURCE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

__all__ = [name for name in vars(_MODULE) if not name.startswith("_")]
globals().update({name: getattr(_MODULE, name) for name in __all__})
