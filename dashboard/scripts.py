"""
dashboard/scripts.py — Dynamic script discovery & argparse introspection.

PLAIN ENGLISH: Scans the project for every .py file, safely reads each one
without executing it, and extracts:
  - the docstring (used as the "What it does" caption)
  - every parser.add_argument() call (flags, types, choices, defaults, help)
  - a category guess (Research / Live trading / Robustness / Infrastructure)

The Run Scripts page uses this to auto-generate a widget per script.  Add a
new script with a clean argparse block and it appears in the dashboard on
the next page reload — no manual wiring needed.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Category map ─────────────────────────────────────────────────────────
# Mirrors the grouping in CLAUDE.md so users see scripts where they expect.
# Anything not matched falls into "Other".
SCRIPT_CATEGORIES: dict[str, set[str]] = {
    "Strategy & research": {
        "backtest", "confidence_calibration", "core_satellite_alpha",
        "core_satellite_nested_walkforward", "core_satellite_tqqq",
        "diagnostics", "feature_quality_diagnostic", "fundamental_features",
        "labels", "leakage_audit", "model", "model_quality",
        "model_self_check", "pipeline_shared", "portfolio_manager",
        "predict", "research", "sentiment_engine", "settings",
        "social_sentiment", "train", "xgb_feature_engineering",
        "ranker_utils", "calibration_stability", "cross_sectional_features",
        "intraday_features", "alternative_data_features",
    },
    "Robustness & validation": {
        "alpha_factor_backtest", "concentration_overlay",
        "core_satellite_drawdown_throttle", "core_satellite_execution_stress",
        "core_satellite_survivorship_audit", "factor_decay_monitor",
        "factor_data_health",
        "feature_health", "nested_cv", "regime_monitor", "robustness_scoring",
        "survivorship_audit", "walkforward_analyzer",
    },
    "Live paper trading": {
        "alpaca_paper_gauntlet", "alpaca_paper_trading", "alpaca_protection",
        "broker_health", "broker_interface", "daily_paper_check",
        "daily_run", "execution_guard", "execution_model", "fill_monitor",
        "paper_gauntlet", "paper_health", "paper_report", "paper_scorecard",
        "publish_live_config_from_csv", "refresh_etf_data", "risk_sizing",
        "signal_freshness", "status", "trade_rules", "moomoo_paper_trading",
    },
    "Infrastructure": {
        "config_health", "data_provider", "data_validation",
        "experiment_ledger", "http_retry", "log_cleanup", "monitor",
        "monitor_heartbeat", "notifications", "options_iv_provider",
        "safe_io",
    },
}


def _categorize(stem: str) -> str:
    for category, names in SCRIPT_CATEGORIES.items():
        if stem in names:
            return category
    return "Other"


# ── Data classes ─────────────────────────────────────────────────────────
@dataclass
class ScriptArg:
    """One CLI flag extracted from argparse."""
    name: str                           # "--ticker"  (long form)
    short: str | None = None            # "-t"
    type: str = "flag"                  # flag / string / int / float / choices
    help: str = ""
    default: Any = None
    choices: list[str] | None = None
    required: bool = False
    nargs: str | None = None            # "+" / "*" / "?" — list-valued

    @property
    def is_flag(self) -> bool:
        return self.type == "flag"


@dataclass
class Script:
    """Metadata extracted from a project .py file."""
    path: Path
    stem: str                           # filename without .py
    description: str = ""               # first non-empty docstring line
    docstring: str = ""                 # full docstring (truncated)
    args: list[ScriptArg] = field(default_factory=list)
    category: str = "Other"
    has_argparse: bool = False
    has_main: bool = False              # has __main__ block

    @property
    def display_name(self) -> str:
        return self.stem.replace("_", " ").title()


# ── AST extraction ───────────────────────────────────────────────────────
def _safe_literal(node: ast.AST) -> Any:
    """Evaluate a literal AST node (string, int, float, tuple, list).

    Returns None on anything dynamic (function calls, names, attributes).
    Keeps argparse extraction sandboxed — we never execute the script.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    """Return the keyword-argument node by name, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _parse_add_argument(call: ast.Call) -> ScriptArg | None:
    """Translate one parser.add_argument(...) call into a ScriptArg.

    Best-effort: anything we can't statically resolve becomes a sane default.
    """
    # First positional args are the flag names.  Could be "--name" or "name".
    positional_names: list[str] = []
    for arg in call.args:
        val = _safe_literal(arg)
        if isinstance(val, str):
            positional_names.append(val)
    if not positional_names:
        return None

    # Pick the long form first; keep short if both present.
    long_name = next((n for n in positional_names if n.startswith("--")), None)
    short_name = next((n for n in positional_names if n.startswith("-") and not n.startswith("--")), None)

    # Positional args (no leading dash) — render as text input.
    if long_name is None and short_name is None:
        # First positional — treat it like --positional_name
        long_name = "--" + positional_names[0].replace("_", "-")

    name = long_name or "--" + (short_name or positional_names[0]).lstrip("-")

    # Keyword extraction
    action_val = _safe_literal(_kw(call, "action")) if _kw(call, "action") else None
    type_node = _kw(call, "type")
    type_val: str = "string"
    if action_val in ("store_true", "store_false"):
        type_val = "flag"
    elif type_node is not None:
        # type=int / type=float
        if isinstance(type_node, ast.Name):
            type_val = type_node.id  # "int", "float", "str"
        elif isinstance(type_node, ast.Attribute):
            type_val = type_node.attr

    choices_val = _safe_literal(_kw(call, "choices")) if _kw(call, "choices") else None
    if choices_val and isinstance(choices_val, (list, tuple)):
        type_val = "choices"
        choices_list: list[str] = [str(c) for c in choices_val]
    else:
        choices_list = None

    default_val = _safe_literal(_kw(call, "default")) if _kw(call, "default") else None
    if action_val == "store_true" and default_val is None:
        default_val = False
    if action_val == "store_false" and default_val is None:
        default_val = True

    help_val = _safe_literal(_kw(call, "help")) if _kw(call, "help") else ""
    required_val = _safe_literal(_kw(call, "required")) if _kw(call, "required") else False
    nargs_val = _safe_literal(_kw(call, "nargs")) if _kw(call, "nargs") else None

    return ScriptArg(
        name=name,
        short=short_name,
        type=type_val,
        help=str(help_val or ""),
        default=default_val,
        choices=choices_list,
        required=bool(required_val),
        nargs=str(nargs_val) if nargs_val is not None else None,
    )


def _extract_main_docstring(tree: ast.Module) -> str:
    """Return the module's docstring (cleaned)."""
    doc = ast.get_docstring(tree, clean=True)
    return (doc or "").strip()


def _has_argparse_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(a.name == "argparse" for a in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module == "argparse":
            return True
    return False


def _has_main_block(tree: ast.Module) -> bool:
    """Detect `if __name__ == "__main__":` so we know it's runnable."""
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            # Match: name == '__main__'  (left or right side)
            if isinstance(test, ast.Compare):
                operands = [test.left] + list(test.comparators)
                strs = [
                    _safe_literal(o) for o in operands if not isinstance(o, ast.Name)
                ]
                names = [o.id for o in operands if isinstance(o, ast.Name)]
                if "__name__" in names and "__main__" in [s for s in strs if isinstance(s, str)]:
                    return True
    return False


def parse_script(path: Path) -> Script | None:
    """Parse a .py file and return a Script metadata object.

    Skips files that fail to parse (syntax errors, partial files).
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None

    args: list[ScriptArg] = []
    # Walk the whole tree — argparse calls can hide inside functions.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            attr = None
            if isinstance(func, ast.Attribute):
                attr = func.attr
            if attr == "add_argument":
                parsed = _parse_add_argument(node)
                if parsed:
                    # Deduplicate by name — some scripts call add_argument
                    # twice with same flag in different code paths.
                    if not any(a.name == parsed.name for a in args):
                        args.append(parsed)

    docstring = _extract_main_docstring(tree)
    # First non-empty line of docstring → "description"
    description = ""
    for line in docstring.splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            description = cleaned
            break

    return Script(
        path=path,
        stem=path.stem,
        description=description,
        docstring=docstring[:600],  # truncate for UI
        args=args,
        category=_categorize(path.stem),
        has_argparse=_has_argparse_import(tree),
        has_main=_has_main_block(tree),
    )


# ── Discovery ─────────────────────────────────────────────────────────────
# Hidden / internal scripts we don't want to expose as "runnable".
HIDDEN_SCRIPTS: set[str] = {
    "dashboard",            # would create a runaway recursion
    "settings",             # imported, not run
    "pipeline_shared",      # imported module
    "broker_interface",     # imported module
    "data_provider", "http_retry", "safe_io", "notifications",
    "model", "labels",      # imported by train.py — rarely run directly
    "execution_model", "trade_rules",
    "options_iv_provider", "sentiment_engine", "social_sentiment",
    "ranker_utils", "robustness_scoring",
    "cross_sectional_features", "intraday_features",
    "alternative_data_features", "fundamental_features",
    "xgb_feature_engineering",
    "feature_health",
}


def discover_scripts(project_root: Path | str) -> list[Script]:
    """Walk the project root and return all parseable Script objects.

    Skips:
      - Files in HIDDEN_SCRIPTS (helper modules not meant to be run)
      - Files with no main block AND no argparse (clearly imported-only)
    """
    root = Path(project_root)
    scripts: list[Script] = []
    for py_path in sorted(root.glob("*.py")):
        if py_path.stem in HIDDEN_SCRIPTS:
            continue
        parsed = parse_script(py_path)
        if parsed is None:
            continue
        # Skip clearly non-runnable modules (no main, no argparse)
        if not parsed.has_main and not parsed.has_argparse:
            continue
        scripts.append(parsed)
    return scripts


def scripts_by_category(scripts: list[Script]) -> dict[str, list[Script]]:
    """Group discovered scripts into the CLAUDE.md categories."""
    grouped: dict[str, list[Script]] = {
        "Live paper trading": [],
        "Strategy & research": [],
        "Robustness & validation": [],
        "Infrastructure": [],
        "Other": [],
    }
    for s in scripts:
        grouped.setdefault(s.category, []).append(s)
    # Drop empty categories so the UI doesn't show empty tabs.
    return {k: v for k, v in grouped.items() if v}
