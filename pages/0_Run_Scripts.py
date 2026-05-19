"""
Run Scripts — control panel for backend operations.

PLAIN ENGLISH: Every .py script in the project root is auto-discovered and
rendered as an expandable card with widgets for each of its argparse flags.
Add a new script with proper argparse and it appears here on next reload —
no manual UI wiring required.

Top of page: Quick Actions for the most-used one-click workflows.
Below: every script grouped by category (live trading / research / robustness
/ infra), with toggles for flags and inputs for parameters.
"""

import sys
from pathlib import Path

import streamlit as st

from dashboard.components import sidebar_refresh, run_script
from dashboard.scripts import (
    Script,
    ScriptArg,
    discover_scripts,
    scripts_by_category,
)


# ─────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Run Scripts", page_icon="•", layout="wide")
sidebar_refresh()

st.title("Run Scripts")
st.caption(
    "Auto-discovered from the project root. Every script's `--flags` and "
    "parameters are introspected and rendered as widgets. Add a new script "
    "with a clean argparse block and it appears here next reload."
)

PY = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def _widget_key(script_stem: str, arg_name: str) -> str:
    """Stable streamlit widget key — must be unique per script+arg."""
    clean_arg = arg_name.lstrip("-").replace("-", "_")
    return f"{script_stem}__{clean_arg}"


def _flag_label(arg: ScriptArg) -> str:
    """Compact widget label that includes the flag literal."""
    base = arg.name
    if arg.short and arg.short != arg.name:
        base = f"{arg.name} / {arg.short}"
    return base


def _render_arg(script: Script, arg: ScriptArg) -> tuple[bool, list[str]]:
    """Render a widget for one argparse arg.

    Returns:
        (is_set, cli_tokens) — is_set is True if the user enabled/changed
        the value from its default; cli_tokens are the CLI strings to add
        when building the command line.
    """
    key = _widget_key(script.stem, arg.name)
    help_text = arg.help or None

    if arg.type == "flag":
        # Boolean toggle
        default_on = bool(arg.default)
        val = st.checkbox(_flag_label(arg), value=default_on, key=key, help=help_text)
        # Only add flag to command if it deviates from the default
        if val and not default_on:
            return True, [arg.name]
        if (not val) and default_on:
            # store_false: the flag suppresses something on by default — we
            # don't have a way to "un-set" without a paired flag.  Skip.
            return False, []
        return False, []

    if arg.type == "choices" and arg.choices:
        options = list(arg.choices)
        default = str(arg.default) if arg.default is not None else options[0]
        if default not in options:
            options = [default] + options
        idx = options.index(default) if default in options else 0
        val = st.selectbox(_flag_label(arg), options, index=idx, key=key, help=help_text)
        if str(val) != str(arg.default):
            return True, [arg.name, str(val)]
        return False, []

    if arg.type in ("int", "integer"):
        default = int(arg.default) if arg.default is not None else 0
        val = st.number_input(
            _flag_label(arg), value=default, step=1, key=key, help=help_text,
        )
        if int(val) != default:
            return True, [arg.name, str(int(val))]
        return False, []

    if arg.type == "float":
        default = float(arg.default) if arg.default is not None else 0.0
        val = st.number_input(
            _flag_label(arg), value=default, step=0.1, format="%.3f",
            key=key, help=help_text,
        )
        if abs(float(val) - default) > 1e-9:
            return True, [arg.name, str(float(val))]
        return False, []

    # Default: text input (string)
    default = "" if arg.default is None else str(arg.default)
    val = st.text_input(_flag_label(arg), value=default, key=key, help=help_text)
    val_stripped = val.strip()
    if val_stripped and val_stripped != default:
        if arg.nargs in ("+", "*"):
            return True, [arg.name, *val_stripped.split()]
        return True, [arg.name, val_stripped]
    return False, []


def _render_script_card(script: Script) -> None:
    """Render one script as an expandable card with widgets + Run button."""
    # Headline = filename + the first sentence of its docstring (or the
    # filename's "Title Case" form if the docstring is empty).
    summary = script.description or script.display_name
    n_args = len(script.args)
    args_label = f"{n_args} option{'s' if n_args != 1 else ''}" if n_args else "no options"

    header = f"**{script.stem}.py**  ·  {args_label}"
    with st.expander(header, expanded=False):
        if summary:
            st.caption(summary)

        cli_tokens: list[str] = []
        if script.args:
            # Split args into flags and parameters for clearer layout
            flags = [a for a in script.args if a.is_flag]
            params = [a for a in script.args if not a.is_flag]

            if flags:
                st.markdown("**Flags**")
                # Render flag toggles in columns of 2
                for i in range(0, len(flags), 2):
                    cols = st.columns(2)
                    for j, arg in enumerate(flags[i : i + 2]):
                        with cols[j]:
                            _, toks = _render_arg(script, arg)
                            cli_tokens.extend(toks)

            if params:
                if flags:
                    st.markdown("**Parameters**")
                for arg in params:
                    _, toks = _render_arg(script, arg)
                    cli_tokens.extend(toks)
        else:
            st.caption("This script takes no CLI arguments — just press Run.")

        cmd = [PY, f"{script.stem}.py", *cli_tokens]
        st.code(" ".join(cmd), language="bash")

        run_key = f"run_btn_{script.stem}"
        if st.button("Run", key=run_key, type="primary", use_container_width=True):
            run_script(cmd, script.display_name, cwd=PROJECT_ROOT)
            st.cache_data.clear()


# ─────────────────────────────────────────────────────────────────────────
# QUICK ACTIONS — pinned at top (curated, not auto-generated)
# ─────────────────────────────────────────────────────────────────────────
st.markdown("### Quick Actions")
qa1, qa2, qa3, qa4 = st.columns(4)

with qa1:
    if st.button("Daily run (Alpaca)", type="primary", use_container_width=True,
                 help="daily_run.py --alpaca --skip-refresh --timeout 600"):
        run_script(
            [PY, "daily_run.py", "--alpaca", "--skip-refresh", "--timeout", "600"],
            "Daily run", cwd=PROJECT_ROOT,
        )
        st.cache_data.clear()

with qa2:
    if st.button("Refresh data + signal", use_container_width=True,
                 help="ETF refresh + research incremental + feature_quality + signal"):
        for cmd, label in [
            ([PY, "refresh_etf_data.py", "--refresh", "--force"], "Refresh ETF"),
            ([PY, "research.py", "--incremental"], "Refresh research"),
            ([PY, "feature_quality_diagnostic.py", "--top", "48"], "Feature quality"),
            ([PY, "core_satellite_alpha.py"], "Generate signal"),
        ]:
            run_script(cmd, label, cwd=PROJECT_ROOT)
        st.cache_data.clear()

with qa3:
    if st.button("Alpaca status", use_container_width=True,
                 help="Read-only — print positions, cash, equity"):
        run_script([PY, "alpaca_paper_trading.py", "--status"], "Account status",
                   cwd=PROJECT_ROOT)
        st.cache_data.clear()

with qa4:
    if st.button("Stable-grid walkforward", use_container_width=True,
                 help="48-config research run — ~20 min"):
        run_script(
            [PY, "core_satellite_nested_walkforward.py",
             "--stable-grid", "--workers", "4", "--no-publish-live-config"],
            "Stable-grid walkforward", cwd=PROJECT_ROOT,
        )
        st.cache_data.clear()

st.divider()


# ─────────────────────────────────────────────────────────────────────────
# DYNAMIC SCRIPT LIST — auto-generated from project root
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _load_scripts() -> dict[str, list[Script]]:
    """Discover scripts (cached briefly so reloads are snappy)."""
    return scripts_by_category(discover_scripts(PROJECT_ROOT))


grouped = _load_scripts()

# Top-of-list summary so user knows what was found
total = sum(len(v) for v in grouped.values())
st.markdown(f"### All Scripts  ·  *{total} discovered*")
st.caption(
    "Each script's argparse is parsed safely (without executing the file). "
    "Booleans → checkboxes · choices → dropdowns · int/float → number inputs · "
    "everything else → text inputs. Use the Refresh button in the sidebar to rescan."
)

# Search filter — handy when scripts get numerous
search = st.text_input(
    "Filter scripts", value="", placeholder="Type to filter by name or description…",
    key="script_search",
).strip().lower()


def _matches(script: Script, query: str) -> bool:
    if not query:
        return True
    return query in script.stem.lower() or query in script.description.lower()


# Tabs per category — order matters (live trading first, then research, etc.)
category_order = [
    "Live paper trading",
    "Strategy & research",
    "Robustness & validation",
    "Infrastructure",
    "Other",
]
present_categories = [c for c in category_order if c in grouped]

tabs = st.tabs(present_categories)

for tab, cat in zip(tabs, present_categories):
    with tab:
        items = grouped.get(cat, [])
        filtered = [s for s in items if _matches(s, search)]
        if not filtered:
            if search:
                st.caption(f"No scripts match '{search}' in this category.")
            else:
                st.caption("No scripts in this category.")
            continue

        # Render two columns of cards on wide screens — keeps the page tight.
        for script in filtered:
            _render_script_card(script)
