"""
regime_monitor.py — Detect and alert on regime changes.

PLAIN ENGLISH: Every time the daily pipeline generates a new signal, the
strategy picks a 'regime' (risk_on, neutral, or risk_off) based on market
conditions.  A regime change means your portfolio allocation shifts
significantly — more/less stocks, different ETF mix, different risk level.

This script compares today's regime to the last known regime.  If it changed,
it sends a macOS notification and logs the transition.  This way you know
immediately when your strategy is shifting gears, without checking logs.

Usage:
    python3 regime_monitor.py          # check both strategies and alert on change
    python3 regime_monitor.py --quiet  # only print if there's a change

How it works:
    1. Reads the current signal CSV(s) for the regime column
    2. Reads a small JSON file (signals/regime_history.json) with the last known regime
    3. If regime changed → send macOS notification, update history file
    4. If regime is the same → do nothing (no spam)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from settings import SIGNAL_DIR

SIGNALS = Path(SIGNAL_DIR)

# Where we store the regime history — a small JSON file that tracks
# the last known regime for each strategy so we can detect changes.
REGIME_HISTORY_PATH = SIGNALS / "regime_history.json"

# Signal files for each strategy
# Both brokers now read the same unified signal file.  Two entries kept so
# regime_history.json preserves per-broker tracking if they ever diverge.
SIGNAL_FILES = {
    "moomoo_core_satellite": SIGNALS / "core_satellite_alpha_signal.csv",
    "alpaca_core_satellite": SIGNALS / "core_satellite_alpha_signal.csv",
}


def _load_regime_history() -> dict:
    """
    Load the last known regime for each strategy from disk.

    PLAIN ENGLISH: Reads a tiny JSON file that remembers what regime each
    strategy was in last time we checked.  If the file doesn't exist yet
    (first run), returns an empty dict.
    """
    if REGIME_HISTORY_PATH.exists():
        try:
            return json.loads(REGIME_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_regime_history(history: dict) -> None:
    """Save the updated regime history to disk."""
    REGIME_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGIME_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )


def _read_signal_regime(signal_path: Path) -> dict | None:
    """
    Read the current regime and related info from a signal CSV.

    PLAIN ENGLISH: Opens the signal file and extracts the regime name,
    core/overlay weights, and when the signal was generated.  Returns
    None if the file doesn't exist or can't be read.
    """
    if not signal_path.exists():
        return None
    try:
        df = pd.read_csv(signal_path)
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "regime": str(row.get("current_regime", "unknown")),
            "core_gross": float(row.get("core_gross", 0)),
            "overlay_gross": float(row.get("overlay_gross", 0)),
            "overlay_tickers": str(row.get("overlay_tickers", "")),
            "predicted_at": str(row.get("predicted_at", "")),
        }
    except Exception:
        return None


def _macos_notification(title: str, message: str) -> None:
    """
    Show a native macOS notification banner.

    PLAIN ENGLISH: Pops a notification on your Mac's screen.
    If it fails (e.g. running on Linux), it just prints a warning.
    """
    try:
        escaped_title = title.replace('"', '\\"')
        escaped_msg = message.replace('"', '\\"')
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{escaped_msg}" with title "{escaped_title}"',
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        print(f"  ⚠ Could not send macOS notification: {e}")


def check_regime_changes(*, quiet: bool = False) -> list[dict]:
    """
    Check all strategies for regime changes and alert if any changed.

    PLAIN ENGLISH: For each strategy (Moomoo and Alpaca), read the current
    signal file, compare the regime to what we saw last time, and if it
    changed, send a notification and update the history.

    Returns a list of change records (empty if nothing changed).
    """
    history = _load_regime_history()
    changes: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")

    for strategy_name, signal_path in SIGNAL_FILES.items():
        current = _read_signal_regime(signal_path)
        if current is None:
            if not quiet:
                print(f"  [{strategy_name}] No signal file found — skipping")
            continue

        new_regime = current["regime"]
        prev_entry = history.get(strategy_name, {})
        prev_regime = prev_entry.get("regime", None)

        if prev_regime is None:
            # First run — just record the current regime, no alert
            if not quiet:
                print(f"  [{strategy_name}] First check — regime is {new_regime} (no previous data)")
            history[strategy_name] = {
                "regime": new_regime,
                "since": now,
                "core_gross": current["core_gross"],
                "overlay_gross": current["overlay_gross"],
            }
            continue

        if new_regime == prev_regime:
            if not quiet:
                print(f"  [{strategy_name}] Regime unchanged: {new_regime}")
            continue

        # REGIME CHANGED — this is the important case
        change = {
            "strategy": strategy_name,
            "from": prev_regime,
            "to": new_regime,
            "changed_at": now,
            "core_gross": current["core_gross"],
            "overlay_gross": current["overlay_gross"],
            "overlay_tickers": current["overlay_tickers"],
        }
        changes.append(change)

        # Describe what the change means in plain english
        if new_regime == "risk_on":
            impact = "More aggressive — heavier QQQ, more overlay stocks"
        elif new_regime == "risk_off":
            impact = "Defensive — more SPY, fewer overlay stocks"
        elif new_regime == "neutral":
            impact = "Balanced — moderate SPY/QQQ mix"
        else:
            impact = "Unknown regime"

        # Print the change
        print(f"\n  🔄 REGIME CHANGE: [{strategy_name}]")
        print(f"     {prev_regime} → {new_regime}")
        print(f"     Impact: {impact}")
        print(f"     Core gross: {current['core_gross']:.0%}")
        print(f"     Overlay gross: {current['overlay_gross']:.0%}")
        if current["overlay_tickers"]:
            print(f"     Overlay stocks: {current['overlay_tickers']}")

        # Send macOS notification
        _macos_notification(
            f"⚡ Regime Change: {prev_regime} → {new_regime}",
            f"{strategy_name}: {impact}. Core={current['core_gross']:.0%} Overlay={current['overlay_gross']:.0%}",
        )

        # Update history
        history[strategy_name] = {
            "regime": new_regime,
            "since": now,
            "previous_regime": prev_regime,
            "core_gross": current["core_gross"],
            "overlay_gross": current["overlay_gross"],
        }

    # Save updated history
    _save_regime_history(history)

    # Log regime change history for reference
    if changes:
        log_path = SIGNALS / "regime_changes_log.csv"
        change_df = pd.DataFrame(changes)
        if log_path.exists():
            # Append to existing log
            existing = pd.read_csv(log_path)
            change_df = pd.concat([existing, change_df], ignore_index=True)
        change_df.to_csv(log_path, index=False)
        print(f"\n  Regime change log updated → {log_path}")

    return changes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for regime changes and send alerts"
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Only print output if a regime change is detected")
    args = parser.parse_args()

    if not args.quiet:
        print(f"\n{'='*50}")
        print(f"  REGIME MONITOR")
        print(f"{'='*50}\n")

    changes = check_regime_changes(quiet=args.quiet)

    if not args.quiet:
        if not changes:
            print(f"\n  No regime changes detected")
        else:
            print(f"\n  {len(changes)} regime change(s) detected!")
        print()


if __name__ == "__main__":
    main()
