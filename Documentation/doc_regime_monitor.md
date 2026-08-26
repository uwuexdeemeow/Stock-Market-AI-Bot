# Regime Monitor

## What it does

`regime_monitor.py` records market-regime changes and reports whether the
strategy has moved between risk-on and defensive states.

## How to run it

Run `python3 regime_monitor.py`; add `--quiet` for machine-oriented use. It
reads the current signal/regime inputs and updates structured regime history in
the signals/log area. Missing or unchanged state produces a clear status rather
than a fabricated transition.

## Key terms

- **Regime:** a broad market state used to change exposure.
- **Risk-on:** normal growth-oriented allocation.
- **Defensive:** reduced or safer allocation during stress.
