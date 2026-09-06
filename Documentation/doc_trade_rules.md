# trade_rules.py - Per-Ticker Trade Rules

## What It Does

`trade_rules.py` defines simple per-ticker rules used to accept or reject
candidate trades. Rules can include confidence thresholds, minimum expected
return, stop loss, take profit, max position size, and whether shorts are
allowed.

Plain language: it stores the "minimum quality bar" a ticker must clear before
the bot should trade it.

## How To Run It

This file is mostly imported by other scripts. You normally do not run it by
itself.

Common outputs when other scripts save rules or reports:

- `models/<TICKER>_trade_rules.json`
- `models/trade_rule_report.csv`

Both outputs are written atomically, so a rule optimization run cannot leave a
half-written rule file or CSV report.

## Key Concepts

- Confidence threshold: minimum model confidence needed before a trade is valid.
- Expected return: model's projected move for the trade horizon.
- Stop loss: exit if price moves too far against the trade.
- Take profit: exit if price reaches the profit target.
- Trade rule report: table summarizing optimized or approved rule candidates.


## Remaining audit repair, September 2026

Rule exits now require the exact entry row and a complete horizon, check entry-day OHLC, and fill adverse opening stop gaps at the adverse open. Missing OHLC no longer becomes a close-only approximation. When intraday stop/profit order is unknowable, the stop is chosen conservatively; a known opening profit gap reaches its limit first. Zero stop/profit settings disable those triggers. Incomplete histories raise an error rather than shortening the holding period or inventing an exit. Run python -m pytest tests/test_corrected_audit.py -q for examples. Corrected daily trailing-stop accounting lives in portfolio_ledger.py.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.
