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
