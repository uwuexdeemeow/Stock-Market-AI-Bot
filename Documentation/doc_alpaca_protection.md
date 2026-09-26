# Alpaca Protection

## What it does

`alpaca_protection.py` manages durable broker-side trailing stops for the core
SPY, QQQ, and TQQQ positions. The stops remain at Alpaca when this computer is
offline. It validates stop percentages, finds stale orders, and repairs missing
or incorrectly sized protection.

## How to use it

This support module is imported by `alpaca_paper_trading.py` and
`execution_guard.py`; do not run it by itself. Configure it with
`GUARD_CORE_STOP`, `GUARD_CORE_TICKERS`, `GUARD_CORE_TRAIL_PCT`, and
`GUARD_TQQQ_TRAIL_PCT`. Expected output is a repair result containing checked,
cancelled, submitted, skipped, and error rows.

## Key terms

- **Trailing stop:** a sell trigger that follows a rising price.
- **Broker-side:** stored at Alpaca instead of depending on a local process.
- **Repair:** make open protection match the broker's current position.

## September 2026: core ETF stops off by default

`CORE_PROTECTION_ENABLED` now defaults to off (`GUARD_CORE_STOP=0` in the
daily workflow too). The backtest never had these stops, and the
pre-registered test H-edge-core-stop (`DELAY_STRESS_PAPER_ADVISORY.md`) found
that 5% stops on SPY/QQQ fire in about 44% of 20-day periods and cut
2013–2022 alpha vs QQQ by more than half. `cancel_core_etf_protective_stops`
is still used: `alpaca_paper_trading.py --submit` calls it to remove stops
left on the account from before the change.
