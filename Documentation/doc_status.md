# status.py

## What This Script Does

`status.py` prints a quick terminal summary of the trading bot. It reads local
files in `signals/`, `logs/`, and `data/` and shows whether the current setup
looks ready, blocked, stale, or missing data.

It does not place trades and it does not call Alpaca. It only reports what the
latest saved files say.

## How To Run It

Full status screen:

```bash
python status.py
```

One-line status for cron logs or a terminal prompt:

```bash
python status.py --short
```

JSON output for scripts:

```bash
python status.py --json
```

Expected output includes:

- Live config approval and age
- Latest signal readiness
- Paper equity, cash, and return since the first recorded equity row
- Current positions
- Today's planned orders
- Factor and feature health
- Broker health
- Data freshness
- Last daily run status
- Next walkforward due date

## Key Concepts

**Signal** means the bot's latest trading instruction file. It says what the
strategy wants to hold.

**paper_ready** means the strategy says the signal is approved for paper
trading.

**gates_all_pass** means the required strategy safety checks passed.

**medium_risk_review_pass** means the medium-risk validation checks passed.

**READY** in `status.py` means all three broker-submit gates are true:
`paper_ready`, `gates_all_pass`, and `medium_risk_review_pass`.

**BLOCKED** means at least one of those gates is false or missing. In that case,
the status line prints the specific blocking field.

**Equity** means the current Alpaca paper account value.

**Gross exposure** means the total market value of open positions divided by
account equity. If the status snapshot does not store this directly, `status.py`
rebuilds it from saved position values.

**Since inception** means the return from the first row in
`signals/alpaca_paper_equity.csv`, not a hardcoded starting balance.

**Walkforward** means the scheduled research process that re-tests and approves
strategy parameters before they are used for new live signals.
