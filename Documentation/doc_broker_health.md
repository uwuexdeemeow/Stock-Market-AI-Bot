# broker_health.py - What It Does and How to Run It

## What This Script Does

`broker_health.py` is a pre-flight check for the Alpaca paper broker. It pings
the account before the daily trading flow tries to submit orders.

It checks:

- Alpaca credentials can connect.
- Account equity can be read.
- Equity is positive and finite, not missing, NaN, infinite, or zero.
- The paper account is active and not blocked from trading.
- Buying power is finite and non-negative.
- Alpaca's market clock endpoint responds with a valid open/closed value.

If the broker is down or returns unusable equity, the check fails and writes a
clear error into `signals/broker_health.json`.

## How To Run It

```bash
python3 broker_health.py
python3 broker_health.py --json
python3 broker_health.py --strict
```

Expected output:

- `healthy=true` when Alpaca is reachable and equity is usable.
- `healthy=false` with an error message when the pre-flight check fails.
- `--strict` also returns a nonzero process exit code when unhealthy. The real
  daily trading pipeline uses this mode so it stops before generating or
  submitting a signal against an unavailable broker.

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Pre-flight check | A small safety check before the main trading job runs. |
| Equity | Account value: cash plus positions. |
| Finite number | A normal usable number, not NaN or infinity. |
| Buying power | Money Alpaca currently permits the paper account to use after reservations. |
| Trading blocked | Alpaca can be reached, but the account is not permitted to place trades. |
| Broker down | The API cannot be reached, credentials fail, the account is restricted, or account data is unusable. |
