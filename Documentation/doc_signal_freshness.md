# signal_freshness.py

## What This Script Does

`signal_freshness.py` holds shared safety checks for broker signal files.  It
does not place trades.  Other scripts import it to answer three plain
questions before trading:

- Is the signal recent enough?
- Is the signal timestamp believable, not far in the future?
- Are the factor data dates recent enough?
- Does the signal match the currently approved live config?

It also reads target weights from the signal into one dictionary.  That keeps
the broker and the sanity checker from disagreeing about what the portfolio is.

Factor-data age is counted using real NYSE sessions.  Weekends and market
holidays do not make yesterday's valid factor data look stale.

Signal timestamps more than `MAX_SIGNAL_FUTURE_MINUTES` minutes ahead of the
runner clock fail closed.  The default tolerance is 5 minutes to allow small
clock skew without accepting a bad future-dated signal.

## How To Run It

This file is normally imported by other scripts:

```bash
python3 alpaca_paper_trading.py --submit
```

For a quick syntax check:

```bash
python3 -m py_compile signal_freshness.py
```

Expected inputs:

- A signal row from `signals/core_satellite_alpha_signal.csv`
- `signals/core_satellite_live_configs.json` when checking config match

Expected outputs:

- `(True, [])` when checks pass
- `(False, ["reason"])` when checks fail

## Key Terms

- **Signal**: the daily instruction file that says what weights to hold.
- **Freshness**: whether the signal and factor data are recent enough to trust.
- **Trading session**: a real NYSE market day.  Holidays are skipped.
- **Live config hash**: a short fingerprint of the approved walkforward config.
  If this changes, old signals are blocked.
- **Gross exposure**: the sum of absolute portfolio weights.  A 100% QQQ plus
  25% stocks portfolio has 125% gross exposure.
- **Overlay**: the individual stock sleeve around the SPY/QQQ core.
