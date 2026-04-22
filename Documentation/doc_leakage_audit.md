# leakage_audit.py — What It Does and How to Run It

## What This Script Does (Plain English)

`leakage_audit.py` checks that your feature pipeline is **not accidentally using future data**. This bug — called data leakage — is the single most dangerous mistake in quant finance. If any feature at bar `t` depends on bar `t+1` or later, the backtest will look spectacular, but live trading will lose money because the future isn't available.

The test: edit tomorrow's price, re-run the feature pipeline for today, and check if any features changed. If they did, those features are leaking the future.

---

## How to Run It

```bash
python leakage_audit.py
```

**Expected output:**
- `logs/leakage_audit.json` with pass/fail per ticker
- Printed summary:
  ```
  [leakage_audit] all tickers passed.          ← Good
  [leakage_audit] 2 tickers have leaking features.  ← Fix these first!
  ```

**Exit code:** 0 = pass, 1 = leaks found, 2 = setup error.

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Data leakage** | A feature at time T uses information that wasn't available until T+1 or later. Makes backtests unrealistically good. |
| **Future perturbation** | The test method: change tomorrow's data and re-check today's features. Honest features should be unchanged. |
| **Forward fill (ffill)** | Carrying the last known value forward when data is missing. Can leak if the "last known value" is actually from a future merge. |
| **Timezone alignment** | Multi-market features (SPY, VIX) from different exchanges must be aligned carefully. A 1-row offset creates 1-day lookahead bias. |
| **Embargo** | The 5-day gap between splits. A feature that changes only on the last row might be legitimate near-future leakage within the embargo window. |

---

## What to Do If Leaks Are Found

1. Read `logs/leakage_audit.json` to see which feature names are listed under `"leaking_features"`.
2. Open `pipeline_shared.py` and search for those feature names.
3. Common culprits:
   - `shift(-k)` for any positive k — explicitly looks ahead
   - `rolling(..., center=True)` — uses future values in the window
   - `merge` or `join` with a ticker that has a 1-row offset in its index
4. Fix, re-run `leakage_audit.py` until exit code is 0.
5. **Do not backtest or go live until all leaks are fixed.**
