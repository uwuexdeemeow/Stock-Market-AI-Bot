# execution_cost_calibration.py

## What This Script Does

This script converts real Alpaca paper-fill measurements into conservative
backtest trading costs. It groups fills by buy or sell, order type, and ticker
liquidity. The backtest uses the 75th-percentile normal-order slippage, never a
number lower than the configured safety floor.

## How To Run It

```bash
python3 execution_cost_calibration.py
```

Input: `signals/alpaca_slippage_reversal_report.json`

Output: `signals/execution_cost_calibration.json`

At least 20 fills are required before the measured calibration becomes active.

## Key Terms

- **Slippage:** The difference between the expected price and actual fill price.
- **Basis point:** One hundredth of one percent.
- **Liquidity:** How easily a ticker can be traded without moving its price.
- **75th percentile:** A conservative value that 75% of measured fills beat.


## Remaining audit repair, September 2026

causal_cost_parameters(report, cutoff) accepts only attributed fills at or before the evaluation cutoff. It estimates residual slippage after subtracting separately modeled half-spread and impact, preventing duplicate costs. Fewer than 20 complete observations use the frozen 0.10% baseline fallback; later fills cannot change an earlier window. The corrected ledger applies this parameter to every stock/ETF fill, with spread, impact and explicit fees charged once. The older recommended turnover-cost output remains legacy diagnostic evidence. Run python -m pytest tests/test_corrected_audit.py -q for pre-cutoff perturbation checks.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.
