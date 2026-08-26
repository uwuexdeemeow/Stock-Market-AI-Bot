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
