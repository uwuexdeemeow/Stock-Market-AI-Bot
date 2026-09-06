# Labels

## What it does

`labels.py` creates the future outcomes that models try to predict: forward
return, volatility-normalized return, and triple-barrier outcomes.

## How to use it

Import the desired label function and pass a chronological price Series (plus
its horizon and barrier settings). The output is an aligned Series; final rows
without enough future data remain missing and must not be trained on.

## Key terms

- **Label:** the answer a supervised model learns to predict.
- **Horizon:** how many future trading days define that answer.
- **Triple barrier:** profit, loss, or time limit—whichever occurs first.


## Remaining audit repair, September 2026

Excess-return and volatility-adjusted excess targets require actual benchmark returns. Missing SPY input raises a clear error instead of inserting zero benchmark returns. Raw-return targets can operate without a benchmark. A benchmark is the comparison investment; zero is a measured no-change return and must never mean unavailable. Run python -m pytest tests/test_corrected_audit.py -q for missing-benchmark/raw-target regressions.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.
