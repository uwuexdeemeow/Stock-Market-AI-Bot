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
