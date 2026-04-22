# Paper Trading

This package includes `paper_trading.py` for manual daily paper trading.

## Daily workflow
1. Run prediction:
   `python predict.py`
2. Run paper trading:
   `python paper_trading.py --run`
3. Check status:
   `python paper_trading.py --status`

## What it does
- reads `signals/signals.csv`
- queues new paper orders above the confidence threshold
- fills pending orders at the next available market open
- exits positions after the configured holding period
- tracks:
  - `signals/paper_state.json`
  - `signals/paper_orders.csv`
  - `signals/paper_positions.csv`
  - `signals/paper_trades.csv`
  - `signals/paper_equity.csv`
  - `signals/paper_daily_summary.json`

## Reset
`python paper_trading.py --reset`
