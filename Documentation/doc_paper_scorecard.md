# paper_scorecard.py - Paper Trading Scorecard

## What It Does

`paper_scorecard.py` compares the actual Alpaca paper-trading account against
the walkforward expectations. In plain language, it asks whether paper trading
is behaving like the backtest said it should.

It looks at paper account equity, QQQ/SPY benchmark returns, drawdown, Sharpe,
alpha, and execution slippage notes. The output helps decide whether live paper
performance is healthy or drifting away from research.

## How To Run It

```bash
python paper_scorecard.py
python paper_scorecard.py --json
python paper_scorecard.py --verbose
python paper_scorecard.py --reset
```

Inputs:

- `signals/alpaca_paper_equity.csv`
- `signals/core_satellite_live_configs.json`
- `signals/alpaca_paper_log.csv`
- ETF parquet data in `data/`

Expected output:

- Terminal scorecard summary
- `logs/paper_scorecard.json`

The JSON file is written atomically. That means the script writes a complete
temporary file first, then swaps it into place so the dashboard never reads a
half-written scorecard.

## Key Concepts

- Paper trading: Simulated broker trading using Alpaca paper account money.
- Walkforward: A backtest method that repeatedly trains on past data and tests
  on future data.
- Alpha: Return above a benchmark such as QQQ.
- Drawdown: The largest peak-to-trough loss during the test period.
- Sharpe: A return-versus-volatility score; higher usually means smoother risk.
- Slippage: Difference between expected trade price and actual fill price.
