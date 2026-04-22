# Weekly Runbook

This runbook is for the current quant pipeline with:
- daily signal generation
- manual paper trading
- weekly retraining
- weekly backtest review

---

## Daily Routine (Monday to Friday)

### Before market open
Run:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

What this does:
- updates today's signals
- fills any pending paper trades due today
- closes any positions due today
- updates paper equity and P&L

### After market close
Review these files:
- `signals/signals.csv`
- `signals/paper_daily_summary.json`
- `signals/paper_positions.csv`
- `signals/paper_trades.csv`

Check:
- which trades were opened
- which trades were closed
- current open positions
- current equity
- whether signal quality is holding up

---

## Weekly Routine

### Monday
Run:

```bash
python model_self_check.py
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

Purpose:
- confirm models and artifacts are healthy
- start the week with fresh signals and paper-trading state

### Tuesday
Run:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

Purpose:
- normal daily operation

### Wednesday
Run:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

Then review:
- open positions
- current drawdown
- sector concentration
- whether the portfolio is becoming too correlated

### Thursday
Run:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

Optional refresh for active tickers:

```bash
python research.py --ticker AAPL
python research.py --ticker NVDA
```

Purpose:
- keep data fresh for the names you care about most

### Friday
Run:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

Then do a weekly review:
- paper P&L
- win rate
- average gain vs average loss
- current drawdown
- alpha vs SPY
- strongest and weakest tickers
- strongest and weakest regimes

---

## Weekend Routine

### Saturday — refresh and retrain
Run:

```bash
python scanner.py
python research.py
python train.py
python model_self_check.py
```

Purpose:
- rebuild the shortlist
- refresh historical research data
- retrain on the latest available market data
- verify the saved artifacts

### Sunday — validate and prepare for Monday
Run:

```bash
python predict.py
python backtest.py
python paper_trading.py --status
```

Purpose:
- confirm the retrained models still generate sensible signals
- inspect backtest metrics
- review the current paper portfolio before the next trading week

---

## Simple Routine

### Every trading day
```bash
python predict.py
python paper_trading.py --run
```

### Every weekend
```bash
python scanner.py
python research.py
python train.py
python model_self_check.py
python backtest.py
```

---

## Best-Practice Cadence

- `predict.py`: daily
- `paper_trading.py --run`: daily
- `research.py`: weekly
- `train.py`: weekly
- `backtest.py`: weekly
- `model_self_check.py`: weekly, or after retraining

---

## Important Notes

### You do not need to keep paper trading running
`paper_trading.py` is stateful. It saves everything to disk and picks up where it left off the next time you run it.

### Weekly retraining is usually enough
You do not need to retrain every day unless:
- you are experimenting heavily
- the market regime is changing very fast
- you are actively tuning features

### Good daily habit
Use this as your default command set:

```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

---

## Files to Watch

### Signal outputs
- `signals/signals.csv`

### Paper trading state
- `signals/paper_state.json`
- `signals/paper_orders.csv`
- `signals/paper_positions.csv`
- `signals/paper_trades.csv`
- `signals/paper_equity.csv`
- `signals/paper_daily_summary.json`

### Model and data health
- `models/`
- `data/`
- `signals/`

---

## Suggested Weekly Review Questions

1. Is the strategy outperforming SPY this week?
2. Are HIGH-quality signals actually performing better than LOW-quality ones?
3. Is one sector dominating risk?
4. Is one ticker repeatedly underperforming?
5. Is the model behaving differently in stable vs defensive regimes?
6. Is confidence aligned with realized outcomes?

---

## Minimal Weekly Workflow

### Monday to Friday
```bash
python predict.py
python paper_trading.py --run
python paper_trading.py --status
```

### Saturday
```bash
python scanner.py
python research.py
python train.py
python model_self_check.py
```

### Sunday
```bash
python predict.py
python backtest.py
python paper_trading.py --status
```
