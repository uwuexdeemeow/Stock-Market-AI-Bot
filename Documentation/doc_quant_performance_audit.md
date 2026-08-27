# Quant performance audit

## What this script does

`quant_performance_audit.py` independently rebuilds the strategy's out-of-sample results one market day at a time. It uses the target ETF and stock weights saved by the strategy plus the raw Open and Close prices in `data/`. This exposes losses hidden between rebalances and prevents the periodic backtest from grading itself.

It reports gross and after-cost results versus QQQ, exact benchmark overlap, elapsed-time CAGR, daily Sharpe and information ratio, full-position turnover, daily drawdown, Newey-West alpha significance, and calendar-year block-bootstrap uncertainty.

The report reconciles the old periodic compound return, Sharpe, and drawdown
to the new daily values. Timing/raw-price changes, full-allocation transaction
costs, daily risk measurement, and continuous fold stitching are listed and
quantified separately. If target weights imply negative cash, the report keeps
historical financing cost as an explicit blocker instead of silently assuming
that borrowing was free.

It also runs only the bounded shadow experiments: top five, 40% overlay, and 80% sticky blending. A combined candidate is tested only after an individual candidate passes every fixed gate. Nothing in this script changes the active Alpaca configuration or places an order.

## How to run it

From the project folder:

```bash
python3 quant_performance_audit.py
```

For the selected-fold reference audit without the slower candidate experiments:

```bash
python3 quant_performance_audit.py --skip-experiments
```

Expected outputs:

- `signals/quant_performance_audit.json`: full evidence, uncertainty, blockers, and recommendation.
- `signals/quant_shadow_experiments.csv`: compact candidate comparison.
- `logs/experiment_ledger.jsonl` and `.csv`: append-only records for every attempted candidate.

## Key concepts

- **Mark-to-market:** value every position at each day's close, not only when it is sold.
- **OOS (out of sample):** dates not used to choose that fold's strategy settings.
- **Information ratio:** average daily return above QQQ divided by the variability of that difference, annualized.
- **Newey-West t-statistic:** an alpha-strength estimate adjusted for serially related daily returns.
- **Block bootstrap:** uncertainty estimated by resampling whole calendar years, preserving within-year market patterns.
- **Turnover:** total change in all target weights, including ETFs and stocks. More turnover normally means more cost.
- **Promotion blocker:** missing or failed evidence that prevents a shadow result from changing the active paper strategy.
