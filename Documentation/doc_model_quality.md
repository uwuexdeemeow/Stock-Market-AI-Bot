# Model Quality

## What it does

`model_quality.py` combines training summaries and walk-forward trade results
into consistent approval checks such as after-cost Sharpe, profit factor,
drawdown, and minimum trade count.

## How to use it

Import its report-building functions from training or evaluation code and pass
the ticker plus trade DataFrame. Expected output is a quality dictionary and,
when the project report path is used, `models/model_quality_report.csv`.

## Key terms

- **Profit factor:** gross winning dollars divided by gross losing dollars.
- **Sharpe ratio:** return relative to variability.
- **Maximum drawdown:** worst peak-to-trough decline.
