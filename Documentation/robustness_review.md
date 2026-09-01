# robustness_review.py

## What It Does

This shared module reads survivorship, execution-stress, and factor-decay JSON
reports and gives one fail-closed answer. Factor status must be `pass` or
`advisory`; `warning`, `block`, missing evidence, and material stress failures
do not pass.

Survivorship has a separate, stricter capital gate. It requires all failed-name
histories, a complete point-in-time universe, adjusted score of at least 0.85,
return damage no worse than 5 percentage points, and drawdown damage no worse
than 2.5 percentage points. Partial evidence may describe a paper stress test
but always reports `capital_approval_pass=false`.

## How To Use It

It is a helper imported by the walk-forward publisher, validation bundle, and
live signal loader. For a quick read-only check, run:

```bash
python3 -c "from robustness_review import medium_risk_review_from_reports; print(medium_risk_review_from_reports())"
```

Input is the three JSON reports in `logs/`. Output is a dictionary containing
the overall result, reasons, and one result per report.

## Key Terms

- **Robustness:** whether a strategy survives tests beyond its normal backtest.
- **Fail closed:** missing or uncertain evidence blocks trading.
- **Factor decay:** weakening of a signal's predictive relationship over time.
