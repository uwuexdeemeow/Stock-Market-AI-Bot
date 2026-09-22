# survivorship_audit.py — Failed-Company Data Check

## What It Does

This script builds and profiles historical data for companies that failed or
were delisted. Those histories test whether a strategy only looks successful
because it was trained on companies that survived.

Ticker symbols can be reused. The script checks that each history begins at
least one year before the company’s known failure date. A newer security using
the same symbol is marked `symbol_reuse` and excluded from the audit.

## How To Run It

Build missing histories and write the report:

```bash
python3 survivorship_audit.py --build --report
```

Force another provider download attempt:

```bash
python3 survivorship_audit.py --build --force --report
```

The report is written to `logs/survivorship_audit.json`. Valid histories may
also create parquet files in `data/`.

## Key Terms

- **Survivorship bias:** judging a strategy only with companies that remained
  available, while silently omitting failures.
- **Symbol reuse:** a ticker being assigned to a different company or fund.
- **Failure date:** the known bankruptcy, receivership, or collapse date used
  to confirm that a price history belongs to the intended failed company.
- **Parquet:** a compact file format used for historical market data.
