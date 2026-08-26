# research.py — What It Does and How to Run It

## What This Script Does (Plain English)

The research script is the **data builder**. For every ticker on the shortlist, it downloads historical price data from Yahoo Finance, then runs it through the full feature-engineering pipeline to produce a rich table of signals.

The result is saved as a `.parquet` file — think of it as a highly compressed spreadsheet that Python can read very fast.

**Output:** `data/<TICKER>.parquet` for each ticker processed.

---

## How to Run It

```bash
# Process every ticker in data/shortlist.csv (run scanner.py first)
python research.py

# Process one specific ticker
python research.py --ticker AAPL

# Quick test with just AAPL
python research.py --test

# Fast daily refresh: only update tickers missing the latest completed session
python research.py --incremental
```

**Expected output:**
- `data/AAPL.parquet` (one file per ticker)
- `logs/research.log` with row/column counts and any errors

`--incremental` treats the `end` date as exclusive, just like Yahoo Finance.
On weekends and NYSE holidays, it checks the latest completed NYSE session first.
If every selected ticker already reaches that session, it exits without running
the slow per-ticker rebuild/download loop.

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Feature** | One column in the output table. Examples: RSI, MACD, news sentiment score. Each row is one trading day. |
| **Technical indicators** | Math formulas applied to price and volume to detect patterns (RSI, MACD, Bollinger Bands, ATR). |
| **Sentiment features** | Scores derived from news headlines and social media. Negative = bad news. Positive = good news. |
| **Macro features** | Market-wide signals like VIX (fear index), SPY trend, yield curve, gold price. Give context to individual stocks. |
| **Parquet** | A file format for tables. Much faster to read than CSV and takes 5–10× less disk space. |
| **EMBARGO_DAYS** | A gap (5 days by default) left between the training set and calibration set to prevent the model from accidentally peeking at near-future data. |

---

## How It Connects

```
scanner.py → data/shortlist.csv
                ↓
           research.py → data/<TICKER>.parquet
                                ↓
                           train.py (reads the parquet)
```

## Data Provenance And Speed

Each ticker parquet receives a JSON sidecar in `data/manifests/` with provider,
adjustment mode, dates, rows, schema, checksum, and quality checks. Incremental
refresh can use multiple workers and calculates shared market inputs once.
Scheduled bulk refresh skips sentiment until an after-cost out-of-sample
ablation proves that it adds value.
