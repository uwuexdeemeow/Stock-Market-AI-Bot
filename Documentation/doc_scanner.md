# scanner.py — What It Does and How to Run It

## What This Script Does (Plain English)

Think of the scanner as a **talent scout**. Every day it looks through a watchlist of stocks, scores each one on several signals, and picks the top candidates worth researching further.

It does NOT predict whether a stock will go up. It only asks: "Is this stock interesting enough to spend time on today?" Interesting means it's showing unusual volume, an extreme RSI, high recent volatility, or a big price move.

**Output:** `data/shortlist.csv` — a ranked list of the top N tickers.

---

## How to Run It

```bash
# Scan all tickers in the watchlist (default from settings.py)
python scanner.py

# Scan a specific ticker
python scanner.py --ticker AAPL

# Scan with extra debug output
python scanner.py --verbose
```

**Expected output:**
- `data/shortlist.csv` with columns: `ticker`, `score`, `volume_spike`, `rsi_score`, `price_change`, `model_status`
- `logs/scanner.log` with a timestamped record of every run

---

## Key Concepts (Beginner Definitions)

| Term | Plain-English Meaning |
|---|---|
| **Watchlist** | The list of stocks the scanner considers. Defined in `settings.py` under `WATCHLIST`. |
| **Volume spike** | Today's trading volume is unusually high compared to the 20-day average. Often a sign something is happening. |
| **RSI** | Relative Strength Index. A number 0–100. Below 30 = "oversold" (maybe cheap). Above 70 = "overbought" (maybe expensive). Extremes score higher. |
| **Shortlist** | The scanner's output: the top N stocks that look most interesting today. |
| **Model status** | Whether a trained model already exists for this ticker in `models/`. Helps prioritize training. |
| **Regime** | The overall market state (bull/bear/crisis), detected via VIX and SPY trend. Affects scoring weights. |

---

## How It Connects to the Rest of the Pipeline

```
scanner.py → data/shortlist.csv
                 ↓
            research.py (reads shortlist, builds features)
```

The scanner runs first. If the shortlist is empty, `research.py` won't know what to process.
