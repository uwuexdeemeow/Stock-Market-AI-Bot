# train.py — What It Does and How to Run It

## What This Script Does (Plain English)

`train.py` is the **teacher**. It takes the parquet files built by `research.py` and uses them to teach the XGBoost model patterns — like: "when these features look like this, the stock went up 5 days later."

Training has strict rules to prevent cheating:
- Data is split into train / calibration / test in chronological order (never shuffled).
- A 5-day embargo gap sits between each split so the model can't accidentally learn from near-future data.
- The calibrator (which converts raw model scores to probabilities) is fit only on the calibration slice.

**Output:** model files saved in `models/` for each ticker.
After every successful save, `models/registry.json` receives a reproducibility
record containing the Git commit, dataset fingerprint, metrics, command, and
checksums of the model artifacts. Failed or incomplete training is not
registered.

---

## How to Run It

```bash
# Train models for all tickers in data/
python train.py

# Train a single ticker
python train.py --ticker AAPL

# Train and show verbose output
python train.py --ticker AAPL --verbose
```

**Expected output:**
- `models/<TICKER>_xgb_dir.json` — direction model (up/down)
- `models/<TICKER>_xgb_ret.json` — return-bucket model (strong_down / down / flat / up / strong_up)
- `models/<TICKER>_scaler.pkl` — feature normalizer
- `models/<TICKER>_calibrator.pkl` — confidence calibrator
- `logs/train.log` — accuracy metrics per ticker
- `models/registry.json` — append-only reproducibility history
- `models/<name>_drift_baseline.json` — compact input distribution baseline

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **XGBoost** | A machine learning algorithm that builds many small decision trees and combines them. Very good at tabular (spreadsheet-style) data. |
| **Direction model** | Predicts UP or DOWN (binary: 1 or 0). |
| **Return-bucket model** | Predicts one of 5 bins: strong_down / moderate_down / flat / moderate_up / strong_up. |
| **Train / calibration / test split** | Three chronologically ordered slices of data. Train = teacher. Calibration = adjust confidence. Test = honest final grade. |
| **Embargo** | 5-day buffer between splits. Prevents the model from "reading ahead" into adjacent time windows. |
| **Scaler** | Normalizes features so no single feature dominates just because its numbers are bigger. |
| **Calibrator** | Maps raw model scores (e.g., 0.72) to true probabilities ("72% chance of going up"). |
| **Isotonic regression** | The type of calibration used here. It learns the shape of the score→probability mapping from data. |
| **Early stopping** | Training automatically halts when accuracy on a held-out set stops improving, preventing the model from memorizing noise. |

---

## How It Connects

```
research.py → data/<TICKER>.parquet
                    ↓
               train.py → models/<TICKER>_xgb_dir.json
                           models/<TICKER>_xgb_ret.json
                           models/<TICKER>_scaler.pkl
                           models/<TICKER>_calibrator.pkl
                                      ↓
                              predict.py / backtest.py
```
