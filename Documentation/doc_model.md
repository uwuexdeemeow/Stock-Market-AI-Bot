# model.py — What It Does and How to Use It

## What This Script Does (Plain English)

`model.py` defines the **neural network architectures** — the deep learning models that live alongside XGBoost in the ensemble. Two architectures are implemented:

1. **LSTMWithAttention** — an LSTM (sequence model) with a multi-head attention layer on top. Good at learning patterns in time-series data.
2. **TransformerEncoder** — a Transformer-based sequence model with positional encoding and recency weighting. The same family of architecture as GPT, but much smaller and purpose-built for financial time series.

Both models output two predictions simultaneously:
- A **direction head**: UP or DOWN (binary classification)
- A **return head**: which of 5 return bins (strong_down / down / flat / up / strong_up)

> ⚠️ As of Phase 2, neural model weight is set to 0.0 in `settings.py`. They're ready to use but disabled until offline experiments confirm they add edge over XGBoost alone.

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **LSTM** | Long Short-Term Memory. A type of neural network designed for sequences. It "remembers" patterns over time, like how prices behaved over the last 90 days. |
| **Attention mechanism** | Lets the model focus on the most relevant time steps in a sequence, not just the most recent. "Pay attention to this day from 3 weeks ago because it matters." |
| **Transformer** | A model architecture (same family as ChatGPT) that uses attention instead of recurrence. Often more powerful than LSTM for long sequences. |
| **Positional encoding** | Since Transformers don't process data sequentially, we embed "day 1, day 2, day 3..." position information into the input. |
| **Recency weighting** | A learned parameter that makes the model naturally give more importance to recent days. |
| **Multi-head attention** | Multiple attention "heads" that each look for different types of patterns in the sequence simultaneously. |
| **Direction head / return head** | Two output layers on the same neural backbone. Sharing the backbone means both tasks reinforce each other's learned features. |
| **Dropout** | Randomly disables neurons during training. Forces the model to learn robust features rather than memorizing training data. |

---

## When to Enable Neural Models

Flip `settings.py` ENSEMBLE_WEIGHTS from:
```python
{"xgboost": 1.0, "neural": 0.0}
```
to something like:
```python
{"xgboost": 0.7, "neural": 0.3}
```

Only after:
1. Running `nested_cv.py` and confirming neural OOS accuracy > XGB-only
2. Verifying `calibration_stability.py` max_ks < 0.10
3. Running `backtest.py` with the combined ensemble and confirming Sharpe improvement
