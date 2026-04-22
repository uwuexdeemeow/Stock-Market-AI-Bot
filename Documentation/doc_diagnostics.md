# diagnostics.py — What It Does and How to Run It

## What This Script Does (Plain English)

`diagnostics.py` is a **deep environment checker**. Where `setup.py` checks that packages are installed, `diagnostics.py` goes further and actually tests them:
- Can XGBoost fit and predict a small dataset?
- Can SHAP generate explanations?
- Is CUDA (GPU) or MPS (Apple Silicon) actually working for PyTorch?
- Are the model artifact files in `models/` valid and loadable?

Run it when something seems wrong but `setup.py` says everything is fine.

---

## How to Run It

```bash
python diagnostics.py
```

**Example output:**
```
── Hardware ──────────────────────────────────────────
  ✓  PyTorch MPS available (Apple Silicon GPU)
  →  Device: mps

── XGBoost ───────────────────────────────────────────
  ✓  XGBoost fit + predict: OK (100 samples, 10 features)
  ✓  SHAP explainability: OK

── Model Artifacts ───────────────────────────────────
  ✓  AAPL_xgb_dir.json loaded OK
  ✗  MSFT_xgb_dir.json missing — run train.py --ticker MSFT
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **CUDA** | NVIDIA's GPU computing platform. If you have an NVIDIA GPU, PyTorch can use it for much faster neural network training. |
| **MPS** | Metal Performance Shaders — Apple Silicon (M1/M2/M3) GPU acceleration for PyTorch. ~3–5× faster than CPU for neural models. |
| **SHAP** | A library for explaining why the model made a prediction. Computationally expensive but crucial for debugging. |
| **Model artifact** | A saved trained model file (`.json` or `.pkl`). If corrupt or missing, `predict.py` will silently produce bad outputs. |
| **nvidia-smi** | A command-line tool that reports NVIDIA GPU status. `diagnostics.py` calls it to verify the GPU is actually visible. |

---

## When to Run Diagnostics

- After installing new packages (did something break an existing dependency?)
- When predictions look wrong (is the model actually loading?)
- On a new machine before running the pipeline for the first time
- When PyTorch model training is unexpectedly slow (is GPU being used?)
