# ensemble_utils.py — What It Does and How to Use It

## What This Script Does (Plain English)

`ensemble_utils.py` contains shared helpers for **combining predictions from multiple models**. Instead of just using XGBoost's output, an ensemble blends XGBoost + LSTM + Transformer predictions together — ideally getting accuracy better than any single model alone.

The key insight: models that make different types of errors cancel each other out when blended.

---

## How to Use It (in Code)

```python
from ensemble_utils import compute_dynamic_weights, has_sufficient_ensemble

# Compute blended weights based on each model's recent accuracy
weights = compute_dynamic_weights(
    model_accs={"xgboost": 61.0, "lstm_attention": 58.0, "transformer": 57.0},
    base_weights={"xgboost": 0.6, "lstm_attention": 0.2, "transformer": 0.2},
    baseline_up_rate=52.0,  # coin flip baseline for this ticker
    blend=0.40,  # 40% dynamic, 60% static base weights
)

# Check if we have at least 2 active models (minimum for a valid ensemble)
if has_sufficient_ensemble(weights):
    # proceed with ensemble prediction
    pass
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Ensemble** | Combining predictions from multiple models. Like asking 3 experts and averaging their opinions. |
| **Base weights** | The starting weights before performance adjustments. Defined in `settings.py` under `ENSEMBLE_WEIGHTS`. |
| **Dynamic weights** | Weights that shift toward whichever model has been most accurate recently. |
| **Blend (40/60)** | The final weight = 40% dynamic + 60% static base. Pure dynamic weights can be unstable; blending anchors them. |
| **Baseline up rate** | The historical % of days this ticker went up. The model only gets "credit" for accuracy above this coin-flip baseline. |
| **has_sufficient_ensemble** | Returns True if at least 2 models are active (weight > 0.01). A single model can't form an ensemble. |

---

## Why Dynamic Weighting?

Different models excel in different market regimes:
- XGBoost handles tabular patterns well in stable markets
- LSTMs/Transformers can detect longer-range temporal patterns
- During crises, the relationship between models shifts

`dynamic_weighting.py` adjusts further based on VIX regime. `ensemble_utils.py` handles the performance-based piece.
