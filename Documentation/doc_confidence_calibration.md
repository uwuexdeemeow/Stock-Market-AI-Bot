# confidence_calibration.py — What It Does and How to Use It

## What This Script Does (Plain English)

Raw model scores are overconfident. A model might output 0.85 but only be right 62% of the time at that score level. `confidence_calibration.py` learns the true relationship between raw scores and actual win rates, then corrects for the gap.

It also automatically finds the minimum confidence threshold at which the model achieves your target precision (default: 58% win rate).

You don't run this directly — `train.py` calls it during training, and `predict.py` uses the saved calibrator at inference time.

---

## How to Use It (in Code)

```python
from confidence_calibration import (
    fit_direction_calibrator,
    calibrate_p_up,
    compute_dynamic_threshold,
)

# Train the calibrator on calibration-set data
calibrator = fit_direction_calibrator(raw_scores, true_labels)

# At prediction time: convert raw score → calibrated probability
p_calibrated = calibrate_p_up(calibrator, raw_score=0.71)
# Returns a number between 0.0 and 1.0

# Find the threshold that gives ≥58% precision
threshold = compute_dynamic_threshold(calibrator, target_precision=0.58)
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Calibration** | The process of making model confidence scores reflect true probabilities. If the model says 70%, it should be right 70% of the time. |
| **Isotonic regression** | A shape-preserving curve fit. It learns that "score 0.6 = 55% win rate, score 0.8 = 71% win rate" from the calibration data. |
| **Precision** | Of all the times the model said "LONG with high confidence," what % actually went up? This is the key metric — not raw accuracy. |
| **Dynamic threshold** | Instead of a fixed 57.5 threshold, this finds the minimum score where precision ≥ 58%. Different for each ticker and each training run. |
| **out_of_bounds="clip"** | If a live prediction has a score outside the range seen during calibration, it's clamped to the edge rather than extrapolating. Safer. |
| **Fallback calibration** | If no calibration candidate improves validation quality, raw probabilities are used. Retraining deletes any old calibrator so a stale curve cannot be applied to the new model. |

---

## Why Not Just Use Raw Scores?

XGBoost probability outputs are not well-calibrated out of the box:
- They tend to cluster near 0.5 (underconfident on clear cases)
- Or push toward extremes (overconfident on noisy signals)

Calibration maps the raw output to a score you can actually trust as a probability.
