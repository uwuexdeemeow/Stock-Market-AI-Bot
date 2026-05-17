# walkforward_analyzer.py — Validate Walkforward Results

## What it does (plain English)

After you run the nested walkforward, this script reads the results CSV
and tells you whether the strategy actually works — or whether the
validation is lying to you.

Just looking at "did it beat QQQ?" misses three sneaky failure modes:

1. **Anti-predictive scoring** — configs that look best in inner folds
   actually do *worse* on the held-out OOS test. This means the scoring
   function is broken; you can fix the strategy code all day but if the
   selector picks the wrong config, you'll never see improvement.
2. **Calibration disaster** — the inner validation predicts the strategy
   beats QQQ 100% of the time, but reality is 30%. The model is
   systematically overconfident.
3. **Concentration vulnerability** — OOS losses cluster in years where
   mega-caps dominated the market (QQQ >> SPY). The strategy structurally
   can't compete with concentrated market momentum.

The analyzer runs four checks, prints PASS/WARN/FAIL for each, and tells
you what to fix first.

## Why it exists

The previous nested walkforward had:

- Correlation between inner score and OOS Sharpe: **-0.327** (anti-predictive)
- Inner predicted beating QQQ 100% of folds; reality: 29%
- All 14 OOS folds picked a different config (zero stability)

If we'd only looked at "compound return = 482%", we'd have thought it
was working. The analyzer catches what raw return numbers hide.

## How to run

```bash
# Default: reads signals/core_satellite_nested_walkforward.csv
python3 walkforward_analyzer.py

# Custom file
python3 walkforward_analyzer.py --csv path/to/results.csv

# Also save a JSON report alongside the CSV
python3 walkforward_analyzer.py --json
```

## What the output looks like

```
── CHECK 1: SCORE PREDICTIVENESS ──
  Verdict:  [FAIL]
  Inner score vs OOS Sharpe correlation: -0.327
  → Inner scoring is anti-predictive — high-scoring configs do WORSE OOS.

── CHECK 2: MODEL CALIBRATION ──
  Verdict:  [FAIL]
  Inner predicts beat QQQ: 100.0%
  OOS actually beats QQQ:  28.6%
  Overconfidence gap:      71.4 pp
  Direction accuracy:      28.6% (50% = coin flip)

── CHECK 3: CONCENTRATION VULNERABILITY ──
  Verdict:  [FAIL]
  Correlation: -0.768
  High-concentration years: [2019, 2020, 2023, ...]
    Mean OOS alpha vs QQQ: -13.4%

── OVERALL RECOMMENDATION ──
  PASS=0  WARN=0  FAIL=4
  → Critical failures detected.  Don't deploy yet.
```

## Pass / Warn / Fail thresholds

| Check | PASS | WARN | FAIL |
|-------|------|------|------|
| Score predictiveness | corr > 0.3 | 0 to 0.3 | < 0 |
| Calibration | direction >= 60% and gap < 30pp | direction >= 40% | otherwise |
| Concentration | corr > -0.3 and delta > -5% | corr > -0.5 | otherwise |
| Config stability | top freq >= 30%, uniqueness < 70% | top freq >= 20% | otherwise |

## Key terms (for beginners)

- **OOS** = out-of-sample. The walkforward tests on years the model never
  saw during training. OOS results are the real-world performance estimate.
- **Inner fold** = a year inside the training window that the walkforward
  treats as a mini test set, used to pick the best config.
- **Correlation** = -1.0 to +1.0 score for how two variables move together.
  +1 = always together, -1 = always opposite, 0 = unrelated.
- **Direction accuracy** = how often the inner score's *sign* (positive
  or negative alpha) matches OOS reality. 50% is a coin flip.
- **Concentration proxy** = QQQ return minus SPY return over a year.
  Positive = mega-cap tech outperformed the broader market.

## When to run this

- After every nested walkforward (`core_satellite_nested_walkforward.py`)
- Before deploying any new config to live trading
- After making changes to the inner scoring function
- When CAGR drops year-over-year without obvious cause
