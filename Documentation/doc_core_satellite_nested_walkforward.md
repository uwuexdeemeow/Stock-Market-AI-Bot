# core_satellite_nested_walkforward.py — Strategy Validation

## What it does (plain English)

This is the script that decides whether the strategy is good enough to
deploy to live trading.  It runs a **nested walk-forward** — a way of
testing strategy configurations that simulates "if you only knew what
you would have known at time X, what's the best you could have done at
time X+1?"

It does this by:

1. Splitting history into yearly "outer folds" (2013, 2014, ..., 2026).
2. For each outer year:
   - Use the years BEFORE it as training data.
   - Run the strategy on every candidate config (384 configs at the time
     of writing).
   - Pick the config that scored best on **inner folds** (years inside
     the training window, held out one at a time for validation).
   - Test that picked config on the outer year — that's the **OOS
     (out-of-sample) result**.
3. Aggregate OOS results across all years → "compound return", "mean
   Sharpe", "alpha vs QQQ", etc.
4. If the aggregate metrics pass approval thresholds, write the winning
   config to `core_satellite_live_configs.json` for live trading to use.

## Why it exists

Without walk-forward validation, it's easy to "discover" a strategy
that looks amazing on historical data but is just curve-fit noise.
The nested structure guards against this — every OOS year is a clean
test on data the config-selector never saw.

The "nested" part is what makes it expensive (hours to run a full grid)
but also what makes it trustworthy.

## How to run

```bash
# Default — runs the full grid on core-alpha strategy
python3 core_satellite_nested_walkforward.py --strategy core-alpha

# Faster: skip the successive-halving screening phase
python3 core_satellite_nested_walkforward.py --strategy core-alpha --full

# Parallel evaluation (faster on multi-core machines)
WALKFORWARD_WORKERS=4 python3 core_satellite_nested_walkforward.py

# Don't push the winning config to live (research mode)
python3 core_satellite_nested_walkforward.py --no-publish-live-config
```

## Inputs

| File | Source | Purpose |
|------|--------|---------|
| `data/*.parquet` | `research.py` | Per-ticker history + factors |
| `data/QQQ.parquet`, `data/SPY.parquet` | `refresh_etf_data.py` | Benchmarks |
| `core_satellite_alpha.py` (imported) | — | Defines the candidate config grid |

## Outputs

| File | What's in it |
|------|--------------|
| `signals/core_satellite_nested_walkforward.csv` | Per-fold OOS metrics (the main result) |
| `signals/core_satellite_nested_walkforward.json` | Same + aggregate stats + approval verdict |
| `signals/core_satellite_live_configs.json` | The winning config (only if approval passes) |
| `signals/walkforward_checkpoint_core_alpha.json` | Resume state if the run is interrupted |

## Key concepts

- **Walk-forward** — a backtest technique that simulates real-life
  knowledge.  At each test point, you only use information that was
  available at that time.
- **Outer fold** — the year being tested.  You don't use ANY data from
  this year to train, so the result is genuinely out-of-sample.
- **Inner fold** — a year inside the training window that's held out
  to pick the best config.  Lets you validate config selection without
  peeking at the outer fold.
- **OOS (out-of-sample)** — performance on data the model never saw.
  The OOS metrics are your real expectations for live trading.
- **Approval gate** — a set of thresholds (min Sharpe, max drawdown,
  config frequency, etc.) that aggregate OOS results must clear before
  the winning config gets promoted to live.
- **Successive halving** — efficiency trick: evaluate every config on
  one fold first, drop the bottom 75%, do the expensive full eval only
  on the top 25%.

## When to run

- **Initial deployment** — before going live for the first time.
- **Periodic re-validation** — every 30-45 days as market conditions
  evolve.  The live config has a `LIVE_CONFIG_MAX_AGE_DAYS=45` expiry.
- **After major code changes** — if you change the strategy logic or
  add features, the historical relationships may have shifted and
  approval needs to be re-earned.

## Recent fixes

- **Stability penalty** (line ~1253) — reduced from 0.35 → 0.10.  The
  old weight crushed concentrated configs (top3) because their per-year
  variance is naturally higher.  At 0.10 it's a tiebreaker, not a
  dominant force.
- **Config momentum bonus disabled** — this used to add +0.15 for
  configs that matched recent outer-fold winners.  The
  `score_predictiveness_audit.py` report showed that bonus was
  anti-predictive on the alphaqqq walkforward, so the field remains in
  reports but the bonus is now 0.0.
- **Cost-stress fallback** — if no config passes the 60% stress pass
  ratio, the fallback path retries with a relaxed gate so we get a
  result rather than a NaN year.
- **min_config_frequency** (line ~334) — relaxed from 0.30 → 0.20
  because 14-fold walk-forwards naturally have more config diversity.
- **Stable family promotion** — live approval now counts repeatable config
  families (`score`, `shape`, `weighting`, `risk`, `tqqq`) instead of only
  exact configs.  This avoids promoting a one-off latest-year winner while
  still letting small tuning knobs come from the freshest fold.
- **Tightened turnover gates** — inner mean cap 600→450%, new inner worst
  cap 525%, and the soft penalty span 4000→1500 (so 1000% turnover now
  costs 0.4 Sharpe instead of 0.15).  Earlier runs let configs with
  inner-mean ~466% win selection, then those configs spiked to 904%
  turnover OOS — blowing through the run-wide 600% live-approval gate
  and dragging cost-stress alpha into negative territory at 5× costs.
  The new caps are sized so OOS turnover stays under 600% even with a
  ~1.3× regime spike on top of the inner mean.
- **Checkpoint fingerprint includes selection filters** — `_ckpt_key`
  now hashes the turnover caps and penalty span alongside the strategy
  and grid.  Changing any of those values automatically invalidates the
  old checkpoint instead of silently reusing fold selections that were
  made under different rules.

## Output to validate

After every run, also run:

```bash
python3 walkforward_analyzer.py
```

It checks four failure modes the raw return numbers hide.  See
`Documentation/doc_walkforward_analyzer.md`.
