# concentration_overlay.py — Dynamic Overlay Sizing

## What it does (plain English)

The "overlay" is the part of the portfolio that holds individual stock
picks instead of just QQQ. When mega-caps dominate the market (e.g. 2019,
2020, 2023), picking individual stocks underperforms QQQ — even when the
picks are good.

This module computes a **multiplier** for the overlay size based on how
concentrated the market is right now:

- **High concentration** (QQQ crushing SPY) → multiplier 0.3 (overlay shrinks)
- **Neutral concentration** → multiplier 1.0 (overlay unchanged)
- **Low concentration** (broad market participation) → multiplier 1.0

So if your base overlay is 50% of the portfolio, and the market is in
extreme concentration mode, the effective overlay becomes 50% × 0.3 = 15%.
The remaining capital goes to QQQ — where the action actually is.

## Why it exists

Historical walkforward data shows:

- Years where QQQ beat SPY by >5%: strategy lost **13.4%** vs QQQ on average
- Years where QQQ-SPY gap was <=5%: strategy gained **+1.5%** vs QQQ
- **14.9 percentage point gap** between high- vs low-concentration regimes

The model can't pick its way out of a mega-cap rally because it's spreading
capital across diversified picks while QQQ is funneling 50%+ of its weight
to a handful of names. The fix is structural: when you can't beat 'em,
join 'em — temporarily reduce the overlay and let QQQ carry the portfolio.

## How to use it

### Standalone (self-test)

```bash
python3 concentration_overlay.py
```

Shows the multiplier curve at various concentration levels — useful for
verifying the tuning knobs look right.

### In a backtest

```python
from concentration_overlay import dynamic_overlay_multipliers

# panel is a DataFrame with concentration_qqq_vs_eqw_20d column
mults = dynamic_overlay_multipliers(panel)
effective_overlay = panel["base_overlay_gross"] * mults
```

### In the live strategy

```python
from concentration_overlay import apply_concentration_overlay

# Inside the per-rebalance loop:
effective_overlay = apply_concentration_overlay(
    base_overlay_gross=config["overlay_gross"],
    panel_row=current_row,
    enable=config.get("concentration_overlay_enabled", False),
)
# Use effective_overlay in your weight calculation.
```

The `enable` flag defaults to opt-in so existing backtests aren't broken.
To turn it on, add `"concentration_overlay_enabled": True` to your config.

## The curve

| Concentration (QQQ-SPY 20d, %) | Multiplier | What it means |
|--------------------------------|-----------|---------------|
| -10 to -2 | 1.00 | Broad market wins; overlay stays at full size |
| -2 to 5 | 1.00 | Neutral; no adjustment |
| 5 to 15 | 1.00 → 0.30 (linear ramp) | Concentration building; overlay shrinks |
| 15+ | 0.30 | Extreme concentration; overlay at floor |

The thresholds (5%, 15%) and floor (0.30) are tunable constants at the
top of the file. They're calibrated from the 2013-2026 walkforward data.

## Key terms (for beginners)

- **Overlay** = the slice of the portfolio that holds individual stock
  picks instead of broad-market ETFs.
- **Overlay gross** = how much of the total portfolio the overlay can
  use (e.g. 0.50 = 50% of capital available for stock picks).
- **Concentration** = a measure of how unevenly returns are distributed
  across stocks. QQQ vs SPY return gap is a simple proxy — when tech
  mega-caps dominate, QQQ rips while SPY (broader, more equal-weighted
  toward smaller names) lags.
- **Multiplier** = a number you multiply by another value to adjust it.
  Multiplier of 1.0 = no change; 0.5 = cut in half; 2.0 = double.

## What this does NOT do

- It does not modify any existing strategy code automatically. The
  integration is opt-in — you have to wire it into the strategy and
  enable the flag.
- It does not predict the future. It reacts to the *current* 20-day
  concentration regime, so there's a short lag.
- It does not turn the overlay off entirely. The minimum multiplier is
  0.30 (not 0) to preserve some active management even in extreme
  regimes.

## Calibration notes

If after the next walkforward the strategy still struggles in high-
concentration years, consider tuning down `MIN_OVERLAY_MULT` from 0.30
toward 0.15 — let QQQ carry even more of the portfolio. Don't go to 0
because then the strategy becomes pure QQQ, which defeats the point of
having a stock-picking layer.
