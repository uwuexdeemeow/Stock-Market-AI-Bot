# shadow_paper_journal.py - Shadow Paper Journal

## What It Does

This script tracks a candidate core-satellite config without sending orders to Alpaca.

It temporarily builds a shadow live-config payload for:

`h=20,ov=0.5,ma=100,vol=percentile:0.3,score=regime_adaptive_riskoff_guard,shape=top3,weighting=sticky_score,tqqq=0.0,risk=off`

Then it runs the same signal-generation logic as the real paper system, writes one tracking row to `signals/shadow_paper_journal.csv`, and restores the normal signal files.

## How To Run

```bash
python shadow_paper_journal.py
```

Useful options:

```bash
python shadow_paper_journal.py --journal-path signals/shadow_paper_journal.csv
python shadow_paper_journal.py --append-duplicate
python shadow_paper_journal.py --ignore-stale
```

Expected output:

- `signals/shadow_paper_journal.csv` gets one row for today.
- `signals/core_satellite_alpha_signal.csv` is restored after the shadow run.
- No Alpaca orders are submitted.

## Key Concepts

**Shadow config**: A candidate config we want to observe before promoting to the real paper-trading config.

**Risk-off guard score**: A scoring route that keeps the normal regime-adaptive score in risk-on/neutral markets, but uses a more defensive score during risk-off regimes.

**Paper journal**: A CSV log of what the shadow config would have targeted each day.

**Overlay gross**: The portion of portfolio exposure assigned to individual stock picks.

**Paper ready**: Whether the signal passed the same live gates used by the normal paper pipeline.

**Sticky holdings**: Existing live holdings that the signal generator tries to keep when they still rank well.
