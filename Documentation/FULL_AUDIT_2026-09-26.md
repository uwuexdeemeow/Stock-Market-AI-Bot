# Full System Audit — 2026-09-26

## Scope and method

- **What runs every day:** the code path was traced from
  `.github/workflows/daily_paper_trading.yml` through `daily_run.py`. That
  covers:
  - the signal (`core_satellite_alpha.py`) and orders (`alpaca_paper_trading.py`);
  - safety guards (`execution_guard.py`, `robustness_review.py`,
    `core_satellite_nested_walkforward.py` approval);
  - health and cost reports (`paper_health.py`, `execution_cost_calibration.py`,
    `alpaca_paper_gauntlet.py`);
  - score building (`alpha_factor_backtest.py`).
- **Math re-checked by hand:** returns, costs, turnover, drawdown, Sharpe,
  CAGR, yearly alpha, regime flags, weights, slippage, halts.
- **Earlier audits:** each finding of `quant_logic_audit_20260906.md` was
  re-checked to see whether it is fixed or still open.
- **Lint:** a bug-pattern `ruff` scan (bugbear, errors, comparisons) was run
  on every root script.
- **Import map:** which scripts the workflows and dashboard actually reach.
- **Tests:** full suite before and after the change: see the end of this
  file. `paper_validation_epoch.py --check-lock` still passes.

The legacy ML scripts (`train.py`, `predict.py`, most of `backtest.py`) were
not audited line by line, because the current system doesn't use them for
trading.

**Severity:** *High* = changes what the paper account does or makes the
research evidence describe a different strategy. *Medium* = biases a gate or a
reported number. *Low* = cosmetic, diagnostic, or a small bias.

**"Locked"** means the file is in `paper_version_lock.json`. Changing it
stops daily paper trading until you approve and re-freeze. Nothing locked was
changed in this pass.

## Status of the September 6 findings

| # | Finding | Status |
|---|---|---|
| 1 | Refreshed quote skipped the spread limit | **Fixed**: every submission quote goes through `_submission_quote` |
| 2 | Regime confirmation used "any" instead of "all" | **Fixed** (`_confirm_regime_flag`) |
| 3 | Costs only on stock-overlay turnover; ETF trades and drift free | **Open** (`core_satellite_alpha.py:1727`) |
| 4 | Backtest and paper trading are different strategies | **Open**, and larger than it looked: see H1 |
| 5 | Drawdown only measured every 20 days | **Open** (`alpha_factor_backtest.py:615`) |
| 6 | Selection used future-price availability | **Fixed** |
| 7 | Weekday math instead of exchange calendar | **Fixed** in the engine; one date label left in the old 5-day harness (`alpha_factor_backtest.py:653`, Low) |
| 8 | Feature list and directions chosen on the full history | **Open**: see M5 |
| 9 | Edge monitor called raw return "alpha" | **Fixed** |
| 10 | ETF price cache returned wrong dates | **Fixed** |

## High

### H1. Paper trading does not run the strategy that was tested

The approved config has a 20-day calendar and
`early_rebalance_on_regime_change: false`. The backtest therefore changes the
market regime and the stock picks only every 20 trading days. The paper
account does something else:

- **Every day** `write_paper_signal` recomputes the regime and the picks
  (`core_satellite_alpha.py:2164`).
- **Every day** `generate_orders` trades anything that drifted more than 3%
  (ETFs) or 1% (stocks) (`alpaca_paper_trading.py:79-80`, `1517`). For
  example, a flip from risk-on to neutral moves QQQ from 60% to 37.5%, so it
  is traded the next morning, not at the next 20-day date.
- **Live-only rules** never appear in any backtest:
  - 8% trailing stops on every stock;
  - the news-sentiment veto;
  - the −8% one-day liquidation (`execution_guard.py:65`);
  - the −12% drawdown liquidation.

**Why it matters:** the walk-forward, the stress tests, the timing-luck study
and the new bake-off all measure the 20-day policy. The paper account's
results can't confirm or reject that evidence, because it trades a different
policy.

**Fix (owner decision):** pick one.

- (a) Make the paper account follow the tested calendar: trade only on the
  20-day dates, or use tranches.
- (b) Make research replay the daily policy, stops and halts included. The
  `accounting_mode="daily-ledger-v1"` adapter in `corrected_audit.py`
  already simulates daily cash and shares, so it is the natural base.

### H2. After an emergency sell-off, trading can never resume on its own — FIXED (see below)

- **What fires:** when the account is 12% below its peak
  (`alpaca_paper_trading.py:4715`), or down 8% in one day
  (`execution_guard.py:434`), `_emergency_liquidate` sells everything and
  writes a halt file.
- **The dead end:** `_maybe_auto_clear_halt` (`:4284`) removes that file only
  when the drawdown is better than −6% (`:4337`). But the account is now 100%
  cash, and cash doesn't grow, so the drawdown stays at about −8% to −12%
  forever. The docstring's example ("next day the account is at −5%") can't
  happen. The research engine's own circuit breaker notes this exact trap and
  re-enters on a risk-on regime instead (`core_satellite_alpha.py:1627`).
- **Result:** one bad day means selling near the low and then sitting in cash
  until someone deletes the file by hand. None of this is backtested.
- **Fix (locked file):** base the restart on the market (for example, regime
  back to risk-on, as in research), or reset the peak at the sell-off. Then
  backtest the rule, or drop it.

### H3. Trailing stop → buy back the next morning — FIXED (see below)

- **What happens:** every stock gets an 8% trailing stop
  (`alpaca_paper_trading.py:3807`). When a stop sells a stock, the next day's
  signal no longer sees it as "held". If the stock still ranks in the top 3,
  it is picked again and bought back. No code keeps a stopped-out stock out
  (no cooldown anywhere).
- **Result:** the stop mostly creates a round trip of trading costs and
  locks in the dip. It gives little protection.
- **Fix (locked files):** keep a stopped-out name out until the next 20-day
  rebalance, or drop the stops. Either way, backtest the choice (see H1).

## Medium

### M1. Sentiment veto: replacements are never checked, and it was never tested

- **Unchecked replacements:** `_apply_sentiment_veto`
  (`core_satellite_alpha.py:299`) downloads news scores for 3 backup stocks
  (`:330-333`) but never uses them. The replacement chosen at `:351` can
  itself have strongly negative news and still be bought.
- **Live only:** the veto is on by default (`CORE_ALPHA_SENTIMENT_VETO`
  defaults to `1`, `:262`) but never used in research.
- **Churn:** one bad-headline day can force a sale, and the stock can be
  bought back the next day.
- **Fix (locked):** apply the veto to replacements too. Better: turn the veto
  off (`CORE_ALPHA_SENTIMENT_VETO=0` in the workflow) until it has
  backtest evidence.

### M2. Trading cost is measured in a way that always looks close to zero

- **The measure used:** `slippage_bps` compares each fill with the average
  price of the **same minute** the order filled
  (`alpaca_paper_trading.py:3423`). The fill is part of that average, so the
  number is near zero by construction. (September's audit saw about 1.2 bps.)
- **Effect on the backtest:** `execution_cost_calibration.py:111` takes its
  cost from this number, so the backtest cost never rises above the 10 bps
  floor.
- **Better measure, already recorded:** `arrival_shortfall_bps` (fill vs the
  quote midpoint when the order was sent) is saved but unused.
- **Fix (not locked, but it changes backtest costs, so owner decision):**
  calibrate from `arrival_shortfall_bps`. Also measure the gap between the
  backtest's assumed price (the next open) and the real fill.

### M3. ETF trades cost nothing in the backtest (September #3, still open)

- **What is charged:** turnover counts only stock-overlay weight changes
  (`core_satellite_alpha.py:1727`). Moving money between QQQ, SPY and cash,
  and drift back to target, are free.
- **Why it's bigger live:** because of H1, the paper account makes these ETF
  trades more often than the backtest assumes.

### M4. Drawdown is only measured every 20 days (September #5, still open)

- **Where:** `portfolio_stats` (`alpha_factor_backtest.py:615`) works on
  equity points spaced 20 days apart, so drops between those points are
  invisible.
- **What it biases:** the −35% stress floor (`robustness_review.py`) and the
  live-versus-backtest drawdown check (`paper_health.py:1164`), which compares
  a daily live drawdown with a sampled backtest one.

### M5. Features and their directions are chosen on the full 2010–2026 history (September #8, still open)

- **The shortlist:** `logs/feature_ic_shortlist.csv` ranks features by
  t-statistics over about 4,100 days, and `preferred_direction` comes from
  the same full sample. Every fold's score reuses them (`alpha_factor_backtest.py:89-126`).
- **The regime scores:** they are built from those clusters (`:368-387`).
- **Effect:** every out-of-sample number, including 2023–2026, has seen
  the future through the feature choice. The bake-off's idea B re-chooses
  features per year for exactly this reason.

### M6. Missing numbers pass approval checks

`float(x or 0.0)` turns a missing value into 0, and 0 passes:

- **Selector uplift:** `core_satellite_nested_walkforward.py:1652-1653`.
  The `-999` default only applies when the key is absent. A stored `None`
  becomes 0, which passes `>= 0`.
- **Walk-forward drawdowns:** the same file, lines `1579-1580`. A missing
  drawdown becomes 0%, which passes.
- **Stress drawdown:** `robustness_review.py:138` (locked). A missing
  drawdown passes the −35% floor.
- **Survivorship deltas:** `robustness_review.py:74-75` (locked). A missing
  delta row counts as a 0 change.

The project's rule is fail-closed. **Fix:** treat a missing value as a
failure. This only makes gates stricter, but it changes gate behavior, so it
is left for owner approval.

## Low

- **L1 (fixed now).** `execution_cost_calibration._liquidity_bucket` compared
  `factor_liquidity_dollar_vol_20d` with $1B and $100M. That feature is
  stored as `log(1 + dollars)` (about 20.7 for $1B), so every stock landed in
  the "low" bucket. It now converts back to dollars first. This affected only
  the diagnostic cost segments, not the cost the backtest uses.
- **L2. Score weights ignore how strong a pick is.** `sticky_score` weights
  are `rank − lowest selected rank + 0.01` (`core_satellite_alpha.py:1387`).
  The last pick always gets close to zero new weight, even when all three
  picks are nearly tied. Equal weight, or fixed weights by rank, would be
  simpler and more stable.
- **L3. Cash earns nothing in the backtest** (`:1794`). This is conservative.
  The risk-off regime holds 20% cash, and T-bills paid about 5% in
  2023–2026, so the backtest undercounts roughly 0.1–0.2% a year. Credit
  BIL's return to cash.
- **L4. The gauntlet judges on 20 days.** `alpaca_paper_gauntlet.py:60-61`
  needs 20 trading days and a Sharpe of 0.5 or more. Over 20 days, the
  uncertainty of an annualized Sharpe is about ±3.5, so pass or fail is
  mostly luck. Use 6–12 months, or a probabilistic Sharpe.
- **L5. Two different sample counts in cost calibration.** "Ready" counts
  trailing-stop fills, but the recommendation excludes them
  (`execution_cost_calibration.py:104-115`). Use one sample for both.
- **L6. "Alpha" means difference in total return.** Examples are +3146% or
  +41.7% "vs QQQ". This is not risk-adjusted, grows with window length, and
  rewards more exposure. Add a risk-adjusted number (excess return per unit
  of tracking error, or regression alpha) next to it.
- **L7. The old 5-day factor harness labels equity dates with weekday
  arithmetic** (`alpha_factor_backtest.py:653`). Labels only.
- **L8. Dead weight.** 21 root scripts aren't reached by any workflow or the
  dashboard. Most are manual tools. The 4,500-line legacy `backtest.py` is
  still imported by the live engine for three helpers
  (`_load_etf_price_frame`, `_newey_west_tstat`, `INITIAL_CAPITAL`). Moving
  those into a small module would let the legacy ML code be archived.

## Improvements to the system (beyond bugs)

1. **One policy, one simulator.** Fixing H1 is the most valuable change. Every
   other result (bake-off, stress tests, paper evidence) depends on the
   backtest and the account doing the same thing. Build on the daily ledger in
   `corrected_audit.py`.
2. **Decide the fate of the live-only rules with data.** Stops, veto, −8%
   day liquidation and −12% halt each need a backtest. If a rule can't show
   value, remove it rather than keep an untested behavior.
3. **Calibrate costs on arrival shortfall** (M2), and charge ETF trades (M3).
4. **Daily marking for drawdown** (M4). The daily ledger gives this for free.
5. **Choose features inside each fold** (M5), as the bake-off's idea B does.
6. **Fail closed on missing numbers** (M6).
7. **Longer paper judgment window** (L4), and risk-adjusted alpha (L6).

## Changed in this pass

| File | Change |
|---|---|
| `execution_cost_calibration.py` | L1: convert the log dollar-volume feature back to dollars before bucketing |
| `tests/test_execution_cost_calibration.py` | New. The first test fails on the old code and passes on the fix. |

Tests: full suite 859 passed, 9 skipped (856 before, plus the 3 new tests).
`paper_validation_epoch.py --check-lock` still passes.

Everything else above is a recommendation. H1–H3 and M1 need locked files or
policy choices, and M2, M3 and M6 change backtest numbers or gate behavior.
All of these are owner decisions.

## Key terms

- **Backtest:** replaying the strategy on past prices to see how it would have done.
- **Policy:** the exact rules for when and how the account trades.
- **Drift band:** trade only when a holding is off target by more than a set amount.
- **Trailing stop:** an order that sells if the price falls a set percent below its recent high.
- **Fail-closed:** when evidence is missing, block rather than allow.
- **Arrival shortfall:** how much worse the fill was than the price when the order was sent.
- **Locked file:** a file fingerprinted by the paper release; changing it pauses paper trading until re-frozen.

## Follow-up: H2 and H3 fixed (owner request, same day)

The owner chose to fix H2 and H3 now and leave H1 for later.

**H2, extra finding while fixing it:** the halt file
`signals/alpaca_halt_active.txt` was never restored or published by the daily
workflow. Each GitHub run starts on a fresh machine, so:

- after a −8% one-day sell-off, the next run had no halt file and bought
  everything back;
- after a −12% halt, the all-time-high check blocked trading every day with
  no way out.

**H2 fix:**

- The halt clears (next day, account flat, no open orders) when the drawdown
  recovered **or** a fresh signal says the regime is `risk_on`.
- Drawdown is then measured from the restart point
  (`alpaca_drawdown_peak_reset.json`).
- The workflow restores and publishes both files, and removes the halt file
  from `signals/latest` once it clears.

**H3 fix:**

- The status snapshot lists stop-loss sells from the last 45 days.
- `daily_run.py` refreshes it right before the signal.
- The signal keeps a stopped-out stock out for one holding period (20
  sessions). It never forces a sale of a stock still held.

**Files changed (locked, so the release must be re-frozen):**
`alpaca_paper_trading.py`, `core_satellite_alpha.py`, `daily_run.py` and
`.github/workflows/daily_paper_trading.yml`.

**Tests:** `tests/test_halt_and_stop_cooldown.py` (new) and
`tests/test_core_satellite_live_signal.py` (end-to-end signal test, which fails
on the old code). Two halt tests in `tests/test_brokers.py` now also pin the
market check and give the fake broker an equity value.
