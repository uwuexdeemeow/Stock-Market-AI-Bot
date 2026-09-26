# Signal Ideas for the Owner — 2026-09-26

This is a **menu**, not a decision. The handoff note says the owner picks the
next signal hypothesis. Nothing here has been run, and nothing here has been
pre-registered in `DELAY_STRESS_PAPER_ADVISORY.md`. After the owner picks one,
its hypothesis and pass/fail rule are written there and committed **before**
any run.

## What the evidence says is wrong with the incumbent

These points come from files already in the repo. They shape every idea below.

1. **The signal is fast, but the trade is slow.** The feature shortlist is
   chosen by how well each feature predicts the **5-day** return
   (`alpha_factor_backtest.HORIZON_DAYS = 5`). Most of the top features in
   `signals/feature_research_summary.csv` are short-term reversal features:
   `ret_1d`, `ret_3d`, `ret_5d`, `dist_ma5`, `dist_ma10`, `ret_vs_sector_5d`.
   The portfolio then holds each pick for **20 days**. A 1–5 day signal is
   mostly used up before the hold ends, so the result depends heavily on the
   exact day you buy. That matches the timing-luck result (a 146-point swing
   from start day alone).
2. **Three stocks is too few.** `top3` makes the book depend on a few lucky or
   unlucky names. The walk-forward analyzer flagged concentration (FAIL), and
   NVDA alone was 20% of overlay profit.
3. **The edge is small and shrinking.** The average feature IC is about 0.01–0.02,
   and with luck averaged out, the 2023–2026 edge over QQQ is about +3 points.
4. **The stock universe is today's survivors.** `BROAD_WATCHLIST` is 147 large
   companies that exist *now* and have 12+ years of history. A backtest that
   starts in 2010 with only today's winners will look better than reality.
   The point-in-time membership gate (400+ names, removed members kept) is not
   met yet, so full-history stock-picking alpha should not be trusted.
5. **Only price and volume data go back far enough.** Analyst revisions, short
   interest, insider trades and options IV (`alternative_data_features.py`,
   `options_iv_provider.py`) are live snapshots only. They cannot be
   backtested, so they cannot be the core of a new signal.

**Design rule that follows:** a new signal should change slowly (so one day
late barely matters), hold more than 3 names, and avoid leaning on
stock-picking history that survivorship bias inflates.

## The ideas

Each idea lists what it is, why it fits the diagnosis, what data it needs,
whether it touches a locked file, and a draft pass/fail rule.

### Idea A — Slow residual momentum, broad basket (recommended first)

- **What:** rank stocks by 12-month return minus the last month
  ("12-1 momentum"), measured *after* removing the market and sector move
  ("residual"). Buy the top 10–15, equal weight, at most 3 per sector, hold 20
  days. Features `factor_mom` / `factor_resid_mom` already exist.
- **Why it fits:** a 12-month signal barely changes from one day to the next,
  so a one-day delay and the choice of start day should matter little. 10–15
  names removes most single-stock luck. Residual momentum is also one of the
  best-documented slow signals in academic research.
- **Data:** existing daily prices. Nothing new.
- **Locked files:** a first test can be a research script (like
  `research_evidence/phase_luck_20260926/tranche_preview.py`) that writes the
  new score into a copy of the panel's `factor_walkforward_score` column and
  runs the engine with `score_source="factor_walkforward"`, without editing it. A new score source in
  `core_satellite_alpha.py` (locked) is needed only if it passes.
- **Weakness:** momentum crashes hard at sharp market turns (2009, 2020
  spring). The existing regime switch may help; it should be tested, not assumed.
- **Draft pass rule:** (1) every stress scenario passes outright; (2) across
  the 20 start days, the spread of 2013–2022 alpha vs QQQ is under 73 points
  and every start day is positive; (3) one-day-late alpha is within 5 points
  of on-time alpha on average; (4) 2023–2026 is a diagnostic only.

### Idea B — Horizon-matched refresh of the current signal

- **What:** keep the engine and families, but rebuild the feature shortlist
  using **20-day** IC measured with a **one-day lag** (signal on day t, return
  from t+1 to t+21). Drop any feature whose IC falls by more than half when it
  is lagged one day.
- **Why it fits:** this directly fixes point 1 above. The research summary
  already marks several features with `horizon_mismatch = True` and
  `optimal_horizon = 20` (the liquidity/illiquidity features).
- **Data:** existing. **Locked files:** the shortlist is built in
  `alpha_factor_backtest.py`, which is not in `paper_version_lock.json`.
  `feature_health.py`, which it calls, is locked, so leave that one alone.
- **Weakness:** it is the smallest change, so it is also the one most likely to
  keep the old problems (small IC, survivor universe). The feature choice must
  happen inside each walk-forward fold, or it leaks the future.
- **Draft pass rule:** same as Idea A, plus the feature list must be chosen
  using data up to each fold's training end only.

### Idea C — Sector ETF rotation (no single stocks)

- **What:** rank the 11 SPDR sector ETFs (XLK, XLF, XLV, …) by 6- and 12-month
  return, keep only those above their 200-day average, hold the top 3 as the
  satellite, rebalance monthly.
- **Why it fits:** ETFs don't go bankrupt or get dropped from an index, so
  there is **no survivorship bias** (point 4). Slow signal (point 1). Each ETF
  already holds dozens of stocks (point 2).
- **Data:** ETF prices via `refresh_etf_data.py`. XLRE starts in 2015 and XLC
  in 2018, so early years have 9–10 choices.
- **Weakness:** honest warning — tech led 2023–2026, so rotating away from it
  may lose to QQQ in exactly the recent period. It may be better at protecting
  in bad years than at beating QQQ.
- **Draft pass rule:** same stress and start-day rules as Idea A.

### Idea D — Earnings-reaction drift (price-only)

- **What:** after a company reports, measure its stock move versus SPY over
  the report day and the next day. Buy the biggest positive reactions on day
  3 and hold about 40 trading days. This uses only earnings **dates**, not
  analyst estimates.
- **Why it fits:** "post-earnings drift" is slow (weeks), so a one-day delay
  should cost little. It is also unlike momentum, so it could later be combined
  with Idea A.
- **Data:** earnings dates (`fundamental_features.build_pead_features`).
  **First check:** how far back the free earnings-date history really goes. If
  it doesn't reach about 2013, the walk-forward has too few years and this idea
  should wait.
- **Weakness:** needs event-driven entries on different days, which the
  current 20-day calendar engine doesn't do. More engineering than A–C.

### Idea E (control, not a signal) — Core regime switch with no stock overlay

- **What:** run the incumbent's QQQ/SPY/cash regime switch with overlay gross
  set to 0.
- **Why:** with the recent overlay edge at about +3 points, the question
  "does picking stocks add anything?" deserves a direct answer. Any new idea
  should beat this baseline, not only QQQ.
- **Weakness:** it will likely trail QQQ in strong bull years because it holds
  cash sometimes. It is a yardstick, not a candidate.

## Ideas to avoid

- **Faster reversal (weekly trading).** The reversal IC is real but lives in
  1–5 days, so it is the most delay-sensitive signal possible.
- **Analyst revisions, short interest, insider trades, options IV** as the main
  signal. They can't be backtested with the free data on hand.
- **A wider grid around top3 / sticky weights.** The fixed research rules
  already forbid widening the grid in response to results.
- **A new ML model on the same features.** Model AUC has been about 0.51 on
  this data; a new model doesn't fix a slow/fast mismatch.

## Suggested order

1. **Idea E** first. It's cheap and gives a yardstick.
2. **Idea A**. It is the best fit to the diagnosis, uses existing data, and the
   first test needs no locked file.
3. **Idea C** as the survivorship-free second opinion.
4. **Idea B** if the owner prefers "refresh the current signal" over "new
   signal".
5. **Idea D** only after checking earnings-date history depth.

## Key terms

- **Signal:** a number per stock that says how good it looks to buy today.
- **IC (information coefficient):** how well a signal's ranking matches later
  returns. 0 = no link, 1 = perfect. Values around 0.02 are small.
- **Horizon:** how far ahead a signal predicts (5 days, 20 days, …).
- **Momentum:** stocks that went up over the past year tend to keep going up
  for a while.
- **Reversal:** stocks that jumped in the last few days tend to fall back a
  little, and the other way round.
- **Residual:** what's left of a stock's return after removing the part
  explained by the whole market or its sector.
- **Survivorship bias:** testing only on companies that still exist today,
  which hides the ones that failed and makes history look too good.
- **Pre-registration:** writing the rule for success *before* running the test,
  so the results can't change the rule.
