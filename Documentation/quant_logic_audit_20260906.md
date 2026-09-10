# Trading logic and quantitative audit — 6 September 2026

## Verdict and scope

Execution record recovery and reconciliation have improved observability. They do not establish a profitable, deployable trading edge. Several confirmed defects affect the strategy's behavior or the reliability of its historical evidence. Repair these before further parameter tuning or increasing capital.

This audit traced the active core/satellite signal, historical return construction, regime selection, portfolio accounting, order submission, execution measurement, research validation, and edge monitoring. ML labels and auxiliary exit logic received spot checks. This is not a claim that every line of every module is correct or that all possible defects have been found.

Reviewed main revision: `3a75517f0ec6709191629044c39ff63775249f1e`. Latest restored data publication reviewed: `e251143` on `signals/latest`. No trading code or settings were changed; no broker orders were sent. Controlled order experiments used a fake broker.

Validation: **565 tests passed, 45 skipped**, with 10 third-party deprecation warnings. Skipped tests are not successful coverage. Seven focused experiments are recorded in [quant_audit_evidence_20260906.json](quant_audit_evidence_20260906.json). That JSON records results; it is not a standalone executable regression suite.

## Priority findings

### 1. High: refreshed first-stage quotes bypass the spread limit

Locations: `alpaca_paper_trading.py:1935–1984`, initial guard at `2251`, second-stage check around `2037`.

The initial spread guard checks an earlier snapshot. Two-stage execution obtains another quote and verifies freshness, but submits its first passive limit without applying the spread threshold again. The second stage does apply a threshold.

Controlled evidence: a fresh bid of 100 and ask of 110 produced a **9.52% spread**, versus a **0.5% configured limit**, and reached the fake broker's submission method. This demonstrates a guard bypass, not a real fill or real loss. Recovered September 3 FCX evidence also contains an approximately 2.38% first-stage spread; that stage did not fill.

Improvement: validate positive, finite, non-crossed bid/ask prices, freshness, and the applicable spread limit on the exact quote used at every submission stage. Add a regression where the first quote is narrow and the refreshed quote is wide.

### 2. High: regime confirmation does not implement consecutive-day confirmation

Location: `core_satellite_alpha.py:1018–1030`.

Rolling minimum is used to identify an all-off window and rolling maximum an all-on window. Those conditions actually mean any-off and any-on. Because the off condition is checked first, one down day switches the state off immediately and keeps it off through the mixed window. The default confirmation period is three days.

Evidence: raw states `[off, on, on, on]` yielded confirmed states `[off, off, off, on]` after an established on period. The documented rule should retain on through that single-day dip.

Improvement: all-off requires rolling maximum zero; all-on requires rolling minimum one. Preserve prior state in mixed windows. Specify startup behavior and test both transitions. Historical and live signals must then be regenerated; this changes strategy behavior.

### 3. High: portfolio trading costs and turnover are incomplete

Locations: `core_satellite_alpha.py:1788–1794`, `1899`.

Costs use changes in overlay target weights only. Core ETF purchases and reallocations are absent. Previous overlay weights are the prior targets, not the drifted weights of actual holdings. Consequently, rebalancing back to unchanged targets after price movement can register zero turnover. Terminal liquidation costs also need an explicit convention.

Evidence: an actual backtest function run with a 100% core allocation reported zero turnover and zero estimated cost for its initial allocation. This is a controlled configuration demonstrating the omission, not the active allocation's measured dollar error.

Improvement: track shares, cash, prices, and drifted weights for every asset. Charge spread, fees, slippage, and calibrated impact on every simulated trade. Recompute execution stress: an extra basis-point charge applied to incomplete turnover is not a whole-portfolio stress test.

### 4. High: historical and paper execution are materially different strategies

Locations: `core_satellite_alpha.py:1534`, `1560`, `1802–1868`; `alpha_factor_backtest.py:182–190`; live defaults in `alpaca_paper_trading.py:104–118`.

The active historical holding interval is 20 sessions. Stock forward returns enter at the next session's open, whereas core ETF returns use the signal-date close at zero entry delay. The regime can use that same close. That is not a reproducible next-session execution assumption for the whole portfolio.

Paper signals refresh daily and order generation reacts to target drift. Live trailing stops and the portfolio drawdown halt are not reproduced by the active historical return path. Stop-outs and subsequent re-entry can change both exposure and costs substantially.

Improvement: use one event ledger with explicit information cutoff, submission time, entry price, rebalance cadence, stop behavior, and re-entry rules. Replay the exact paper policy historically. Separately identify close-auction assumptions if intentionally used; do not mix them with next-open assumptions unnoticed.

### 5. High: sampled equity misses intervening drawdowns

Locations: equity construction near the end of `core_satellite_alpha.run_core_satellite`; statistics at `alpha_factor_backtest.py:596`.

Equity observations occur at holding-period exits. Maximum drawdown calculated from those points does not measure daily or intraday peak-to-trough risk. Reported drawdown and drawdown stress therefore cannot be treated as daily risk bounds.

Evidence: the statistics helper reports a 40% drawdown for `[100, 60, 100]` and zero for its endpoints `[100, 100]`. This illustrates the blind spot; it does not assert that this strategy actually experienced an unreported 40% loss.

Improvement: mark all holdings and cash daily, including stops and costs. Calculate CAGR from elapsed time, and use a return frequency consistent with annualization. Variable holding periods and serial dependence require additional care with Sharpe uncertainty.

### 6. High: historical selection uses future-price availability

Location: `core_satellite_alpha.py:1289–1290`.

Historical ranking removes rows missing their forward-return label before selecting holdings. Thus a stock's future data availability influences whether it can be selected today. This is especially dangerous for delistings or interrupted histories. It differs from legitimately excluding an unfinished final evaluation period for the entire portfolio.

Improvement: select from information available on the decision date. Resolve later missing prices through an explicit delisting, liquidation, suspension, or conservative missing-data policy. Audit corporate actions and adjusted price consistency.

The saved survivorship audit also remains incomplete: failed-name coverage is **5 of 17**, and the point-in-time membership table has **zero rows**. Its stressed versus baseline total-return difference is about **1,095.62 percentage points**, not a 1,095% realized loss. Those figures are outputs of the existing model, whose accounting problems above still apply. Existing real-capital blocking is appropriate.

### 7. High: weekday arithmetic can mis-purge forward labels

Locations: `core_satellite_alpha.py:1542–1552`, `1629`, terminal exit at `1811`; forward labels in `alpha_factor_backtest.py:182–190`.

Labels shift actual price rows, but some evaluation boundary checks and terminal exits use `BDay`, which excludes weekends but does not represent exchange holidays. A label can finish after the nominal boundary used to admit it into a fold.

Evidence on the available SPY calendar: December 2, 2024 plus 20 weekdays is December 30; 20 exchange sessions reaches December 31. Ordinary next-rebalance exits often use observed panel dates; this finding specifically concerns boundary/purge and fallback terminal calculations, not every exit.

Improvement: store actual entry and label-end timestamps, use the exchange calendar, and purge by those timestamps. Test holidays, missing ticker sessions, and fold boundaries.

### 8. High research limitation: feature selection is not isolated inside folds

Locations: `alpha_factor_backtest.py:89–126`; `core_satellite_nested_walkforward.py:3447–3449`, grid rationale near `2532`.

The current shortlist exists and includes statistics over roughly 4,100 days. Features are globally ranked by absolute t-statistic, with globally supplied directions, before outer fold construction. Current feature-health enrichment is also applied at this stage. Unless an artifact is demonstrably trained only before every relevant fold, the outer history is not a clean test of the complete selection process. Shifting later IC weights cannot undo earlier feature selection.

Code also describes choosing grid dimensions from previous walk-forward results. Reusing those same outer years evaluates a research process already informed by them.

Improvement: fit feature selection, direction, health weighting, calibration, and parameters within each training fold. Version their data cutoffs. Reserve a genuinely unused final period and prospective paper cohort. Record all trials; apply multiple-testing diagnostics only after repairing the return ledger. The exact performance inflation is not quantified here.

### 9. Medium: the edge monitor calls raw return “alpha”

Location: `factor_decay_monitor.py:333–364`, classification around `138`.

`overlay_alpha_sum_pct` is identical to the sum of overlay returns. It subtracts neither an exposure-matched benchmark nor trading costs, and it is not compounded wealth. A strategy can rise with the market, underperform its alternative, and still report positive “alpha.” This field feeds edge classification.

Evidence: the helper labels a positive 1% overlay return as positive 1% alpha without a benchmark comparison. The saved monitor's 60/120-day ranking correlations are approximately **−0.066 / −0.077**, based on only **2 / 5 cohorts**. That is weak, sparse evidence, not proof that the strategy has no edge. Raw positive returns do not resolve it.

Improvement: report net portfolio return, exposure-matched excess return, and regression alpha separately. Use consistent return aggregation and confidence intervals that preserve dependence across dates/cohorts. Require enough matured independent observations before a healthy-edge conclusion.

### 10. Medium: ETF cache can return the wrong requested dates

Location: `backtest.py:3680–3688`.

The cache key contains symbols and date-range endpoints, but not the complete requested index. Cached results return without reindexing. Two requests with identical endpoints but different interior dates can receive the first request's rows.

Evidence: requesting two endpoints followed by all three dates returned only two rows on the second call. This can make a research run depend on request order. Its impact on the saved headline result is not yet quantified.

Improvement: cache the underlying full price series and align each request explicitly, or key the entire index. Avoid backward filling tradable prices from future observations; specify missing-price behavior.

## Execution evidence: what improved, and what is still missing

Restored records reconcile eight parent orders and ten attempts, including partial-fill/cancel/retry paths. The scorecard has nine rebalance child fills over two sessions, no unmatched fills, and approximately 1.161 basis points of fill-minute slippage. Six fills predate the current paper epoch. That is useful operational evidence, but not nine independent strategy experiments or proof of current-epoch profitability.

Current checks show the configuration lock and position alignment passing, and protective stops covering the four holdings. Keep these controls. The collection gate of 20 fills across three sessions is an operational minimum, not a statistical threshold proving edge.

Measurement improvements:

- Preserve the decision-time bid/ask, feed, timestamps, target, and original requested quantity. Measure arrival-price implementation shortfall alongside fill-minute VWAP. The fill's minute includes market activity after the fill and is a different benchmark.
- Separate passive fills, retries, cancellations, unfilled opportunity cost, and protective-stop executions. Report parent completion and child execution costs without double counting.
- Paginate broker order history. `alpaca_paper_trading.py:3068–3086` currently caps the query at 100 orders; coverage within the returned subset does not prove complete coverage once volume grows. No present truncation was demonstrated.
- Measure distributions by spread, liquidity, order size, side, and session, not just one mean. Paper simulation does not model all live queue, impact, and latency effects; see [Alpaca's paper-trading limitations](https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading).

## Additional helper risks and scope limits

`labels.py:49` substitutes zero benchmark returns when the benchmark column is absent. This silently changes an excess-return target into raw return. Make missing required benchmark data a visible failure or invalid label. This is an auxiliary ML-path issue, not evidence that the active factor strategy used that fallback.

The auxiliary `trade_rules.resolve_rule_exit` warrants a separate gap-through-stop/entry-day execution test before its results support deployment: assuming a stop-level fill despite an adverse opening gap is optimistic. This helper is not the active core/satellite ledger, so its effect on that ledger is not asserted.

This audit did not recalibrate market impact, rerun the entire historical search, estimate corrected net alpha, or verify every upstream raw price and corporate action. Do not interpret the existing headline CAGR, drawdown, or stress outputs as corrected results.

## Recommended improvement sequence

1. **Close the submission-time spread hole**, with a fake-broker regression and no live test orders.
2. **Repair regime confirmation, calendars, cache alignment, and forward-availability selection.** Add focused tests that demonstrate the old failures.
3. **Build the common daily holdings/cash ledger** for ETF and stock trades, drift, costs, stops, and timing. Reconcile it against recovered paper orders before rerunning research.
4. **Repeat causal validation** with fold-local feature selection, point-in-time universes, and an untouched holdout. Compare against buy-and-hold and risk/exposure-matched ETF alternatives. Preserve the original results as superseded evidence rather than silently replacing them.
5. **Repair edge and execution measurement**, then gather prospective paper observations under a newly documented strategy version. Keep old observations for diagnostics while distinguishing the changed policy's evidence.

Only after those steps should model complexity, new features, optimizer parameters, or increased capital be considered. The next useful project is the execution guard plus trustworthy accounting, not another search for a better historical score. For background on repeated selection, see the primary paper [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
