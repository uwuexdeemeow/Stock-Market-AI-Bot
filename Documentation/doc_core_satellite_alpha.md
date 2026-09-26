# core_satellite_alpha.py

`--validation-refresh` is a research-only maintenance option used while
rebuilding robustness evidence for an already walk-forward-approved config.
It can generate local signal artifacts before the replacement evidence bundle
is complete, but it never connects to the broker or submits orders. Normal
daily runs never use this option.

The generated metrics also retain the deployment gross-exposure ceiling so
all downstream robustness reports identify the exact tested configuration.

Generated signals include a metadata-only `run_id`. It links the target to
orders and reports without changing a score, target weight, or trading rule.
The exact latest feature rows used by that calculation are saved to
`signals/core_satellite_alpha_input_snapshot.csv` for later reproduction; live
trading never reads this snapshot.

Approved evidence paths are portable between Windows development machines and
Linux GitHub runners. A stored backslash is normalized before the validation
bundle is opened, so valid published evidence is not rejected as missing.

## What This Script Does

`core_satellite_alpha.py` builds the daily core-satellite strategy signal.  In
plain language, it keeps a core ETF allocation such as SPY/QQQ, then adds a
small stock overlay chosen from factor scores.

For live or paper trading, it loads the approved config from
`signals/core_satellite_live_configs.json`, evaluates that config, and writes
`signals/core_satellite_alpha_signal.csv`.  The signal now includes the live
config hash and creation time so the broker can reject old signals after a new
config is published.

Before writing the signal, it checks factor-data freshness against the latest
completed NYSE session so weekends and holidays do not create false stale-data
warnings.

Research configs can optionally use a **concentration overlay target**.  That
rule reads the trailing 120-trading-day QQQ-vs-SPY return gap.  If QQQ is
strongly leading SPY, the config can raise the overlay gross for a compact
top-three basket; otherwise it can keep a smaller overlay in broader markets.

The optional `sticky_blend` research setting controls how much of a retained
position's old weight is kept. The approved live configuration omits it and
therefore remains frozen at the historical 0.65 default. The quant audit uses
0.80 only in a named shadow experiment.

Nested validation can set `deployment_max_gross_exposure=1.00`. In that mode,
the research engine finishes selection, caps, and sticky weighting first, then
scales core and stock weights together just like the paper signal. The trade
report records the raw gross, the final deployment scale, and the deployed
gross so reviewers can verify parity.

## How To Run It

Generate the daily signal:

```bash
python3 core_satellite_alpha.py
```

Offline or CI dry-run without live news sentiment retries:

```bash
CORE_ALPHA_SENTIMENT_VETO=0 python3 core_satellite_alpha.py
```

Typical safe flow:

```bash
python3 factor_data_health.py --strict
python3 core_satellite_alpha.py
python3 alpaca_paper_trading.py --submit --dry-run
```

Expected inputs:

- Factor parquet files in `data/factors/`
- ETF price data in `data/`
- Feature specs and quality reports in `signals/`
- Approved live config in `signals/core_satellite_live_configs.json`

Expected outputs:

- `signals/core_satellite_alpha_signal.csv`
- `signals/core_satellite_alpha_metrics.json`
- `signals/core_satellite_alpha_equity.csv`
- `signals/core_satellite_alpha_trades.csv`

These signal/report artifacts are written atomically, so the broker,
dashboard, and daily gates never read half-written CSV or JSON files if the
script is interrupted.

## Key Terms

- **Core**: the ETF part of the portfolio, usually SPY/QQQ/TQQQ.
- **Satellite / overlay**: individual stock picks added around the ETF core.
- **Regime**: market state such as risk-on, neutral, or risk-off.
- **Concentration overlay target**: a rule that changes overlay size based on
  whether QQQ is strongly leading SPY.
- **Factor score**: a ranking number made from market features.
- **Research score route**: an explicit score-source experiment.  For example,
  `regime_adaptive_riskoff_guard` keeps normal risk-on and neutral scores but
  lets the risk-off score-health guard choose between defensive and
  walk-forward rankings from shifted trailing history.  It keeps the live
  default unchanged until fixed validation proves whether the route is useful.
- **Sentiment veto**: optional live-news check that can remove a selected stock
  when fresh headlines are strongly negative.
- **Nested walkforward approval**: the validation result that decides whether a
  config is allowed to generate paper/live signals.
- **Live config hash**: a short ID for the approved config used to create the
  signal.

## Validation Source

Paper signals verify the tracked validation bundle checksum and config
fingerprint. An ignored scratch walk-forward cannot replace it. Simulated cost
uses conservative Alpaca fill calibration after enough observations, otherwise
the configured floor. The strategy remains `paper_provisional`.

## Flat-Start Research Windows

`run_core_satellite()` accepts optional `evaluation_start` and
`evaluation_end` dates. The full input table remains available for moving
averages and rankings, but the simulated account starts in cash at the first
eligible date. Only positions that both enter and exit inside the window are
counted. The returned metrics show the effective dates and how many crossing
positions were deliberately removed.

Live configuration loading also rereads the current robustness reports. A
missing, stale, changed, warning, or blocked report makes `paper_ready=false`,
even if an older configuration file contains a copied passing review.

## Exact returns and yearly alpha

If early regime-change rebalancing is enabled, stock returns now use the
observed entry Open and actual early-exit Close. The engine never estimates a
short return by multiplying a 20-day return by a fraction of 20 days.

Yearly alpha also compounds every strategy and benchmark period within the
calendar year before subtracting benchmark performance. This captures gains
and losses multiplicatively, as a real account experiences them.

## September 2026 submission and historical-data repair

Trend and high-volatility flags now change only after a full window of consecutive agreeing observations. The first observation initializes the state; mixed or incomplete windows keep it unchanged. `regime_confirm_days=1` uses each raw observation directly. Regime calculations use NYSE sessions, so holidays do not count as confirmation days.

Backtests retain candidates even when their future returns are missing. Selection uses decision-date information, then a selected holding with unavailable or invalid required prices/returns stops the run with its ticker and date. Do not replace that stock, assume zero return, or remove its row to make the run pass; restore and investigate the price history first.

Holding schedules, delayed ETF entries, terminal exits, and fold boundaries use NYSE sessions. Forward-label endpoints come from actual ticker observations, including missing ticker sessions. Whole periods whose outcomes extend beyond the evaluation cutoff are excluded. Without an explicit cutoff, the last panel date is used. Missing scheduled decision data stops the run. Early exits use actual stock Open and Close prices and keep their planned exit when the next trade is excluded by a fold boundary.

Trade output adds `label_end_date`, the latest outcome date for the evaluated period; `exit_date` retains the ETF/early-exit date. Stock next-session-open and ETF signal-close entry conventions remain unchanged. This is not the separate daily holdings/cash accounting repair.

ETF prices use the shared source cache and return independent copies. Run `python -m pytest tests/test_submission_history_guards.py tests/test_audit_three_fixes.py -q` for offline examples and expected passing checks. A regime is a market condition; a fold is a dated evaluation window; purging means excluding a whole trade period whose outcome is outside that window.

Historical performance and live signals produced with the old confirmation or date rules must be regenerated and validated before they are treated as current evidence. This repair does not itself publish replacement operational evidence.


## Remaining audit repair, September 2026

The explicit accounting_mode=daily-ledger-v1 adapter runs corrected daily cash/share accounting and preserves the existing equity/events/metrics return tuple. Programmatic callers must provide panel.attrs['ledger_inputs'] with verified bars, actions, provenance and membership_path; use corrected_audit.py for the complete offline workflow and fold-local scores. The existing active paper path is not switched by this adapter. Daily simulation uses prior-session decisions and next-open fills for stocks and ETFs; costs include initial ETF trades, drift, stops and evaluation terminal sales. Old period-return results are superseded diagnostics and need regeneration. Verification: python -m pytest tests/test_corrected_audit.py -q.

Historical results affected by these changes must be regenerated. Original audit evidence is preserved; no corrected historical claim is made when source checks are blocked.

The September 2026 daily-run repair excludes zero-weight ETF placeholders from
price requirements. A configuration that holds no TQQQ in any regime does not
need pre-inception TQQQ prices. Any nonzero allocation retains strict coverage
checks. Research dates, allocations and missing-price safeguards are unchanged.

The live loader uses the same approval-identity checks as the evidence audit.
A nested rejection or mismatched nested bundle hash blocks daily signals even
if top-level flags say approved. Research-only validation refresh remains
separately labeled and does not grant trading permission.

## September 2026 fix: signal shows today's stress results

The signal file used to embed the survivorship, execution-stress and
factor-decay reviews copied from the published live config. The daily
workflow refreshes those reports just before the signal is written, so the
embedded values were one refresh old. The signal now shows today's reviews
(read from `logs/`) and keeps the published copy in `*_review_published`.
`robustness_review_source` says which one is shown (`current_reports` or
`published_live_config`).

This only changes what is displayed. The pass/fail gate still uses the
published approval, and `robustness_snapshot_gate.py` still checks today's
reports before any order.

Test: `python -m pytest tests/test_locked_audit_fixes.py -q`.

## Known limit: the earnings blackout is currently inactive (2026-09-26)

The approved config sets `earnings_blackout_days = 5` ("don't buy a stock
within 5 days of its earnings report"). In practice the rule never triggers.
`settings.py` has `USE_EARNINGS_DATA = False`, so
`pipeline_shared.build_earnings_features_context` gives every stock a
placeholder `days_to_next_earnings = 60` on every date. That value is never
between 0 and 5, so nothing is skipped, in backtests or in the paper signal.

**Decision: leave it as is for now, and don't count it as protection.**

- Every backtest, walk-forward and stress result was produced with the rule
  inactive. Switching it on only for live trading would make paper trades
  differ from the strategy that was tested, and paper evidence would stop
  measuring the tested strategy.
- It can't be switched on in backtests honestly: the free earnings-date
  history (`fundamental_features.build_pead_features`, about 40 reports) does
  not reach back to the 2010 start.
- It only checks on the buy day. With 20-day holds, most reports that fall
  inside a hold (days 6–20) would still be held through.
- Research finds stocks earn slightly more, on average, around earnings
  ("earnings announcement premium"), so skipping them is not a free gain.
- The code involved is locked (`paper_version_lock.json`), and the incumbent
  is already on the paper advisory while a replacement is researched.

If a future strategy needs a real earnings filter, add it with earnings
history that covers the whole test period, and validate it like any other
rule change.

**Key term — "earnings blackout":** a rule that avoids buying a stock just
before the company reports its quarterly results, when the price can jump a
lot in either direction.

## September 2026 fix: no buy-back after a stop-loss

Every stock gets an 8% trailing stop in paper trading. Before this fix, a stock
sold by its stop was often bought right back the next morning, because it
still ranked near the top. That paid trading costs twice and undid the stop.

Now the signal reads `recent_protective_exits` from Alpaca's status file. A
stock sold by a stop sits out for one holding period (`holding_days`, 20
trading days). The next-ranked stock takes its slot. Stocks still held are
never affected, so this rule can block a re-entry but never forces a sale.
The signal records `stop_cooldown_tickers`, `stop_cooldown_source` and
`stop_cooldown_json`. If the exit list is missing, the rule is skipped and
`stop_cooldown_source` says why.

`daily_run.py` refreshes the status file right before the signal, so stops
that fired after yesterday's run are seen.

The backtest still has no stops at all (audit finding H1, left for later).

Test: `python -m pytest tests/test_core_satellite_live_signal.py -k stop -q`.

**Key term — cooldown:** a waiting period before the same stock can be bought again.
