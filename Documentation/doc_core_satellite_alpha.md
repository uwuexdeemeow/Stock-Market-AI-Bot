# core_satellite_alpha.py

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

When a config enables the concentration overlay guard, the script reduces the
stock-picking overlay during mega-cap concentration and rotates that freed
exposure into the ETF core instead of leaving it idle.

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

## Key Terms

- **Core**: the ETF part of the portfolio, usually SPY/QQQ/TQQQ.
- **Satellite / overlay**: individual stock picks added around the ETF core.
- **Regime**: market state such as risk-on, neutral, or risk-off.
- **Factor score**: a ranking number made from market features.
- **Research score route**: an explicit score-source experiment.  For example,
  `regime_adaptive_riskoff_guard` keeps normal risk-on and neutral scores but
  lets the risk-off score-health guard choose between defensive and
  walk-forward rankings from shifted trailing history.  It keeps the live
  default unchanged until fixed validation proves whether the route is useful.
- **Sentiment veto**: optional live-news check that can remove a selected stock
  when fresh headlines are strongly negative.
- **Concentration overlay guard**: lowers individual-stock overlay exposure
  when QQQ-style mega-cap concentration is high, then shifts the freed sleeve
  back into the core ETF allocation.
- **Nested walkforward approval**: the validation result that decides whether a
  config is allowed to generate paper/live signals.
- **Live config hash**: a short ID for the approved config used to create the
  signal.
