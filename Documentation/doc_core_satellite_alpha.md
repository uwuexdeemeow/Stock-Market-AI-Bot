# core_satellite_alpha.py

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
