# publish_live_config_from_csv.py

## What This Script Does

`publish_live_config_from_csv.py` promotes an already-created nested
walkforward CSV into the live config files the paper/live trading bot reads.
It is a fast path for when the expensive walkforward run already finished and
you only need to choose which validated config should be used next.

By default it uses `stable_family` selection.  A stable family groups configs
by the big behavior choices: score source, shape, weighting, risk mode, and
TQQQ weight.  Smaller knobs like holding days and overlay size are ignored for
the stability count, then the newest fold inside the winning family supplies
those smaller values.

The publisher now preserves the selected risk mode exactly.  If the winning
config says `risk=off`, the live config keeps both `drawdown_circuit_breaker`
and `vol_target` at `0.0`.  If it says `risk=defensive`, both controls are set
to the defensive values that the nested walkforward tested.

It also records `walkforward_analyzer.py` warnings in the approval payload.
WARN verdicts flag research issues that should be monitored at paper size.
FAIL verdicts now block approval unless you explicitly use `--force`, because
a real analyzer failure means the validation method itself is not trustworthy
enough to promote.

The publisher also blocks large selection-bias gaps.  In plain language, if
the inner validation years looked much better than the outer unseen years, the
selector may be overfitting.  The gap is measured as average
`inner_mean_sharpe - oos_sharpe`, so both sides use Sharpe units.

If a selected config includes a dynamic concentration overlay such as
`conc_ov=qqq_spy_dynamic:0.3-0.7`, the publisher carries those exact controls
into the live config and keeps that family separate from configs without the
overlay.

## How To Run It

Dry-run first:

```bash
python3 publish_live_config_from_csv.py --source stable_family --dry-run
```

Publish after review:

```bash
python3 publish_live_config_from_csv.py --source stable_family
```

Other selection modes remain available:

```bash
python3 publish_live_config_from_csv.py --source most_common
python3 publish_live_config_from_csv.py --source latest
python3 publish_live_config_from_csv.py --source best_sharpe
python3 publish_live_config_from_csv.py --source top_family
```

Expected inputs:

- `signals/core_satellite_nested_walkforward.csv`

Expected outputs:

- `signals/core_satellite_live_configs.json`
- `signals/core_satellite_nested_walkforward.json`

Both JSON outputs are written atomically after the backup step, so the trading
bot never sees a half-written live config while publishing is in progress.

## Key Terms

- **Nested walkforward**: a test where each outer year is held out as unseen
  data, while earlier years choose the config.
- **Config**: one exact set of strategy settings.
- **Stable family**: a group of similar configs that share the important
  behavior choices.
- **Selection-bias gap**: how much better the selected config looked in inner
  validation than it later looked in the held-out outer year.
- **Dynamic concentration overlay**: a rule that changes overlay exposure when
  QQQ is far ahead of SPY, used as a market-concentration risk control.
- **TQQQ**: a leveraged Nasdaq ETF.  The script keeps TQQQ and no-TQQQ configs
  in different families so leverage does not sneak into a safer family.
- **Dry run**: prints what would happen without writing files.
