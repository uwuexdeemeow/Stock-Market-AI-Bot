# robustness_scoring.py - Score Candidate Strategy Configs

## What it does in plain language

This script gives every candidate strategy config a single robustness score.
The nested walkforward uses that score to choose which config gets tested on
the next out-of-sample fold.

The score tries to avoid configs that only look good because of one lucky
metric. It combines reward metrics, such as Sharpe and QQQ alpha, with risk
penalties, such as drawdown, turnover, and unstable validation flags.

## Current default objective

The default objective is `alpha_vs_qqq`:

```text
score = (alpha_vs_qqq_pct / 10)
      - drawdown penalty
      - instability penalty
```

The QQQ-alpha objective won the recent comparison against the hybrid
experiment. Hybrid had lower compound return, lower CAGR, lower QQQ alpha,
higher worst turnover, and score predictiveness still failed.

## Hybrid experiment

The optional `hybrid` objective is:

```text
score = 0.50 * Sharpe
      + 0.50 * (alpha_vs_qqq_pct / 10)
      - drawdown penalty
      - 0.25 * turnover penalty
      - instability penalty
```

This is kept as an opt-in experiment:

- Sharpe keeps the selector tied to risk-adjusted quality.
- QQQ alpha keeps the selector focused on beating the opportunity-cost
  benchmark.
- A small turnover penalty discourages churn without letting turnover dominate
  the decision.

## Reverting or comparing objectives

You can switch objectives without changing code by setting
`ROBUSTNESS_OBJECTIVE` before a walkforward run.

PowerShell examples:

```powershell
$env:ROBUSTNESS_OBJECTIVE = "hybrid"
python core_satellite_nested_walkforward.py --recent-alpha-grid --workers 1 --no-resume --no-publish-live-config --output-prefix core_satellite_nested_walkforward_hybrid

$env:ROBUSTNESS_OBJECTIVE = "alpha_vs_qqq"
python core_satellite_nested_walkforward.py --recent-alpha-grid --workers 1 --no-resume --no-publish-live-config --output-prefix core_satellite_nested_walkforward_alphaqqq_check

$env:ROBUSTNESS_OBJECTIVE = "sharpe"
python core_satellite_nested_walkforward.py --recent-alpha-grid --workers 1 --no-resume --no-publish-live-config --output-prefix core_satellite_nested_walkforward_sharpe_check
```

Supported objectives:

- `alpha_vs_qqq`: Current default. It uses QQQ alpha as the primary metric and
  drops the turnover penalty.
- `hybrid`: Opt-in blend of Sharpe and QQQ alpha.
- `sharpe`: Original behavior. It uses Sharpe as the primary metric with full
  drawdown, turnover, and instability penalties.

## How to inspect results

After each walkforward, run the analyzer with the matching objective:

```powershell
python walkforward_analyzer.py --csv signals\core_satellite_nested_walkforward_hybrid.csv --objective hybrid --json
python walkforward_analyzer.py --csv signals\core_satellite_nested_walkforward_alphaqqq_check.csv --objective alpha_vs_qqq --json
python walkforward_analyzer.py --csv signals\core_satellite_nested_walkforward_sharpe_check.csv --objective sharpe --json
```

Compare these fields first:

- `score_predictiveness.verdict`
- `corr_inner_score_vs_oos_objective`
- `compound_return_pct`
- `cagr_pct`
- `mean_oos_sharpe`
- `mean_oos_alpha_vs_qqq_pct`
- `worst_oos_turnover_pct`
- live approval reasons in the walkforward checkpoint JSON

## Key terms

- Sharpe: Return adjusted for volatility. Higher usually means cleaner risk.
- Alpha vs QQQ: How much the strategy beat or trailed QQQ.
- Drawdown: Peak-to-trough loss. A large negative drawdown is risky.
- Turnover: How much the portfolio changes. High turnover can create trading
  costs and live execution risk.
- Instability flag: A validation warning that says a config failed a robustness
  check.
- Objective: The formula used to turn many metrics into one selection score.

## Expected outputs

This script is usually imported by other scripts rather than run directly.
The main function, `robustness_score_components`, returns a dictionary with:

- `robustness_score`
- `primary_metric`
- `drawdown_penalty`
- `turnover_penalty`
- `instability_penalty`
- `objective`
- `sharpe_component`
- `alpha_vs_qqq_component`
