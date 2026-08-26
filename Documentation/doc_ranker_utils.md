# Ranker Utilities

## What it does

`ranker_utils.py` prepares same-day groups for XGBoost ranking, measures daily
Spearman information coefficient, and calculates adaptive research weights.

## How to run it

Run `python3 ranker_utils.py` for synthetic self-tests. In model code, import
`build_rank_groups_from_dates` and `daily_rank_ic`. Inputs are sorted dates,
scores, and future outcomes. Expected output includes group sizes, mean IC,
t-statistic, and the number of usable days.

## Key terms

- **Ranker:** a model that orders stocks rather than only predicting up/down.
- **IC:** correlation between predicted and realized ranks.
- **Group:** stocks compared with each other on one date.
