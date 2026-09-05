# Saved September 3 execution example

The order plan and two journal rows come from immutable repository commit
`364a9e5daa69e70a997313c402a4c7ebae4b8b58`, Daily Paper Trading run 232
(`33764832587`). The plan is reduced to identity and quantity columns; the
journal retains the original attempt and cash-clamp details.

The three broker child-fill measurements come from commit `b0cda94`,
September 4 post-market reporting. Only September 3 orders are included.
These are historical paper records, not new trades or a synthetic backtest.

INTC: intended 205, first fill 155, second fill 50. FCX: planned 43,
cash-clamped to 30, first attempt canceled without a fill, second filled 30.
No test uses credentials or contacts Alpaca.
