# portfolio_ledger.py

This offline module keeps cash and shares for every stock and ETF. A **ledger**
is a chronological record of money and ownership changes. **Equity** is cash,
documented dividend receivables, and the current value of owned shares.

Run it through `python corrected_audit.py --spec corrected_shadow_spec.json`,
after supplying the verified inputs described in `doc_corrected_audit.md`.
For examples that use only fake data, run
`python -m pytest tests/test_corrected_audit.py -q`.

`simulate_daily` returns a `LedgerResult` with events, daily holdings, daily
equity, metrics and data-quality findings. Decisions use the prior completed
NYSE session. Stocks and ETFs fill at the next open. This approximates the
9:35 paper workflow; it does not claim identical execution. Each independent
fold starts in cash. Evaluation-only final sales occur at the last eligible
close and include costs. Prospective observations omit that artificial sale.

Orders compare actual holdings with targets, including drift. Sells precede
buys; buys preserve the cash reserve. Turnover is absolute filled notional
divided by pre-trade equity. Spread, slippage and impact change fill prices;
per-share fees change cash once. The frozen fallback uses a 5-basis-point full
spread, 0.10% baseline slippage, the existing impact formula
`1000 * sqrt(shares / prior_average_daily_volume)` basis points, and $0.005
per share. One **basis point** is 0.01%. Stress multiplies all simulated costs,
including first ETF purchases, stops, drift trades and final sales.

Splits change shares and stop prices. Dividend rights are recorded on the
ex-date, remain receivables after a sale, and become cash on payment. Raw
execution prices avoid counting adjusted returns and dividends twice. Symbol
changes and documented cash liquidations are explicit actions; unsupported or
missing evidence stops evaluation.

Stock stops operate on the entry day. Existing adverse opening gaps execute
before new purchases; the stopped name cannot be repurchased that day. Both
open-high-low-close and open-low-high-close paths are examined, retaining the
worse long-position outcome and an ambiguity event. Drawdown uses every daily
close, even when a later recovery hides the loss. A close-time halt sells at
the next open; becoming cash never resets the peak or manufactures recovery.
Daily drawdown is not a measurement of intraday drawdown.

`replay_events` instead applies documented individual fills and actions, with
actual quantities, prices, fees, cancellations and child orders. Daily marks
are inserted in time order. Opening and closing balances are required for
reconciliation: shares must match within 1e-8 and cash within one cent. Unknown
fees or balances produce gaps, not assumed zeros. No terminal broker sale is
invented. `known_share_changes` can describe a partial fixture but is not a
closing balance or a successful reconciliation.

The schema is `daily-ledger-v1`. Earlier period-return backtests must be
regenerated; they cannot stand in for corrected ledger validation.
