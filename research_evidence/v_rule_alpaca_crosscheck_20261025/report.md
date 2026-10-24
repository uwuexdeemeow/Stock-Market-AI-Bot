# Rule V: independent Alpaca cross-check (2026-10-25)

**Question:** do the project's rule V numbers come from correct code and
correct data? Rule V is QQQ at min(1, 20% ÷ 63-day volatility), with the rest
in BIL, decided at each month-end close and traded at the next open. To
check, the rule was rebuilt from scratch (`run.py`, no project imports) on a
second data source (Alpaca CLI bars). It was then compared with the project's
code and data (`compare.py`).

**Answer: yes.** No code bug and no look-ahead were found, and the data
agrees. This is a check only. It does not re-judge V; that is still done by
`Documentation/V_RULE_FORWARD_TEST.md`.

## Cross-check result (common window 2016-05-02 → 2026-10-20)

| Code + data | CAGR | Sharpe (over BIL) | Max drawdown | Total return |
|---|---:|---:|---:|---:|
| Project code + project data (`data/*.parquet`) | 17.35% | 0.805 | −28.56% | 433.8% |
| Project code + Alpaca data | 17.36% | 0.806 | −28.56% | 434.3% |
| `run.py` (independent) + Alpaca data | 17.32% | 0.804 | −28.54% | 432.1% |

- **Month-end weights (126 signal days):**
  - Gap between `run.py` and the project's `vol_weight`: at most 0.0000005. They are the same.
  - Gap between `run.py` and the live signal helper `v_rule_helper.qqq_weight`: at most 0.0000005. They are the same.
  - Gap between project data and Alpaca data: at most 0.0009 (2018-03-29), 0.00007 on average. The data sources agree.
- **Yearly returns:** the three rows differ by at most 0.22 points in any
  year. `run.py` is slightly lower each year. That is expected: it charges
  5 bps on both the sell leg and the buy leg, plus Alpaca's regulatory fees.
  The project charges 5 bps per unit of weight changed.
- **Calendar:** the live signal (`v_rule_signal.month_end_schedule`) and
  Alpaca's exchange calendar give the same month-ends for 2016–2026. The
  next signal is **2026-10-30**, traded **2026-11-02**, which matches
  `V_RULE_FORWARD_TEST.md`.

## `run.py` alone (Alpaca data, 2016-05-02 → 2026-10-23)

| | Total return | Ann. return | Max drawdown | Sharpe (rf 0) | Sharpe (over BIL) | Final equity |
|---|---:|---:|---:|---:|---:|---:|
| **V** | 422.3% | 17.1% | −28.5% | 0.91 | 0.79 | $522,291 |
| QQQ buy-and-hold | 616.9% | 20.7% | −35.0% | 0.96 | 0.86 | $716,891 |
| BIL buy-and-hold | 25.8% | 2.2% | −0.2% | — | — | $125,783 |
| 50/50 QQQ+BIL, no rebalance | 321.3% | 14.7% | −27.5% | 0.94 | 0.80 | $421,337 |

- **Trading:** 181 fills on 126 rebalance days. Fees paid: $34. Slippage paid: $1,704. Turnover: 13.2× average equity.
- **Win rate (90%) and profit factor (203):** these come from FIFO lot
  closures. They are **not meaningful** for a rebalancing rule. V only trims
  or adds to an ETF that rose over the period, so most trimmed lots show a
  gain. Don't read them as a stock-picking hit rate.
- **First trade:** 2016-05-02, buy QQQ at the open ($98.20, filled at $98.25). Signal day 2016-04-29.
- **Last trade:** 2026-10-01, buy QQQ back to 100%. Signal day 2026-09-30.
- **What this window shows:** in 2016–2026, V lost to QQQ on return
  (−3.6 points a year) and on Sharpe (−0.06). It had a smaller worst fall
  (−28.5% vs −35.0%). This matches the project's own exam window. V's pass
  rests on drawdown, plus a Sharpe within 0.20 of QQQ's. It does not rest on
  beating QQQ.

## Data fingerprint

| | Bars | First | Last | Close sum |
|---|---:|---|---|---:|
| QQQ | 2,718 | 2016-01-04 | 2026-10-23 | 841,838.93 |
| BIL | 2,718 | 2016-01-04 | 2026-10-23 | 213,121.93 |

- Settings: Alpaca CLI 0.0.14, feed `sip`, adjustment `all`, 1Day bars, regular hours only.
- File hashes are in `data_fingerprint.json`.

## Caveats

- Alpaca daily data starts in 2016. The check covers 2016–2026 only, not
  2007–2022, the project's decision window. It confirms the code and the
  data. It cannot confirm the earlier years.
- Fills use bar opens plus 5 bps, not historical quotes.
- Fees come from Alpaca's fee schedule PDF (revised 2026-10-12), as read
  from the extracted text. The FINRA TAF line was read as $0 a share for
  equities, because the PDF text layout was ambiguous there. The amounts
  are tiny either way.
- Positions are sized from the fill-day open value, as the live bot does,
  not from the signal-day close.

> **Important disclosure**
> This backtest is a hypothetical historical simulation and does not represent actual trading performance. Backtested results do not guarantee future results. Results depend on market-data quality, data feed selection, corporate-action handling, fees, slippage, liquidity, taxes, execution assumptions, and implementation details. This material is for research and educational purposes only and is not investment advice, a recommendation, an offer, or a solicitation to buy or sell securities, options, cryptocurrencies, or any other financial product. All investments involve risk and may lose value. Review Alpaca's disclosures and agreements at [alpaca.markets/disclosures](https://alpaca.markets/disclosures). Fee schedule: https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf
