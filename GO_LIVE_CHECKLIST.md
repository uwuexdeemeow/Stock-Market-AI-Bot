# Go-Live Checklist (rule V)

**Do not trade real money until every box is ticked.** The most dangerous
items come first.

This list is for **rule V**, the strategy the paper account has traded since
the 2026-10-30 signal: QQQ + BIL, rebalanced at each month-end with a
volatility target (`v_rule_signal.py`). The old XGBoost / machine-learning
checklist is retired. That layer is in `legacy/` and trades nothing.

**Where things stand today:** paper only. The code refuses any broker address
that is not Alpaca's paper address (`is_paper_endpoint` in
`alpaca_sdk_adapter.py`), so real-money trading is not possible without a
deliberate code change (section 4).

**Key terms**

- **Paper trading:** trading with pretend money on a real broker's test system.
- **Forward test:** judging a strategy only on data that did not exist when it
  was designed.
- **Drawdown:** how far the account has fallen from its highest value.
- **Halt (kill switch):** a hard limit in code. When it is crossed the bot
  sells everything and blocks new orders, whatever the strategy says.
- **Rebalance:** moving the account back to the target mix of QQQ and BIL.
- **Refreeze:** re-recording the fingerprints of the locked trading files in
  `paper_version_lock.json` after an approved change.

## 1. Forward-test checkpoint passed

The rules were fixed before the first trade, in
`Documentation/V_RULE_FORWARD_TEST.md`.

- [ ] 12 V rebalances are complete (about November 2027).
- [ ] `python forward_test_monitor.py --rule v` reports a **PASS** on all three:
  - [ ] **V1:** the account's worst fall is not worse than buy-and-hold QQQ's.
  - [ ] **V2:** the account's Sharpe ≥ QQQ's Sharpe − 0.20.
  - [ ] **V3:** mean trading cost ≤ 20 bps (QQQ and BIL fills only).
- [ ] No early operational stop is open (V3 above 40 bps after 3 rebalances, a
      V4 gap above 2 points in 3 months of any 6, or a rebalance more than 3
      sessions late).
- [ ] The 20% volatility target and the 63-day window were never changed
      during the test.
- [ ] The owner has read the result and agreed to go on. A pass allows a
      real-money discussion. It is not an automatic switch.

## 2. Drawdown halt verified

A hard limit must not depend on the strategy behaving. The account has two
halts. Both sell everything.

| Halt | Setting | Paper value (rule V) | Code |
|---|---|---|---|
| Fall from peak | `ALPACA_DD_HALT_PCT` | 40% | `check_portfolio_drawdown` in `alpaca_paper_trading.py` |
| One-day loss | `GUARD_PNL_HALT_PCT` | −12% | `execution_guard.py` |

Both values are set in `.github/workflows/daily_paper_trading.yml`.
`MAX_DRAWDOWN_HALT_PCT = 0.99` in `settings.py` is **not** this halt. It
belongs to the old backtest risk manager and is effectively off.

- [ ] The manufactured-crash tests are green:
      `python -m pytest tests/test_v_rule_drawdown_halt.py tests/test_halt_and_stop_cooldown.py tests/test_paper_rebalance_calendar.py`
- [ ] The halt level for **real money** is decided and written down before the
      first live order.
  - The paper value is 40% because rule V's worst fall in its test was −38%.
    A halt inside that range sells at a point where the tested rule held on.
  - A tighter real-money halt (for example 25%) is a different rule from the
    one that was tested. Write its own pre-registered check first, in
    `Documentation/DELAY_STRESS_PAPER_ADVISORY.md`.
- [ ] The live workflow sets both halt values explicitly. With no value set,
      the code falls back to 12% and −8%, the core-alpha settings.
- [ ] One paper drill is done: lower `ALPACA_DD_HALT_PCT` for one manual run
      until the halt fires, then confirm the sell orders appear in Alpaca and
      `signals/alpaca_halt_active.txt` is written. Restore the value after.
- [ ] You know how a halt clears: on account recovery or by owner review. The
      "market is risk-on again" restart does not apply to rule V.

## 3. Live keys separate from paper keys

- [ ] New Alpaca **live** keys are created. The paper keys are not reused.
- [ ] Live keys sit in their own GitHub secrets, with different names from
      `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (the paper pair).
- [ ] Only the live workflow reads the live secrets. Research, data-refresh and
      monitoring workflows keep the paper keys.
- [ ] No key is in the repository: `.env` is gitignored and
      `git log -p -- .env` prints nothing.
- [ ] The Alpaca account has two-factor login switched on.
- [ ] Repository access is limited to people you trust. Anyone who can edit a
      workflow can read its secrets.

## 4. Code and operations ready

- [ ] `python -m pytest tests/` is green.
- [ ] `python paper_validation_epoch.py --check-lock` passes (no locked trading
      file changed without a refreeze).
- [ ] The change that allows a live broker address is a separate, reviewed
      release with its own refreeze. Paper stays the default.
- [ ] The live workflow has only a `workflow_dispatch:` trigger and its timer
      is a cron-job.org job (never a GitHub `schedule:`).
- [ ] The paper account keeps running beside the live one, so the two can be
      compared.
- [ ] Alerts reach you: a failed run, a missed rebalance and a fired halt each
      send a notification (`notifications.py`).

## 5. Small first allocation

- [ ] The money at risk is money you can afford to lose.
- [ ] The first live size is small: about **10% of the target amount**.
- [ ] It stays at that size for at least **3 monthly rebalances**. Rule V
      trades once a month, so one month shows only one trade.
- [ ] Before any increase, live fills are compared with paper: cost per trade
      (V3) and the monthly tracking gap (V4) are in the same range.
- [ ] You have a written manual stop: pause the cron-job.org job, then close
      the positions in the Alpaca dashboard. You have tried the pause once.
- [ ] You have written down the loss at which you stop live trading and go
      back to paper.

**When every box is ticked:** start at the small size, review after each
rebalance, and increase in steps.
