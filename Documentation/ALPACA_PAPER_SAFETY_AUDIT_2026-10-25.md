# Alpaca paper bot: safety audit against Alpaca's paper-trading skill (2026-10-25)

**What this is:** a read-only check of the live paper bot against the rules
in Alpaca's official paper-trading skill (`alpacahq/alpaca-skills`, pinned
commit `89d0943`). Nothing was changed, and no order was placed.
`alpaca_paper_trading.py` and `alpaca_sdk_adapter.py` are both in
`paper_version_lock.json`, so any fix needs a refreeze. Whether and when to
do that is the owner's call. The first rule V trade is 2026-11-02.

**Key term — fail closed:** when the bot can't prove something is safe, it
stops instead of carrying on.

## Account snapshot (Alpaca CLI, read only, 2026-10-25)

| | |
|---|---|
| Endpoint | `https://paper-api.alpaca.markets` (`alpaca doctor`) |
| Status | ACTIVE |
| Equity / cash | $104,346 / $687 |
| Positions | AMAT 35, AMD 39, CAT 8, QQQ 92 (the 2026-10-15 core-alpha book, as `V_RULE_FORWARD_TEST.md` expects) |
| Open orders | 0 |
| Multiplier | 4 (margin account; the bot never uses margin, see below) |

**What 2026-11-02 should look like:** `generate_orders` takes every ticker
the account holds plus every ticker in the signal (`alpaca_paper_trading.py:1641`).
AMAT, AMD and CAT have a target of 0, so they get sold. On the 2026-10-23
close, QQQ's 63-day volatility was 19.0%, which would mean 100% QQQ. That is
close to the 20% line, so the 10/30 close could still put a few percent in
BIL. Rough order set if the weight stays at 100%:
- sell AMAT 35, AMD 39 and CAT 8;
- buy about 56 QQQ (to about 148 shares).

## What already meets the skill

| Skill rule | Where in the bot |
|---|---|
| One door for orders | Every order and cancel goes through `AlpacaBroker` (`alpaca_paper_trading.py:1043`). The gauntlet, protection, execution guard, broker truth, broker health and dashboard all build it. |
| Refuse a non-paper endpoint at startup | `alpaca_paper_trading.py:1071` raises before connecting. |
| Account must be active and unblocked | `alpaca_paper_trading.py:1088-1096` |
| Unique client order IDs | `bot_client_order_id` (`:1974`), plus the same-day duplicate-run check (`:5190`) |
| Check buying power | Spending is capped at min(cash, buying power) (`:2652`), so 4× margin is never used. |
| Market clock | `is_market_open` (`:1243`) |
| Retry-After on 429 | Data calls only: `http_retry.py:102` |
| Rule V can't trade the wrong signal | `PAPER_MODE_STRATEGY` must match the signal type (`:4950`). |
| Secrets | Read from environment / GitHub Secrets; never printed |

## Gaps (ranked)

### G1 — The paper check is a text match on a configurable URL (medium)
`"paper-api.alpaca.markets" not in ALPACA_BASE_URL.lower()`
(`alpaca_paper_trading.py:1071`, also `:1551`, `:3899`). The adapter then
sets `paper = ... in base_url` (`alpaca_sdk_adapter.py:47`). A URL such as
`https://paper-api.alpaca.markets.example.com` passes the check, and the
paper keys would be sent to another host. Live money is not at risk,
because paper keys don't work on the live API. The real risk is leaking
the keys. The skill asks for the resolved host to equal the paper host, and
for a literal `paper=True`.
**Fix:** `urllib.parse.urlsplit(url).hostname == "paper-api.alpaca.markets"`
and `scheme == "https"`, in one helper used at all three places, and
`TradingClient(..., paper=True, ...)` written literally.

### G2 — An order sent again after a timeout isn't looked up first (medium-low)
`place_order` has `@_broker_retry` (`:1158`), which retries on "timeout".
If the first try did reach Alpaca, the retry is rejected as a duplicate
client order ID. So no double order happens, but the bot logs a failure for
an order that is live. The two-stage limit flow then loses track of it
until broker truth reconciles. The skill says to query the order by client
order ID before any retry.
**Fix:** on a timeout or a duplicate-ID error, call
`GET /v2/orders:by_client_order_id` and return that order's ID. The adapter
needs a `get_order_by_client_id` method.

### G3 — Broker retries ignore Retry-After / X-RateLimit-Reset (low)
`_broker_retry` (`:1006`) waits a fixed 1, 2, then 4 seconds. Rule V makes
very few calls a month, so the risk is small.

### G4 — Only `ACTIVE` status is accepted (low)
Alpaca documents `PAPER_ONLY` as a valid status too. Today the account is
`ACTIVE`. If Alpaca ever reports `PAPER_ONLY`, the bot would fail closed.
That is safe, but the run would stop.

## Advice

None of G1–G4 can cause a wrong trade on 2026-11-02, so the release can stay
frozen through the first V rebalance. G1 and G2 are worth one bundled fix
with tests and a single refreeze, after the 11/2 fills are reconciled. The
fix changes plumbing, not what is traded, so the FV clock keeps running.

> Paper trading is simulated and may differ from live trading in fills, market impact, liquidity, fees and latency. This is not investment advice. Alpaca disclosures: https://alpaca.markets/disclosures
