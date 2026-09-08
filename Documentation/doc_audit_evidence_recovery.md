# audit_evidence_recovery.py

This script recovers evidence using read-only web requests. It never submits,
cancels or replaces an order. A **reconciliation** compares reconstructed cash
and shares against an independent account snapshot. A **candidate source** is
downloaded material that has not passed production data requirements.

Run from the project folder:

```powershell
python audit_evidence_recovery.py --paper --sources --price-probes SPY QQQ SIVB FRC
```

The paper option uses existing ALPACA_API_KEY and ALPACA_SECRET_KEY values from
environment variables or .env, exclusively against paper-api.alpaca.markets.
Price probes use the data API with adjustment=raw. Do not put keys on a command
line. Sources downloads two public membership repositories and their licenses.
Every run creates a timestamped directory under ignored data/audit_recovery;
private account responses are never automatically published or committed.

Activity pagination follows IDs until completion and rejects repeated IDs,
interruptions and safety-limit exhaustion. Fills use actual execution prices.
Separately posted fee debits become separate cash events, charged once. Unknown
activity types stop replay rather than disappearing. Recorded negative cash
is retained as historical margin evidence; simulation still rejects borrowing.

The report distinguishes matching arithmetic from verified opening balances.
Pre-trade equity can support a flat-start reconstruction but is not a historical
cash-and-position statement. This inferred opening cannot certify a freeze.
current_balance_snapshot.json records observed current cash and positions for
a future replay interval; it cannot replace an older opening balance.

Membership intervals preserve exits and re-entry and stop at the last source
date. Conflicting same-date snapshots fail conversion. Files stay candidates,
outside data/universe_membership.csv. Raw-price probes report first/last dates,
empty history and pagination remaining; partial or ambiguous ticker histories
are never labeled complete.

Offline verification:

```powershell
python -m pytest tests/test_audit_evidence_recovery.py tests/test_corrected_audit.py -q
```

Sources: [Alpaca account activities](https://docs.alpaca.markets/us/docs/account-activities),
[membership candidate](https://github.com/fja05680/sp500), and
[comparison candidate](https://github.com/hanshof/sp500_constituents).
Complete data checks, certified replay and corrected historical/stress runs
must precede a prospective freeze.

## Independently documented balance interval

```bash
python3 audit_evidence_recovery.py --paper --opening-balances inputs/opening.json
```

Optionally pass `--closing-balances inputs/closing.json`; otherwise an unchanged
current paper API snapshot closes the interval. Both JSON objects must contain
`cash`, `holdings`, `verified: true`, an attributed `source`, and an exact UTC
`observed_at` timestamp. Set verified only when actual independent records support
it. Never derive opening cash by subtracting trades from closing cash.

The runner retrieves the complete activity stream and pages all broker orders.
Activities after opening and through closing are replayed once. Date-only cash
entries overlapping the interval, unknown activity types, changing account
snapshots and incomplete order pagination block certification. Separate fee
activities are charged once. Missing arrival quotes are not reconstructed.

Private `events.csv`, balances, `orders.json`, `broker_history_report.json` and
`replay_reconciliation.json` stay in the ignored timestamped recovery folder.
A certified replay still cannot freeze or approve a strategy by itself. Pass its
summary to the unified evidence report using `--reconciliation-report`.

### Complete market-data pagination and action probes

```sh
python audit_evidence_recovery.py --price-probes SPY QQQ SIVB --action-probes SPY QQQ
```

These read-only requests use existing credentials and store original candidate
records under a new private `data/audit_recovery/` directory. A **page token** is
a provider's pointer to the next batch. The script follows tokens even on short
pages and rejects repeated tokens and duplicate identities. Raw price requests
record `feed=sip`, `adjustment=raw`, and `asof=-` to avoid silently renaming old
symbols using today's identity. Reports preserve request parameters, retrieval
times and hashes. They do not create an approved raw manifest.

Action probes retain the provider's original event type and all available fields.
They report missing dividend payment dates separately; no processing date is
substituted. Complete pagination means all returned pages were collected, not
that the provider covers every historical event. The report remains unverified
until issuer evidence establishes coverage and ledger-compatible semantics.

Paper interval reports now distinguish a no-activity balance check from an
interval containing recorded activity, with explicit fill/fee counts. Neither
scope by itself establishes profitability or starts a prospective freeze.

## Evidence closure update

Broker activity and order pages are saved privately before interpretation. Failed pagination or unsupported events cannot erase received evidence. Run python audit_evidence_recovery.py --paper --opening-balances PATH --action-probes SPY QQQ for read-only recovery. The reviewed dividend_payment_facts.json collection can match missing payment dates by symbol, CUSIP, ex-date and exact amount. dividend_payment_review.json records supported dates separately; original API rows remain unchanged and full action coverage remains unverified. A CUSIP identifies a security; it is distinct from the trading symbol.

Documented cash-only activity types (deposits, withdrawals, dividends, interest,
cash journals and cash transfers) map to their signed broker net amounts. Stock
transfers and reorganizations remain blocked until their share movements are
verified. Date-only cash records cannot certify an intraday interval boundary.
Type definitions: https://docs.alpaca.markets/us/docs/account-activities.
