# Remaining audit repair delivery

This document supplements, and does not overwrite, the original September 6
audit report and evidence JSON. Older historical/stress outputs are superseded
diagnostics until regenerated with verified corrected inputs.

| Finding | Corrected code path | Evidence limit |
| --- | --- | --- |
| 3–5: incomplete portfolio accounting | `portfolio_ledger.py`, `paper_policy.py`, explicit core adapter | Daily OHLC simulation approximates 9:35 execution; recorded fills are replayed separately. |
| 8: fitted research leaking across folds | `causal_research.py`, corrected nested entry point | New fold-local artifacts; existing inspected history remains retrospective. |
| 9: raw returns called alpha | `edge_evidence.py`, migrated factor monitor | No healthy conclusion without paired confidence bounds and 20 matured cohorts. |
| Historical universe/actions | `corrected_data.py` importer and strict provenance gates | Verified historical constituents, raw prices, and complete action coverage are not present locally; affected validation remains blocked. |
| Execution measurement | `broker_history.py`, paper report, scorecard completeness gates | Missing historical arrival quotes, fees and account balances remain explicit gaps. |
| Auxiliary labels/stops | `labels.py`, `trade_rules.py` | Missing required benchmarks/history fail; conservative daily ordering is an assumption. |

The recovered September 3 fixture has three child fills: INTC +155 and +50,
and FCX +30. Known share changes reconcile to INTC +205 and FCX +30. The fixture
does not document complete opening/closing account balances or fees, and its
historical execution report rounds prices. It cannot establish cash reconciliation
within one cent. No balances or fee zeros were invented to make it pass.

`corrected_shadow_spec.json` supplies a reviewable candidate with the existing
allocation and paper sizing defaults, a fixed raw feature family fitted inside
training, three outer historical folds and separate selected-configuration cost
stresses. Dated VIX, earnings and sector context is required. This candidate is
not an automatic replacement for the current paper strategy.

The 252-session clock has **not started**. Complete verified data, corrected
evaluation, recorded-event reconciliation and an explicit freeze must precede
new shadow observations. Code/protocol changes restart the period. A 252-session
review with fewer than 20 independent 20-session cohorts remains inconclusive.

Offline regressions cover cash/share conservation, initial ETF costs, drift,
terminal costs, interim drawdown, stops, gaps, splits/dividends, partial fills,
missing balances/fees, fold-local artifacts, future-data perturbations, source
gates, benchmark underperformance, pagination and parent/unfilled accounting.
No test broker orders or strategy cutover are part of this delivery.

Final offline suite: **633 passed, 45 skipped**. Ten existing third-party
date-frequency deprecation warnings remain. The source audit reports 20 gaps;
no corrected historical returns or prospective freeze were manufactured.
The new specification, module syntax, documentation coverage and workflow YAML
were also checked. Operational output stays outside the code commit.
