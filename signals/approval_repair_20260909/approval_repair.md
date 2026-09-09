# Approval identity repair

Code is on main at 9d42ba37c9e2af2306637aeb26ef9fbc2bb49f8a.

The daily loader and audit now share one approval-identity validator. Nested
rejections and different bundle references block both paths. Rebuilding from
matching source evidence publishes the actual decision consistently at both
levels, including rejection. Migration output reports the written decision.

Verification: 715 tests passed, 37 skipped; all four required CI jobs passed.
The register preserves all 92 original findings and records 24 verified repairs.

[Latest audit](strategy_evidence_report.md) · [Closure register](gap_register.md)
· [Named test proof](code_verification.json)

Overall status remains blocked. Matching historical inputs and successful
corrected evaluation are still missing; the existing lock has not been replaced,
no freeze has started, and this repair submitted no orders.
