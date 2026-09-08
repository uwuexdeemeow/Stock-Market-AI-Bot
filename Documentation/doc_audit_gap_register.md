# audit_gap_register.py

## Evidence closure update

Creates stable IDs for original and new findings. Disappearance never closes a gap. The exact lock-difference report compares hashes without editing the lock. It is a library called by corrected_audit.py --evidence-report. Outputs are gap_register.json and gap_register.md. A dependency is evidence or an observation needed before another result can be verified.

Closure evidence can be supplied with corrected_audit.py --evidence-report
--verification-report PATH. The JSON identifies source_commit, code_files
(relative filename to SHA-256), test_report (relative JUnit XML path), its
SHA-256, and fixes (stable id, reason, fully qualified test names). Named tests
must actually pass, including no skip/error, and code bytes must match. A new
checkout requires re-verification; prior closure evidence remains attached.
Changing a next-action sentence or an observation count does not change a gap ID.
Normal-session evidence can be awaiting_new_observations; unavailable external
inputs remain blocked. Original source failures are not closed by code tests.
