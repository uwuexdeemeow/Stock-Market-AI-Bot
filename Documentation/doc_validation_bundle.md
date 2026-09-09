
Daily loading and the unified audit now share `validate_live_approval_identity`.
They require matching configuration and bundle fingerprints and consistent
approval states at the top and selected-strategy levels. A rejected record
cannot be overridden by another approval. Rebuilding from matching evidence
copies the new bundle's actual decision to both levels; it never substitutes a
passing decision for a rejected bundle. No lock or freeze changes occur.
