# causal_research.py

This module fits feature directions, selection and weights using only completed
training outcomes. A **feature** is an input such as past momentum. A **fold**
is a training window followed by a separate evaluation window. A **label** is
the future outcome used to assess an input; its actual end date determines
whether it was already known at the training cutoff.

Run through `python corrected_audit.py --spec corrected_shadow_spec.json`.
The input is a raw feature panel, feature names, configurations and dated
inner/outer folds. Output includes frozen `FoldArtifact` records and evaluated
folds. `python -m pytest tests/test_corrected_audit.py -q` checks that changing
held-out outcomes cannot change fitted artifacts or earlier scores.

Each inner fold selects and weights features independently, using daily rank
correlations, training coverage and training stability. Directions are fitted
there too. Within-day ranking is the scaling rule; no global scaler or external
shortlist is loaded. There is no probability calibration in this score model.
Inner validation selects a configuration; outer training then fits its own
artifact before outer evaluation. At least 20 usable training dates are needed.
Labels whose actual endpoint exceeds a training cutoff are purged.

Artifact and checkpoint identities include selected features, directions,
weights, training data, policy, costs, configuration and code versions. The
runner writes a fresh immutable trial directory; old research caches are not
accepted as corrected evidence. Previously inspected years remain retrospective
diagnostics regardless of their resulting returns.
