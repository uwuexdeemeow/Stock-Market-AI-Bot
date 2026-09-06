"""Fit feature directions and weights inside training windows; freeze before evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from portfolio_ledger import LEDGER_VERSION

CAUSAL_VERSION = "fold-local-v1"


def fingerprint(value) -> str:
    """Stable identities stop old checkpoints from masquerading as new evidence."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class FoldArtifact:
    cutoff: str
    label: str
    features: tuple[str, ...]
    directions: tuple[float, ...]
    weights: tuple[float, ...]
    training_fingerprint: str
    version: str = CAUSAL_VERSION

    @property
    def identity(self):
        return fingerprint(asdict(self))


def fit_features(panel: pd.DataFrame, features: list[str], *, cutoff, label: str, maximum=20, minimum_dates=20) -> FoldArtifact:
    """Use only matured training outcomes; never read an external shortlist."""
    end_column = f"{label}_end_date"
    if end_column not in panel:
        raise ValueError(f"Actual label endpoints required: {end_column}")
    cutoff = pd.Timestamp(cutoff)
    train = panel.loc[(pd.to_datetime(panel.date) <= cutoff) & (pd.to_datetime(panel[end_column]) <= cutoff)].copy()
    train = train.sort_values(["date", "ticker"])
    if train.empty:
        raise ValueError("No matured training labels")
    ranked = []
    for feature in sorted(features):
        if feature not in train:
            continue
        correlations = []
        for _, group in train.groupby("date"):
            clean = group[[feature, label]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) >= 3 and clean[feature].nunique() > 1 and clean[label].nunique() > 1:
                correlations.append(clean[feature].corr(clean[label], method="spearman"))
        values = np.asarray(correlations, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < minimum_dates:
            continue
        mean = float(values.mean())
        # Direction, stability and availability all come from this training fold.
        stability = abs(mean) / (float(values.std()) + .05)
        coverage = float(train[feature].notna().mean())
        score = stability * coverage
        if mean != 0 and score > 0:
            ranked.append((score, feature, float(np.sign(mean))))
    ranked = sorted(ranked, key=lambda x: (-x[0], x[1]))[:maximum]
    if not ranked:
        raise ValueError("No eligible fold-local features")
    columns = ["date", "ticker", label, end_column] + sorted(set(features) & set(train.columns))
    identity = hashlib.sha256(pd.util.hash_pandas_object(train[columns], index=False).values.tobytes()).hexdigest()
    total = sum(x[0] for x in ranked)
    return FoldArtifact(str(cutoff), label, tuple(x[1] for x in ranked), tuple(x[2] for x in ranked), tuple(x[0] / total for x in ranked), identity)


def score_features(panel: pd.DataFrame, artifact: FoldArtifact) -> pd.DataFrame:
    """Within-day ranks use today's candidates; fitted directions stay frozen."""
    out = panel.copy()
    score = pd.Series(0., index=out.index)
    available = pd.Series(0., index=out.index)
    for feature, direction, weight in zip(artifact.features, artifact.directions, artifact.weights):
        ranks = out.groupby("date")[feature].rank(pct=True)
        score += ((ranks - .5) * direction * weight).fillna(0.)
        available += ranks.notna() * weight
    out["causal_score"] = score.div(available.replace(0, np.nan))
    out.attrs["feature_artifact"] = asdict(artifact)
    return out


def checkpoint_identity(*, code, data, policy, artifact, costs, config):
    return fingerprint({"code": code, "data": data, "policy": policy, "feature_selection": artifact,
                        "costs": costs, "config": config, "ledger": LEDGER_VERSION, "causal": CAUSAL_VERSION})


def nested_evaluate(panel, features, configurations, folds, evaluate, *, label, record_trial):
    """Refit every inner training window, then freeze an outer-training artifact.

    evaluate(scored_panel, configuration, start, end, artifact) must use the
    daily ledger; the returned net return selects a configuration inside training.
    """
    outputs = []
    for fold in folds:
        if not pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["start"]) <= pd.Timestamp(fold["end"]):
            raise ValueError("Outer training must end before evaluation begins")
        candidates = []
        for config in configurations:
            results = []
            for inner in fold["inner"]:
                if not pd.Timestamp(inner["train_end"]) < pd.Timestamp(inner["start"]) <= pd.Timestamp(inner["end"]) <= pd.Timestamp(fold["train_end"]):
                    raise ValueError("Inner evaluation must remain within outer training")
                artifact = fit_features(panel, features, cutoff=inner["train_end"], label=label)
                scored = score_features(panel.loc[pd.to_datetime(panel.date) <= pd.Timestamp(inner["end"])], artifact)
                result = evaluate(scored, config, inner["start"], inner["end"], artifact)
                record_trial(config, artifact, inner, result)
                results.append(float(result["total_return_pct"]))
            if not results:
                raise ValueError("At least one inner validation window is required")
            candidates.append((float(np.mean(results)), fingerprint(config), config))
        selected = sorted(candidates, key=lambda x: (-x[0], x[1]))[0][2]
        artifact = fit_features(panel, features, cutoff=fold["train_end"], label=label)
        scored = score_features(panel.loc[pd.to_datetime(panel.date) <= pd.Timestamp(fold["end"])], artifact)
        result = evaluate(scored, selected, fold["start"], fold["end"], artifact)
        record_trial(selected, artifact, fold, result)
        outputs.append({"fold": fold, "configuration": selected, "artifact": asdict(artifact), "metrics": result,
                        "interpretation": "retrospective_diagnostic"})
    return outputs
