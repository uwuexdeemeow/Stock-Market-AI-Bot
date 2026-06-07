"""
feature_health.py - redundancy and decay controls for factor scoring.

This module turns the feature diagnostics into a machine-readable profile used
by the scoring path.  The important invariant is that one economic signal gets
one vote, even if it appears as raw values, market ranks, sector ranks, or close
benchmark-relative variants.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_json
from settings import SIGNAL_DIR


CORRELATION_CLUSTER_THRESHOLD = 0.70
DECAY_QUARANTINE_RATIO = 0.50
DECAY_WATCH_RATIO = 0.80
STRENGTHENING_RATIO = 1.50
# Live trading needs several independent "signal families" so one crowded idea
# cannot dominate the overlay.  Keep these shared with alpha_factor_backtest.py
# so the report and the live capital gate judge the same thing.
MIN_ACTIVE_CLUSTERS = 6
MAX_CLUSTER_WEIGHT = 0.25


_RANK_PREFIXES = ("xs_rank_market_", "xs_rank_sector_")


def canonical_feature_root(feature: str) -> str:
    """Return the economic root used for raw/rank duplicate detection."""
    root = str(feature)
    changed = True
    while changed:
        changed = False
        for prefix in _RANK_PREFIXES:
            if root.startswith(prefix):
                root = root[len(prefix):]
                changed = True
    if root.startswith("rel_"):
        root = f"sector_{root}"

    match = re.fullmatch(r"ret_vs_(spy|qqq)_(\d+d)", root)
    if match:
        return f"ret_{match.group(2)}"
    return root


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {str(item): str(item) for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot == rroot:
            return
        keep, drop = sorted((lroot, rroot))
        self.parent[drop] = keep


def _load_quality_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_research_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if df.empty or "feature" not in df.columns:
        return {}
    return {str(row["feature"]): row for row in df.to_dict("records")}


def _health_state(row: dict | None) -> tuple[str, float | None, float | None, float | None]:
    if not row:
        return "healthy", None, None, None
    full_ic = pd.to_numeric(row.get("full_ic"), errors="coerce")
    recent_ic = pd.to_numeric(row.get("recent_ic"), errors="coerce")
    full_f = float(full_ic) if pd.notna(full_ic) else None
    recent_f = float(recent_ic) if pd.notna(recent_ic) else None
    if full_f is not None and recent_f is not None and abs(full_f) > 1e-9:
        ratio_f = recent_f / full_f
    else:
        ratio = pd.to_numeric(row.get("recent_vs_full_trend"), errors="coerce")
        ratio_f = float(ratio) if pd.notna(ratio) else None
    if ratio_f is None or full_f is None:
        return "healthy", ratio_f, recent_f, full_f
    if abs(full_f) > 0.002 and ratio_f < DECAY_QUARANTINE_RATIO:
        return "quarantined", ratio_f, recent_f, full_f
    if abs(full_f) > 0.002 and ratio_f < DECAY_WATCH_RATIO:
        return "watchlist", ratio_f, recent_f, full_f
    if ratio_f > STRENGTHENING_RATIO and recent_f is not None and abs(recent_f) > 0.002:
        return "strengthening", ratio_f, recent_f, full_f
    return "healthy", ratio_f, recent_f, full_f


def build_feature_health_profile(
    features: Iterable[str],
    *,
    quality_report_path: str | Path | None = None,
    research_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> dict:
    """Build and optionally persist the feature health profile."""
    feature_list = [str(f) for f in features]
    signal_dir = Path(output_dir) if output_dir is not None else Path(SIGNAL_DIR)
    quality_path = Path(quality_report_path) if quality_report_path is not None else signal_dir / "feature_quality_report.json"
    research_path = Path(research_summary_path) if research_summary_path is not None else signal_dir / "feature_research_summary.csv"

    quality = _load_quality_report(quality_path)
    research = _load_research_summary(research_path)

    uf = _UnionFind(feature_list)
    by_root: dict[str, list[str]] = {}
    for feature in feature_list:
        by_root.setdefault(canonical_feature_root(feature), []).append(feature)
    for root_features in by_root.values():
        first = root_features[0]
        for other in root_features[1:]:
            uf.union(first, other)

    feature_set = set(feature_list)
    for pair in quality.get("correlation_clusters", []):
        f1 = str(pair.get("feature_1", ""))
        f2 = str(pair.get("feature_2", ""))
        corr = pd.to_numeric(pair.get("correlation"), errors="coerce")
        if f1 in feature_set and f2 in feature_set and pd.notna(corr) and abs(float(corr)) >= CORRELATION_CLUSTER_THRESHOLD:
            uf.union(f1, f2)

    groups_by_rep: dict[str, list[str]] = {}
    for feature in feature_list:
        groups_by_rep.setdefault(uf.find(feature), []).append(feature)

    cluster_ids: dict[str, str] = {}
    clusters = []
    for idx, (_rep, members) in enumerate(groups_by_rep.items(), 1):
        cluster_id = f"cluster_{idx:02d}"
        for member in members:
            cluster_ids[member] = cluster_id
        clusters.append({
            "cluster_id": cluster_id,
            "canonical_roots": sorted({canonical_feature_root(m) for m in members}),
            "features": members,
        })

    feature_rows = []
    for feature in feature_list:
        state, ratio, recent_ic, full_ic = _health_state(research.get(feature))
        feature_rows.append({
            "feature": feature,
            "cluster_id": cluster_ids[feature],
            "canonical_root": canonical_feature_root(feature),
            "health_state": state,
            "recent_vs_full_ratio": ratio,
            "recent_ic": recent_ic,
            "full_ic": full_ic,
            "active_candidate": state != "quarantined",
        })

    root_states: dict[str, set[str]] = {}
    for row in feature_rows:
        root_states.setdefault(str(row["canonical_root"]), set()).add(str(row["health_state"]))
    quarantined_roots = {root for root, states in root_states.items() if "quarantined" in states}
    for row in feature_rows:
        if row["canonical_root"] in quarantined_roots and row["health_state"] != "quarantined":
            row["health_state"] = "quarantined"
            row["active_candidate"] = False
            row["root_quarantine_inherited"] = True
        else:
            row["root_quarantine_inherited"] = False

    by_cluster = {c["cluster_id"]: [] for c in clusters}
    for row in feature_rows:
        by_cluster[row["cluster_id"]].append(row)

    active_cluster_weights: dict[str, float] = {}
    for cluster in clusters:
        rows = by_cluster[cluster["cluster_id"]]
        healthy = [r for r in rows if r["health_state"] in {"healthy", "strengthening"}]
        watch = [r for r in rows if r["health_state"] == "watchlist"]
        quarantined = [r for r in rows if r["health_state"] == "quarantined"]
        if healthy:
            cluster_state = "active"
            raw_weight = 1.0
            contributing = [r["feature"] for r in healthy]
        elif watch:
            cluster_state = "watchlist"
            raw_weight = 0.5
            contributing = [r["feature"] for r in watch]
        else:
            cluster_state = "quarantined"
            raw_weight = 0.0
            contributing = []
        active_cluster_weights[cluster["cluster_id"]] = raw_weight
        cluster.update({
            "health_state": cluster_state,
            "raw_weight": raw_weight,
            "contributing_features": contributing,
            "quarantined_features": [r["feature"] for r in quarantined],
        })

    total_weight = sum(active_cluster_weights.values())
    for cluster in clusters:
        effective_weight = (cluster["raw_weight"] / total_weight) if total_weight > 0 else 0.0
        cluster["effective_weight"] = round(float(effective_weight), 6)

    weight_by_cluster = {c["cluster_id"]: float(c["effective_weight"]) for c in clusters}
    contributing_by_cluster = {c["cluster_id"]: set(c["contributing_features"]) for c in clusters}
    for row in feature_rows:
        row["contributes_to_score"] = row["feature"] in contributing_by_cluster.get(row["cluster_id"], set())
        row["effective_weight"] = weight_by_cluster.get(row["cluster_id"], 0.0) if row["contributes_to_score"] else 0.0

    active_clusters = [c for c in clusters if c["raw_weight"] > 0]
    quarantined_features = [r["feature"] for r in feature_rows if r["health_state"] == "quarantined"]
    watchlist_features = [r["feature"] for r in feature_rows if r["health_state"] == "watchlist"]
    max_cluster_weight = max((float(c["effective_weight"]) for c in clusters), default=0.0)
    summary = {
        "raw_feature_count": len(feature_rows),
        "cluster_count": len(clusters),
        "active_cluster_count": len(active_clusters),
        "effective_cluster_count": len(active_clusters),
        "quarantined_features": quarantined_features,
        "watchlist_features": watchlist_features,
        "max_cluster_weight": round(float(max_cluster_weight), 6),
        "feature_health_gate_pass": bool(
            len(active_clusters) >= MIN_ACTIVE_CLUSTERS
            and max_cluster_weight <= MAX_CLUSTER_WEIGHT + 1e-12
        ),
        "feature_health_gate_reasons": [],
    }
    if len(active_clusters) < MIN_ACTIVE_CLUSTERS:
        summary["feature_health_gate_reasons"].append(
            f"active_cluster_count {len(active_clusters)} < {MIN_ACTIVE_CLUSTERS}"
        )
    if max_cluster_weight > MAX_CLUSTER_WEIGHT + 1e-12:
        summary["feature_health_gate_reasons"].append(
            f"max_cluster_weight {max_cluster_weight:.3f} > {MAX_CLUSTER_WEIGHT:.2f}"
        )

    profile = {
        "purpose": "feature_health_profile",
        "correlation_cluster_threshold": CORRELATION_CLUSTER_THRESHOLD,
        "decay_quarantine_ratio": DECAY_QUARANTINE_RATIO,
        "decay_watch_ratio": DECAY_WATCH_RATIO,
        "min_active_clusters": MIN_ACTIVE_CLUSTERS,
        "max_cluster_weight_allowed": MAX_CLUSTER_WEIGHT,
        "summary": summary,
        "features": feature_rows,
        "clusters": clusters,
    }

    if write_outputs:
        # Write both live gate inputs through the crash-safe helper so a killed
        # refresh never leaves a half-written JSON/CSV file for trading to read.
        atomic_write_json(profile, signal_dir / "feature_health_profile.json")
        atomic_write_csv(pd.DataFrame(feature_rows), signal_dir / "feature_health_profile.csv")

    return profile


def enrich_feature_specs(specs: list[dict], *, write_outputs: bool = True) -> tuple[list[dict], dict]:
    """Attach health metadata to feature specs and return the profile."""
    profile = build_feature_health_profile(
        [str(spec.get("feature", "")) for spec in specs],
        write_outputs=write_outputs,
    )
    by_feature = {row["feature"]: row for row in profile.get("features", [])}
    enriched = []
    for spec in specs:
        updated = dict(spec)
        row = by_feature.get(str(spec.get("feature", "")), {})
        updated.update({
            "cluster_id": row.get("cluster_id"),
            "canonical_root": row.get("canonical_root", canonical_feature_root(str(spec.get("feature", "")))),
            "health_state": row.get("health_state", "healthy"),
            "recent_vs_full_ratio": row.get("recent_vs_full_ratio"),
            "effective_weight": row.get("effective_weight", 0.0),
            "contributes_to_score": bool(row.get("contributes_to_score", True)),
        })
        enriched.append(updated)
    return enriched, profile


def feature_health_summary_from_specs(specs: list[dict]) -> dict:
    profile = build_feature_health_profile(
        [str(spec.get("feature", "")) for spec in specs],
        write_outputs=False,
    )
    return dict(profile.get("summary", {}))


def _format_feature_list(features: list[str], limit: int) -> str:
    """Return a short comma-separated feature list for console output."""
    if not features:
        return "none"
    shown = features[:limit]
    suffix = "" if len(features) <= limit else f", ... (+{len(features) - limit} more)"
    return ", ".join(shown) + suffix


def _print_profile_summary(profile: dict, *, output_dir: Path, limit: int) -> None:
    """Print the health profile in a human-readable CLI summary."""
    summary = dict(profile.get("summary") or {})
    clusters = list(profile.get("clusters") or [])
    gate_pass = bool(summary.get("feature_health_gate_pass"))
    status = "PASS" if gate_pass else "FAIL"

    print("\nFEATURE HEALTH")
    print("=" * 70)
    print(f"Gate:                 {status}")
    print(f"Raw features:         {summary.get('raw_feature_count', 0)}")
    print(f"Clusters:             {summary.get('cluster_count', 0)}")
    print(f"Active clusters:      {summary.get('active_cluster_count', 0)} / {MIN_ACTIVE_CLUSTERS} min")
    print(f"Max cluster weight:   {summary.get('max_cluster_weight', 0)} / {MAX_CLUSTER_WEIGHT:.2f} max")

    reasons = list(summary.get("feature_health_gate_reasons") or [])
    if reasons:
        print("Gate reasons:")
        for reason in reasons:
            print(f"  - {reason}")

    quarantined = [str(x) for x in summary.get("quarantined_features", [])]
    watchlist = [str(x) for x in summary.get("watchlist_features", [])]
    print(f"Quarantined features: {len(quarantined)}")
    print(f"  {_format_feature_list(quarantined, limit)}")
    print(f"Watchlist features:   {len(watchlist)}")
    print(f"  {_format_feature_list(watchlist, limit)}")

    ranked_clusters = sorted(
        clusters,
        key=lambda row: float(row.get("effective_weight", 0.0) or 0.0),
        reverse=True,
    )
    print("\nTop active clusters")
    for cluster in ranked_clusters[:limit]:
        weight = float(cluster.get("effective_weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        contributors = [str(x) for x in cluster.get("contributing_features", [])]
        print(
            f"  {cluster.get('cluster_id')}: weight={weight:.3f}, "
            f"state={cluster.get('health_state')}, "
            f"features={_format_feature_list(contributors, 3)}"
        )

    print("\nWrote:")
    print(f"  {output_dir / 'feature_health_profile.json'}")
    print(f"  {output_dir / 'feature_health_profile.csv'}")


def main() -> None:
    """CLI entry point for refreshing and printing the feature health profile."""
    parser = argparse.ArgumentParser(
        description=(
            "Refresh feature-health clustering/quarantine outputs and print "
            "a short gate summary."
        )
    )
    parser.add_argument(
        "--max-specs",
        type=int,
        default=48,
        help="Maximum feature specs to load from the factor shortlist.",
    )
    parser.add_argument(
        "--output-dir",
        default=SIGNAL_DIR,
        help="Directory for feature_health_profile.json/csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum features/clusters to print in each summary section.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing profile files.",
    )
    args = parser.parse_args()

    # PLAIN ENGLISH: Use the same feature loader as the backtest/live strategy
    # so this command audits the exact feature set the strategy would score.
    from alpha_factor_backtest import load_feature_specs

    specs = load_feature_specs(max_specs=int(args.max_specs))
    if not specs:
        raise SystemExit(
            "No feature specs found. Run feature research first or check logs/feature_ic_shortlist.csv."
        )
    output_dir = Path(args.output_dir)
    profile = build_feature_health_profile(
        [str(spec.get("feature", "")) for spec in specs],
        output_dir=output_dir,
        write_outputs=not bool(args.no_write),
    )
    _print_profile_summary(profile, output_dir=output_dir, limit=max(1, int(args.limit)))


if __name__ == "__main__":
    main()
