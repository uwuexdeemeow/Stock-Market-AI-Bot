from __future__ import annotations

import json

import pandas as pd

from feature_health import build_feature_health_profile, canonical_feature_root


def test_canonical_roots_collapse_rank_and_benchmark_return_variants():
    assert canonical_feature_root("xs_rank_market_factor_liquidity_dollar_vol_20d") == "factor_liquidity_dollar_vol_20d"
    assert canonical_feature_root("xs_rank_sector_ret_5d") == "ret_5d"
    assert canonical_feature_root("ret_vs_spy_5d") == "ret_5d"
    assert canonical_feature_root("ret_vs_qqq_5d") == "ret_5d"


def test_feature_health_clusters_and_quarantines_decaying_features(tmp_path):
    features = [
        "factor_liquidity_dollar_vol_20d",
        "xs_rank_market_factor_liquidity_dollar_vol_20d",
        "factor_illiquidity_amihud_20d",
        "ret_5d",
        "ret_vs_spy_5d",
        "ret_vs_qqq_5d",
        "xs_rank_market_ret_5d",
        "dist_ma5",
        "ret_3d",
        "dist_ma10",
        "factor_mom_12_1",
        "vol_trend_20_60",
    ]
    quality_path = tmp_path / "feature_quality_report.json"
    quality_path.write_text(json.dumps({
        "correlation_clusters": [
            {
                "feature_1": "factor_liquidity_dollar_vol_20d",
                "feature_2": "xs_rank_market_factor_liquidity_dollar_vol_20d",
                "correlation": 1.0,
            },
            {
                "feature_1": "factor_liquidity_dollar_vol_20d",
                "feature_2": "factor_illiquidity_amihud_20d",
                "correlation": -0.93,
            },
            {"feature_1": "dist_ma5", "feature_2": "ret_3d", "correlation": 0.93},
            {"feature_1": "dist_ma5", "feature_2": "dist_ma10", "correlation": 0.81},
            {"feature_1": "dist_ma10", "feature_2": "ret_5d", "correlation": 0.89},
        ],
    }))
    rows = []
    for feature in features:
        ratio = 1.0
        full_ic = 0.01
        recent_ic = 0.01
        if feature in {"dist_ma5", "ret_3d", "ret_5d", "dist_ma10", "ret_vs_spy_5d", "ret_vs_qqq_5d", "xs_rank_market_ret_5d"}:
            ratio = 0.40
            recent_ic = 0.004
        rows.append({
            "feature": feature,
            "recent_vs_full_trend": ratio,
            "recent_ic": recent_ic,
            "full_ic": full_ic,
        })
    research_path = tmp_path / "feature_research_summary.csv"
    pd.DataFrame(rows).to_csv(research_path, index=False)

    profile = build_feature_health_profile(
        features,
        quality_report_path=quality_path,
        research_summary_path=research_path,
        output_dir=tmp_path,
    )
    by_feature = {row["feature"]: row for row in profile["features"]}

    liquidity_cluster = by_feature["factor_liquidity_dollar_vol_20d"]["cluster_id"]
    assert by_feature["xs_rank_market_factor_liquidity_dollar_vol_20d"]["cluster_id"] == liquidity_cluster
    assert by_feature["factor_illiquidity_amihud_20d"]["cluster_id"] == liquidity_cluster

    ret_cluster = by_feature["ret_5d"]["cluster_id"]
    assert by_feature["ret_vs_spy_5d"]["cluster_id"] == ret_cluster
    assert by_feature["ret_vs_qqq_5d"]["cluster_id"] == ret_cluster
    assert by_feature["xs_rank_market_ret_5d"]["cluster_id"] == ret_cluster

    for feature in ("dist_ma5", "ret_3d", "ret_5d", "dist_ma10", "ret_vs_spy_5d", "ret_vs_qqq_5d"):
        assert by_feature[feature]["health_state"] == "quarantined"
        assert by_feature[feature]["contributes_to_score"] is False

    assert (tmp_path / "feature_health_profile.json").exists()
    assert (tmp_path / "feature_health_profile.csv").exists()
