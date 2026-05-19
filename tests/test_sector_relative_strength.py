from __future__ import annotations

import numpy as np
import pandas as pd

import cross_sectional_features as xs
from feature_health import canonical_feature_root
from fundamental_features import build_sector_strength_features
from pipeline_shared import add_technical_features


def _price_frame(periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    stock_close = pd.Series(np.linspace(100.0, 220.0, periods), index=dates)
    sector_close = pd.Series(np.linspace(100.0, 150.0, periods), index=dates)
    df = pd.DataFrame(
        {
            "Open": stock_close,
            "High": stock_close * 1.01,
            "Low": stock_close * 0.99,
            "Close": stock_close,
            "Volume": 1_000_000,
            "sector_close": sector_close,
            "spy_ret5d": 0.0,
        },
        index=dates,
    )
    df = add_technical_features(df)
    for horizon in (1, 5, 20, 60, 120):
        df[f"sector_ret{horizon}d"] = sector_close.pct_change(horizon).fillna(0.0)
    return df


def test_sector_relative_strength_positive_for_stock_outperforming_sector():
    out = build_sector_strength_features("AAA", _price_frame())
    latest = out.iloc[-1]

    assert latest["sector_rel_return_20d"] > 0
    assert latest["sector_rel_return_60d"] > 0
    assert latest["sector_rel_return_120d"] > 0
    assert latest["sector_rel_mom_20d"] > 0
    assert latest["sector_rel_mom_60d"] > 0
    assert latest["sector_rel_trend_strength"] > 0


def test_sector_relative_strength_is_point_in_time():
    base = _price_frame()
    changed_future = base.copy()
    target_date = base.index[80]
    future_dates = base.index[100:]
    changed_future.loc[future_dates, "Close"] *= 4.0
    changed_future.loc[future_dates, "sector_close"] *= 0.25
    for horizon in (1, 5, 20, 60, 120):
        changed_future[f"ret_{horizon}d"] = changed_future["Close"].pct_change(horizon)
        changed_future[f"sector_ret{horizon}d"] = changed_future["sector_close"].pct_change(horizon).fillna(0.0)

    before = build_sector_strength_features("AAA", base)
    after = build_sector_strength_features("AAA", changed_future)

    cols = [
        "sector_rel_return_20d",
        "sector_rel_return_60d",
        "sector_rel_mom_20d",
        "sector_rel_mom_60d",
        "sector_rel_trend_strength",
    ]
    pd.testing.assert_series_equal(before.loc[target_date, cols], after.loc[target_date, cols])


def test_sector_strength_missing_sector_inputs_returns_neutral_values():
    dates = pd.bdate_range("2026-01-02", periods=5)
    out = build_sector_strength_features(
        "AAA",
        pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=dates),
    )
    assert set(
        [
            "sector_rel_return_20d",
            "sector_rel_return_60d",
            "sector_rel_return_120d",
            "sector_rel_mom_20d",
            "sector_rel_mom_60d",
            "sector_rel_trend_strength",
        ]
    ).issubset(out.columns)
    assert float(out.abs().sum().sum()) == 0.0


def test_cross_sectional_rank_adds_clean_sector_relative_rank_names(tmp_path):
    dates = pd.bdate_range("2026-01-02", periods=2)
    tickers = [f"T{i}" for i in range(6)]
    for i, ticker in enumerate(tickers):
        pd.DataFrame(
            {"sector_rel_return_20d": [float(i), float(i + 1)]},
            index=dates,
        ).to_parquet(tmp_path / f"{ticker}.parquet")

    summary = xs.apply_cross_sectional_rank_features(
        tickers,
        data_dir=str(tmp_path),
        sector_map={ticker: "XLK" for ticker in tickers},
        source_cols=["sector_rel_return_20d"],
    )

    assert summary["updated"] == 6
    assert "xs_rank_sector_rel_return_20d" in summary["new_cols"]
    assert "xs_rank_sector_sector_rel_return_20d" not in summary["new_cols"]

    low = pd.read_parquet(tmp_path / "T0.parquet")
    high = pd.read_parquet(tmp_path / "T5.parquet")
    assert high["xs_rank_sector_rel_return_20d"].iloc[0] > low["xs_rank_sector_rel_return_20d"].iloc[0]
    assert high["xs_rank_market_sector_rel_return_20d"].iloc[0] > low["xs_rank_market_sector_rel_return_20d"].iloc[0]


def test_sector_relative_rank_alias_clusters_with_raw_feature():
    assert canonical_feature_root("xs_rank_sector_rel_return_20d") == "sector_rel_return_20d"
