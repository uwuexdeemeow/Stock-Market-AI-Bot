from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline_shared


def _price_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=80)
    close = pd.Series(range(100, 180), index=idx, dtype=float)
    return pd.DataFrame({
        "Open": close,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": 1_000_000,
    }, index=idx)


def _empty_frame_for_dates(*args, **kwargs) -> pd.DataFrame:
    dates = args[1] if len(args) > 1 else kwargs.get("dates")
    if dates is None:
        dates = pd.bdate_range("2026-01-01", periods=80)
    return pd.DataFrame(index=dates)


def _patch_live_feature_dependencies(monkeypatch):
    monkeypatch.setattr(pipeline_shared, "fetch_price_data", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(pipeline_shared, "build_multi_timeframe", lambda close, dates: pd.DataFrame(index=dates))
    monkeypatch.setattr(pipeline_shared, "build_vix_features", lambda dates, *args, **kwargs: pd.DataFrame(index=dates))
    monkeypatch.setattr(pipeline_shared, "build_multi_market", lambda ticker, dates, *args, **kwargs: pd.DataFrame(index=dates))
    monkeypatch.setattr(pipeline_shared, "build_gap_features", lambda df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(pipeline_shared, "build_volume_features", lambda df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(pipeline_shared, "build_point_in_time_valuation_features", lambda ticker, df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(pipeline_shared, "build_market_breadth_features", lambda dates, *args, **kwargs: pd.DataFrame(index=dates))
    monkeypatch.setattr(pipeline_shared, "build_earnings_features_context", lambda ticker, dates: pd.DataFrame(index=dates))
    monkeypatch.setattr(pipeline_shared, "build_literature_factor_features", lambda df: pd.DataFrame(index=df.index))
    monkeypatch.setattr(pipeline_shared, "apply_live_fundamental_sector_zscores", lambda ticker, frame: frame)
    monkeypatch.setattr(pipeline_shared, "build_sector_strength_features", lambda ticker, frame: pd.DataFrame(index=frame.index))
    monkeypatch.setattr(pipeline_shared, "build_options_features_context", lambda ticker, dates, live=False: pd.DataFrame(index=dates))


def test_social_safety_marks_fallback_without_model_social_columns(monkeypatch):
    _patch_live_feature_dependencies(monkeypatch)
    monkeypatch.setattr(pipeline_shared, "USE_NEWS_SENTIMENT", False)
    monkeypatch.setattr(pipeline_shared, "SOCIAL_SENTIMENT_ALPHA_ENABLED", False)
    monkeypatch.setattr(pipeline_shared, "SOCIAL_SENTIMENT_SAFETY_ENABLED", True)
    monkeypatch.setattr(
        pipeline_shared,
        "get_live_social_signal",
        lambda ticker: {
            "social_available": True,
            "combined": 0.42,
            "message_volume": 0.0,
            "x_tweets": 7,
        },
    )
    monkeypatch.setattr(
        pipeline_shared,
        "build_social_sentiment_features",
        lambda *args, **kwargs: pytest.fail("social alpha features should not be built in safety mode"),
    )

    out = pipeline_shared.build_live_features_with_latest_news("AAPL", feature_cols=[])

    assert out is not None and not out.empty
    assert float(out["sentiment_fallback_available"].iloc[-1]) == 1.0
    assert float(out["sentiment_fresh_headline_count"].iloc[-1]) == 7.0
    assert not any(c.startswith("social_") or c.startswith("x_") for c in out.columns)


def test_conservative_filter_intentionally_strips_social_columns():
    frame = pd.DataFrame({
        "Close": [100.0, 101.0],
        "social_combined": [0.2, 0.3],
        "social_message_volume": [1.0, 2.0],
        "x_mention_count": [3.0, 4.0],
        "x_sentiment_score": [0.1, 0.2],
        "sent_z_news_sentiment": [0.0, 0.1],
    })

    out = pipeline_shared.keep_conservative_feature_set(frame)

    assert "Close" in out.columns
    assert "sent_z_news_sentiment" in out.columns
    assert not any(c.startswith("social_") or c.startswith("x_") for c in out.columns)


def test_legacy_social_enabled_flag_does_not_inject_alpha(monkeypatch):
    _patch_live_feature_dependencies(monkeypatch)
    monkeypatch.setattr(pipeline_shared, "USE_NEWS_SENTIMENT", False)
    monkeypatch.setattr(pipeline_shared, "SOCIAL_SENTIMENT_ENABLED", True)
    monkeypatch.setattr(pipeline_shared, "SOCIAL_SENTIMENT_ALPHA_ENABLED", False)
    monkeypatch.setattr(pipeline_shared, "SOCIAL_SENTIMENT_SAFETY_ENABLED", False)
    monkeypatch.setattr(
        pipeline_shared,
        "build_social_sentiment_features",
        lambda *args, **kwargs: pytest.fail("legacy flag must not inject social alpha features"),
    )

    out = pipeline_shared.build_live_features_with_latest_news("AAPL", feature_cols=[])

    assert out is not None and not out.empty
    assert not any(c.startswith("social_") or c.startswith("x_") for c in out.columns)


@pytest.mark.skip(reason="Scaffold for future social-alpha validation, not enabled by default.")
def test_future_social_alpha_validation_gate():
    """
    Before SOCIAL_SENTIMENT_ALPHA_ENABLED can be enabled, require:
    nonzero live/historical coverage, no leakage in strict audit, positive IC,
    and walk-forward improvement versus the no-social baseline.
    """
    raise AssertionError("Define and pass social-alpha validation before enabling.")
