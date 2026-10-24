"""Legacy predict.py sentiment-health tests, split out of tests/test_sentiment_fallback.py."""
from __future__ import annotations

import pandas as pd

import predict


def test_sentiment_health_partial_when_only_fallback_has_fresh_news():
    df = pd.DataFrame({
        "sent_news_sentiment": [0, 0, 0, 0, 0],
        "sentiment_primary_available": [0, 0, 0, 0, 0],
        "sentiment_fallback_available": [0, 0, 0, 0, 1],
        "sentiment_fresh_headline_count": [0, 0, 0, 0, 1],
    })

    health, _, _ = predict.evaluate_sentiment_health(df)

    assert health == "PARTIAL"


def test_sentiment_health_degraded_when_all_providers_unavailable():
    df = pd.DataFrame({
        "sent_news_sentiment": [0, 0, 0, 0, 0],
        "sentiment_primary_available": [0, 0, 0, 0, 0],
        "sentiment_fallback_available": [0, 0, 0, 0, 0],
        "sentiment_fresh_headline_count": [0, 0, 0, 0, 0],
    })

    health, _, _ = predict.evaluate_sentiment_health(df)

    assert health == "DEGRADED"


def test_sentiment_health_full_when_primary_has_neutral_fresh_news():
    df = pd.DataFrame({
        "sent_news_sentiment": [0, 0, 0, 0, 0],
        "sentiment_primary_available": [0, 0, 0, 0, 1],
        "sentiment_fallback_available": [0, 0, 0, 0, 0],
        "sentiment_fresh_headline_count": [0, 0, 0, 0, 1],
    })

    health, _, _ = predict.evaluate_sentiment_health(df)

    assert health == "FULL"
