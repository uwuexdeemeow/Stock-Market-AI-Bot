from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "Stock picking scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core_satellite_alpha
import predict
import sentiment_engine


class _StaticEngine:
    def __init__(self, scores):
        self._scores = list(scores)

    def score_batch(self, texts):
        return self._scores[: len(texts)]


def _provider(name, *, primary=False, headlines=None, ok=True):
    headlines = headlines or []
    return {
        "name": name,
        "provider_type": "finnhub_api" if primary else "rss",
        "is_primary": primary,
        "ok": ok,
        "error": "",
        "fresh_count": len(headlines),
        "headlines": headlines,
    }


def test_score_todays_news_fallback_only_marks_partial_coverage(monkeypatch):
    now = datetime.now(timezone.utc)

    monkeypatch.setitem(
        sentiment_engine.score_todays_news.__globals__,
        "fetch_premarket_news_by_provider",
        lambda *args, **kwargs: [
            _provider("yahoo_finance", headlines=[(now, "AAPL product update")]),
            _provider("finnhub", primary=True, headlines=[], ok=False),
        ],
    )

    out = sentiment_engine.score_todays_news("AAPL", _StaticEngine([0.0]))

    assert out["headline_count"] == 1
    assert out["primary_available"] is False
    assert out["fallback_available"] is True
    assert out["composite_score"] == 0.0

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


def test_core_satellite_veto_uses_negative_fallback_sentiment(monkeypatch):
    def fake_score_todays_news(ticker, engine, verbose=False):
        return {
            "composite_score": -0.4 if ticker == "BAD" else 0.0,
            "fallback_available": True,
            "primary_available": False,
        }

    monkeypatch.setattr(sentiment_engine, "score_todays_news", fake_score_todays_news)

    scores = core_satellite_alpha._fetch_live_sentiment(["BAD", "OK"])

    assert scores["BAD"] < core_satellite_alpha.SENTIMENT_VETO_THRESHOLD
    assert scores["OK"] == 0.0
