from __future__ import annotations

import os
import sys
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core_satellite_alpha
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
    # (The matching predict.evaluate_sentiment_health check moved to
    # tests/legacy/test_predict_sentiment_health.py with predict.py.)


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
