"""
sentiment_engine.py — Drop-in Sentiment Scorer
================================================
Replaces VADER in 02_research_v4.py with a better financial scorer.

THREE LEVELS — use whichever fits your setup:

  Level 1: FinVADER  (pip install finvader)
    Same speed as VADER. Adds 7,300 financial terms + earnings vocabulary.
    Accuracy on financial text: ~65% (vs VADER's ~56%).
    Best for: quick upgrade, no GPU needed, runs fast.

  Level 2: FinBERT   (pip install transformers torch)
    Deep learning model pre-trained on financial news and filings.
    Accuracy on financial text: ~89–91%.
    Best for: highest quality sentiment signal, runs on CPU but slow.
    Recommended: run with GPU for reasonable speed (~0.05s per headline).

  Level 3: GPT-4 via API (requires OpenAI key)
    Best possible accuracy, understands full context and sarcasm.
    Cost: ~$0.002 per 1000 headlines (very cheap).
    Best for: production-grade pipeline with budget for API calls.

HOW TO SWAP IN:
  In 02_research_v4.py, replace:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(title)["compound"]

  With:
    from sentiment_engine import SentimentEngine
    engine = SentimentEngine(level="finbert")   # or "finvader" or "gpt4"
    score = engine.score(title)

  That's it. Everything else stays the same.

WHY VADER UNDERPERFORMS ON FINANCIAL TEXT:
  VADER was designed for Twitter and product reviews. It works by looking
  up each word in a sentiment dictionary and adding up the scores.
  This fails on financial text because:

  1. Domain-specific words score wrong:
       "beats" → neutral (VADER doesn't know this means earnings beat)
       "liability" → slightly negative (actually neutral in finance)
       "raised guidance" → VADER misses that this is strongly positive

  2. No context understanding:
       "Unemployment near historic lows" → VADER scores neutral or negative
       (because "unemployment" is a negative word in its dictionary)
       FinBERT understands this is a positive economic signal.

  3. Compound phrases ignored:
       "revenue miss but raised guidance" → complex mixed signal
       VADER: sums individual word scores → confused result
       FinBERT: reads the whole sentence as context → correct score

ACCURACY BENCHMARKS (from academic literature):
  VADER on financial news:     ~56% accuracy
  FinVADER on financial news:  ~65% accuracy
  FinBERT on financial news:   ~89–91% accuracy
  GPT-4 on financial news:     ~93%+ accuracy (best available)
"""

import json
import re
import time
import logging
from collections import defaultdict
from functools import lru_cache

log = logging.getLogger("sentiment_engine")


def _parse_llm_sentiment_number(raw: str) -> float | None:
    """Extract a value in [-1, 1] from model text (handles extra words)."""
    if not raw:
        return None
    s = raw.strip().replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", s, re.I)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return max(-1.0, min(1.0, v))
    except ValueError:
        return None


class SentimentEngine:
    """
    Unified sentiment scoring interface.
    Automatically loads the right model based on `level`.
    Falls back to the next level down if a model fails to load.

    Usage:
        engine = SentimentEngine(level="finbert")
        score  = engine.score("Apple beats Q3 earnings by 15%")
        # Returns float between -1.0 (very negative) and +1.0 (very positive)

        scores = engine.score_batch(["headline 1", "headline 2", ...])
        # Returns list of floats — much faster than calling score() in a loop
    """

    def __init__(self, level: str = "finvader"):
        """
        level: "finvader"  — fast, good (recommended default)
               "finbert"   — slow, best accuracy (recommended if GPU available)
               "gpt4"      — best accuracy, costs money per call
               "vader"     — original VADER (fallback only)
        """
        self.level   = level
        self.model   = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model for the specified level with graceful fallback."""
        if self.level == "finvader":
            self._load_finvader()
        elif self.level == "finbert":
            self._load_finbert()
        elif self.level == "gpt4":
            self._load_gpt4()
        else:
            self._load_vader_fallback()

    def _load_finvader(self):
        """
        FinVADER: VADER extended with two financial lexicons.
          SentiBignomics: 7,300 financial-domain terms with polarity scores
          Henry's lexicon: 189 terms specific to earnings press releases

        Install: pip install finvader
        Speed: same as VADER (~0.001s per headline)
        Accuracy: ~65% on financial text (vs VADER's ~56%)
        """
        try:
            from finvader import finvader
            self._finvader_fn = finvader
            self.model        = "finvader"
            log.info("Sentiment engine: FinVADER loaded (fast, financial-aware)")
        except ImportError:
            log.warning("FinVADER not installed (pip install finvader). Falling back to VADER.")
            self._load_vader_fallback()

    def _load_finbert(self):
        """
        FinBERT: BERT model pre-trained on financial news, earnings calls, and filings.
        HuggingFace model: ProsusAI/finbert

        Install: pip install transformers torch
        Speed: ~0.05s/headline on GPU, ~0.5s on CPU
        Accuracy: ~89-91% on financial text

        FinBERT classifies text as POSITIVE, NEGATIVE, or NEUTRAL and returns
        a confidence score. We map this to a compound score in [-1, +1]:
          POSITIVE with confidence 0.9  →  +0.9
          NEGATIVE with confidence 0.8  →  -0.8
          NEUTRAL                       →   0.0 (or small score based on confidence)
        """
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch

            model_name = "ProsusAI/finbert"
            log.info(f"Loading FinBERT ({model_name}) — first run downloads ~440MB...")

            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model     = BertForSequenceClassification.from_pretrained(model_name)
            self.model.eval()

            # Use GPU if available — makes FinBERT ~10x faster
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model   = self.model.to(self._device)

            # FinBERT label mapping
            # The model outputs logits for [positive, negative, neutral]
            self._label_map = {0: 1.0, 1: -1.0, 2: 0.0}  # positive, negative, neutral

            import torch as _torch
            self._torch = _torch

            self.level = "finbert"
            log.info(f"Sentiment engine: FinBERT loaded on {self._device}")
            log.info("Accuracy on financial text: ~89-91% (vs VADER ~56%)")

        except ImportError:
            log.warning("transformers not installed (pip install transformers). Falling back to FinVADER.")
            self._load_finvader()
        except Exception as e:
            log.warning(f"FinBERT load failed: {e}. Falling back to FinVADER.")
            self._load_finvader()

    def _load_gpt4(self):
        """
        GPT-4 via OpenAI API — best accuracy (~93%+), costs ~$0.002/1000 headlines.

        Requires: pip install openai
        Requires: OPENAI_API_KEY environment variable set

        We use a carefully crafted prompt that instructs GPT-4 to:
          1. Focus on the financial implications, not surface sentiment
          2. Return a score in [-1, +1] with reasoning
          3. Handle complex cases: "miss but raised guidance", "beat but stock down"
        """
        try:
            import openai
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                log.warning("OPENAI_API_KEY not set. Falling back to FinBERT.")
                self._load_finbert()
                return

            self._openai_client = openai.OpenAI(api_key=api_key)
            self.model = "gpt4"
            log.info("Sentiment engine: GPT-4 loaded (~$0.002 per 1000 headlines)")

        except ImportError:
            log.warning("openai not installed (pip install openai). Falling back to FinBERT.")
            self._load_finbert()

    def _load_vader_fallback(self):
        """Original VADER — fallback of last resort."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()
        self.model  = "vader"
        log.warning("Sentiment engine: Using original VADER (lowest accuracy on financial text)")

    def score(self, text: str) -> float:
        """
        Score a single headline. Returns float in [-1.0, +1.0].
          +1.0 = very positive (e.g. "Massive earnings beat, guidance raised")
          0.0  = neutral     (e.g. "Apple to release earnings next week")
         -1.0 = very negative (e.g. "SEC investigation launched into accounting fraud")
        """
        if not text or not isinstance(text, str):
            return 0.0

        try:
            if self.model == "finvader":
                return self._score_finvader(text)
            elif self.model == "finbert":
                return self._score_finbert(text)
            elif self.model == "gpt4":
                return self._score_gpt4(text)
            else:
                return self._score_vader(text)
        except Exception as e:
            log.debug(f"Scoring error for '{text[:50]}': {e}")
            return 0.0

    def score_batch(self, texts: list) -> list:
        """
        Score a list of headlines efficiently.

        For FinBERT: processes in batches of 32 (much faster than one-by-one).
        For others: applies score() to each item.

        Returns list of floats, same length as input.
        """
        if not texts:
            return []

        if self.model == "finbert":
            return self._score_finbert_batch(texts, batch_size=32)

        return [self.score(t) for t in texts]

    def _score_finvader(self, text: str) -> float:
        """
        FinVADER scoring.
        use_sentibignomics=True: activates the 7,300-term financial lexicon
        use_henry=True: activates the earnings-specific 189-term lexicon
        """
        return float(self._finvader_fn(
            text,
            use_sentibignomics=True,
            use_henry=True,
            indicator="compound"
        ))

    def _score_finbert(self, text: str) -> float:
        """FinBERT scoring for a single headline."""
        import torch
        with torch.no_grad():
            inputs  = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs  = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # probs: [P(positive), P(negative), P(neutral)]
        # Compound score: P(positive) - P(negative), scaled by max confidence
        compound = float(probs[0] - probs[1])

        # If model is very confident it's neutral, return near-zero
        if probs[2] > 0.6:
            compound *= (1 - probs[2])

        return compound

    def _score_finbert_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        FinBERT batch scoring — much faster than one-by-one for large lists.

        Processes `batch_size` headlines at once through the model.
        Typically 10-20x faster than individual calls.
        """
        import torch
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Filter empty strings
            batch = [t if t else " " for t in batch]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs  = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for p in probs:
                compound = float(p[0] - p[1])
                if p[2] > 0.6:
                    compound *= (1 - p[2])
                results.append(compound)

        return results

    def _score_gpt4(self, text: str) -> float:
        """
        GPT scoring with strict prompt, regex parse, and one retry on bad output.
        """
        prompt = f"""You are a financial sentiment analyst. Rate the financial market sentiment of this headline on a scale from -1.0 to +1.0 where:
  +1.0 = very positive for the stock price (e.g. huge earnings beat, major contract won)
   0.0 = neutral or unclear market impact
  -1.0 = very negative for the stock price (e.g. fraud, major miss, bankruptcy)

Consider: earnings beats/misses, guidance changes, regulatory news, macro impacts.
Reply with exactly one number between -1.0 and 1.0 and nothing else (no words).

Headline: {text}"""

        strict = (
            'Output format: a single JSON object only, like {{"score": -0.35}} '
            "where score is a number from -1.0 to 1.0."
        )

        for attempt in range(2):
            try:
                user_content = prompt if attempt == 0 else f"{prompt}\n\n{strict}"
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=32,
                    temperature=0,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = _parse_llm_sentiment_number(raw)
                if parsed is not None:
                    return parsed
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "score" in obj:
                        v = float(obj["score"])
                        return max(-1.0, min(1.0, v))
                except Exception:
                    pass
            except Exception as e:
                log.debug(f"GPT scoring attempt {attempt + 1} failed: {e}")
        return 0.0

    def _score_vader(self, text: str) -> float:
        """Original VADER fallback."""
        return float(self._vader.polarity_scores(text)["compound"])

    def get_info(self) -> dict:
        """Return info about the currently loaded model."""
        accuracy_map = {
            "finvader": "~65% on financial text",
            "finbert" : "~89-91% on financial text",
            "gpt4"    : "~93%+ on financial text",
            "vader"   : "~56% on financial text (not recommended)",
        }
        return {
            "model"    : self.model,
            "accuracy" : accuracy_map.get(self.model, "unknown"),
            "device"   : str(getattr(self, "_device", "cpu")),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED NEWS → DAILY SENTIMENT (research + live predict)
# Same logic as 02_research_v4: per-source columns, weighted blend, rollups.
# Generic RSS feeds (CNBC, MarketWatch, BI) filter headlines with
# headline_mentions_ticker() so FinBERT does not score unrelated stories.
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_RSS_SOURCES = [
    {
        "name": "yahoo_finance",
        "weight": 1.1,
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "filter_headlines": False,
    },
    {
        "name": "cnbc",
        "weight": 1.0,
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "filter_headlines": True,
    },
    {
        "name": "marketwatch",
        "weight": 1.0,
        "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
        "filter_headlines": True,
    },
    {
        "name": "google_news",
        "weight": 0.8,
        "url": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        "filter_headlines": False,
    },
    {
        "name": "business_insider",
        "weight": 0.7,
        "url": "https://feeds.businessinsider.com/custom/all",
        "filter_headlines": True,
    },
]


def _rss_http_headers() -> dict:
    """Browser-like User-Agent so RSS/CDN endpoints return feeds (not empty)."""
    try:
        from config_v2 import HEADERS as h
        return dict(h)
    except ImportError:
        try:
            from config import HEADERS as h
            return dict(h)
        except ImportError:
            pass
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }


def _normalize_trading_index(dates) -> "pd.DatetimeIndex":
    """Sorted, timezone-naive midnight index matching the price DataFrame."""
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=False).normalize())
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    return idx


def _align_pub_to_trading_day(pub, trading_idx: "pd.DatetimeIndex"):
    """
    Map a headline's calendar time to a trading row in `dates`.

    Price data has only **trading days**. Headlines often fall on nights/weekends.
    Without this step, `reindex(dates)` yields all zeros because Saturday's
    Timestamp never appears in the index.

    Rule: use the **first trading session on or after** the headline date
    (US/Eastern calendar for UTC timestamps from Finnhub).
    """
    import pandas as pd

    if trading_idx is None or len(trading_idx) == 0:
        return None

    p = pd.Timestamp(pub)
    if p.tzinfo is not None:
        try:
            p = p.tz_convert("America/New_York").tz_localize(None)
        except Exception:
            p = p.tz_localize(None)
    p = p.normalize()

    if p > trading_idx[-1]:
        return None
    if p < trading_idx[0]:
        return trading_idx[0]

    pos = trading_idx.searchsorted(p, side="left")
    if pos >= len(trading_idx):
        return None
    return trading_idx[pos]


def headline_mentions_ticker(ticker: str, title: str) -> bool:
    """
    True if the headline likely refers to this symbol.
    Used to drop noise from broad market RSS feeds.
    """
    if not ticker or not title:
        return False
    t = ticker.upper().strip()
    u = title.upper()
    if t in u:
        return True
    if "-" in t:
        base = t.split("-")[0]
        if len(base) >= 2 and base in u:
            return True
        if "BTC" in t and ("BITCOIN" in u or "BTC" in u):
            return True
        if "ETH" in t and ("ETHEREUM" in u or "ETH" in u):
            return True
    if t == "GOOGL" and ("GOOGLE" in u or "ALPHABET" in u or "GOOG" in u):
        return True
    if t == "GOOG" and ("GOOGLE" in u or "ALPHABET" in u or "GOOGL" in u):
        return True
    return False


def _fetch_rss_headlines(url: str, ticker: str) -> list:
    """Return list of (published_date, title). Uses HTTP GET + headers so feeds are not empty."""
    import feedparser
    import pandas as pd

    filled = url.format(ticker=ticker, ticker_lower=ticker.lower())
    try:
        try:
            import requests

            resp = requests.get(
                filled, headers=_rss_http_headers(), timeout=30
            )
            resp.raise_for_status()
            raw = resp.content
        except Exception:
            raw = None

        feed = feedparser.parse(raw if raw is not None else filled)
        out = []
        for e in feed.entries[:80]:
            try:
                raw_t = e.get("published", e.get("updated", ""))
                pub = pd.to_datetime(raw_t, errors="coerce")
                if pd.isna(pub):
                    continue
                if pub.tzinfo is not None:
                    pub = (
                        pub.tz_convert("America/New_York")
                        .tz_localize(None)
                        .normalize()
                    )
                else:
                    pub = pub.normalize()
                title = e.get("title", e.get("summary", "")).strip()
                if title:
                    out.append((pub, title))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _fetch_finnhub_headlines(
    ticker: str, client, dates: "pd.DatetimeIndex",
) -> list:
    """Company news for [min(dates), max(dates)] (capped at today)."""
    import pandas as pd

    if client is None:
        return []
    try:
        dmin = pd.Timestamp(dates.min()).normalize()
        dmax = pd.Timestamp(dates.max()).normalize()
        today = pd.Timestamp.today().normalize()
        dmax = min(dmax, today)
        start_d = dmin.strftime("%Y-%m-%d")
        end_d = dmax.strftime("%Y-%m-%d")
        news = client.company_news(ticker, _from=start_d, to=end_d)
        time.sleep(0.4)
        out = []
        for item in news[:200]:
            try:
                ts_utc = pd.to_datetime(
                    item.get("datetime", 0), unit="s", utc=True
                )
                ts = (
                    ts_utc.tz_convert("America/New_York")
                    .tz_localize(None)
                    .normalize()
                )
                title = item.get("headline", "").strip()
                if title:
                    out.append((ts, title))
            except Exception:
                continue
        return out
    except Exception:
        return []


def build_sentiment_feature_dataframe(
    ticker: str,
    dates,
    *,
    finnhub_client=None,
    engine_level: str = "finbert",
    sleep_rss: float = 0.25,
    logger=None,
) -> "pd.DataFrame":
    """
    Build the same sentiment columns as 02_research_v4 build_sentiment_features.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading days to align (index of price DataFrame).
    finnhub_client : optional Finnhub client (or None).
    """
    import numpy as np
    import pandas as pd

    lg = logger or log
    engine = SentimentEngine(level=engine_level)
    sources_data: dict[str, list] = {}
    weights: dict[str, float] = {}

    if finnhub_client is not None:
        fh_list = _fetch_finnhub_headlines(ticker, finnhub_client, dates)
        sources_data["finnhub"] = fh_list
        weights["finnhub"] = 1.3
        lg.info(f"    finnhub: {len(fh_list)} headlines")

    for src in SENTIMENT_RSS_SOURCES:
        name = src["name"]
        headlines = _fetch_rss_headlines(src["url"], ticker)
        if src.get("filter_headlines"):
            headlines = [
                (d, t)
                for d, t in headlines
                if headline_mentions_ticker(ticker, t)
            ]
        sources_data[name] = headlines
        weights[name] = src["weight"]
        lg.info(f"    {name}: {len(headlines)} headlines (after filter)")
        if sleep_rss > 0:
            time.sleep(sleep_rss)

    # Flatten for batch scoring (preserves order for grouping)
    meta: list[tuple[str, object]] = []
    texts: list[str] = []
    for name, headlines in sources_data.items():
        for pub_date, title in headlines:
            meta.append((name, pub_date))
            texts.append(title if title else " ")

    scores_list: list[float] = []
    if texts:
        scores_list = engine.score_batch(texts)

    total_headlines = sum(len(h) for h in sources_data.values())
    trading_idx = _normalize_trading_index(dates)
    date_axis = pd.DatetimeIndex(pd.to_datetime(dates, utc=False).normalize())
    source_scores = {n: defaultdict(list) for n in sources_data}
    headline_counts: dict = defaultdict(int)
    dropped_align = 0

    for (name, pub_date), score in zip(meta, scores_list):
        adj = _align_pub_to_trading_day(pub_date, trading_idx)
        if adj is None:
            dropped_align += 1
            continue
        source_scores[name][adj].append(score)
        headline_counts[adj] += 1

    if dropped_align > 0:
        lg.info(
            f"  [{ticker}] Mapped headlines to trading days; "
            f"{dropped_align} outside range skipped."
        )

    result = pd.DataFrame(index=dates)
    src_cols: list[tuple[str, float]] = []

    for name in sources_data:
        col = f"sent_{name}"
        daily = {
            d: float(np.mean(v))
            for d, v in source_scores[name].items()
            if v
        }
        series = pd.Series(daily, dtype=float)
        series.index = pd.DatetimeIndex(series.index).normalize()
        series = series.reindex(date_axis).fillna(0.0)
        result[col] = series.values
        src_cols.append((col, weights[name]))

    lg.info(f"  [{ticker}] Total headlines scored: {total_headlines}")

    total_w = sum(w for _, w in src_cols)
    if total_w > 0 and src_cols:
        result["news_sentiment"] = sum(
            result[col] * w for col, w in src_cols
        ) / total_w
    else:
        result["news_sentiment"] = 0.0

    only_cols = [c for c, _ in src_cols]
    if only_cols:
        result["sentiment_disagreement"] = result[only_cols].std(axis=1).fillna(0)
    else:
        result["sentiment_disagreement"] = np.zeros(len(dates), dtype=float)

    vol_s = pd.Series(dict(headline_counts), dtype=float)
    vol_s.index = pd.DatetimeIndex(vol_s.index).normalize()
    vol_s = vol_s.reindex(date_axis).fillna(0)
    rmean = vol_s.rolling(20).mean().replace(0, 1)
    result["headline_volume"] = (vol_s / rmean).fillna(1.0)
    result["sentiment_3d"] = result["news_sentiment"].rolling(3).mean().fillna(0)
    result["sentiment_7d"] = result["news_sentiment"].rolling(7).mean().fillna(0)
    result["sentiment_delta"] = result["news_sentiment"].diff().fillna(0)
    result["sentiment_accel"] = result["sentiment_delta"].diff().fillna(0)

    nz = int((result["news_sentiment"].abs() > 1e-9).sum())
    lg.info(
        f"  [{ticker}] Trading days with non-zero news_sentiment: "
        f"{nz} / {len(dates)}"
    )
    if total_headlines == 0:
        lg.warning(
            f"  [{ticker}] No headlines from any source — check FINNHUB_API_KEY, "
            "network, or install: pip install requests feedparser"
        )
    elif nz == 0:
        lg.warning(
            f"  [{ticker}] Headlines were scored but news_sentiment is all zero — "
            "check date alignment and FinBERT/finvader install."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME NEWS FETCHER
# Fetches headlines published TODAY (pre-market) so predictions are based
# on this morning's news, not last week's stale headlines.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_premarket_news(ticker: str, max_age_hours: int = 16) -> list:
    """
    Fetch only RECENT headlines (published within the last 16 hours).

    WHY THIS MATTERS:
    The current pipeline fetches all headlines from the last 30 days.
    That means a prediction made Monday morning is weighted by news from
    two weeks ago, not this morning's pre-market earnings release.

    For the PREDICTOR (04_predict_v2.py), we only want:
      - Pre-market news (6 AM - 9:30 AM EST today)
      - After-hours news from yesterday (4 PM - 6 AM today)
      - Any breaking news in the last 16 hours

    This function filters headlines by age and returns only the recent ones.
    The sentiment from these headlines is what should drive today's prediction.

    Parameters:
      ticker        : stock symbol e.g. "AAPL"
      max_age_hours : only return headlines newer than this many hours (default 16)

    Returns:
      List of (published_datetime, headline_text) tuples, newest first.
    """
    import feedparser
    import requests
    from datetime import datetime, timedelta, timezone
    import pandas as pd

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    recent_headlines = []

    # Sources to check for real-time news
    sources = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    ]

    for url in sources:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                try:
                    pub_str = entry.get("published", entry.get("updated", ""))
                    pub     = pd.to_datetime(pub_str).to_pydatetime()

                    # Normalise timezone
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)

                    # Only keep headlines newer than cutoff
                    if pub >= cutoff:
                        title = entry.get("title", "").strip()
                        if title:
                            recent_headlines.append((pub, title))

                except Exception:
                    continue
        except Exception:
            continue

    # Deduplicate by title
    seen  = set()
    dedup = []
    for pub, title in recent_headlines:
        if title not in seen:
            seen.add(title)
            dedup.append((pub, title))

    # Sort newest first
    dedup.sort(key=lambda x: x[0], reverse=True)

    return dedup


def score_todays_news(ticker: str, engine: SentimentEngine) -> dict:
    """
    Fetch today's headlines and return a sentiment summary for the predictor.

    Returns a dict with:
      composite_score   : weighted average of recent headline scores
      headline_count    : how many headlines found in the last 16 hours
      strongest_headline: the headline with the highest absolute score
      strongest_score   : its score
      is_breaking_news  : True if > 3 headlines in the last 2 hours (unusual activity)

    This is what the predictor should use for TODAY's news signal, not the
    stale 30-day rolling average from training.
    """
    headlines = fetch_premarket_news(ticker, max_age_hours=16)

    if not headlines:
        return {
            "composite_score"   : 0.0,
            "headline_count"    : 0,
            "strongest_headline": "",
            "strongest_score"   : 0.0,
            "is_breaking_news"  : False,
        }

    texts  = [h[1] for h in headlines]
    scores = engine.score_batch(texts)

    # Weight more recent headlines higher (exponential decay)
    # A headline from 1 hour ago matters more than one from 14 hours ago
    from datetime import datetime, timezone
    now    = datetime.now(timezone.utc)
    weights = []
    for pub, _ in headlines:
        age_hours = (now - pub).total_seconds() / 3600
        weight    = max(0.1, 1.0 - (age_hours / 16))  # linear decay over 16 hours
        weights.append(weight)

    total_weight   = sum(weights) + 1e-9
    composite      = sum(s * w for s, w in zip(scores, weights)) / total_weight

    # Find the strongest single headline
    abs_scores     = [abs(s) for s in scores]
    if abs_scores:
        strongest_idx  = abs_scores.index(max(abs_scores))
        strongest_text = texts[strongest_idx]
        strongest_scr  = scores[strongest_idx]
    else:
        strongest_text = ""
        strongest_scr  = 0.0

    # Breaking news: more than 3 headlines in last 2 hours
    from datetime import timedelta
    two_hours_ago  = datetime.now(timezone.utc) - timedelta(hours=2)
    recent_2h      = sum(1 for pub, _ in headlines if pub >= two_hours_ago)
    is_breaking    = recent_2h > 3

    return {
        "composite_score"   : round(composite, 4),
        "headline_count"    : len(headlines),
        "strongest_headline": strongest_text,
        "strongest_score"   : round(strongest_scr, 4),
        "is_breaking_news"  : is_breaking,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify your sentiment engine works
# python sentiment_engine.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                         format="%(levelname)s  %(message)s")

    print("\n" + "="*60)
    print(" SENTIMENT ENGINE TEST")
    print("="*60)

    # Test headlines — the engine should score these correctly
    test_cases = [
        # (headline, expected_direction)
        ("Apple reports record quarterly earnings, beats estimates by 18%",  "positive"),
        ("SEC launches investigation into accounting irregularities",         "negative"),
        ("Company announces quarterly results next week",                     "neutral"),
        ("Revenue miss but raised full-year guidance",                        "mixed/slight positive"),
        ("Unemployment claims remain near historically low levels",           "positive"),
        ("Stock falls 12% after CEO resigns amid fraud allegations",          "negative"),
        ("Fed holds rates steady as expected",                                "neutral"),
        ("Nvidia crushes estimates, AI demand accelerates guidance raise",    "positive"),
    ]

    # Try FinVADER first (fast, always available after pip install finvader)
    print("\nTesting FinVADER (fast mode):")
    try:
        engine_fv = SentimentEngine(level="finvader")
        info = engine_fv.get_info()
        print(f"  Model: {info['model']}  |  Expected accuracy: {info['accuracy']}\n")

        print(f"  {'Score':>7}  Headline")
        print(f"  {'-'*55}")
        for headline, expected in test_cases:
            score = engine_fv.score(headline)
            print(f"  {score:>+7.3f}  {headline[:52]}...")
    except Exception as e:
        print(f"  FinVADER test failed: {e}")

    # Try FinBERT if available (slower, more accurate)
    print("\n\nTesting FinBERT (accurate mode — may take a moment to load):")
    try:
        engine_fb = SentimentEngine(level="finbert")
        info = engine_fb.get_info()
        print(f"  Model: {info['model']}  |  Device: {info['device']}  |  Accuracy: {info['accuracy']}\n")

        print(f"  {'Score':>7}  Headline")
        print(f"  {'-'*55}")
        for headline, expected in test_cases:
            score = engine_fb.score(headline)
            print(f"  {score:>+7.3f}  {headline[:52]}...")
    except Exception as e:
        print(f"  FinBERT not available: {e}")
        print("  Install with: pip install transformers torch")

    # Fetch real-time news for AAPL
    print("\n\nFetching today's AAPL headlines (last 16 hours):")
    try:
        engine = SentimentEngine(level="finvader")
        result = score_todays_news("AAPL", engine)
        print(f"  Headlines found  : {result['headline_count']}")
        print(f"  Composite score  : {result['composite_score']:+.4f}")
        print(f"  Breaking news    : {result['is_breaking_news']}")
        if result['strongest_headline']:
            print(f"  Strongest signal : {result['strongest_score']:+.4f}")
            print(f"  Headline         : {result['strongest_headline'][:70]}")
    except Exception as e:
        print(f"  Real-time fetch failed: {e}")

    print("\n" + "="*60)
    print(" To use in your pipeline:")
    print("   1. pip install finvader           (quick upgrade)")
    print("   2. pip install transformers torch  (best accuracy)")
    print("   3. from sentiment_engine import SentimentEngine")
    print("   4. engine = SentimentEngine(level='finbert')")
    print("   5. score  = engine.score(headline)")
    print("="*60 + "\n")
