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
import os
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
            # finvader imports cleanly even when NLTK's vader_lexicon is absent;
            # in that case every scoring call fails and score() silently returns
            # 0.0.  Self-test now so we can fall back to bundled vaderSentiment
            # instead of producing all-zero sentiment for every ticker.
            _ = float(self._finvader_fn(
                "Company beats earnings and raises guidance",
                use_sentibignomics=True,
                use_henry=True,
                indicator="compound",
            ))
            self.level        = "finvader"
            self.model        = "finvader"
            log.info("Sentiment engine: FinVADER loaded (fast, financial-aware)")
        except ImportError:
            log.warning("FinVADER not installed (pip install finvader). Falling back to VADER.")
            self._load_vader_fallback()
        except Exception as e:
            log.warning(f"FinVADER unavailable ({e}). Falling back to VADER.")
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
        requested_level = self.level
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()
        self.level  = "vader"
        self.model  = "vader"
        msg = "Sentiment engine: Using original VADER (lowest accuracy on financial text)"
        if requested_level == "vader":
            log.info(msg)
        else:
            log.warning(msg)

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
            if self.level == "finvader":
                return self._score_finvader(text)
            elif self.level == "finbert":
                return self._score_finbert(text)
            elif self.level == "gpt4":
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

        if self.level == "finbert":
            return self._score_finbert_batch(texts, batch_size=32)

        return [self.score(t) for t in texts]

    def score_batch_detail(self, texts: list) -> list:
        """
        Like score_batch but returns a list of dicts with separate positive /
        negative probabilities instead of just the compound score.

        Each dict has:
            compound  — P(pos) - P(neg) weighted by confidence (existing signal)
            p_pos     — raw probability of positive sentiment (0–1)
            p_neg     — raw probability of negative sentiment (0–1)

        Separating p_pos and p_neg lets the model learn asymmetric reactions:
        negative news often causes sharper moves than equivalent positive news,
        and short signals need their own dedicated feature column.

        For FinVADER / GPT (no raw probability output), p_pos and p_neg are
        derived from the compound score: if compound > 0, p_pos=compound else
        p_neg=abs(compound).  This is a simplification but preserves direction.
        """
        if not texts:
            return []

        if self.level == "finbert":
            return self._score_finbert_batch_detail(texts, batch_size=32)

        # Fallback for FinVADER / GPT: derive pos/neg from compound
        compounds = [self.score(t) for t in texts]
        return [
            {
                "compound": c,
                "p_pos": max(c, 0.0),
                "p_neg": max(-c, 0.0),
            }
            for c in compounds
        ]

    def _score_finbert_batch_detail(self, texts: list, batch_size: int = 32) -> list:
        """FinBERT batch scoring returning raw P(pos) and P(neg) per headline."""
        import torch
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i: i + batch_size]]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch, return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                probs = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
            for p in probs:
                compound = float(p[0] - p[1])
                if p[2] > 0.6:
                    compound *= (1 - p[2])
                results.append({
                    "compound": compound,
                    "p_pos": float(p[0]),
                    "p_neg": float(p[1]),
                })
        return results

    def _score_finvader(self, text: str) -> float:
        """
        FinVADER scoring.
        use_sentibignomics=True: activates the 7,300-term financial lexicon
        use_henry=True: activates the earnings-specific 189-term lexicon
        """
        try:
            return float(self._finvader_fn(
                text,
                use_sentibignomics=True,
                use_henry=True,
                indicator="compound"
            ))
        except Exception:
            if hasattr(self, "_vader"):
                return self._score_vader(text)
            raise

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
            "model"    : self.level,
            "accuracy" : accuracy_map.get(self.level, "unknown"),
            "device"   : str(getattr(self, "_device", "cpu")),
        }


@lru_cache(maxsize=4)
def get_sentiment_engine(level: str = "finvader") -> SentimentEngine:
    """Return a cached sentiment engine so FinBERT is not reloaded per ticker."""
    return SentimentEngine(level=level)


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


_MARKET_OPEN_HOUR  = 9.5    # 9:30 ET
_MARKET_CLOSE_HOUR = 16.0   # 16:00 ET


def _align_pub_to_trading_day(pub, trading_idx: "pd.DatetimeIndex"):
    """
    Map a headline's publication time to a trading row and classify its session.

    Returns (trading_day_timestamp, session_str)  or  None if out of range.

    Session classification (US Eastern time):
        "premarket"  — before 09:30 ET on a trading day
        "regular"    — 09:30–16:00 ET (market hours)
        "afterhours" — after 16:00 ET → pushed to the NEXT trading session
                       because the market is closed and the impact will be felt
                       when it next opens.
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

    hour = p.hour + p.minute / 60.0

    # Determine which calendar date this headline belongs to for trading purposes.
    if hour >= _MARKET_CLOSE_HOUR:
        # After market close → impacts the NEXT session. Use date + 1 day so
        # searchsorted finds the next trading day.
        cal_date = (p + pd.Timedelta(days=1)).normalize()
        session = "afterhours"
    else:
        cal_date = p.normalize()
        session = "premarket" if hour < _MARKET_OPEN_HOUR else "regular"

    if cal_date > trading_idx[-1]:
        # After-hours news on the final indexed trading day gets pushed to the
        # next calendar day. Keep it on the last session instead of dropping it.
        if session == "afterhours" and (cal_date - trading_idx[-1]).days <= 3:
            return trading_idx[-1], session
        return None
    if cal_date < trading_idx[0]:
        return trading_idx[0], session

    pos = trading_idx.searchsorted(cal_date, side="left")
    if pos >= len(trading_idx):
        return None
    return trading_idx[pos], session


def headline_mentions_ticker(ticker: str, title: str) -> bool:
    """
    True if the headline likely refers to this symbol.
    Used to drop noise from broad market RSS feeds.
    """
    if not ticker or not title:
        return False
    u = str(title).upper()
    for alias in _ticker_news_aliases(ticker):
        a = str(alias).upper().strip()
        if a and a in u:
            return True
    return False


# -----------------------------------------------------------------------------
# Ticker News Aliases and Live Google News Queries
# -----------------------------------------------------------------------------

def _ticker_news_aliases(ticker: str) -> list[str]:
    """Balanced alias list so live news can match both bullish and bearish catalysts."""
    t = (ticker or "").upper().strip()
    aliases = [t]
    alias_map = {
        "AAPL": ["Apple"],
        "MSFT": ["Microsoft"],
        "NVDA": ["Nvidia"],
        "AMD": ["AMD", "Advanced Micro Devices"],
        "GOOGL": ["Google", "Alphabet", "GOOG"],
        "GOOG": ["Google", "Alphabet", "GOOGL"],
        "META": ["Meta", "Facebook"],
        "AMZN": ["Amazon"],
        "TSLA": ["Tesla"],
        "JPM": ["JPMorgan", "JPMorgan Chase"],
        "GS": ["Goldman Sachs", "Goldman"],
        "BAC": ["Bank of America"],
        "UNH": ["UnitedHealth", "UnitedHealth Group"],
        "JNJ": ["Johnson & Johnson"],
        "ABBV": ["AbbVie"],
        "XOM": ["Exxon", "ExxonMobil"],
        "CVX": ["Chevron"],
        "CAT": ["Caterpillar"],
        "DE": ["Deere", "John Deere"],
        "MU": ["Micron", "Micron Technology"],
        "INTC": ["Intel"],
        "BTC-USD": ["Bitcoin", "BTC", "crypto"],
        "ETH-USD": ["Ethereum", "ETH", "crypto"],
    }
    aliases.extend(alias_map.get(t, []))
    if "-" in t:
        aliases.append(t.split("-")[0])
    # Preserve order, remove blanks/duplicates.
    dedup = []
    seen = set()
    for a in aliases:
        a = str(a).strip()
        if a and a not in seen:
            dedup.append(a)
            seen.add(a)
    return dedup


def _google_news_live_queries(ticker: str) -> list[str]:
    """
    Symmetric live-news queries that can surface both bullish and bearish catalysts.
    We do NOT bias the search toward only negative or only positive stories.
    """
    aliases = _ticker_news_aliases(ticker)
    primary = aliases[0]
    alias_or = " OR ".join(f'"{a}"' for a in aliases[:4])
    return [
        f"({alias_or}) (stock OR shares OR earnings OR guidance OR analyst)",
        f"({alias_or}) (upgrade OR downgrade OR investigation OR lawsuit OR sec OR forecast)",
        f"({alias_or}) (profit OR revenue OR margin OR demand OR warning)",
        f'"{primary}"',
    ]


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
                # Convert to ET and drop tz info but KEEP the time-of-day so
                # _align_pub_to_trading_day can split premarket vs after-hours.
                if pub.tzinfo is not None:
                    pub = pub.tz_convert("America/New_York").tz_localize(None)
                # No .normalize() — preserve hour/minute
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

    try:
        dmin = pd.Timestamp(dates.min()).normalize()
        dmax = pd.Timestamp(dates.max()).normalize()
        today = pd.Timestamp.today().normalize()
        dmax = min(dmax, today)
        out = []

        if client is not None:
            payload = client.company_news(ticker, _from=dmin.strftime("%Y-%m-%d"), to=dmax.strftime("%Y-%m-%d"))[:1000]
            time.sleep(0.4)
        else:
            key = _get_finnhub_api_key()
            if not key:
                return []
            import requests

            payload = []
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sentiment_cache")
            os.makedirs(cache_dir, exist_ok=True)
            chunk_start = dmin
            while chunk_start <= dmax:
                chunk_end = min(chunk_start + pd.Timedelta(days=89), dmax)
                cache_path = os.path.join(
                    cache_dir,
                    f"{ticker.upper()}_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}_finnhub.json",
                )
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, encoding="utf-8") as f:
                            chunk_payload = json.load(f)
                    except Exception:
                        chunk_payload = []
                else:
                    resp = requests.get(
                        "https://finnhub.io/api/v1/company-news",
                        params={
                            "symbol": ticker,
                            "from": chunk_start.strftime("%Y-%m-%d"),
                            "to": chunk_end.strftime("%Y-%m-%d"),
                            "token": key,
                        },
                        headers=_rss_http_headers(),
                        timeout=30,
                    )
                    resp.raise_for_status()
                    chunk_payload = resp.json() or []
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(chunk_payload, f)
                    time.sleep(0.4)
                payload.extend(chunk_payload)
                chunk_start = chunk_end + pd.Timedelta(days=1)

        seen_titles: set[str] = set()
        for item in payload:
            try:
                ts_utc = pd.to_datetime(item.get("datetime", 0), unit="s", utc=True)
                # Keep time-of-day (no .normalize()) so we can detect
                # premarket vs after-hours headlines downstream.
                ts = ts_utc.tz_convert("America/New_York").tz_localize(None)
                title = str(item.get("headline", "") or "").strip()
                title_key = title.lower()
                if title and title_key not in seen_titles:
                    out.append((ts, title))
                    seen_titles.add(title_key)
            except Exception:
                continue
        return out
    except Exception:
        return []


# Helper: FINNHUB API KEY lookup (robust)
def _get_finnhub_api_key() -> str:
    """Best-effort FINNHUB_API_KEY lookup without crashing if dotenv/config is absent."""
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if key:
        return key
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        key = os.environ.get("FINNHUB_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    try:
        from config_v2 import FINNHUB_API_KEY as key  # type: ignore
        if key:
            return str(key).strip()
    except Exception:
        pass
    try:
        from config import FINNHUB_API_KEY as key  # type: ignore
        if key:
            return str(key).strip()
    except Exception:
        pass
    return ""


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
    engine = get_sentiment_engine(engine_level)
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

    # Flatten all headlines for batch scoring in one pass
    meta: list[tuple[str, object]] = []   # (source_name, pub_datetime)
    texts: list[str] = []
    for name, headlines in sources_data.items():
        for pub_date, title in headlines:
            meta.append((name, pub_date))
            texts.append(title if title else " ")

    lg.info(f"  [{ticker}] Headlines to score after filtering: {len(texts)}")
    detail_list: list[dict] = []
    if texts:
        detail_list = engine.score_batch_detail(texts)
        compounds = [d["compound"] for d in detail_list]
        if compounds:
            lg.info(
                f"  [{ticker}] Score sample: min={min(compounds):.4f}, "
                f"max={max(compounds):.4f}, mean={float(np.mean(compounds)):.4f}"
            )

    total_headlines = sum(len(h) for h in sources_data.values())
    trading_idx = _normalize_trading_index(dates)
    date_axis = pd.DatetimeIndex(pd.to_datetime(dates, utc=False).normalize())

    # Per-day accumulators — track compound, p_pos, p_neg per session separately
    source_scores:    dict = {n: defaultdict(list) for n in sources_data}
    day_pos:          dict = defaultdict(list)   # positive scores per day
    day_neg:          dict = defaultdict(list)   # negative scores per day
    day_premarket:    dict = defaultdict(list)   # premarket compound scores
    day_afterhours:   dict = defaultdict(list)   # after-hours compound scores
    headline_counts:  dict = defaultdict(int)
    dropped_align = 0

    for (name, pub_date), detail in zip(meta, detail_list):
        aligned = _align_pub_to_trading_day(pub_date, trading_idx)
        if aligned is None:
            dropped_align += 1
            continue
        adj, session = aligned
        compound = detail["compound"]
        p_pos    = detail["p_pos"]
        p_neg    = detail["p_neg"]
        w        = weights.get(name, 1.0)

        source_scores[name][adj].append(compound)
        headline_counts[adj] += 1

        if p_pos > 0:
            day_pos[adj].append(p_pos * w)
        if p_neg > 0:
            day_neg[adj].append(p_neg * w)

        if session == "premarket":
            day_premarket[adj].append(compound * w)
        elif session == "afterhours":
            day_afterhours[adj].append(compound * w)

    if dropped_align > 0:
        lg.info(
            f"  [{ticker}] {dropped_align} headlines outside date range skipped."
        )

    result = pd.DataFrame(index=dates)
    src_cols: list[tuple[str, float]] = []

    # Per-source net compound score (backward-compat columns: sent_finnhub etc.)
    for name in sources_data:
        col = f"sent_{name}"
        daily = {d: float(np.mean(v)) for d, v in source_scores[name].items() if v}
        series = pd.Series(daily, dtype=float)
        series.index = pd.DatetimeIndex(series.index).normalize()
        series = series.reindex(date_axis).fillna(0.0)
        result[col] = series.values
        src_cols.append((col, weights[name]))

    lg.info(f"  [{ticker}] Total headlines scored: {total_headlines}")

    # ── Weighted net compound score ───────────────────────────────────────────
    total_w = sum(w for _, w in src_cols)
    if total_w > 0 and src_cols:
        result["news_sentiment"] = sum(result[col] * w for col, w in src_cols) / total_w
    else:
        result["news_sentiment"] = 0.0

    # ── Separate positive / negative raw signals ──────────────────────────────
    def _day_series(day_dict: dict) -> pd.Series:
        daily = {d: float(np.mean(v)) for d, v in day_dict.items() if v}
        s = pd.Series(daily, dtype=float)
        s.index = pd.DatetimeIndex(s.index).normalize()
        return s.reindex(date_axis).fillna(0.0)

    result["sent_pos"]        = _day_series(day_pos).values      # avg P(positive) on positive-news days
    result["sent_neg"]        = _day_series(day_neg).values      # avg P(negative) on negative-news days
    result["sent_premarket"]  = _day_series(day_premarket).values  # before 9:30 ET
    result["sent_afterhours"] = _day_series(day_afterhours).values # after 16:00 ET, already next-session

    # ── Disagreement across sources ───────────────────────────────────────────
    only_cols = [c for c, _ in src_cols]
    result["sentiment_disagreement"] = (
        result[only_cols].std(axis=1).fillna(0) if only_cols
        else pd.Series(0.0, index=dates)
    )

    # ── Headline volume ───────────────────────────────────────────────────────
    vol_s = pd.Series(dict(headline_counts), dtype=float)
    vol_s.index = pd.DatetimeIndex(vol_s.index).normalize()
    vol_s = vol_s.reindex(date_axis).fillna(0)
    rmean = vol_s.rolling(20, min_periods=5).mean().replace(0, 1)
    result["headline_volume"] = (vol_s / rmean).fillna(1.0)

    # Headline volume spike: how many std devs above the 20-day average
    vstd = vol_s.rolling(20, min_periods=5).std().replace(0, 1)
    result["headline_vol_spike"] = ((vol_s - rmean) / vstd).clip(-3, 3).fillna(0.0).values

    # ── Exponential decay (more recent = more weight) ─────────────────────────
    # EWM span=2 ≈ half-life 1 trading day (fast news decay)
    # EWM span=5 ≈ half-life ~3.5 trading days (slower carry-forward)
    net = result["news_sentiment"]
    result["sent_decay_1d"]  = net.ewm(span=2,  min_periods=1).mean().fillna(0.0)
    result["sent_decay_3d"]  = net.ewm(span=5,  min_periods=1).mean().fillna(0.0)
    result["sent_neg_decay"] = result["sent_neg"].ewm(span=5, min_periods=1).mean().fillna(0.0)
    result["sent_pos_decay"] = result["sent_pos"].ewm(span=5, min_periods=1).mean().fillna(0.0)

    # ── Legacy rolling features (kept for backward compat) ───────────────────
    result["sentiment_3d"]    = net.rolling(3).mean().fillna(0)
    result["sentiment_7d"]    = net.rolling(7).mean().fillna(0)
    result["sentiment_delta"] = net.diff().fillna(0)
    result["sentiment_accel"] = result["sentiment_delta"].diff().fillna(0)

    nz = int((net.abs() > 1e-9).sum())
    lg.info(f"  [{ticker}] Trading days with non-zero news_sentiment: {nz} / {len(dates)}")
    if total_headlines == 0:
        lg.warning(
            f"  [{ticker}] No headlines from any source — check FINNHUB_API_KEY, "
            "network, or install: pip install requests feedparser"
        )
    elif nz == 0:
        lg.warning(
            f"  [{ticker}] Headlines scored but news_sentiment is all zero — "
            "check date alignment and FinBERT/finvader install."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME NEWS FETCHER
# Fetches headlines published TODAY (pre-market) so predictions are based
# on this morning's news, not last week's stale headlines.
# ─────────────────────────────────────────────────────────────────────────────

def _empty_provider_result(name: str, provider_type: str, is_primary: bool, error: str = "") -> dict:
    return {
        "name": name,
        "provider_type": provider_type,
        "is_primary": bool(is_primary),
        "ok": False,
        "error": error,
        "fresh_count": 0,
        "headlines": [],
    }


def _provider_status(result: dict) -> dict:
    return {
        "name": str(result.get("name", "")),
        "provider_type": str(result.get("provider_type", "")),
        "is_primary": bool(result.get("is_primary", False)),
        "ok": bool(result.get("ok", False)),
        "fresh_count": int(result.get("fresh_count", 0) or 0),
        "error": str(result.get("error", ""))[:160],
    }


def _dedupe_recent_headlines(headlines: list) -> list:
    """Deduplicate headline tuples by title, keeping the newest occurrence."""
    dedup_map: dict[str, object] = {}
    for pub, title in headlines:
        prev = dedup_map.get(title)
        if prev is None or pub > prev:
            dedup_map[title] = pub
    dedup = [(pub, title) for title, pub in dedup_map.items()]
    dedup.sort(key=lambda x: x[0], reverse=True)
    return dedup


def fetch_premarket_news_by_provider(
    ticker: str,
    max_age_hours: int = 16,
    verbose: bool = False,
) -> list[dict]:
    """
    Fetch recent headlines grouped by provider with health metadata.

    Finnhub is marked primary. RSS providers are fallback sources, so an outage
    in Finnhub can be distinguished from a complete news outage.
    """
    from datetime import datetime, timedelta, timezone
    import pandas as pd

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    provider_results: list[dict] = []

    rss_sources = [
        ("yahoo_finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"),
    ]

    for i, q in enumerate(_google_news_live_queries(ticker), start=1):
        rss_sources.append(
            (
                f"google_news_{i}",
                "https://news.google.com/rss/search?q={ticker_query}&hl=en-US&gl=US&ceid=US:en".replace(
                    "{ticker_query}", q.replace(" ", "+")
                ),
            )
        )

    for source_name, url in rss_sources:
        result = _empty_provider_result(source_name, "rss", False)
        try:
            fetched = _fetch_rss_headlines(url, ticker)
            recent_headlines: list[tuple[datetime, str]] = []
            kept = 0
            for pub, title in fetched:
                try:
                    ts = pd.Timestamp(pub)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("America/New_York").tz_convert("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    py_dt = ts.to_pydatetime()
                    if py_dt >= cutoff and title:
                        # For broader Google/Yahoo feeds, keep only headlines that appear
                        # relevant to the ticker or one of its aliases.
                        if source_name.startswith("google_news") or source_name == "yahoo_finance":
                            aliases = _ticker_news_aliases(ticker)
                            u = title.upper()
                            if not any(str(a).upper() in u for a in aliases):
                                continue
                        recent_headlines.append((py_dt, title.strip()))
                        kept += 1
                except Exception:
                    continue
            result.update({
                "ok": True,
                "fresh_count": kept,
                "headlines": _dedupe_recent_headlines(recent_headlines),
            })
            if verbose:
                print(f"[{ticker}] {source_name} recent headlines: {kept}")
        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"[{ticker}] {source_name} fetch failed: {e}")
        provider_results.append(result)

    finnhub_key = _get_finnhub_api_key()
    finnhub_result = _empty_provider_result("finnhub", "finnhub_api", True)
    if finnhub_key:
        try:
            import requests

            today_utc = datetime.now(timezone.utc).date()
            start_utc = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).date()
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start_utc.isoformat(),
                "to": today_utc.isoformat(),
                "token": finnhub_key,
            }
            resp = requests.get(url, params=params, headers=_rss_http_headers(), timeout=30)
            resp.raise_for_status()
            payload = resp.json() or []

            kept = 0
            recent_headlines: list[tuple[datetime, str]] = []
            for item in payload[:100]:
                try:
                    ts = pd.to_datetime(item.get("datetime", 0), unit="s", utc=True)
                    if pd.isna(ts):
                        continue
                    py_dt = ts.to_pydatetime()
                    if py_dt < cutoff:
                        continue
                    title = str(item.get("headline", "") or "").strip()
                    if not title:
                        continue
                    recent_headlines.append((py_dt, title))
                    kept += 1
                except Exception:
                    continue
            finnhub_result.update({
                "ok": True,
                "fresh_count": kept,
                "headlines": _dedupe_recent_headlines(recent_headlines),
            })
            if verbose:
                print(f"[{ticker}] finnhub recent headlines: {kept}")
        except Exception as e:
            finnhub_result["error"] = str(e)
            if verbose:
                print(f"[{ticker}] finnhub fetch failed: {e}")
    else:
        finnhub_result["error"] = "FINNHUB_API_KEY not found"
        if verbose:
            print(f"[{ticker}] finnhub skipped: FINNHUB_API_KEY not found")
    provider_results.append(finnhub_result)

    return provider_results


def fetch_premarket_news(ticker: str, max_age_hours: int = 16, verbose: bool = False) -> list:
    """
    Fetch only RECENT headlines (published within the last `max_age_hours`).

    Uses the stronger RSS helper with browser-like headers and also attempts
    Finnhub live company news when an API key is available.

    Returns:
      List of (published_datetime, headline_text) tuples, newest first.
    """
    providers = fetch_premarket_news_by_provider(ticker, max_age_hours=max_age_hours, verbose=verbose)
    dedup = _dedupe_recent_headlines([
        headline
        for provider in providers
        for headline in provider.get("headlines", [])
    ])

    if verbose:
        print(f"[{ticker}] total recent headlines after dedupe: {len(dedup)}")
    return dedup


def score_todays_news(ticker: str, engine: SentimentEngine, verbose: bool = False) -> dict:
    """
    Fetch today's headlines and return a sentiment summary for the predictor.

    Returns a dict with:
      composite_score   : weighted average of recent headline scores
      headline_count    : how many headlines found in the last 16 hours
      strongest_headline: the headline with the highest absolute score
      strongest_score   : its score
      is_breaking_news  : True if > 3 headlines in the last 2 hours
      breaking_count    : number of headlines in the last 2 hours
      avg_abs_score     : mean absolute headline sentiment score
      provider_status   : per-provider health summaries
      primary_available : True when Finnhub returned fresh headlines
      fallback_available: True when at least one non-Finnhub source returned fresh headlines
    """
    provider_results = fetch_premarket_news_by_provider(ticker, max_age_hours=16, verbose=verbose)
    provider_status = [_provider_status(p) for p in provider_results]
    primary_available = any(
        bool(p.get("is_primary")) and int(p.get("fresh_count", 0) or 0) > 0
        for p in provider_results
    )
    fallback_available = any(
        not bool(p.get("is_primary")) and int(p.get("fresh_count", 0) or 0) > 0
        for p in provider_results
    )
    headlines = _dedupe_recent_headlines([
        headline
        for provider in provider_results
        for headline in provider.get("headlines", [])
    ])

    if not headlines:
        return {
            "composite_score": 0.0,
            "headline_count": 0,
            "strongest_headline": "",
            "strongest_score": 0.0,
            "is_breaking_news": False,
            "breaking_count": 0,
            "avg_abs_score": 0.0,
            "provider_status": provider_status,
            "primary_available": primary_available,
            "fallback_available": fallback_available,
            "fresh_headline_count": 0,
        }

    texts = [h[1] for h in headlines]
    scores = engine.score_batch(texts)

    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    weights = []
    for pub, _ in headlines:
        age_hours = (now - pub).total_seconds() / 3600
        weight = max(0.1, 1.0 - (age_hours / 16))
        weights.append(weight)

    total_weight = sum(weights) + 1e-9
    composite = sum(s * w for s, w in zip(scores, weights)) / total_weight

    abs_scores = [abs(s) for s in scores]
    avg_abs_score = float(sum(abs_scores) / len(abs_scores)) if abs_scores else 0.0
    if abs_scores:
        strongest_idx = abs_scores.index(max(abs_scores))
        strongest_text = texts[strongest_idx]
        strongest_scr = scores[strongest_idx]
    else:
        strongest_text = ""
        strongest_scr = 0.0

    two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
    recent_2h = sum(1 for pub, _ in headlines if pub >= two_hours_ago)
    is_breaking = recent_2h > 3

    return {
        "composite_score": round(composite, 4),
        "headline_count": len(headlines),
        "strongest_headline": strongest_text,
        "strongest_score": round(strongest_scr, 4),
        "is_breaking_news": is_breaking,
        "breaking_count": int(recent_2h),
        "avg_abs_score": round(avg_abs_score, 4),
        "provider_status": provider_status,
        "primary_available": primary_available,
        "fallback_available": fallback_available,
        "fresh_headline_count": len(headlines),
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
