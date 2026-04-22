"""
settings.py — XGBoost-only unified settings
"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # loads .env file from project root into os.environ

# ─────────────────────────────────────────────────────────────────────────────
# API / GENERAL
# ─────────────────────────────────────────────────────────────────────────────

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST / SCANNER
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "GS", "BAC", "UNH", "JNJ", "ABBV", "XOM", "CVX",
    "CAT", "DE", "MU", "INTC", "BTC-USD",
]

TOP_N_STOCKS = 10
MIN_PRICE = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_START = "2015-01-01"
TRAIN_END = datetime.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# MARKET / SECTOR MAPS
# ─────────────────────────────────────────────────────────────────────────────

MULTI_MARKET = {
    "spy": "SPY",
    "qqq": "QQQ",
    "dia": "DIA",
    "vix": "^VIX",
    "gld": "GLD",
    "tnx": "^TNX",
    "iwm": "IWM",
    "ewl": "EWL",
    "uup": "UUP",
    "hyg": "HYG",
    "tlt": "TLT",
    "eem": "EEM",
}

SECTOR_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMD": "XLK",
    "GOOGL": "XLK",
    "META": "XLK",
    "INTC": "XLK",
    "MU": "XLK",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "JPM": "XLF",
    "GS": "XLF",
    "BAC": "XLF",
    "JNJ": "XLV",
    "UNH": "XLV",
    "ABBV": "XLV",
    "XOM": "XLE",
    "CVX": "XLE",
    "CAT": "XLI",
    "DE": "XLI",
}

# ─────────────────────────────────────────────────────────────────────────────
# NEWS SOURCES
# ─────────────────────────────────────────────────────────────────────────────

NEWS_SOURCES = [
    {"name": "finnhub", "weight": 1.3, "type": "finnhub_api", "url": None},
    {
        "name": "yahoo_finance",
        "weight": 1.0,
        "type": "rss",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    },
    {
        "name": "cnbc",
        "weight": 1.1,
        "type": "rss",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    },
    {
        "name": "marketwatch",
        "weight": 1.0,
        "type": "rss",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    },
    {
        "name": "google_news",
        "weight": 0.8,
        "type": "rss",
        "url": "https://news.google.com/rss/search?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SIGNAL_DIR = os.path.join(BASE_DIR, "signals")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for _d in (DATA_DIR, MODEL_DIR, SIGNAL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

SHORTLIST_FILE = os.path.join(DATA_DIR, "shortlist.csv")
SIGNALS_FILE = os.path.join(SIGNAL_DIR, "signals.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FLAGS
# ─────────────────────────────────────────────────────────────────────────────

USE_MULTI_TIMEFRAME = True
USE_VIX_TERM = True
USE_OPTIONS_DATA = True
USE_ORDER_BOOK = True
USE_ANALYST_DATA = True
USE_EARNINGS_DATA = True
USE_YIELD_CURVE = True
SOCIAL_SENTIMENT_ENABLED = True
USE_NEWS_SENTIMENT = True

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS / SPLITS
# ─────────────────────────────────────────────────────────────────────────────

RETURN_HORIZON_DAYS = 5
RETURN_BINS = [-0.03, -0.01, 0.01, 0.03]
RETURN_BIN_LABELS = [
    "strong_down",
    "moderate_down",
    "flat",
    "moderate_up",
    "strong_up",
]

TRAIN_CALIBRATION_SPLIT = 0.70
CALIBRATION_TEST_SPLIT = 0.85
EMBARGO_DAYS = RETURN_HORIZON_DAYS

CONFIDENCE_TARGET_PRECISION = 0.58
DEFAULT_FIXED_CONFIDENCE_THRESHOLD = 57.5

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.7
XGB_EARLY_STOP_ROUNDS = 30
XGB_MIN_CHILD_WEIGHT = 3
XGB_GAMMA = 0.1
XGB_REG_ALPHA = 0.1
XGB_REG_LAMBDA = 1.0

# kept for compatibility with old code paths
SHAP_MIN_IMPORTANCE = 0.0001
SHAP_MAX_FEATURES = 120

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Phase 2: neural weight stays 0 until offline validation shows positive edge vs
# XGB-only. Flip to e.g. {"xgboost": 0.7, "neural": 0.3} after nested-CV confirms
# the neural branch improves out-of-sample Sharpe.
ENSEMBLE_WEIGHTS = {"xgboost": 1.0, "neural": 0.0}

# ─────────────────────────────────────────────────────────────────────────────
# REGIME / RISK / BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

VIX_HIGH_THRESHOLD = 25.0
VIX_EXTREME_THRESHOLD = 35.0
BEAR_REGIME_MULT = 0.40
HIGH_VIX_MULT = 0.70
EXTREME_VIX_MULT = 0.40

SLIPPAGE_BASE_PCT = 0.0010
COMMISSION_PER_SHARE = 0.005

MAX_GROSS_EXPOSURE = 1.00
MAX_NET_EXPOSURE = 0.60
MAX_SECTOR_EXPOSURE = 0.35
MAX_SINGLE_NAME_EXPOSURE = 0.20
MAX_PAIR_CORRELATION = 0.85
MAX_DRAWDOWN_HALT_PCT = 0.15