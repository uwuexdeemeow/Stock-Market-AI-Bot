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
USE_OPTIONS_DATA = False
USE_ORDER_BOOK = True
USE_ANALYST_DATA = False
USE_EARNINGS_DATA = False
USE_YIELD_CURVE = True
SOCIAL_SENTIMENT_ENABLED = False
USE_NEWS_SENTIMENT = True
SENTIMENT_ENGINE_LEVEL = "finbert"

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS / SPLITS
# ─────────────────────────────────────────────────────────────────────────────

RETURN_HORIZON_DAYS = 10
RETURN_BINS = [-0.03, -0.01, 0.01, 0.03]

# ── Prediction target ─────────────────────────────────────────────────────────
# "direction"        — original: UP if stock_ret > DIRECTION_LABEL_THRESHOLD
# "excess_return"    — UP if (stock_ret - spy_ret) > EXCESS_RETURN_MIN_PCT
#                      Removes beta noise; model only gets credit for alpha.
# "vol_adjusted"     — UP if (stock_ret - spy_ret) / hvol_20d_scaled > threshold
#                      Rewards the same excess return more during calm markets.
# "triple_barrier"   — López de Prado: +1 if price hits +2*ATR first,
#                      -1 if price hits -1*ATR first, 0 on time-out.
#                      Sharper class separation than fixed-horizon returns.
PREDICTION_TARGET = "triple_barrier"

# Minimum excess return vs SPY (after estimated slippage) to label a row UP.
# 0.5 % clears the spread + commission hurdle for a $10k position.
EXCESS_RETURN_MIN_PCT = 0.005

# Sharpe-proxy threshold used when PREDICTION_TARGET = "vol_adjusted".
# A row is UP only if annualised excess Sharpe > this value.
VOL_ADJUSTED_SHARPE_THRESHOLD = 0.3
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

# ── Pooled / cross-sectional settings ─────────────────────────────────────────
# POOLED_TRAINING: train ONE model across all tickers (9× more data) instead of
# separate per-ticker models.  Adds ticker one-hot features so the model still
# learns ticker-specific patterns.
POOLED_TRAINING = True

# Cross-sectional ranking: at each signal date, rank all tickers by predicted
# excess return and trade the top/bottom N instead of filtering by a confidence
# threshold.  This sidesteps the broken calibration entirely.
CROSS_SECTIONAL_TOP_N = 5   # raised from 3: was 9.5% in-market; 5 targets ~16%
# In pooled cross-sectional mode, the relative rank is the primary selector.
# Allow all quality tiers so the ranker can see all candidates.
BACKTEST_ALLOWED_SIGNAL_QUALITIES = ("LOW", "MEDIUM", "HIGH")

# Label threshold: 0.0 means any positive 5-day return counts as UP.
# Tested 0.003 (0.3% min move) — hurt performance by cutting too many
# profitable trades and reducing the universe to 7 tickers.
DIRECTION_LABEL_THRESHOLD = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
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
SHAP_MAX_FEATURES = 15

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PRUNING
# ─────────────────────────────────────────────────────────────────────────────

# After an initial XGBoost fit, keep only the top-K base features ranked by
# gain importance, then retrain.  Ticker one-hot dummies are always kept.
# Reduces dimensionality from ~400 rolled columns to a stable, informative set.
FEATURE_IMPORTANCE_TOP_K = 30

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

# When True, train_pooled() runs a nested walk-forward search over
# max_depth / min_child_weight / reg_lambda before the final model fit.
# Best params are saved to models/pooled_best_xgb_params.json and reloaded
# by backtest.py so both pipelines use the same tuned configuration.
# Pass --no-tune to python3 train.py to skip this step for faster runs.
TUNE_HYPERPARAMS = True

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Production default: XGBoost is the only active model until the neural branch
# proves incremental out-of-sample value in nested CV / walk-forward testing.
#
# Keep both keys for backward compatibility with older code, but expose the
# active 3-model stack explicitly for the newer dynamic-weighting path.
ENSEMBLE_WEIGHTS = {"xgboost": 1.0, "neural": 0.0}
ACTIVE_MODEL_WEIGHTS = {
    "xgboost": 1.0,
    "lstm_attention": 0.0,
    "transformer": 0.0,
}
LIVE_SHORTS_ENABLED = False   # shorts off until calibration is healthy; long_only avoids bull-market short disaster

# Live approval is generated from models/model_quality_report.csv by backtest.py.
# Keep these compatibility names empty so old imports fail safe instead of
# silently trading a stale hand-maintained list.
APPROVED_TICKERS: list[str] = []
DEFAULT_APPROVED_LIVE_TICKERS: list[str] = []

# ─────────────────────────────────────────────────────────────────────────────
# REGIME / RISK / BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

VIX_HIGH_THRESHOLD = 25.0
VIX_EXTREME_THRESHOLD = 35.0
BEAR_REGIME_MULT = 0.40
HIGH_VIX_MULT = 0.70
EXTREME_VIX_MULT = 0.40

# Regime filter: block all new trades when market is unfavorable.
# Set MARKET_REGIME_FILTER_ENABLED = False to disable during testing.
MARKET_REGIME_FILTER_ENABLED = True
MARKET_REGIME_VIX_MAX = 25.0        # Block trades when VIX >= this value
MARKET_REGIME_SPY_MA200_REQUIRED = True  # Block trades when SPY is below its 200-day moving average

# Backtest-specific regime controls.
# Regime filter is ON: prevents long entries when SPY < MA200 or VIX >= 25.
# This avoids bear-market LONG losses (2022 base rate was 43% up — below random).
# The NW t-stat gate is already measured vs 0 (cash), not vs SPY, so blocking
# entries during hostile regimes does not penalise the Sharpe unfairly.
BACKTEST_MARKET_REGIME_FILTER_ENABLED = True
BACKTEST_REGIME_SIZE_MULTIPLIER_ENABLED = False

SLIPPAGE_BASE_PCT = 0.0010
COMMISSION_PER_SHARE = 0.005
BID_ASK_SPREAD_BPS = 5.0           # Base spread — scaled up by realized vol in execution_model
CAPACITY_ADV_THRESHOLD = 0.05

# Annual borrow rate for short positions (as a decimal, e.g. 0.005 = 0.5%).
# Most liquid large-caps are 0.3–0.5%; hard-to-borrow names can be 5%+.
BORROW_COST_ANNUAL_DEFAULT = 0.005
BORROW_COSTS: dict[str, float] = {
    # Add per-ticker overrides here when broker rates are known.
    # e.g. "GME": 0.50,  # 50% annualized borrow for a hard-to-borrow name
}

MAX_GROSS_EXPOSURE = 1.00
MAX_NET_EXPOSURE = 0.60
MAX_SECTOR_EXPOSURE = 0.35
MAX_SINGLE_NAME_EXPOSURE = 0.20
MAX_PAIR_CORRELATION = 0.85
MAX_DRAWDOWN_HALT_PCT = 0.15

# "legacy_confidence" matches the earlier gate-passing backtests.
# "vol_kelly" uses the newer blended vol-target + fractional-Kelly sizing.
POSITION_SIZING_MODE = "legacy_confidence"

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT / MONITORING
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_PSI_CAUTION = 0.10
DRIFT_PSI_ALERT = 0.25
DRIFT_KS_CAUTION = 0.05
DRIFT_KS_ALERT = 0.10
