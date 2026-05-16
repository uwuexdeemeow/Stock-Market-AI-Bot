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

# Tradier API — free with any brokerage account.  Used for real options IV data.
# Get your token at https://web.tradier.com/user/api
# Set TRADIER_USE_SANDBOX=1 for delayed/test data without a funded account.
TRADIER_API_TOKEN = os.environ.get("TRADIER_API_TOKEN", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST / SCANNER
# ─────────────────────────────────────────────────────────────────────────────

CORE_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "GS", "BAC", "UNH", "JNJ", "ABBV", "XOM", "CVX",
    "CAT", "DE", "MU", "INTC", "BTC-USD",
    # Added for larger training pool — diverse sectors, high liquidity
    "LLY", "V", "MA", "HD", "COST", "PG", "BRK-B",
]

PHASE3_EXPANSION_WATCHLIST = [
    # Phase 3 expansion — defensive, real estate, materials, and transport
    "NEE", "DUK", "SO", "PLD", "AMT", "EQIX", "KO", "WMT",
    "PEP", "MCD", "LIN", "FCX", "UPS", "UNP",
]

# Broad cross-sectional universe (~140 names, ~10–15 per sector across 11 GICS
# sectors).  Switching to this universe is the structural fix when AUC stays
# at ~0.51 on a narrow mega-cap universe: cross-sectional rank features need
# enough names per sector to produce real dispersion, not coin-flip rank
# percentiles between 1–3 sector peers.
#
# Constraints used when picking names:
#   * ≥ $1B average daily dollar volume (executable size)
#   * ≥ 12 years of clean yfinance history (backtest from 2010 onwards)
#   * Single-listing US-domiciled equities (no ADRs, dual-listings, BRK-A)
#   * One ticker per company (no class-A/B duplicates)
#
# Sector counts:  XLK 18 · XLY 14 · XLF 18 · XLV 18 · XLE 11 · XLI 16 ·
#                 XLP 12 · XLU 10 · XLRE 10 · XLB 10 · XLC 10  →  147 names
BROAD_WATCHLIST = [
    # ─ XLK Technology ──────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "INTC", "MU",
    "CRM", "ORCL", "ADBE", "AVGO", "TXN", "QCOM", "AMAT", "CSCO", "IBM", "NOW",
    # ─ XLY Consumer Discretionary ──────────────────────────────────────────
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG", "TJX",
    "LOW", "MAR", "GM", "F", "ROST", "EBAY",
    # ─ XLF Financials ──────────────────────────────────────────────────────
    "JPM", "GS", "BAC", "V", "MA", "BRK-B", "WFC", "MS",
    "C", "AXP", "BLK", "SCHW", "USB", "SPGI", "COF", "PNC", "TRV", "AIG",
    # ─ XLV Healthcare ──────────────────────────────────────────────────────
    "UNH", "JNJ", "ABBV", "LLY", "PFE", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "SYK", "BSX", "CI",
    # ─ XLE Energy ──────────────────────────────────────────────────────────
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "OXY", "PSX", "VLO", "KMI", "WMB",
    # ─ XLI Industrials ─────────────────────────────────────────────────────
    "CAT", "DE", "UPS", "UNP", "BA", "HON", "GE", "RTX",
    "LMT", "MMM", "FDX", "NOC", "EMR", "ETN", "PH", "CSX",
    # ─ XLP Consumer Staples ────────────────────────────────────────────────
    "COST", "PG", "KO", "WMT", "PEP", "MO", "PM", "CL",
    "KMB", "GIS", "SYY", "HSY",
    # ─ XLU Utilities ───────────────────────────────────────────────────────
    "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "D", "PEG", "ED",
    # ─ XLRE Real Estate (REITs) ────────────────────────────────────────────
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "EQR", "AVB",
    # ─ XLB Materials ───────────────────────────────────────────────────────
    "LIN", "FCX", "APD", "SHW", "ECL", "NEM", "NUE", "DOW", "DD", "IFF",
    # ─ XLC Communication Services ──────────────────────────────────────────
    "T", "VZ", "NFLX", "CMCSA", "DIS", "TMUS", "CHTR", "EA", "OMC", "LYV",
]

# Rejected-by-default subset from the first 42-ticker experiment. These names
# were individually positive in the full run, but the candidate universe still
# worsened raw stock Sharpe versus core. Keep only for reproducible A/B tests.
PHASE3_CANDIDATE_WATCHLIST = ["LIN", "DUK", "KO"]

# Survivorship-bias audit candidates. These are not part of production training
# by default; survivorship_audit.py tries aliases because delisted symbols often
# move to OTC/Q tickers or disappear from free data vendors.
SURVIVORSHIP_AUDIT_TICKERS = {
    "SIVB": ["SIVB", "SIVBQ"],      # Silicon Valley Bank
    "FRC":  ["FRC", "FRCB"],        # First Republic Bank
    "BBBY": ["BBBY", "BBBYQ"],      # Bed Bath & Beyond
    "WE":   ["WE", "WEWKQ"],        # WeWork
    "RIDE": ["RIDE", "RIDEQ"],      # Lordstown Motors
    "SHLD": ["SHLD", "SHLDQ"],      # Sears Holdings
    "JCP":  ["JCP", "JCPNQ"],       # J.C. Penney
    "HTZ":  ["HTZ", "HTZGQ"],       # Hertz pre-2021 bankruptcy line
    "CHK":  ["CHK", "CHKAQ"],       # Chesapeake Energy pre-2021 line
    "WLL":  ["WLL", "WLLAW"],       # Whiting Petroleum pre-bankruptcy line
    "CRC":  ["CRC", "CRCQQ"],       # California Resources pre-bankruptcy line
    "DO":   ["DO", "DOFSQ"],        # Diamond Offshore
    "MDR":  ["MDR", "MDRIQ"],       # McDermott International
    "FTR":  ["FTR", "FTRCQ"],       # Frontier Communications pre-bankruptcy line
    "WIN":  ["WIN", "WINMQ"],       # Windstream Holdings
    "WPG":  ["WPG", "WPGGQ"],       # Washington Prime Group
    "TUP":  ["TUP", "TUPBQ"],       # Tupperware Brands
}

# Crashed/delisted tickers that are included in training data ONLY.
# These contribute rows to the pooled model so it learns pre-crash patterns
# without survivorship bias. They never appear in signals or live trading.
# Only tickers confirmed to have available parquet data are listed here
# (SIVB, WE, RIDE are missing; see logs/survivorship_audit.json).
SURVIVORSHIP_TRAINING_TICKERS: list[str] = ["FRC", "BBBY"]

# Lean universe (~60 names): keeps every ticker that has been picked as an
# overlay stock, plus enough per sector for cross-sectional ranking to work
# (minimum 5 per GICS sector).  Cuts research.py runtime by ~55% vs broad.
#
# PLAIN ENGLISH: The strategy picks ~3 overlay stocks each rebalance, and they
# come from several sectors.  Cross-sectional ranking needs peers to compare
# against, so we keep at least 5 per sector — but don't need 18.
# Sectors that NEVER produce picks (XLV, XLE, XLU, XLRE) are trimmed to 5
# so their ranking context exists but doesn't waste download time.
LEAN_WATCHLIST = [
    # ─ XLK Technology (top source of overlay picks — keep strongest 10) ────
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "INTC", "MU",
    "AVGO", "AMAT",
    # ─ XLY Consumer Discretionary (TSLA picked) ───────────────────────────
    "AMZN", "TSLA", "HD", "COST", "BKNG",
    # ─ XLF Financials (C picked) ──────────────────────────────────────────
    "JPM", "GS", "BAC", "V", "MA", "C",
    # ─ XLV Healthcare (no picks, keep 5 for ranking context) ──────────────
    "UNH", "JNJ", "ABBV", "LLY", "MRK",
    # ─ XLE Energy (no picks, keep 5 for ranking context) ──────────────────
    "XOM", "CVX", "COP", "EOG", "SLB",
    # ─ XLI Industrials (CAT picked) ───────────────────────────────────────
    "CAT", "DE", "HON", "GE", "RTX",
    # ─ XLP Consumer Staples (HSY picked) ──────────────────────────────────
    "PG", "KO", "PEP", "WMT", "HSY",
    # ─ XLU Utilities (no picks, keep 5 for ranking context) ───────────────
    "NEE", "DUK", "SO", "AEP", "EXC",
    # ─ XLRE Real Estate (no picks, keep 5 for ranking context) ────────────
    "PLD", "AMT", "EQIX", "CCI", "PSA",
    # ─ XLB Materials (FCX, NEM, IFF picked) ───────────────────────────────
    "LIN", "FCX", "NEM", "APD", "IFF", "SHW",
    # ─ XLC Communication Services (CHTR, LYV picked) ─────────────────────
    "NFLX", "CMCSA", "DIS", "CHTR", "LYV",
]

UNIVERSE_MODE = os.environ.get("STOCK_UNIVERSE_MODE", "lean").strip().lower()
if UNIVERSE_MODE == "broad":
    # 147-name cross-sectional universe with 10-18 names per GICS sector.
    # Required for cross-sectional rank features to have real dispersion.
    WATCHLIST = list(BROAD_WATCHLIST)
elif UNIVERSE_MODE == "lean":
    # ~60-name universe — keeps all overlay-picked tickers + 5 per sector
    # for ranking.  Much faster research.py runs.
    WATCHLIST = list(LEAN_WATCHLIST)
elif UNIVERSE_MODE == "expanded":
    WATCHLIST = CORE_WATCHLIST + PHASE3_EXPANSION_WATCHLIST
elif UNIVERSE_MODE == "candidate":
    WATCHLIST = CORE_WATCHLIST + PHASE3_CANDIDATE_WATCHLIST
else:   # "core" or anything else
    WATCHLIST = CORE_WATCHLIST

TOP_N_STOCKS = 10
MIN_PRICE = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_START = "2010-01-01"
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
    "irx": "^IRX",
    "iwm": "IWM",
    "ewl": "EWL",
    "uup": "UUP",
    "hyg": "HYG",
    "ief": "IEF",
    "tlt": "TLT",
    "copper": "CPER",
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
    # New tickers
    "LLY":   "XLV",   # Healthcare
    "V":     "XLF",   # Financials
    "MA":    "XLF",   # Financials
    "HD":    "XLY",   # Consumer Discretionary
    "COST":  "XLP",   # Consumer Staples
    "PG":    "XLP",   # Consumer Staples
    "BRK-B": "XLF",   # Financials
    # Phase 3 expansion
    "NEE":  "XLU",    # Utilities
    "DUK":  "XLU",    # Utilities
    "SO":   "XLU",    # Utilities
    "PLD":  "XLRE",   # Real Estate / REITs
    "AMT":  "XLRE",   # Real Estate / REITs
    "EQIX": "XLRE",   # Real Estate / REITs
    "KO":   "XLP",    # Consumer Staples
    "WMT":  "XLP",    # Consumer Staples
    "PEP":  "XLP",    # Consumer Staples
    "MCD":  "XLY",    # Consumer Discretionary
    "LIN":  "XLB",    # Materials
    "FCX":  "XLB",    # Materials
    "UPS":  "XLI",    # Industrials
    "UNP":  "XLI",    # Industrials
    # Survivorship-bias audit tickers
    "SIVB": "XLF",    # Financials
    "FRC":  "XLF",    # Financials
    "BBBY": "XLY",    # Consumer Discretionary
    "WE":   "XLY",    # Consumer Discretionary
    "RIDE": "XLY",    # Consumer Discretionary
    "SHLD": "XLY",    # Consumer Discretionary
    "JCP":  "XLY",    # Consumer Discretionary
    "HTZ":  "XLI",    # Industrials
    "CHK":  "XLE",    # Energy
    "WLL":  "XLE",    # Energy
    "CRC":  "XLE",    # Energy
    "DO":   "XLE",    # Energy
    "MDR":  "XLE",    # Energy
    "FTR":  "XLC",    # Communication Services
    "WIN":  "XLC",    # Communication Services
    "WPG":  "XLRE",   # Real Estate
    "TUP":  "XLY",    # Consumer Discretionary
    # ── BROAD_WATCHLIST additions (146 names) ────────────────────────────
    # XLK Technology
    "CRM":  "XLK", "ORCL": "XLK", "ADBE": "XLK", "AVGO": "XLK",
    "TXN":  "XLK", "QCOM": "XLK", "AMAT": "XLK", "CSCO": "XLK",
    "IBM":  "XLK", "NOW":  "XLK",
    # XLY Consumer Discretionary
    "NKE":  "XLY", "SBUX": "XLY", "BKNG": "XLY", "TJX":  "XLY",
    "LOW":  "XLY", "MAR":  "XLY", "GM":   "XLY", "F":    "XLY",
    "ROST": "XLY", "EBAY": "XLY",
    # XLF Financials
    "WFC":  "XLF", "MS":   "XLF", "C":    "XLF", "AXP":  "XLF",
    "BLK":  "XLF", "SCHW": "XLF", "USB":  "XLF", "SPGI": "XLF",
    "COF":  "XLF", "PNC":  "XLF", "TRV":  "XLF", "AIG":  "XLF",
    # XLV Healthcare
    "PFE":  "XLV", "MRK":  "XLV", "TMO":  "XLV", "ABT":  "XLV",
    "DHR":  "XLV", "BMY":  "XLV", "AMGN": "XLV", "GILD": "XLV",
    "CVS":  "XLV", "MDT":  "XLV", "ISRG": "XLV", "SYK":  "XLV",
    "BSX":  "XLV", "CI":   "XLV",
    # XLE Energy
    "COP":  "XLE", "EOG":  "XLE", "SLB":  "XLE", "MPC":  "XLE",
    "OXY":  "XLE", "PSX":  "XLE", "VLO":  "XLE", "KMI":  "XLE",
    "WMB":  "XLE",
    # XLI Industrials
    "BA":   "XLI", "HON":  "XLI", "GE":   "XLI", "RTX":  "XLI",
    "LMT":  "XLI", "MMM":  "XLI", "FDX":  "XLI", "NOC":  "XLI",
    "EMR":  "XLI", "ETN":  "XLI", "PH":   "XLI", "CSX":  "XLI",
    # XLP Consumer Staples
    "MO":   "XLP", "PM":   "XLP", "CL":   "XLP", "KMB":  "XLP",
    "GIS":  "XLP", "SYY":  "XLP", "HSY":  "XLP",
    # XLU Utilities
    "AEP":  "XLU", "EXC":  "XLU", "XEL":  "XLU", "SRE":  "XLU",
    "D":    "XLU", "PEG":  "XLU", "ED":   "XLU",
    # XLRE Real Estate (REITs)
    "CCI":  "XLRE", "PSA":  "XLRE", "O":    "XLRE", "WELL": "XLRE",
    "SPG":  "XLRE", "EQR":  "XLRE", "AVB":  "XLRE",
    # XLB Materials
    "APD":  "XLB", "SHW":  "XLB", "ECL":  "XLB", "NEM":  "XLB",
    "NUE":  "XLB", "DOW":  "XLB", "DD":   "XLB", "IFF":  "XLB",
    # XLC Communication Services
    "T":    "XLC", "VZ":   "XLC", "NFLX": "XLC", "CMCSA": "XLC",
    "DIS":  "XLC", "TMUS": "XLC", "CHTR": "XLC", "EA":   "XLC",
    "OMC":  "XLC", "LYV":  "XLC",
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
# Social sentiment is safety-only by default.
#
# StockTwits/X data is useful as a live fallback/diagnostic signal, but the
# free feeds do not provide point-in-time history good enough for model alpha
# without a separate validation project.  Keep the alpha flag off unless that
# validation exists.
SOCIAL_SENTIMENT_SAFETY_ENABLED = os.environ.get("SOCIAL_SENTIMENT_SAFETY_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
SOCIAL_SENTIMENT_ALPHA_ENABLED = os.environ.get("SOCIAL_SENTIMENT_ALPHA_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
# Backward-compatible alias for older code.  Legacy SOCIAL_SENTIMENT_ENABLED is
# intentionally mapped to alpha mode, not safety mode.
SOCIAL_SENTIMENT_ENABLED = SOCIAL_SENTIMENT_ALPHA_ENABLED
USE_NEWS_SENTIMENT = True
# Point-in-time valuation features from dated yfinance quarterly statements.
# Values are exposed after a conservative reporting lag, then z-scored within
# sector by research.py after all selected tickers are built.
USE_FUNDAMENTAL_ZSCORES = os.environ.get("USE_FUNDAMENTAL_ZSCORES", "1").strip().lower() in {"1", "true", "yes", "on"}
FUNDAMENTAL_REPORT_LAG_DAYS = int(os.environ.get("FUNDAMENTAL_REPORT_LAG_DAYS", "45"))
# Experimental: give the h5 short-horizon model its own sentiment-protected
# feature set. Initial A/B worsened the core baseline, so it is opt-in.
H5_SENTIMENT_FEATURES_ENABLED = os.environ.get("H5_SENTIMENT_FEATURES_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
# Which NLP model to use for scoring news headlines during research.py builds.
# "finbert" = best accuracy (~89%) but needs torch+transformers (2 GB).
# "finvader" = decent (~65%), lightweight, financial-aware VADER extension.
# "vader" = fastest, no extra deps, ~56% accuracy on financial text.
# Override via SENTIMENT_ENGINE_LEVEL env var on CI where torch isn't installed
# to skip the slow fallback chain (finbert import fail → finvader → vader).
SENTIMENT_ENGINE_LEVEL = os.environ.get("SENTIMENT_ENGINE_LEVEL", "finbert")
# Live predictions avoid FinBERT/PyTorch crashes and FinVADER's NLTK lexicon
# download dependency. Research/backfills still use SENTIMENT_ENGINE_LEVEL above.
LIVE_SENTIMENT_ENGINE_LEVEL = os.environ.get("LIVE_SENTIMENT_ENGINE_LEVEL", "vader")

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS / SPLITS
# ─────────────────────────────────────────────────────────────────────────────

# RETURN_HORIZON_DAYS: dropped from 20 to 5 after the diagnostic ladder showed
# OOS AUC = 0.505 across 28-ticker AND 147-ticker universes regardless of
# architecture (pooling, regime split, h5 ensemble, macro/xs_rank protection,
# triple-barrier labels, calibration tweaks). Technical features (RSI, MACD,
# Bollinger, momentum) carry information on 3–10 day mean-reversion / momentum
# windows; at 20d the signal washes out into pure beta. Override via env var
# RETURN_HORIZON_DAYS=20 to A/B the old setting.
RETURN_HORIZON_DAYS = int(os.environ.get("RETURN_HORIZON_DAYS", "5"))
RETURN_BINS = [-0.03, -0.01, 0.01, 0.03]

# ── Prediction target ─────────────────────────────────────────────────────────
# "direction"        — original: UP if stock_ret > DIRECTION_LABEL_THRESHOLD
# "excess_return"    — UP if (stock_ret - spy_ret) > EXCESS_RETURN_MIN_PCT
#                      Removes beta noise; model only gets credit for alpha.
# "vol_adjusted"     — UP if (stock_ret - spy_ret) / hvol_20d_scaled > threshold
#                      Rewards the same excess return more during calm markets.
# "triple_barrier"   — López de Prado: +1 if price hits +PT*ATR first,
#                      -1 if price hits -SL*ATR first, 0 on time-out.
#                      Sharper class separation than fixed-horizon returns.
PREDICTION_TARGET = "excess_return"
# Diagnostic ladder result (28-ticker pooled run, 2026-04-28):
#   excess_return: train AUC 0.500, OOS Direction AUC 0.510, OOS LONG AUC 0.511
#   direction    : train AUC ~0.50, OOS Direction AUC 0.486, OOS LONG AUC 0.493
# Both targets gave AUC ~0.50 — the feature set lacks ranking signal regardless
# of label. Reverted to excess_return because (a) it slightly outperformed
# direction OOS, and (b) excess_return is the right target for a cross-sectional
# ranker against SPY benchmark. The fix is on the input side: cross-sectional
# rank-within-sector features added in research.py rebuild_cross_sectional_features().

# Triple-barrier barrier widths (multiplied by 14-day ATR).
# SYMMETRIC barriers keep the UP rate near 40-50%, which is critical for
# isotonic calibration.  Asymmetric barriers (e.g. pt=2.5, sl=1.0) produce
# only ~25-30% UP labels — isotonic then maps almost all raw p_up values
# below 0.5 because it shrinks toward the prior, making direction_vote
# permanently SHORT and breaking all calibration downstream.
TRIPLE_BARRIER_PT_MULT = 2.0   # profit-take barrier = entry + PT_MULT × ATR
TRIPLE_BARRIER_SL_MULT = 2.0   # stop-loss barrier   = entry - SL_MULT × ATR

# Minimum excess return vs SPY (after estimated slippage) to label a row UP.
# Scaled proportionally to RETURN_HORIZON_DAYS: at 20d horizon 0.5% was the
# spread+commission hurdle; at 5d that's ~0.15% (sqrt(5/20) * 0.5% ≈ 0.25%
# accounting for vol scaling, but we use the linear hurdle since spread is
# horizon-independent). Keeps the UP-rate baseline near 45-50%.
# Auto-scale with horizon: 0.5% at 20d, ~0.15% at 5d (linear scaling).
# Override via env var EXCESS_RETURN_MIN_PCT for A/B tests.
_excess_default = round(0.005 * (RETURN_HORIZON_DAYS / 20.0), 5)
EXCESS_RETURN_MIN_PCT = float(os.environ.get("EXCESS_RETURN_MIN_PCT", str(_excess_default)))

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
# Raw XGBoost probabilities in this project often compress into a narrow
# 50-56% band. A 57.5% fixed fallback creates a dead zone, so the fallback is a
# low coverage floor; live trading still requires the separate confidence
# validation gate before trusting this score as edge.
DEFAULT_FIXED_CONFIDENCE_THRESHOLD = 52.5

# ── Pooled / cross-sectional settings ─────────────────────────────────────────
# POOLED_TRAINING: train ONE model across all tickers (9× more data) instead of
# separate per-ticker models.  Adds ticker one-hot features so the model still
# learns ticker-specific patterns.
POOLED_TRAINING = True

# Cross-sectional ranking: at each signal date, rank all tickers by predicted
# excess return and trade the top/bottom N instead of filtering by a confidence
# threshold.  This sidesteps the broken calibration entirely.
# Env override exists for test-gated ranker sweeps:
#   CROSS_SECTIONAL_TOP_N=1 POOLED_RANKER_EXPERIMENT_ENABLED=1 python3 backtest.py --output summary
# Raised from 4 → 10. The model cannot pick stocks (rank_spearman ≈ 0) but
# CAN time the market (NW t-stat 3.3 vs cash). Holding 10 names instead of
# 4 adds diversification that raises raw stock return from +13% to +17% and
# Sharpe from 0.19 to 0.25 by reducing single-stock noise.
CROSS_SECTIONAL_TOP_N = max(1, int(os.environ.get("CROSS_SECTIONAL_TOP_N", "10")))
# signal_quality is assigned by rank inside apply_cross_sectional_ranking:
#   rank 0 → HIGH, rank 1 → MEDIUM, rank 2 → LOW.
# The quality label is for reporting only; the gate was removed (it was
# redundant — top-N selection already controls which tickers enter the backtest).

# Meta-labelling: second-stage model predicts whether the primary direction
# signal is likely to be correct.  The ranker only sees rows with enough meta
# confidence.  This sacrifices coverage to improve precision.
META_LABELING_ENABLED = True
META_LABEL_MIN_CONFIDENCE = 0.60

# Multi-horizon ensemble: blend the 20-day primary model with the 5-day
# secondary model.  Weights must sum to 1.0.
# p_up_raw = H20_WEIGHT * p_up_h20 + H5_WEIGHT * p_up_h5
# Blend ablation favoured 50/50 over the earlier 65/35 heuristic
# (Sharpe=0.736, return=277.33%, trades=396).
MULTI_HORIZON_H20_WEIGHT = 0.50
MULTI_HORIZON_H5_WEIGHT  = 0.50

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

# Legacy compatibility names only — NOT used in current training pipeline.
# Feature pruning in train.py uses XGBoost gain importance (feature_importances_),
# not SHAP values.  These constants are kept so older scripts that import them
# do not crash.  Do not rely on them for new code.
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
#
# Fixed: switched nested CV scoring from accuracy_score → roc_auc_score so
# class imbalance (~30% UP triple-barrier labels) no longer biases the search
# toward majority-class-predicting models. Also tightened param grid to exclude
# known-bad underregularised values (min_child_weight=1, reg_lambda=0.1).
TUNE_HYPERPARAMS = True

# Experimental train-only ranker path.  When enabled with `python3 train.py --ranker`
# or env POOLED_RANKER_EXPERIMENT_ENABLED=1, train.py fits a separate XGBRanker
# on continuous forward excess returns and reports daily rank IC.  It writes
# ranker artifacts separately and does not replace the production classifier
# until backtest/live support proves out.
POOLED_RANKER_EXPERIMENT_ENABLED = os.environ.get("POOLED_RANKER_EXPERIMENT_ENABLED", "0") == "1"

# Simple factor composite ranker: bypass XGBoost entirely for cross-sectional
# ranking and instead use a fixed-weight average of volatility/beta factors
# that have proven OOS rank IC.  The factor_baseline_diagnostic.py showed:
#   - Equal-weight vol/beta composite: OOS IC=+0.055, t=+7.85 (28 tickers)
#   - XGBoost walk-forward:            OOS IC=+0.007, t=+1.82 (28 tickers)
# XGBoost destroys the signal by overfitting noise features each block.
# When True, apply_cross_sectional_ranking uses raw feature z-scores as
# rank_score instead of XGBoost predictions.
SIMPLE_FACTOR_RANKING = os.environ.get("SIMPLE_FACTOR_RANKING", "0").strip().lower() in {"1", "true", "yes", "on"}

# Factors and weights for the simple composite.  Derived from walk-forward
# IC-weighted estimation (factor_baseline_diagnostic.py).  All factors are
# the "higher = better future return" direction (signs already corrected).
# These are the raw parquet column names; the ranking code handles
# cross-sectional z-scoring internally.
SIMPLE_FACTOR_COLS = [
    "factor_idio_vol_252_spy",    # idiosyncratic volatility (low-vol anomaly inverted)
    "hvol_20d",                   # 20d historical volatility
    "atr_norm",                   # normalised ATR
    "factor_beta_252_spy",        # market beta
    "xs_rank_sector_hvol_20d",    # within-sector vol rank
]
# Weights from the walk-forward IC-weighted estimation.  They sum to 1.0.
# These are the FALLBACK weights — the adaptive updater (Fix 3) writes fresh
# IC-weighted weights to ADAPTIVE_WEIGHTS_FILE after each research run.
SIMPLE_FACTOR_WEIGHTS = {
    "factor_idio_vol_252_spy": 0.25,
    "hvol_20d":                0.20,
    "atr_norm":                0.20,
    "factor_beta_252_spy":     0.20,
    "xs_rank_sector_hvol_20d": 0.15,
}

# ── Adaptive factor weight settings ─────────────────────────────────────────
# PLAIN ENGLISH: After each research.py run, we recompute factor weights from
# trailing IC (information coefficient = how predictive each factor was recently).
# Factors that have been working well get more weight; those that lost edge get
# less.  This prevents slow alpha decay from stale static weights.
ADAPTIVE_WEIGHTS_FILE = os.path.join(SIGNAL_DIR, "adaptive_factor_weights.json")
ADAPTIVE_WEIGHT_HALFLIFE = int(os.environ.get("ADAPTIVE_WEIGHT_HALFLIFE", "63"))  # days
ADAPTIVE_WEIGHT_FLOOR = float(os.environ.get("ADAPTIVE_WEIGHT_FLOOR", "0.05"))   # min per factor

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

# ── Cash / benchmark overlay ─────────────────────────────────────────────────
# When the model has no approved single-name exposure, put part of idle capital
# into a simple SPY/QQQ sleeve during broad-market uptrends.  This keeps the
# portfolio from sitting fully idle in bull regimes while preserving cash in
# hostile regimes.
CASH_OVERLAY_ENABLED = True
CASH_OVERLAY_MAX_ALLOC = 0.75
CASH_OVERLAY_REQUIRE_SPY_MA200 = True
CASH_OVERLAY_WEIGHTS = {"SPY": 0.60, "QQQ": 0.40}

# Main benchmark used for alpha reporting. Individual SPY/QQQ/IWM comparisons
# are still written under benchmark_comparisons.
PRIMARY_BENCHMARK_WEIGHTS = {"SPY": 0.60, "QQQ": 0.40}

# Ticker eligibility filter: prior walk-forward contribution determines whether
# a ticker can enter the cross-sectional ranking.  Missing reports fail open so
# the first research run can still generate model_quality_report.csv.
TICKER_ELIGIBILITY_FILTER_ENABLED = True
TICKER_ELIGIBILITY_MIN_TRADES = 20
TICKER_ELIGIBILITY_MIN_TOTAL_PNL = 0.0
TICKER_ELIGIBILITY_MIN_WIN_RATE = 0.52
# Only keep tickers with a demonstrated, causal edge in recent selected trades.
TICKER_ELIGIBILITY_MIN_SHARPE = 0.40
TICKER_ELIGIBILITY_MIN_PROFIT_FACTOR = 1.15
TICKER_ELIGIBILITY_MAX_DRAWDOWN_PCT = -25.0

# Experimental rank repair is disabled by default. A noisy short-window flip can
# make selection worse even when the final report says confidence/rank is weak.
RANK_ORIENTATION_FLIP_ENABLED = False

# Live approval is generated from models/model_quality_report.csv by backtest.py.
# Keep these compatibility names empty so old imports fail safe instead of
# silently trading a stale hand-maintained list.
APPROVED_TICKERS: list[str] = []
DEFAULT_APPROVED_LIVE_TICKERS: list[str] = []

# ─────────────────────────────────────────────────────────────────────────────
# REGIME / RISK / BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

# VIX term structure inversion threshold.
# PLAIN ENGLISH: When VIX > VIX3M (ratio > 1.0), the term structure is
# "inverted" (backwardation).  This means the market expects MORE volatility
# in the short term than the medium term — a strong crash signal.  When
# the ratio exceeds this threshold, the live regime is forced to "risk_off"
# regardless of trend signals.  1.05 means VIX must be 5% above VIX3M.
VIX_INVERSION_THRESHOLD = float(os.environ.get("VIX_INVERSION_THRESHOLD", "1.05"))

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

# Split-conformal direction filter. When enabled, a trade must be a singleton
# conformal prediction from the calibration window. Keep off by default until an
# A/B backtest proves it improves raw stock edge.
CONFORMAL_TRADE_GATE_ENABLED = os.environ.get("CONFORMAL_TRADE_GATE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
CONFORMAL_ALPHA = float(os.environ.get("CONFORMAL_ALPHA", "0.20"))
CONFORMAL_MIN_CALIBRATION_ROWS = int(os.environ.get("CONFORMAL_MIN_CALIBRATION_ROWS", "100"))

# ── Regime-conditional models ─────────────────────────────────────────────────
# When True, train_pooled() and the backtest walk-forward each train two
# separate XGBoost direction classifiers — one on bull-regime rows
# (regime=+1: SPY above 200MA AND VIX < 20) and one on bear-regime rows
# (regime=-1: SPY below 200MA OR VIX >= 28).  At prediction time the model
# that matches the current market regime is used.  Falls back to the combined
# (pooled) model if either regime partition has fewer than REGIME_MIN_TRAIN_ROWS
# rows, preventing overfitting on tiny subsets.
REGIME_CONDITIONAL_MODELS = os.environ.get("REGIME_CONDITIONAL_MODELS", "0").strip().lower() in {"1", "true", "yes", "on"}
REGIME_MIN_TRAIN_ROWS = 800   # raised from 200: bear fold had only 668 rows total
                              # in the 28-ticker run, so early walk-forward blocks
                              # were training the bear specialist on < 200 rows and
                              # producing inverted ranks. 800 lets the bear model
                              # only fire when there is enough signal to learn from.

# ── Drop market-wide-constant features from candidate set ─────────────────────
# A "market-wide-constant" feature is one whose value is the SAME for every
# ticker on a given date — e.g. spy_ret_5d, vix_close, macro_yield_curve,
# regime.  These are useful for absolute direction prediction, but for a
# CROSS-SECTIONAL ranker (which compares stocks to each other on day D) they
# contribute zero discriminative gradient: every stock sees the same value, so
# the split "spy_ret_5d > 0" partitions zero stocks within a date.
#
# In the diagnostic ladder these features ate ~310 / 1120 candidate slots
# (~28%) without lifting AUC above 0.51.  When True, _select_feature_cols
# drops them BEFORE pruning so the per-stock + xs_rank features get the full
# top-K budget instead of competing with constants.
#
# Set to False to A/B against the previous behaviour (kept-with-protection).
EXCLUDE_MARKET_CONSTANTS = os.environ.get("EXCLUDE_MARKET_CONSTANTS", "1").strip().lower() in {"1", "true", "yes", "on"}

# Edge audit / causal edge gates.  Audit is diagnostic-only and safe to keep on.
# Gates are opt-in until A/B tests prove they improve raw stock Sharpe.
EDGE_AUDIT_ENABLED = os.environ.get("EDGE_AUDIT_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
REGIME_EDGE_GATE_ENABLED = os.environ.get("REGIME_EDGE_GATE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
RANK_DECILE_GATE_ENABLED = os.environ.get("RANK_DECILE_GATE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
EDGE_GATE_MIN_TRADES = int(os.environ.get("EDGE_GATE_MIN_TRADES", "30"))
EDGE_GATE_MIN_PROFIT_FACTOR = float(os.environ.get("EDGE_GATE_MIN_PROFIT_FACTOR", "1.05"))
EDGE_GATE_MIN_AVG_PNL = float(os.environ.get("EDGE_GATE_MIN_AVG_PNL", "0.0"))

# ── SPY/QQQ market-timing mode ────────────────────────────────────────────────
# The model can time the market (NW t-stat 3.3 vs cash) but can't pick stocks
# (rank_spearman ≈ 0).  When SPY_TIMING_MODE is on:
#   - predict.py counts how many tickers have LONG direction votes
#   - If the fraction exceeds SPY_TIMING_BULL_THRESHOLD → output "BUY SPY"
#   - Otherwise → "STAY CASH"
# This eliminates single-stock risk entirely and captures the timing edge.
SPY_TIMING_MODE = os.environ.get("SPY_TIMING_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
# Fraction of tickers that must vote LONG for a "BUY SPY" signal.
# 0.50 = majority rule.  Higher = more conservative (fewer signals, higher precision).
SPY_TIMING_BULL_THRESHOLD = float(os.environ.get("SPY_TIMING_BULL_THRESHOLD", "0.50"))
# What to buy when the signal fires.  Default is 60/40 SPY/QQQ matching benchmark.
SPY_TIMING_INSTRUMENTS = {"SPY": 0.60, "QQQ": 0.40}
# Position size when fully invested (fraction of equity).
SPY_TIMING_INVEST_PCT = float(os.environ.get("SPY_TIMING_INVEST_PCT", "0.95"))
# ── Smoothing: reduces whipsaw from daily signal flips ─────────────────────
# Rolling average window for bull_fraction — smooths 1-day noise.
# At 5, the signal reflects the past week's model consensus, not today's alone.
SPY_TIMING_SMOOTH_WINDOW = int(os.environ.get("SPY_TIMING_SMOOTH_WINDOW", "1"))
# Hysteresis: separate entry/exit thresholds to avoid flipping on noise.
# Enter when smoothed bull_fraction ≥ ENTER_THRESHOLD, exit when ≤ EXIT_THRESHOLD.
# Set both to 0.50 for no hysteresis (pure majority rule).
SPY_TIMING_ENTER_THRESHOLD = float(os.environ.get("SPY_TIMING_ENTER_THRESHOLD", "0.50"))
SPY_TIMING_EXIT_THRESHOLD = float(os.environ.get("SPY_TIMING_EXIT_THRESHOLD", "0.50"))
# Minimum holding period (trading days).  Once invested, hold at least this many
# days before considering an exit.  Set to 0 for no constraint.
SPY_TIMING_MIN_HOLD_DAYS = int(os.environ.get("SPY_TIMING_MIN_HOLD_DAYS", "5"))
# Transaction cost per unit of turnover for ETFs.  SPY/QQQ have ~1-2 cent
# spreads on $500+ prices, so round-trip cost is ~1-2bps.  This is used to
# compute after-cost returns in the backtest.  Stock-level slippage (10bps)
# would overstate costs 10x for liquid ETFs.
SPY_TIMING_ETF_COST_BPS = float(os.environ.get("SPY_TIMING_ETF_COST_BPS", "1.5"))
# Paper-readiness gates for the timing-only strategy.  These are separate from
# the stock-sleeve kill criteria so an index-timing product can be evaluated
# without pretending the single-name ranker has edge.
MIN_TIMING_SHARPE = float(os.environ.get("MIN_TIMING_SHARPE", "0.70"))
MAX_TIMING_DRAWDOWN_PCT = float(os.environ.get("MAX_TIMING_DRAWDOWN_PCT", "-20.0"))
MIN_TIMING_NW_TSTAT_VS_CASH = float(os.environ.get("MIN_TIMING_NW_TSTAT_VS_CASH", "2.0"))
MIN_TIMING_TRADES_OR_EXPOSURE_DAYS = int(os.environ.get("MIN_TIMING_TRADES_OR_EXPOSURE_DAYS", "100"))

# ── Benchmark-beating ETF rotation mode ──────────────────────────────────────
# This is a stricter index-allocation product than SPY_TIMING_MODE.  The goal is
# not just to beat cash, but to beat buy-and-hold SPY, QQQ, and the configured
# 60/40 SPY/QQQ benchmark with explicit benchmark-relative gates.
ETF_ROTATION_MODE = os.environ.get("ETF_ROTATION_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
ETF_ROTATION_ASSETS = ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD")
ETF_ROTATION_MAX_GROSS_EXPOSURE = float(os.environ.get("ETF_ROTATION_MAX_GROSS_EXPOSURE", "1.25"))
ETF_ROTATION_RISK_ON_ENTER = float(os.environ.get("ETF_ROTATION_RISK_ON_ENTER", "0.55"))
ETF_ROTATION_RISK_ON_EXIT = float(os.environ.get("ETF_ROTATION_RISK_ON_EXIT", "0.45"))
ETF_ROTATION_NEUTRAL_THRESHOLD = float(os.environ.get("ETF_ROTATION_NEUTRAL_THRESHOLD", "0.50"))
ETF_ROTATION_MIN_HOLD_DAYS = int(os.environ.get("ETF_ROTATION_MIN_HOLD_DAYS", "5"))
ETF_ROTATION_REBALANCE_BAND = float(os.environ.get("ETF_ROTATION_REBALANCE_BAND", "0.10"))
ETF_ROTATION_DEFAULT_VOL_TARGET = float(os.environ.get("ETF_ROTATION_DEFAULT_VOL_TARGET", "0.12"))
ETF_ROTATION_VOL_TARGETS = (0.10, 0.12, 0.15)
ETF_ROTATION_PRESETS = {
    "conservative_unlevered": {
        "risk_on": {"SPY": 0.55, "QQQ": 0.45},
        "neutral": {"SPY": 0.45, "QQQ": 0.20, "BIL": 0.35},
        "risk_off": {"BIL": 1.00},
        "max_gross": 1.00,
    },
    "limited_leverage": {
        "risk_on": {"SPY": 0.50, "QQQ": 0.75},
        "neutral": {"SPY": 0.55, "QQQ": 0.25, "BIL": 0.20},
        "risk_off": {"BIL": 1.00},
        "max_gross": ETF_ROTATION_MAX_GROSS_EXPOSURE,
    },
    "defensive_trend": {
        "risk_on": {"SPY": 0.60, "QQQ": 0.50, "GLD": 0.10},
        "neutral": {"SPY": 0.45, "IEF": 0.35, "GLD": 0.20},
        "risk_off": {"BIL": 0.50, "IEF": 0.35, "GLD": 0.15},
        "max_gross": 1.20,
    },
}
ETF_ROTATION_DEFENSIVE_MIXES = {
    "bil_only": {"BIL": 1.00},
    "bil_ief": {"BIL": 0.60, "IEF": 0.40},
    "bil_ief_gld": {"BIL": 0.50, "IEF": 0.35, "GLD": 0.15},
}
MIN_ETF_ROTATION_ALPHA_VS_SPY_PCT = float(os.environ.get("MIN_ETF_ROTATION_ALPHA_VS_SPY_PCT", "0.0"))
MIN_ETF_ROTATION_ALPHA_VS_QQQ_PCT = float(os.environ.get("MIN_ETF_ROTATION_ALPHA_VS_QQQ_PCT", "0.0"))
MIN_ETF_ROTATION_ALPHA_VS_BLEND_PCT = float(os.environ.get("MIN_ETF_ROTATION_ALPHA_VS_BLEND_PCT", "0.0"))
MIN_ETF_ROTATION_NW_TSTAT_VS_BLEND = float(os.environ.get("MIN_ETF_ROTATION_NW_TSTAT_VS_BLEND", "0.0"))

# Paper trading defaults to the index-timing product. Single-name paper trades
# must be explicitly enabled and still pass raw stock-sleeve gates in the latest
# backtest metrics.
PAPER_MODE_STRATEGY = os.environ.get("PAPER_MODE_STRATEGY", "core_satellite_alpha").strip().lower()
SINGLE_NAME_PAPER_TRADING_ENABLED = os.environ.get("SINGLE_NAME_PAPER_TRADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}

# ── Earnings blackout gate ─────────────────────────────────────────────────────
# PLAIN ENGLISH: Don't open new positions when earnings are within this many
# trading days.  Earnings announcements are binary events with unpredictable
# outcomes — the model cannot predict gap risk, so we simply avoid it.
# Set to 0 to disable.  Applies to both predict.py per-ticker "actionable" gate
# and the core_satellite_alpha overlay selection.
EARNINGS_BLACKOUT_DAYS = int(os.environ.get("EARNINGS_BLACKOUT_DAYS", "3"))

SINGLE_NAME_CRASH_FILTER_ENABLED = True
SINGLE_NAME_CRASH_MIN_PRICE = 2.0
SINGLE_NAME_CRASH_MAX_DRAWDOWN_20D = -0.45
SINGLE_NAME_CRASH_MAX_DRAWDOWN_60D = -0.60
SINGLE_NAME_CRASH_MAX_DIST_MA200 = -0.65

# Long-term distress veto for long entries. This is intentionally separate
# from the acute crash filter above: it catches slow bankruptcy-style downtrends
# that are far from prior highs, below short-term trend, and still falling.
SINGLE_NAME_DISTRESS_FILTER_ENABLED = True
SINGLE_NAME_DISTRESS_MAX_DRAWDOWN_252D = -0.35
SINGLE_NAME_DISTRESS_MAX_DIST_MA50 = -0.10
SINGLE_NAME_DISTRESS_MAX_RET_20D = 0.0

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

MAX_GROSS_EXPOSURE = 1.00          # max 5 concurrent positions at 20% each = 100% exposure
MAX_NET_EXPOSURE = 0.80
MAX_SECTOR_EXPOSURE = 0.40
MAX_SINGLE_NAME_EXPOSURE = 0.20   # 20% per name; 5 names × 20% = 100% max exposure
MAX_PAIR_CORRELATION = 0.85
MAX_DRAWDOWN_HALT_PCT = 0.99      # effectively disabled — regime filter handles protection; halt is a last resort

# "legacy_confidence" matches the earlier gate-passing backtests (ATR-based,
#  fixed 15% per slot). Replaced by "vol_kelly".
# "vol_kelly" — vol-target base (15% annual vol contribution per position) ×
#  fractional Kelly multiplier (1 + quarter_Kelly(confidence)) × quality boost.
#  High-vol tickers (NVDA) get smaller slots; high-confidence signals get larger
#  ones. Replaces fixed 15% with a risk-equalised, conviction-weighted size.
POSITION_SIZING_MODE = "vol_kelly"

# Target annual volatility contribution per position (vol_kelly mode only).
# vol-target weight = VOL_TARGET_PCT / asset_vol_annual.
# Must be lower than MAX_SINGLE_NAME_EXPOSURE × min_expected_vol to leave room
# for Kelly to differentiate.  With our universe vol 20–50%:
#   VOL_TARGET=0.08: NVDA(40%)=20% cap, AMD(45%)=17.8%, TSLA(55%)=14.5%  ← visible differentiation
#   VOL_TARGET=0.15: everything exceeds 20% cap before Kelly can act          ← no differentiation
VOL_TARGET_PCT = 0.08

# Kelly safety multiplier (vol_kelly mode only).  Full Kelly is theoretically
# optimal but too aggressive when edge estimates are noisy.
# 0.25 = quarter-Kelly (industry standard). 0.50 = half-Kelly (more aggressive).
KELLY_SAFETY_FRACTION = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT / MONITORING
# ─────────────────────────────────────────────────────────────────────────────

DRIFT_PSI_CAUTION = 0.10
DRIFT_PSI_ALERT = 0.25
DRIFT_KS_CAUTION = 0.05
DRIFT_KS_ALERT = 0.10
    
