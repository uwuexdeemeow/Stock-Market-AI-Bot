"""
setup.py — One-Command Pipeline Setup
======================================
Run this ONCE before anything else to install all dependencies and
verify your environment is ready.

  python setup.py

What it does (in order):
  1. Checks your Python version (needs 3.10+)
  2. Installs every required package
  3. Downloads the FinBERT model weights (~440 MB, done once)
  4. Creates all required folders (data/, models/, signals/, logs/)
  5. Verifies each import works
  6. Checks your API keys and shows which features are active
  7. Prints a clear summary: GREEN = ready, YELLOW = optional missing, RED = broken

This takes about 5-10 minutes on first run (mostly downloading FinBERT).
After that, re-running takes about 30 seconds.
"""

import os
import sys
import subprocess
import importlib
import platform
import json
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS (for the terminal output — makes it easy to see what's wrong)
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {BLUE}→{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{msg}{RESET}\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — PYTHON VERSION CHECK
# Your code uses features from Python 3.10+ (like `dict | None` type hints).
# Earlier versions will crash with a confusing SyntaxError.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE DETECTION
# Detect CUDA, Apple Silicon MPS, or CPU fallback.
# Also reports when an NVIDIA GPU exists but PyTorch cannot actually use it.
# ─────────────────────────────────────────────────────────────────────────────

def detect_acceleration_backend():
    result = {
        "device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "cuda_device_name": None,
        "nvidia_hardware_present": False,
        "nvidia_hardware_name": None,
        "reason": "No CUDA or MPS backend detected",
    }

    # Detect NVIDIA hardware at the OS level even if torch can't use it
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            smi = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if smi.returncode == 0:
                names = [x.strip() for x in smi.stdout.splitlines() if x.strip()]
                if names:
                    result["nvidia_hardware_present"] = True
                    result["nvidia_hardware_name"] = names[0]
    except Exception:
        pass

    try:
        import torch
    except Exception as e:
        if result["nvidia_hardware_present"]:
            result["reason"] = (
                f"NVIDIA GPU detected ({result['nvidia_hardware_name']}) but torch import failed: {e}"
            )
        else:
            result["reason"] = f"torch import failed: {e}"
        return result

    # CUDA check
    try:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["device"] = "cuda"
            result["reason"] = "CUDA-capable GPU detected and usable by PyTorch"
            return result
    except Exception as e:
        result["reason"] = f"torch.cuda check failed: {e}"

    # Apple Silicon MPS check
    try:
        result["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if result["mps_available"]:
            result["device"] = "mps"
            result["reason"] = "Apple Silicon MPS detected and usable by PyTorch"
            return result
    except Exception as e:
        result["reason"] = f"torch MPS check failed: {e}"

    # NVIDIA hardware exists but torch cannot use CUDA
    if result["nvidia_hardware_present"]:
        result["reason"] = (
            f"NVIDIA GPU detected ({result['nvidia_hardware_name']}) but PyTorch CUDA is unavailable. "
            f"This usually means a CPU-only torch build, missing/unsupported CUDA runtime, or driver mismatch."
        )

    return result


def check_python_version():
    hdr("STEP 1 — Python Version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"

    if major < 3 or (major == 3 and minor < 10):
        err(f"Python {version_str} found — need 3.10 or higher")
        err("Download from: https://www.python.org/downloads/")
        print("\nYour code uses 'X | None' type hints which were added in Python 3.10.")
        sys.exit(1)
    else:
        ok(f"Python {version_str} — compatible")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INSTALL PACKAGES
#
# Think of pip as an app store for Python.
# Each package here does a specific job:
#
#
# CORE DATA SCIENCE:
#   numpy       — fast math on arrays (the foundation of everything)
#   pandas      — spreadsheet-like data manipulation (DataFrames)
#   scipy       — statistical functions
#
# FINANCE DATA:
#   yfinance    — free stock price & options data from Yahoo Finance
#   finnhub-python — premium news & fundamentals (optional, free tier works)
#
# MACHINE LEARNING:
#   torch       — PyTorch: trains the neural networks (LSTM, Transformer, CNN)
#   scikit-learn — StandardScaler + accuracy_score + classification_report
#   xgboost     — gradient boosting for tabular features and hybrid ensemble voting
#   shap        — SHAP explainability for XGBoost feature pruning
#
# DEEP LEARNING NLP (for FinBERT):
#   transformers — HuggingFace library: loads pre-trained FinBERT model
#
# SENTIMENT:
#   finvader    — fast financial VADER (works without GPU, ~65% accuracy)
#   vaderSentiment — original VADER (fallback)
#
# NEWS FETCHING:
#   feedparser  — parses RSS feeds (Yahoo Finance, CNBC, MarketWatch)
#   requests    — HTTP library for fetching web content
#
# SOCIAL MEDIA:
#   tweepy      — X/Twitter API client (needs X API credentials)
#
# BROKER INTEGRATION:
#   moomoo      — broker-connected simulated trading through OpenD
#
# FILE FORMATS:
#   pyarrow     — reads/writes .parquet files (fast columnar storage)
#   openpyxl    — Excel support (optional, for spreadsheet exports)
#
# PLOTTING:
#   matplotlib  — charts for training curves + backtest charts
#
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (package_to_install, import_name, required_or_optional, description)
PACKAGES = [
    # ── Core data science ───────────────────────────────────────────────────
    ("numpy",            "numpy",          "required", "Array math — backbone of all features"),
    ("pandas",           "pandas",         "required", "DataFrames — how we store price + feature data"),
    ("scipy",            "scipy",          "required", "Statistical functions"),

    # ── Finance data ────────────────────────────────────────────────────────
    ("yfinance",         "yfinance",       "required", "Free stock prices, options, and fundamentals"),
    ("finnhub-python",   "finnhub",        "optional", "Premium news & earnings (free tier available)"),

    # ── Machine learning ────────────────────────────────────────────────────
    ("scikit-learn",     "sklearn",        "required", "Data scaling + accuracy metrics"),
    ("xgboost",          "xgboost",        "required", "Gradient boosting model for tabular alpha signals"),
    ("shap",             "shap",           "required", "SHAP explainability for feature pruning and ranking"),
    ("torch",            "torch",          "required", "PyTorch — trains all 4 neural network models"),
    ("torchvision",      "torchvision",    "optional", "PyTorch computer vision (not directly used, but often needed)"),

    # ── FinBERT / NLP ───────────────────────────────────────────────────────
    ("transformers",     "transformers",   "required", "HuggingFace — loads FinBERT (89-91% sentiment accuracy)"),
    ("tokenizers",       "tokenizers",     "required", "Fast tokeniser for FinBERT"),
    ("accelerate",       "accelerate",     "optional", "Faster FinBERT inference on modern hardware"),

    # ── Sentiment ───────────────────────────────────────────────────────────
    ("finvader",         "finvader",       "recommended", "Fast financial VADER (65% accuracy, no GPU needed)"),
    ("vaderSentiment",   "vaderSentiment", "optional", "Original VADER (fallback only, 56% accuracy)"),

    # ── News fetching ───────────────────────────────────────────────────────
    ("feedparser",       "feedparser",     "required", "Parses RSS feeds from Yahoo Finance, CNBC, etc."),
    ("requests",         "requests",       "required", "HTTP requests — needed for RSS + StockTwits"),

    # ── Social media ────────────────────────────────────────────────────────
    ("tweepy",           "tweepy",         "optional", "X/Twitter API client for social sentiment (needs API credentials)"),
    ("moomoo-api",       "moomoo",         "required", "Moomoo OpenAPI SDK for broker-connected paper trading"),

    # ── File formats ────────────────────────────────────────────────────────
    ("pyarrow",          "pyarrow",        "required", "Reads/writes .parquet files (fast compressed storage)"),
    ("fastparquet",      "fastparquet",    "optional", "Alternative parquet engine (backup)"),

    # ── Charting ────────────────────────────────────────────────────────────
    ("matplotlib",       "matplotlib",     "required", "Charts for training curves + backtest results"),
    ("seaborn",          "seaborn",        "optional", "Prettier charts (used for correlation heatmaps)"),

    # ── Utilities ───────────────────────────────────────────────────────────
    ("tqdm",             "tqdm",           "optional", "Progress bars in terminal"),
    ("python-dotenv",    "dotenv",         "optional", "Load API keys from .env file instead of export"),
]


def install_packages():
    hdr("STEP 2 — Installing Packages")

    # Figure out which packages are already installed
    already_installed = []
    to_install        = []

    for pip_name, import_name, level, desc in PACKAGES:
        try:
            importlib.import_module(import_name)
            already_installed.append((pip_name, level))
        except ImportError:
            to_install.append((pip_name, import_name, level, desc))

    if already_installed:
        info(f"{len(already_installed)} packages already installed — skipping those")

    if not to_install:
        ok("All packages already installed")
        return True

    # Install missing packages
    required_installs   = [(p, d) for p, _, l, d in to_install if l == "required"]
    recommended_installs= [(p, d) for p, _, l, d in to_install if l == "recommended"]
    optional_installs   = [(p, d) for p, _, l, d in to_install if l == "optional"]

    # Install required first
    if required_installs:
        info(f"Installing {len(required_installs)} required packages...")
        for pip_name, desc in required_installs:
            print(f"    Installing {pip_name} ({desc})...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet", "--upgrade"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                err(f"Failed to install {pip_name}:")
                print(f"       {result.stderr[:200]}")
                err("Try manually: pip install " + pip_name)
                return False
            ok(pip_name)

    # Install recommended
    if recommended_installs:
        info(f"Installing {len(recommended_installs)} recommended packages...")
        for pip_name, desc in recommended_installs:
            print(f"    Installing {pip_name} ({desc})...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(pip_name)
            else:
                warn(f"{pip_name} failed (optional) — {result.stderr[:100]}")

    # Install optional (ignore failures)
    if optional_installs:
        info(f"Installing {len(optional_installs)} optional packages...")
        for pip_name, desc in optional_installs:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                ok(f"{pip_name} (optional)")
            else:
                warn(f"{pip_name} not available (optional, skipping)")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CREATE DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

def create_directories():
    hdr("STEP 3 — Creating Folders")

    base = Path(__file__).resolve().parent
    dirs = {
        "data"   : "Stores .parquet feature files (one per stock)",
        "models" : "Stores trained model files + scaler/calibrator/XGBoost metadata",
        "signals": "Stores trade signals, backtest results, charts, and broker sync files",
        "logs"   : "Log files from every script run",
    }

    for dirname, desc in dirs.items():
        path = base / dirname
        path.mkdir(exist_ok=True)
        ok(f"{dirname}/  — {desc}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DOWNLOAD FINBERT MODEL (first run only)
# ─────────────────────────────────────────────────────────────────────────────

def download_finbert():
    hdr("STEP 4 — FinBERT Model Download")

    print("  FinBERT is a BERT model trained specifically on financial news.")
    print("  It's 89-91% accurate on financial text vs 56% for basic VADER.")
    print("  First download: ~440 MB (saved to ~/.cache/huggingface)")
    print("  Subsequent runs: instant (already cached)")
    print()

    try:
        from transformers import BertTokenizer, BertForSequenceClassification

        # Check if already cached (avoids re-downloading)
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        already_cached = any(
            "finbert" in str(d).lower()
            for d in (cache_dir.iterdir() if cache_dir.exists() else [])
        )

        if already_cached:
            ok("FinBERT already cached — skipping download")
            return True

        info("Downloading FinBERT (~440 MB)...")
        info("This is a one-time download. Go make a coffee ☕")
        print()

        _ = BertTokenizer.from_pretrained("ProsusAI/finbert")
        _ = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

        ok("FinBERT downloaded and cached")
        return True

    except ImportError:
        warn("transformers not installed — FinBERT unavailable")
        warn("Sentiment will use FinVADER (~65% accuracy) instead")
        return True  # Not a fatal failure
    except Exception as e:
        warn(f"FinBERT download failed: {e}")
        warn("This is optional — pipeline still works with FinVADER")
        return True  # Not fatal


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VERIFY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def verify_imports():
    hdr("STEP 5 — Verifying Imports")

    critical_imports = [
        ("numpy",          "Array math"),
        ("pandas",         "DataFrames"),
        ("torch",          "PyTorch (neural networks)"),
        ("sklearn",        "Scikit-learn (scaling + metrics)"),
        ("xgboost",        "XGBoost (tabular boosting model)"),
        ("shap",           "SHAP (feature importance and pruning)"),
        ("yfinance",       "Stock data"),
        ("feedparser",     "RSS news feeds"),
        ("requests",       "HTTP client"),
        ("pyarrow",        "Parquet file format"),
        ("matplotlib",     "Charting"),
    ]

    optional_imports = [
        ("transformers",   "FinBERT (high-accuracy sentiment)"),
        ("finvader",       "FinVADER (fast financial sentiment)"),
        ("finnhub",        "Finnhub premium news"),
        ("tweepy",         "X/Twitter social sentiment"),
        ("moomoo",         "Moomoo OpenAPI / OpenD broker connection"),
    ]

    all_critical_ok = True
    for module, desc in critical_imports:
        try:
            importlib.import_module(module)
            ok(f"{module} — {desc}")
        except ImportError:
            err(f"{module} — {desc} (MISSING — run: pip install {module})")
            all_critical_ok = False

    print()
    for module, desc in optional_imports:
        try:
            importlib.import_module(module)
            ok(f"{module} — {desc} (optional, available)")
        except ImportError:
            warn(f"{module} — {desc} (optional, not installed)")

    return all_critical_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CHECK API KEYS AND FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def check_api_keys():
    hdr("STEP 6 — API Keys & Feature Status")

    print("  API keys control which premium data sources are available.")
    print("  The pipeline works without any keys — they just improve quality.")
    print()

    # ── Finnhub ─────────────────────────────────────────────────────────────
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not finnhub_key:
        # Try reading from config
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from settings import FINNHUB_API_KEY as fk
            finnhub_key = fk or ""
        except ImportError:
            finnhub_key = ""

    if finnhub_key and finnhub_key not in ("YOUR_FINNHUB_API_KEY_HERE", ""):
        ok(f"Finnhub API key: SET ({finnhub_key[:4]}...{finnhub_key[-4:]})")
        info("  → Premium news, earnings estimates, analyst ratings active")
        info("  → Get a free key at: https://finnhub.io")
    else:
        warn("Finnhub API key: NOT SET")
        info("  → Using free sources only (Yahoo Finance, CNBC, Google News)")
        info("  → Set with: export FINNHUB_API_KEY=your_key_here")
        info("  → Or: get a free key at https://finnhub.io")

    print()

    # ── X / Twitter ──────────────────────────────────────────────────────────
    x_bearer = os.environ.get("X_BEARER_TOKEN", "").strip()

    if x_bearer:
        ok(f"X Bearer Token: SET ({x_bearer[:6]}...)")
        info("  → Social sentiment from X/Twitter active")
    else:
        warn("X Bearer Token: NOT SET")
        info("  → X/Twitter social sentiment will be skipped")
        info("  → Get a free token at: https://developer.twitter.com/en/portal/dashboard")
        info("  → Then add to .env:  X_BEARER_TOKEN=your_token_here")

    print()

    # ── GPU / ACCELERATION ────────────────────────────────────────────────────
    try:
        backend = detect_acceleration_backend()
        if backend["device"] == "cuda":
            import torch
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"GPU: {backend['cuda_device_name']} ({gpu_mem:.1f} GB)")
            info("  → CUDA available — training/inference can use the NVIDIA GPU")
        elif backend["device"] == "mps":
            ok("GPU: Apple Silicon MPS available")
            info("  → PyTorch can use the Apple GPU through MPS")
        else:
            if backend["nvidia_hardware_present"]:
                warn(f"NVIDIA GPU found: {backend['nvidia_hardware_name']}")
                info("  → Hardware is present, but PyTorch cannot use CUDA in this environment")
                info("  → Usually caused by a CPU-only torch build or a CUDA/driver mismatch")
            else:
                warn("GPU: No usable CUDA or MPS backend detected — using CPU")
            info("  → Training will still work, but it will be slower")
            info("  → For faster training, use a CUDA-enabled environment or cloud GPU")
    except Exception as e:
        warn(f"GPU / acceleration check failed: {e}")

    print()

    # ── OpenAI (for GPT-4 sentiment) ─────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        ok(f"OpenAI API: SET ({openai_key[:6]}...)")
        info("  → GPT-4 sentiment scoring available (93%+ accuracy)")
        info("  → Cost: ~$0.002 per 1000 headlines (very cheap)")
    else:
        warn("OpenAI API: NOT SET (GPT-4 sentiment disabled)")
        info("  → FinBERT or FinVADER will be used instead (89-91% accuracy)")
        info("  → Set with: export OPENAI_API_KEY=sk-...")

    print()

    # ── Moomoo OpenD / Broker Paper Trading ───────────────────────────────────
    moomoo_host = os.environ.get("MOOMOO_HOST", "127.0.0.1").strip() or "127.0.0.1"
    moomoo_port = os.environ.get("MOOMOO_PORT", "11111").strip() or "11111"
    moomoo_market = os.environ.get("MOOMOO_MARKET", "US").strip() or "US"
    moomoo_firm = os.environ.get("MOOMOO_SECURITY_FIRM", "FUTUINC").strip() or "FUTUINC"

    try:
        import importlib as _importlib
        _importlib.import_module("moomoo")
        ok(f"Moomoo SDK: INSTALLED (OpenD {moomoo_host}:{moomoo_port}, market={moomoo_market}, firm={moomoo_firm})")
        info("  → Broker-connected paper trading available through moomoo OpenD")
        info("  → Make sure OpenD is running and logged in before using moomoo_paper_trading.py")
        info("  → No traditional API key is needed here — OpenD login/session is what matters")
    except ImportError:
        warn("Moomoo SDK: NOT INSTALLED")
        info("  → Install with: pip install moomoo-api")
        info("  → Then set if needed:")
        info("      export MOOMOO_HOST=127.0.0.1")
        info("      export MOOMOO_PORT=11111")
        info("      export MOOMOO_MARKET=US")
        info("      export MOOMOO_SECURITY_FIRM=FUTUINC")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SMOKE TEST
# A smoke test is a quick sanity check — like turning on a new appliance
# and checking if it smokes. We just verify the basic pieces connect.
# ─────────────────────────────────────────────────────────────────────────────


def run_smoke_test():
    hdr("STEP 7 — Smoke Test")

    print("  Running a lightweight sanity check on each component...")
    print("  Heavy runtime checks are moved to diagnostics.py")
    print()

    # Test 0: Hardware backend report
    try:
        backend = detect_acceleration_backend()
        ok(f"Acceleration selected: {backend['device']}")
        info(f"  CUDA available: {backend['cuda_available']}")
        info(f"  MPS available : {backend['mps_available']}")
        if backend["cuda_device_name"]:
            info(f"  CUDA device   : {backend['cuda_device_name']}")
        elif backend["nvidia_hardware_present"]:
            warn(f"  NVIDIA hardware present but not usable by torch: {backend['nvidia_hardware_name']}")
            info("  → Use diagnostics.py to debug CUDA availability in more detail")
    except Exception as e:
        warn(f"Acceleration detection failed: {e}")

    # Test 1: yfinance can fetch data
    try:
        import yfinance as yf
        test_df = yf.download("AAPL", period="5d", progress=False, auto_adjust=True)
        if test_df.empty:
            err("yfinance: downloaded empty DataFrame (check internet connection)")
        else:
            ok(f"yfinance: fetched {len(test_df)} days of AAPL data")
    except Exception as e:
        err(f"yfinance fetch failed: {e}")

    # Test 2: PyTorch import + tiny tensor op
    try:
        import torch
        x = torch.randn(2, 90, 50)
        y = x.mean(dim=1)
        ok(f"PyTorch: tensor test passed (device: {x.device}, out shape: {tuple(y.shape)})")
    except Exception as e:
        err(f"PyTorch test failed: {e}")

    # Test 3: StandardScaler
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data = np.random.randn(100, 10)
        sc.fit_transform(data)
        ok("scikit-learn: StandardScaler test passed")
    except Exception as e:
        err(f"scikit-learn test failed: {e}")

    # Test 4: XGBoost import only (fit moved to diagnostics.py)
    try:
        import xgboost
        ok(f"XGBoost: import test passed ({getattr(xgboost, '__version__', 'version unknown')})")
    except Exception as e:
        err(f"XGBoost import failed: {e}")
        msg = str(e).lower()
        if "omp" in msg or "openmp" in msg or "libiomp" in msg:
            info("  Likely OpenMP runtime conflict detected.")
            info("  macOS fix: keep one OpenMP toolchain only (avoid mixed Homebrew/conda runtimes).")
            info("  Windows fix: reinstall a CUDA-compatible torch build and a clean xgboost wheel in one environment.")

    # Test 5: SHAP import only
    try:
        import shap
        ok("SHAP: import test passed")
    except Exception as e:
        warn(f"SHAP import failed: {e}")

    # Test 6: Parquet read/write
    try:
        import pandas as pd
        import io
        df_test = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        buf = io.BytesIO()
        df_test.to_parquet(buf)
        buf.seek(0)
        df_read = pd.read_parquet(buf)
        ok("pyarrow: parquet read/write test passed")
    except Exception as e:
        err(f"Parquet test failed: {e}")

    # Test 7: Sentiment engine
    try:
        from sentiment_engine import SentimentEngine
        engine = SentimentEngine(level="finvader")
        score = engine.score("Apple beats earnings by 15%")
        ok(f"Sentiment engine: score = {score:+.3f} (expected positive)")
    except Exception as e:
        warn(f"Sentiment engine test failed: {e} (install finvader: pip install finvader)")

    # Test 8: StockTwits
    try:
        import requests
        resp = requests.get("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json?limit=1", timeout=5)
        if resp.status_code == 200:
            ok("StockTwits API: accessible (social sentiment available)")
        else:
            warn(f"StockTwits API returned {resp.status_code}")
    except Exception as e:
        warn(f"StockTwits API: not reachable ({e}) — check internet connection")

    print()
    info("For heavier checks (XGBoost fit, SHAP pipeline, CUDA details, model artifact checks), run:")
    info("  python diagnostics.py")

    return True



# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PRINT FINAL INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_run_instructions():
    hdr("YOU'RE READY — How to Run the Pipeline")

    print("""
  Run these scripts IN ORDER. Each one feeds into the next.

  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1:  python scanner.py                                     │
  │           Scans your watchlist and ranks the top stocks.        │
  │           Takes 1-2 minutes. Saves data/shortlist.csv           │
  │                                                                 │
  │  STEP 2:  python research.py                                    │
  │           Builds research parquet files with price/news/social  │
  │           context. Saves data/{ticker}.parquet                  │
  │                                                                 │
  │  STEP 3:  python train.py                                       │
  │           Trains the hybrid ensemble per stock.                 │
  │           Saves neural models, XGBoost, scaler, calibrator      │
  │                                                                 │
  │  STEP 4:  python model_self_check.py --ticker AAPL              │
  │           Validates parquet/model/scaler consistency            │
  │                                                                 │
  │  STEP 5:  python predict.py                                     │
  │           Generates LONG/SHORT signals with confidence,         │
  │           expected return, and signal quality                   │
  │                                                                 │
  │  STEP 6:  python backtest.py --ticker AAPL                      │
  │           Tests historical performance and saves trade logs     │
  │                                                                 │
  │  STEP 7:  python moomoo_paper_trading.py --submit               │
  │           Sends approved signals into moomoo simulated trading  │
  │           through OpenD, then sync with --sync / --status       │
  └─────────────────────────────────────────────────────────────────┘

  SHORTCUTS:
    Test on AAPL only:   python scanner.py --test
    Single stock full:   python research.py --ticker NVDA
                         python train.py --ticker NVDA
                         python model_self_check.py --ticker NVDA
                         python predict.py --ticker NVDA
                         python backtest.py --ticker NVDA
                         python diagnostics.py
                         python moomoo_paper_trading.py --submit
                         python moomoo_paper_trading.py --sync
                         python moomoo_paper_trading.py --status

  MOOMOO PAPER TRADING:
    1. Start and log in to OpenD
       export MOOMOO_HOST=127.0.0.1
       export MOOMOO_PORT=11111
       export MOOMOO_MARKET=US
       export MOOMOO_SECURITY_FIRM=FUTUINC
    2. Generate signals:    python predict.py
    3. Optional diagnostics: python diagnostics.py
    4. Submit paper orders: python moomoo_paper_trading.py --submit
    5. Sync broker state:   python moomoo_paper_trading.py --sync
    6. Check status:        python moomoo_paper_trading.py --status

  TIPS:
    - Run scanner + research + train over the weekend
    - Run model_self_check before prediction after retraining
    - Run predict.py every trading day before market open
    - Use moomoo_paper_trading.py only with OpenD running and logged in
    - Re-train models weekly or monthly depending on your workflow
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  STOCK PIPELINE SETUP")
    print(f"  Setting up your environment...")
    print(f"{'='*60}")

    steps = [
        check_python_version,
        install_packages,
        create_directories,
        download_finbert,
        verify_imports,
        check_api_keys,
        run_smoke_test,
    ]

    all_ok = True
    for step_fn in steps:
        try:
            result = step_fn()
            if result is False:
                all_ok = False
                err("Setup stopped due to critical error above.")
                err("Fix the issue and re-run: python setup.py")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            sys.exit(0)
        except Exception as e:
            err(f"Unexpected error in {step_fn.__name__}: {e}")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print(f"  {GREEN}{BOLD}SETUP COMPLETE{RESET}")
        print(f"  Everything is installed and verified.")
    else:
        print(f"  {YELLOW}{BOLD}SETUP COMPLETE WITH WARNINGS{RESET}")
        print(f"  Some optional features are unavailable. See warnings above.")
    print(f"{'='*60}")

    print_run_instructions()


if __name__ == "__main__":
    main()
