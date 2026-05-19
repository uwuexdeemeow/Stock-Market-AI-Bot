"""
options_iv_provider.py -- Real implied volatility data from Tradier API.

WHY THIS FILE EXISTS
--------------------
The system previously used a "proxy" IV rank computed from historical
volatility (HV).  That's like guessing the weather by looking at yesterday's
temperature -- it correlates, but it's not the real thing.  REAL implied
volatility comes from options markets and reflects what traders are ACTUALLY
paying for protection.  It's forward-looking, not backward-looking.

This module fetches real IV data from Tradier's free API (included with any
Tradier brokerage account, even a $0 one).  It provides:
  - iv_atm:           at-the-money implied volatility (annualized)
  - iv_rank:          where current IV sits vs its 252-day range (0-1)
  - put_call_ratio:   put OI / call OI (>1 = bearish sentiment)
  - iv_skew:          put IV - call IV (positive = demand for downside protection)

SETUP
-----
1. Open a free Tradier account at https://tradier.com
2. Get your API token at https://web.tradier.com/user/api
3. Set the environment variable:
       export TRADIER_API_TOKEN="your-token-here"
   For sandbox/testing (delayed data), also set:
       export TRADIER_USE_SANDBOX="1"

PLAIN ENGLISH
-------------
  - "Implied volatility" = how much the options market expects a stock to
    move.  High IV means traders expect big moves (scared or excited).
  - "IV rank" = is IV currently high or low compared to the past year?
    0.0 = yearly low (calm), 1.0 = yearly high (turbulent).
  - "Put/call ratio" = are more people buying puts (downside protection)
    or calls (upside bets)?  Above 1.0 = more puts = bearish.
  - "IV skew" = are puts more expensive than calls?  Positive skew means
    the market is paying more for crash protection.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# Tradier API token -- get one free at https://web.tradier.com/user/api
TRADIER_API_TOKEN = os.environ.get("TRADIER_API_TOKEN", "").strip()

# Use sandbox (delayed data, no real account needed for testing)
TRADIER_USE_SANDBOX = os.environ.get("TRADIER_USE_SANDBOX", "").strip().lower() in {
    "1", "true", "yes",
}

# Base URL: sandbox for testing, live for real data
TRADIER_BASE_URL = (
    "https://sandbox.tradier.com/v1"
    if TRADIER_USE_SANDBOX
    else "https://api.tradier.com/v1"
)

# Cache directory for IV history (needed to compute IV rank over 252 days)
# We store daily ATM IV snapshots so we can build the rank without
# needing 252 API calls on the first run.
try:
    from settings import SIGNAL_DIR
    IV_CACHE_DIR = Path(SIGNAL_DIR) / "iv_cache"
except ImportError:
    IV_CACHE_DIR = Path("signals") / "iv_cache"


def _tradier_get(endpoint: str, params: dict) -> dict | None:
    """Make a GET request to the Tradier API.

    PLAIN ENGLISH: This is the low-level function that talks to Tradier's
    servers.  It handles authentication (the API token), retries on
    temporary failures, and returns the JSON response as a Python dict.
    """
    if not TRADIER_API_TOKEN:
        return None

    import requests

    url = f"{TRADIER_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_TOKEN}",
        "Accept": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited -- wait and retry
                time.sleep(2 ** attempt)
                continue
            else:
                logger.debug("Tradier %s returned %d", endpoint, resp.status_code)
                return None
        except Exception as e:
            logger.debug("Tradier request error: %s", e)
            if attempt < 2:
                time.sleep(1)
    return None


def _get_nearest_expiration(ticker: str) -> str | None:
    """Get the nearest options expiration date for a ticker.

    PLAIN ENGLISH: Options expire on specific dates (usually Fridays).
    We want the nearest expiration that's at least 7 days away -- too close
    and the options are about to expire, making their IV unreliable.
    """
    data = _tradier_get("markets/options/expirations", {"symbol": ticker})
    if not data:
        return None

    expirations = data.get("expirations", {})
    if not expirations:
        return None

    # Handle both list and dict responses
    dates = expirations.get("date", [])
    if isinstance(dates, str):
        dates = [dates]
    if not dates:
        return None

    today = datetime.now()
    min_days = 7  # skip expiration that are too close (noisy)
    max_days = 60  # don't go too far out (less liquid)

    for exp_str in sorted(dates):
        try:
            exp_date = datetime.strptime(str(exp_str), "%Y-%m-%d")
            days_out = (exp_date - today).days
            if min_days <= days_out <= max_days:
                return str(exp_str)
        except ValueError:
            continue

    # Fallback: just take the first expiration
    return str(dates[0]) if dates else None


def _get_options_chain(ticker: str, expiration: str) -> list[dict]:
    """Fetch the full options chain for a ticker and expiration.

    PLAIN ENGLISH: An options chain is the list of all call and put
    contracts available for a stock at a given expiration date, at
    different strike prices.  Each contract has a price, IV, greeks, etc.
    We request greeks=true so Tradier includes IV data from ORATS.
    """
    data = _tradier_get(
        "markets/options/chains",
        {"symbol": ticker, "expiration": expiration, "greeks": "true"},
    )
    if not data:
        return []

    options = data.get("options", {})
    if not options:
        return []

    chain = options.get("option", [])
    if isinstance(chain, dict):
        chain = [chain]
    return chain


def _get_stock_price(ticker: str) -> float | None:
    """Get current stock price from Tradier quotes endpoint.

    PLAIN ENGLISH: We need the current price to find the at-the-money
    strike (the strike closest to where the stock is trading now).
    """
    data = _tradier_get("markets/quotes", {"symbols": ticker})
    if not data:
        return None

    quotes = data.get("quotes", {})
    quote = quotes.get("quote", {})
    if isinstance(quote, list):
        quote = quote[0] if quote else {}

    price = quote.get("last") or quote.get("close") or quote.get("prevclose")
    return float(price) if price else None


def fetch_live_iv_features(ticker: str) -> dict[str, float]:
    """Fetch real implied volatility features from Tradier for one ticker.

    PLAIN ENGLISH: This is the main function.  It:
    1. Gets the nearest options expiration
    2. Fetches the full options chain
    3. Finds the at-the-money (ATM) call and put
    4. Extracts their implied volatility
    5. Computes put/call ratio from open interest
    6. Computes IV skew (put IV - call IV)
    7. Computes IV rank from cached daily history

    Returns a dict of feature_name -> value.  If anything fails, returns
    neutral defaults so the model still works.
    """
    # Neutral defaults -- returned if API is unavailable.
    # iv_source tells you WHICH path produced the IV: 'default' (no real
    # data), 'tradier_mid_iv' (preferred), 'tradier_smv_vol' (ORATS
    # smoothed fallback), 'tradier_greeks_mid_iv', 'tradier_bid_iv'
    # (last-resort, illiquid contracts), or 'no_token' (API key missing).
    # This is logged to logs/options_iv_source.csv so you can audit why
    # IV looks weird on a given day without re-running the live fetch.
    result = {
        "iv_atm": 0.25,          # 25% vol is roughly average for large-caps
        "iv_rank": 0.50,         # middle of range
        "put_call_ratio": 1.0,   # neutral
        "iv_skew": 0.0,          # no skew
        "iv_has_real_data": 0.0, # flag: 0 = defaults, 1 = real data
        "iv_source": "default",  # which path produced the value (diagnostic)
    }

    if not TRADIER_API_TOKEN:
        result["iv_source"] = "no_token"
        _record_iv_source(ticker, result)
        return result

    try:
        # Step 1: Get current stock price
        price = _get_stock_price(ticker)
        if price is None or price <= 0:
            return result

        # Step 2: Get nearest expiration (7-60 days out)
        expiration = _get_nearest_expiration(ticker)
        if not expiration:
            return result

        # Step 3: Fetch options chain with greeks
        chain = _get_options_chain(ticker, expiration)
        if not chain:
            return result

        # Step 4: Separate calls and puts
        calls = [c for c in chain if str(c.get("option_type", "")).lower() == "call"]
        puts = [c for c in chain if str(c.get("option_type", "")).lower() == "put"]

        if not calls or not puts:
            return result

        # Step 5: Find ATM strike (closest to current price)
        all_strikes = sorted(set(
            float(c.get("strike", 0)) for c in chain if c.get("strike")
        ))
        if not all_strikes:
            return result

        atm_strike = min(all_strikes, key=lambda s: abs(s - price))

        # Step 6: Get ATM call and put
        atm_calls = [c for c in calls if float(c.get("strike", 0)) == atm_strike]
        atm_puts = [c for c in puts if float(c.get("strike", 0)) == atm_strike]

        if not atm_calls or not atm_puts:
            return result

        atm_call = atm_calls[0]
        atm_put = atm_puts[0]

        # Step 7: Extract IV from the chain
        # Tradier provides mid_iv (mid implied vol) and smv_vol (ORATS smoothed)
        # Prefer mid_iv, fall back to smv_vol, then greeks, then bid_iv.
        # _iv_with_source returns (value, source_tag) so we can record
        # which fallback path actually fired — useful for debugging when
        # IV readings look odd (e.g. an illiquid bid_iv-only fallback
        # often produces noisy values).
        call_iv, call_src = _iv_with_source(atm_call)
        put_iv, put_src = _iv_with_source(atm_put)

        if call_iv and put_iv and call_iv > 0 and put_iv > 0:
            # ATM IV = average of call and put IV
            atm_iv = (call_iv + put_iv) / 2.0
            result["iv_atm"] = round(float(np.clip(atm_iv, 0.01, 5.0)), 4)
            result["iv_skew"] = round(float(np.clip(put_iv - call_iv, -1.0, 1.0)), 4)
            result["iv_has_real_data"] = 1.0
            # Record the WORST-quality of the two paths (mid_iv > smv_vol >
            # greeks > bid_iv) so the diagnostic flags any contract that
            # had to use a lower-quality field.
            result["iv_source"] = call_src if _IV_SOURCE_RANK[call_src] >= _IV_SOURCE_RANK[put_src] else put_src

        # Step 8: Put/call ratio from open interest across all strikes
        total_call_oi = sum(int(c.get("open_interest", 0) or 0) for c in calls)
        total_put_oi = sum(int(p.get("open_interest", 0) or 0) for p in puts)
        if total_call_oi > 0:
            result["put_call_ratio"] = round(
                float(np.clip(total_put_oi / total_call_oi, 0.1, 5.0)), 3
            )

        # Step 9: Compute IV rank from cached history
        if result["iv_has_real_data"] > 0:
            result["iv_rank"] = _compute_iv_rank(ticker, result["iv_atm"])

    except Exception as e:
        logger.debug("fetch_live_iv_features(%s) error: %s", ticker, e)

    _record_iv_source(ticker, result)
    return result


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if it's missing or invalid.

    PLAIN ENGLISH: API responses sometimes have None, empty strings, or
    weird values.  This safely converts to float without crashing.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) and f > 0 else None
    except (ValueError, TypeError):
        return None


# Lower rank = higher quality.  Used to pick the WORST of the call/put
# source tags for the diagnostic — so if a contract used bid_iv (rank 4)
# the whole result is flagged that way.
_IV_SOURCE_RANK = {
    "tradier_mid_iv": 0,
    "tradier_smv_vol": 1,
    "tradier_greeks_mid_iv": 2,
    "tradier_bid_iv": 3,
    "default": 4,
    "no_token": 4,
}


def _iv_with_source(contract: dict) -> tuple[float | None, str]:
    """Return (iv_value, source_tag) for one option contract.

    Walks the same mid_iv → smv_vol → greeks → bid_iv cascade as before,
    but reports which step actually produced a usable number.  This is
    the diagnostic that audit #22 was asking for — when IV looks weird,
    you can check logs/options_iv_source.csv to see whether the value
    came from a clean mid_iv or a noisy bid_iv last-resort.
    """
    v = _safe_float(contract.get("mid_iv"))
    if v:
        return v, "tradier_mid_iv"
    v = _safe_float(contract.get("smv_vol"))
    if v:
        return v, "tradier_smv_vol"
    v = _safe_float((contract.get("greeks") or {}).get("mid_iv"))
    if v:
        return v, "tradier_greeks_mid_iv"
    v = _safe_float(contract.get("bid_iv"))
    if v:
        return v, "tradier_bid_iv"
    return None, "default"


def _record_iv_source(ticker: str, result: dict) -> None:
    """Append one row to logs/options_iv_source.csv for diagnostics.

    PLAIN ENGLISH: One row per (ticker, day) recording which path
    produced today's IV.  Lets you grep historical issues — e.g. "why
    was AAPL's IV 0.95 last Tuesday?" → check this CSV and see it fell
    through to bid_iv (last-resort fallback on illiquid quotes).
    Failures here are silently ignored — diagnostics must never break
    the live signal path.
    """
    try:
        import csv
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        log_path = _Path("logs") / "options_iv_source.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "ticker", "iv_source",
                            "iv_atm", "iv_has_real_data"])
            w.writerow([
                _dt.now().isoformat(timespec="seconds"),
                ticker.upper(),
                result.get("iv_source", "unknown"),
                result.get("iv_atm", 0.0),
                int(result.get("iv_has_real_data", 0.0)),
            ])
    except Exception:
        pass


def _compute_iv_rank(ticker: str, current_iv: float) -> float:
    """Compute IV rank from cached daily IV history.

    PLAIN ENGLISH: IV rank tells you where current IV sits relative to
    the past year.  If IV was between 15% and 45% over the past year
    and it's currently at 30%, the IV rank is:
        (30 - 15) / (45 - 15) = 0.50 (middle of the range)

    We cache daily IV snapshots to a local file so we don't need 252
    API calls.  Each day we append the current IV to the cache.
    """
    IV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = IV_CACHE_DIR / f"{ticker.upper()}_iv_history.json"

    # Load existing history
    history: list[dict] = []
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Append today's IV (one entry per calendar day)
    today_str = datetime.now().strftime("%Y-%m-%d")
    if not history or history[-1].get("date") != today_str:
        history.append({"date": today_str, "iv": round(current_iv, 4)})

    # Keep only last 300 entries (~1.2 years of trading days)
    history = history[-300:]

    # Save updated history
    try:
        with open(cache_path, "w") as f:
            json.dump(history, f)
    except Exception:
        pass

    # Compute IV rank from the history
    iv_values = [float(h["iv"]) for h in history if h.get("iv")]
    if len(iv_values) < 20:
        # Not enough history yet -- return neutral
        return 0.50

    iv_min = min(iv_values)
    iv_max = max(iv_values)
    iv_range = iv_max - iv_min

    if iv_range < 0.001:
        return 0.50

    rank = (current_iv - iv_min) / iv_range
    return round(float(np.clip(rank, 0.0, 1.0)), 3)


def batch_fetch_iv_features(tickers: list[str], delay: float = 0.5) -> dict[str, dict]:
    """Fetch IV features for multiple tickers with rate limiting.

    PLAIN ENGLISH: Tradier doesn't have a strict rate limit for market
    data, but we add a small delay between tickers to be polite and
    avoid getting throttled.  For 147 tickers at 0.5s delay, this takes
    ~75 seconds -- fine for a once-daily batch run.

    Returns a dict of ticker -> features dict.
    """
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_live_iv_features(ticker)
        if i < len(tickers) - 1 and delay > 0:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            logger.info("IV fetch progress: %d/%d tickers", i + 1, len(tickers))
    return results


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    if not TRADIER_API_TOKEN:
        print("No TRADIER_API_TOKEN set. Set it to test:")
        print("  export TRADIER_API_TOKEN='your-token'")
        print("  export TRADIER_USE_SANDBOX='1'  # optional, for testing")
        print()
        print("Showing neutral defaults:")
        result = fetch_live_iv_features("AAPL")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
    else:
        print(f"Using {'SANDBOX' if TRADIER_USE_SANDBOX else 'LIVE'} API")
        print()

        for ticker in ["AAPL", "NVDA", "SPY", "TQQQ"]:
            print(f"--- {ticker} ---")
            result = fetch_live_iv_features(ticker)
            for k, v in sorted(result.items()):
                print(f"  {k}: {v}")
            print()
