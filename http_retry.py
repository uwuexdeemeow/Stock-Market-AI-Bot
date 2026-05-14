"""
Shared HTTP retry helper with exponential backoff.

PLAIN ENGLISH: When you call an external API (Finnhub, StockTwits, Alpaca,
RSS feeds, etc.), the request can fail for temporary reasons — network
hiccup, API rate limit (HTTP 429), or a server-side error (HTTP 500-503).
Instead of immediately giving up, we retry a few times with increasing
delays between attempts.  This makes the whole pipeline more resilient
to intermittent failures.

Usage:
    from http_retry import retry_request

    # Simple GET with 3 retries and exponential backoff:
    resp = retry_request("GET", url, timeout=10)

    # With custom params and headers:
    resp = retry_request("GET", url, params={"symbol": "AAPL"},
                         headers={"User-Agent": "MyBot"}, timeout=15)

    # POST with JSON body:
    resp = retry_request("POST", url, json={"text": "hello"}, timeout=10)

    # Returns None if all retries fail (instead of raising).
"""

import logging
import time

log = logging.getLogger(__name__)

# ── Default retry settings ────────────────────────────────────────────
# These work well for financial APIs like Finnhub (60 calls/min free tier).
DEFAULT_MAX_RETRIES = 3          # total attempts = max_retries + 1 (first try + retries)
DEFAULT_BACKOFF_BASE = 1.0       # first retry waits 1s, second 2s, third 4s
DEFAULT_BACKOFF_FACTOR = 2.0     # multiply delay by this each retry (exponential)
# HTTP status codes that are worth retrying (transient errors)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def retry_request(
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retryable_codes: set[int] | None = None,
    **kwargs,
):
    """Make an HTTP request with automatic retry on transient failures.

    PLAIN ENGLISH: Tries the request up to (max_retries + 1) times.
    If the request fails due to a network error or a retryable HTTP
    status code (429 rate-limit, 500+ server error), it waits a bit
    and tries again.  The wait time doubles each retry (exponential
    backoff) so we don't hammer a struggling server.

    Parameters
    ----------
    method : str
        HTTP method: "GET", "POST", etc.
    url : str
        The URL to request.
    max_retries : int
        How many times to retry after the first failure (default 3).
    backoff_base : float
        Initial delay in seconds before first retry (default 1.0).
    backoff_factor : float
        Multiply delay by this each retry (default 2.0).
        Delays: 1s → 2s → 4s with defaults.
    retryable_codes : set[int] | None
        HTTP status codes to retry on.  Default: {429, 500, 502, 503, 504}.
    **kwargs
        Passed directly to requests.request() — supports params, headers,
        json, data, timeout, etc.

    Returns
    -------
    requests.Response or None
        The response object if successful, None if all retries exhausted.
    """
    import requests

    if retryable_codes is None:
        retryable_codes = RETRYABLE_STATUS_CODES

    # Set a default timeout so we never hang forever
    kwargs.setdefault("timeout", 15)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)

            # If the status code is retryable and we have retries left,
            # wait and try again instead of returning the error response.
            if resp.status_code in retryable_codes and attempt < max_retries:
                delay = backoff_base * (backoff_factor ** attempt)
                # For 429 (rate limit), check if server tells us how long
                # to wait via the Retry-After header.
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                log.debug(
                    "HTTP %s %s returned %d, retrying in %.1fs (attempt %d/%d)",
                    method, url[:80], resp.status_code, delay, attempt + 1, max_retries + 1,
                )
                time.sleep(delay)
                continue

            # Non-retryable status or last attempt — return whatever we got
            return resp

        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                delay = backoff_base * (backoff_factor ** attempt)
                log.debug(
                    "HTTP %s %s failed (%s), retrying in %.1fs (attempt %d/%d)",
                    method, url[:80], type(e).__name__, delay, attempt + 1, max_retries + 1,
                )
                time.sleep(delay)
            else:
                log.warning(
                    "HTTP %s %s failed after %d attempts: %s",
                    method, url[:80], max_retries + 1, e,
                )
        except Exception as e:
            # Non-transient error (e.g., invalid URL) — don't retry
            log.warning("HTTP %s %s non-retryable error: %s", method, url[:80], e)
            return None

    log.warning("HTTP %s %s: all %d retries exhausted (last error: %s)",
                method, url[:80], max_retries + 1, last_error)
    return None
