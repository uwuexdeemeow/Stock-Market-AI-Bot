# HTTP Retry

## What it does

`http_retry.py` retries temporary web failures with bounded exponential backoff.
It is shared by data providers so brief rate limits and server errors do not
break a whole research refresh.

## How to use it

Import `retry_request` and pass the HTTP method, URL, and normal request
arguments. The result is a response when successful or the documented failure
value after all attempts. Callers must still validate response status and data.

## Key terms

- **Retry:** repeat a failed temporary operation.
- **Backoff:** wait progressively longer between attempts.
- **Rate limit:** a provider temporarily refuses too many requests.
