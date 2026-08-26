# data_provider.py

## What It Does

This script gives the project one common historical-price downloader.
YahooQuery and Yahoo are normalized to adjusted OHLCV data. Raw Stooq prices
cannot be silently mixed into adjusted training history.

## How It Is Used

Other scripts import `download_history()` rather than normally running this
file directly. `STOCKBOT_PRICE_PROVIDER_ORDER` controls the allowed source
order. The returned table identifies which provider supplied it.

## Key Terms

- **OHLCV:** open, high, low, close, and volume for one market session.
- **Adjusted price:** a historical price made consistent for splits and payouts.
- **Provider:** the service that supplied the market data.
