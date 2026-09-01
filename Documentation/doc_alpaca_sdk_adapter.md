# Alpaca SDK adapter

## What it does

`alpaca_sdk_adapter.py` translates the bot's existing broker calls into the current `alpaca-py` SDK. It keeps the surrounding order-safety logic small and readable while removing the retired `alpaca-trade-api` dependency.

## How to use it

Install the project requirements, then run the paper trader normally:

```bash
python3 alpaca_paper_trading.py --status
```

The adapter is created automatically. It requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. The trading script still refuses a live-money endpoint.

## Key terms

- **SDK:** A library used to communicate with a service such as Alpaca.
- **Adapter:** A small translator between an old interface and a new interface.
- **Request object:** A typed container describing an order or market-data query.
- **Paper account:** A simulated brokerage account that uses no real capital.
