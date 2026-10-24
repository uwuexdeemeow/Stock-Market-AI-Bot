# Alpaca SDK adapter

## What it does

`alpaca_sdk_adapter.py` translates the bot's existing broker calls into the current `alpaca-py` SDK. It keeps the surrounding order-safety logic small and readable while removing the retired `alpaca-trade-api` dependency.

## How to use it

Install the project requirements, then run the paper trader normally:

```bash
python3 alpaca_paper_trading.py --status
```

The adapter is created automatically. It requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. `ALPACA_DATA_FEED` selects the explicit quote source and defaults to `iex`; choose `sip` only when the account subscription permits it. The adapter itself refuses any endpoint that is not exactly `https://paper-api.alpaca.markets` (`is_paper_endpoint`). It checks the host name, not a text match. It then creates the SDK client with `paper=True` written in the code, never read from a setting. `get_order_by_client_id` finds an order by the ID the bot gave it. The trading script uses it after a timeout to check whether an order already arrived before sending it again.

## Key terms

- **SDK:** A library used to communicate with a service such as Alpaca.
- **Adapter:** A small translator between an old interface and a new interface.
- **Request object:** A typed container describing an order or market-data query.
- **Paper account:** A simulated brokerage account that uses no real capital.
- **Data feed:** The named exchange-data source used for trades and quotes.

Model detection reads `model_fields` from the object's class, as required by
current Pydantic, avoiding deprecated instance access. This changes neither
broker requests nor order execution.
