# Broker Interface

## What this script does

`broker_interface.py` defines the common order, position, and fill shapes used
by broker adapters. It also provides a small in-memory broker for tests.

An **order** is an instruction to buy or sell. Supported order types are market,
limit, fixed stop, and trailing stop. A **fixed stop** activates at one price;
a **trailing stop** follows a rising position by a chosen percentage.

## How to run it

This is a support module and normally is not run directly. Run its tests with:

```bash
python3 -m pytest tests/test_broker_interface.py -q
```

Import `Broker`, `Order`, `Position`, or `Fill` from it when building an adapter.
The `DryRunBroker` accepts orders in memory and returns simulated positions and
fills. It does not connect to Alpaca or submit a real order.

## Expected outputs

Imported classes and predictable in-memory test results. The file itself does
not write reports or trade logs.
