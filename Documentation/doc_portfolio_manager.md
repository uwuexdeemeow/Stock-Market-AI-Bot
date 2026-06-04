# portfolio_manager.py — What It Does and How to Use It

## What This Script Does (Plain English)

`portfolio_manager.py` is the **risk cop**. Before any trade is actually placed, it asks: "Does adding this position break any of our risk rules?" If yes, the trade is blocked.

Think of it like a financial compliance officer who reviews every trade before it hits the market.

You don't run this script directly — `backtest.py` uses it while testing
single-name strategies.

---

## How to Use It (in Code)

```python
from portfolio_manager import PortfolioRiskManager, ProposedTrade
import pandas as pd

manager = PortfolioRiskManager()

# Define a proposed trade
trade = ProposedTrade(
    ticker="AAPL",
    date=pd.Timestamp("2024-01-15"),
    signal="LONG",
    confidence=0.72,
    expected_return=0.025,
    requested_position_pct=0.15,  # 15% of portfolio
)

# Check if the trade is approved (pass current prices + equity history).
# Existing holdings can be passed too, so sector/concentration gates count
# what is already in the book before approving a new trade.
approved = manager.approve_day(
    [trade],
    price_history,
    equity_curve,
    open_tickers={"MSFT"},
    open_position_weights={"MSFT": 0.20},
)
```

---

## Risk Rules Enforced

| Rule | Default Limit | What It Prevents |
|---|---|---|
| **Max gross exposure** | 100% | Borrowing more than portfolio value |
| **Max net exposure** | 80% | Being too directionally biased |
| **Max sector exposure** | 40% | Being too concentrated in tech/energy/etc |
| **Max single name** | 20% | One stock dominating the portfolio |
| **Max pair correlation** | 0.85 | Owning two stocks that always move together |
| **Max drawdown halt** | 99% | Last-resort halt; main protection comes from regime filters |
| **Trade input validation** | fail closed | Blocks invalid signal text or non-finite requested weights before exposure math |
| **Existing position accounting** | enabled when weights are passed | Counts already-held positions in sector and diversification checks |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Gross exposure** | Total value of all positions (long + short), as a % of portfolio |
| **Net exposure** | Long positions minus short positions, as a % of portfolio |
| **Sector exposure** | How much of the portfolio is in one industry (e.g., tech stocks) |
| **Correlation** | How closely two stocks move together. 1.0 = identical. 0.0 = unrelated. High correlation = hidden concentration risk |
| **Drawdown** | How far the portfolio has fallen from its peak value |
| **Regime filter** | A market-state rule that can block new entries when conditions are bad |

---

## Existing Holdings

When `backtest.py` already has open positions, it passes both:

- `open_tickers` so the same ticker is not opened twice.
- `open_position_weights` so the sector and diversification gates count what
  is already held before adding a new trade.

That prevents a hidden concentration bug where the manager respected total
gross exposure but added another stock from a sector that was already near the
sector limit.
