# portfolio_manager.py — What It Does and How to Use It

## What This Script Does (Plain English)

`portfolio_manager.py` is the **risk cop**. Before any trade is actually placed, it asks: "Does adding this position break any of our risk rules?" If yes, the trade is blocked.

Think of it like a financial compliance officer who reviews every trade before it hits the market.

You don't run this script directly — `backtest.py`, `predict.py`, and the paper-trading pipeline use it.

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

# Check if the trade is approved (pass current prices + equity history)
approved = manager.approve_day([trade], price_history, equity_curve)
```

---

## Risk Rules Enforced

| Rule | Default Limit | What It Prevents |
|---|---|---|
| **Max gross exposure** | 100% | Borrowing more than portfolio value |
| **Max net exposure** | 60% | Being too directionally biased |
| **Max sector exposure** | 35% | Being too concentrated in tech/energy/etc |
| **Max single name** | 20% | One stock dominating the portfolio |
| **Max pair correlation** | 0.85 | Owning two stocks that always move together |
| **Max drawdown halt** | 15% | Hard stop: stop all trading if portfolio drops 15% from peak |
| **Trade input validation** | fail closed | Blocks invalid signal text or non-finite requested weights before exposure math |
| **Soft de-risking band** | 8–15% DD | Gradually reduces exposure as drawdown grows toward the hard stop |

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **Gross exposure** | Total value of all positions (long + short), as a % of portfolio |
| **Net exposure** | Long positions minus short positions, as a % of portfolio |
| **Sector exposure** | How much of the portfolio is in one industry (e.g., tech stocks) |
| **Correlation** | How closely two stocks move together. 1.0 = identical. 0.0 = unrelated. High correlation = hidden concentration risk |
| **Drawdown** | How far the portfolio has fallen from its peak value |
| **Soft de-risking** | Between -8% and -15% drawdown, position sizes are scaled down linearly. Protects capital before the hard halt triggers |

---

## Phase 3 Addition: Soft De-Risking Band

```
Portfolio at peak → drawdown starts
  At -8%:  gross/net budgets start shrinking
  At -15%: hard halt, no new trades approved
```

This prevents a sudden cliff from full exposure to zero. The gradual reduction gives the model time to close positions naturally rather than being forced to dump everything at once.
