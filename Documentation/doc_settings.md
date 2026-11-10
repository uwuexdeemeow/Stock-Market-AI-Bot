# settings.py — What It Does and How to Configure It

## What This Script Does (Plain English)

`settings.py` is the **control panel** for the entire project. Every important number — split ratios, risk limits, model hyperparameters, API keys, file paths — lives here. Changing a value here changes it everywhere.

You should never have to change a number buried inside a script (such as the legacy `legacy/train.py` or `legacy/backtest.py`). If you want to adjust something, look here first.

---

## How to Use It

You don't run `settings.py` directly. Other scripts import from it:

```python
from settings import WATCHLIST, RETURN_HORIZON_DAYS, SLIPPAGE_BASE_PCT
```

**To configure your API keys:**
1. Copy `.env.example` to `.env`
2. Open `.env` and fill in your keys:
   ```
   OPENAI_API_KEY=sk-your-key-here
   FINNHUB_API_KEY=your-finnhub-key-here
   ```
3. `settings.py` loads `.env` automatically on import via `load_dotenv()`.

---

## Key Settings Explained

### Trading Universe
| Setting | Default | Meaning |
|---|---|---|
| `WATCHLIST` | 21 stocks | The stocks the scanner considers each day |
| `TOP_N_STOCKS` | 10 | How many make it onto the shortlist |
| `MIN_PRICE` | $5.00 | Ignore penny stocks |

`SURVIVORSHIP_FAILURE_DATES` records when each failed-company audit example
failed. The audit uses these dates to reject recycled symbols, such as the
modern SHLD ETF being mistaken for historical Sears shares.

### Time Windows
| Setting | Default | Meaning |
|---|---|---|
| `TRAIN_START` | `"2015-01-01"` | How far back to pull historical data |
| `RETURN_HORIZON_DAYS` | 5 | Predict 5 trading days into the future |
| `EMBARGO_DAYS` | 5 | Gap between train/calibration/test splits |

### Data Splits
| Setting | Default | Meaning |
|---|---|---|
| `TRAIN_CALIBRATION_SPLIT` | 0.70 | 70% of data used for training |
| `CALIBRATION_TEST_SPLIT` | 0.85 | Next 15% for calibration; last 15% for testing |

### Confidence
| Setting | Default | Meaning |
|---|---|---|
| `CONFIDENCE_TARGET_PRECISION` | 0.58 | Target 58% win rate on high-confidence signals |
| `DEFAULT_FIXED_CONFIDENCE_THRESHOLD` | 57.5 | Fallback threshold when calibration can't be computed |

### Costs (Crucial for Honest Backtesting)
| Setting | Default | Meaning |
|---|---|---|
| `SLIPPAGE_BASE_PCT` | 0.0010 | 0.10% slippage per trade |
| `COMMISSION_PER_SHARE` | $0.005 | Broker fee per share |

### Risk Limits
| Setting | Default | Meaning |
|---|---|---|
| `MAX_GROSS_EXPOSURE` | 100% | Total long + short can't exceed portfolio value |
| `MAX_NET_EXPOSURE` | 80% | Long minus short can't exceed 80% |
| `MAX_SECTOR_EXPOSURE` | 40% | No single sector can exceed 40% |
| `MAX_SINGLE_NAME_EXPOSURE` | 20% | No single stock can exceed 20% |
| `MAX_DRAWDOWN_HALT_PCT` | 99% | Effectively off. Used only by the old backtest risk manager (`portfolio_manager.py`) and the `monitor.py` alert. **Not the paper account's kill switch** (see below). |

**Where the real kill switch is.** The Alpaca paper account is protected by
two other settings, which are not in `settings.py`:

| Setting | Read by | Rule V value | What it does |
|---|---|---|---|
| `ALPACA_DD_HALT_PCT` | `alpaca_paper_trading.py` | 40% | Account more than 40% below its peak: sell everything, block new orders. |
| `GUARD_PNL_HALT_PCT` | `execution_guard.py` | −12% | Account down 12% in one day: sell everything. |

Both values are set in `.github/workflows/daily_paper_trading.yml`. The
reasons for 40% and −12% are in `V_RULE_FORWARD_TEST.md`. The test
`tests/test_v_rule_drawdown_halt.py` fakes a crash and checks the halt fires.

### VIX Regime Thresholds
| Setting | Default | Meaning |
|---|---|---|
| `VIX_HIGH_THRESHOLD` | 25.0 | Above this = "defensive" regime; reduce positions |
| `VIX_EXTREME_THRESHOLD` | 35.0 | Above this = "crisis" regime; minimum positions |

---

## Key Concept: VIX

VIX is the "fear index" — it measures how much the options market expects the S&P 500 to move in the next 30 days. VIX < 20 = calm. VIX 20–35 = elevated concern. VIX > 35 = panic (2008, March 2020).
