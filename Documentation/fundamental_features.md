# fundamental_features.py — Documentation

## What this script does

`fundamental_features.py` adds four new sets of features to the trading model — all downloadable for free. These features give the model information that pure price/technical analysis misses:

| Feature Group | What it tells the model |
|---|---|
| PEAD (earnings surprise) | Did the company beat or miss earnings? By how much? How long ago? |
| IV Rank proxy | Are options/volatility expensive or cheap right now? |
| Sector relative strength | Is this stock outperforming its sector, or just riding a sector wave? |
| Market breadth | Is the whole market healthy, or only a few big names rising? |

---

## How to use it

This file is **not run directly**. It is called automatically by `pipeline_shared.py` whenever you build training data or make a live prediction.

To rebuild your training parquet files with the new features:
```bash
python3 research.py --ticker AAPL
# or for all tickers:
python3 research.py
```

Then retrain the models:
```bash
python3 train.py
```

---

## Feature-by-feature explanation

### 1. PEAD — Post-Earnings Announcement Drift

**Why it matters:**
Research shows stocks continue drifting in the direction of an earnings surprise for days or weeks after the announcement. A stock that beat earnings by 10% tends to keep rising even after the news is out — that's PEAD.

**Features:**
- `eps_surprise_pct` — How much did actual EPS beat or miss analyst estimates? (e.g. 5.0 = beat by 5%, -3.2 = missed by 3.2%)
- `days_since_earnings` — How many calendar days since the last earnings report (capped at 120 days)
- `days_to_next_earnings` — How many calendar days until the next expected report (capped at 120 days). Stocks behave differently close to earnings (higher uncertainty = higher volatility)

**Data source:** `yf.Ticker(ticker).get_earnings_dates()` — free from Yahoo Finance

**Leakage guard:** Earnings figures are only assigned to dates AFTER the report date (strictly `price_date > earnings_date`), so the model never sees future earnings data.

---

### 2. IV Rank Proxy

**Why it matters:**
When options are expensive (high implied volatility), it often signals that large players expect an imminent move. When options are cheap, markets are complacent. Knowing where volatility sits in its recent range helps the model decide how confident to be in a prediction.

**The problem:** Historical implied volatility data costs money to download.

**Our free solution:** Use *historical volatility percentile* as a proxy. Historical volatility (HV) is computed entirely from price data and tracks implied volatility fairly closely.

**Features:**
- `iv_rank_proxy` — Where is today's 20-day historical vol in its past 252-day range? 0.0 = at a yearly low (calm), 1.0 = at a yearly high (fearful)
- `iv_hv_spread` — For live prediction only: actual ATM implied vol minus realized HV. A positive spread means options are priced rich vs recent moves (market scared). Set to 0.0 in historical training data.

---

### 3. Sector Relative Strength

**Why it matters:**
A tech stock rising 3% when the XLK tech ETF is flat is a much stronger signal than a 3% rise when XLK itself is up 4%. The first case means traders specifically want *this* stock. The second case is just the whole sector lifting everything.

**Features:**
- `ret_vs_sector_5d` — Ticker's 5-day return minus its sector ETF's 5-day return (e.g. AAPL vs XLK)
- `ret_vs_sector_20d` — Same over 20 days (longer-term relative momentum)
- `sector_vs_spy_5d` — Sector ETF's 5-day return minus SPY's 5-day return (sector rotation signal — which sector is leading?)

**Data source:** The sector ETF prices are already downloaded inside `build_multi_market()`. No new network calls needed — these features are computed from existing data.

**Sector mappings** are defined in `settings.py` under `SECTOR_MAP`. BTC-USD and some tickers without a sector entry get 0.0 for these columns.

---

### 4. Market Breadth

**Why it matters:**
In a healthy bull market, most stocks participate. In a fragile rally, only mega-cap names rise while the average stock drifts sideways or lower. Breadth features give the model a read on "market health."

**Features:**
- `breadth_rsp_vs_spy_5d` — RSP (equal-weight S&P 500) 5-day return minus SPY (cap-weighted). Positive = broad participation (many stocks rising). Negative = only large caps are rising (narrow/fragile rally)
- `breadth_iwm_vs_spy_5d` — IWM (Russell 2000 small caps) 5-day return minus SPY. Small cap leadership historically signals risk appetite is broad and healthy
- `breadth_rsp_ma20_dist` — RSP's distance from its own 20-day moving average. Positive = equal-weight market is in an uptrend
- `breadth_slope_5d` — 5-day rate of change of the RSP/SPY ratio. Rising = breadth expanding; falling = breadth contracting

**Data source:** `RSP`, `SPY`, and `IWM` from Yahoo Finance — all free. Downloaded as a group once per research run.

---

## Key concepts glossary

**EPS (Earnings Per Share):** A company's profit divided by its share count. Analysts forecast this number before each quarterly report; the difference between forecast and actual is the "earnings surprise."

**Implied Volatility (IV):** The market's expectation of future price swings, extracted from options prices. High IV = market expects big moves.

**Historical Volatility (HV):** The actual realized price swing over a past period, computed from daily returns. A free substitute for IV in historical research.

**IV Rank:** A percentile — "how expensive are options today compared to the past year?" An IV rank of 80 means options are more expensive than 80% of the time in the past year.

**Market Breadth:** The proportion of stocks participating in a market move. High breadth = healthy; low breadth = fragile.

**RSP:** The Invesco S&P 500 Equal Weight ETF. Unlike SPY where Apple is ~7% of the index, in RSP every S&P 500 company is ~0.2%. RSP vs SPY is a classic breadth measure.

**PEAD:** Post-Earnings Announcement Drift. The documented tendency for stocks to keep moving in the direction of their earnings surprise for weeks after the report.

---

## Files modified

| File | Change |
|---|---|
| `fundamental_features.py` | New file — all four feature-building functions |
| `pipeline_shared.py` | Imports new functions; calls them inside `build_research_feature_frame` and `build_live_features_with_latest_news`; adds `sector_ret20d` to `build_multi_market` |

---

## Next step after running

After rebuilding parquets with `research.py` and retraining with `train.py`, re-run the backtest:
```bash
python3 backtest.py
```
Check if Sharpe and NW t-stat improve. PEAD and sector relative strength are the highest-expected-impact features. Market breadth is a regime filter that should help the model avoid trades during narrow/fragile markets.
