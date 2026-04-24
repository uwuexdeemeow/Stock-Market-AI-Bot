# Moomoo integration

This package includes:
- `moomoo_paper_trading.py`

What changed:
- paper/backtest slippage is unified through `SLIPPAGE_BASE_PCT`
- Moomoo paper sizing uses broker-reported account equity instead of a fixed $2,000 notional

Recommended workflow:
```bash
python predict.py
python moomoo_paper_trading.py --submit
python moomoo_paper_trading.py --sync
python moomoo_paper_trading.py --status
```
