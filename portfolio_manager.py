"""
portfolio_manager.py — Portfolio-level exposure, sector, and correlation controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from settings import (
    MAX_GROSS_EXPOSURE, MAX_NET_EXPOSURE, MAX_SECTOR_EXPOSURE,
    MAX_SINGLE_NAME_EXPOSURE, MAX_PAIR_CORRELATION, MAX_DRAWDOWN_HALT_PCT,
    SECTOR_MAP,
)


@dataclass
class ProposedTrade:
    ticker: str
    date: pd.Timestamp
    signal: str
    confidence: float
    expected_return: float
    requested_position_pct: float


class PortfolioRiskManager:
    def __init__(
        self,
        max_gross_exposure: float = MAX_GROSS_EXPOSURE,
        max_net_exposure: float = MAX_NET_EXPOSURE,
        max_sector_exposure: float = MAX_SECTOR_EXPOSURE,
        max_single_name_exposure: float = MAX_SINGLE_NAME_EXPOSURE,
        max_pair_correlation: float = MAX_PAIR_CORRELATION,
        max_drawdown_halt_pct: float = MAX_DRAWDOWN_HALT_PCT,
    ):
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_single_name_exposure = max_single_name_exposure
        self.max_pair_correlation = max_pair_correlation
        self.max_drawdown_halt_pct = max_drawdown_halt_pct

    def _signed(self, signal: str, weight: float) -> float:
        return weight if signal == "LONG" else -weight

    def approve_day(
        self,
        candidates: List[ProposedTrade],
        price_history: Dict[str, pd.Series],
        equity_curve: pd.Series,
    ) -> List[ProposedTrade]:
        approved: List[ProposedTrade] = []

        current_peak = equity_curve.cummax().iloc[-1] if not equity_curve.empty else 1.0
        current_val = equity_curve.iloc[-1] if not equity_curve.empty else 1.0
        drawdown = (current_val / current_peak - 1.0) if current_peak > 0 else 0.0
        if drawdown <= -self.max_drawdown_halt_pct:
            return approved

        soft_dd = 0.08
        if drawdown < -soft_dd:
            scale = max(0.0, (self.max_drawdown_halt_pct + drawdown) / (self.max_drawdown_halt_pct - soft_dd))
            soft_gross = self.max_gross_exposure * scale
            soft_net = self.max_net_exposure * scale
        else:
            soft_gross = self.max_gross_exposure
            soft_net = self.max_net_exposure

        candidates = sorted(candidates, key=lambda x: (x.confidence, abs(x.expected_return)), reverse=True)

        gross = 0.0
        net = 0.0
        sector_alloc: Dict[str, float] = {}
        chosen_tickers: List[str] = []
        returns_cache: Dict[str, pd.Series] = {}

        for ticker, series in price_history.items():
            try:
                px = pd.Series(series).dropna()
                returns_cache[ticker] = px.pct_change().dropna().tail(60)
            except Exception:
                returns_cache[ticker] = pd.Series(dtype=float)

        for trade in candidates:
            weight = min(max(float(trade.requested_position_pct), 0.0), self.max_single_name_exposure)
            if weight <= 0:
                continue
            signed = self._signed(trade.signal, weight)

            if gross + abs(weight) > soft_gross:
                continue
            if abs(net + signed) > soft_net:
                continue

            sector = SECTOR_MAP.get(trade.ticker, "OTHER")
            if sector_alloc.get(sector, 0.0) + abs(weight) > self.max_sector_exposure:
                continue

            too_correlated = False
            rt = returns_cache.get(trade.ticker, pd.Series(dtype=float))
            for chosen in chosen_tickers:
                rc = returns_cache.get(chosen, pd.Series(dtype=float))
                joined = pd.concat([rt, rc], axis=1).dropna()
                if len(joined) >= 20:
                    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    if pd.notna(corr) and abs(corr) >= self.max_pair_correlation:
                        too_correlated = True
                        break
            if too_correlated:
                continue

            approved.append(trade)
            chosen_tickers.append(trade.ticker)
            gross += abs(weight)
            net += signed
            sector_alloc[sector] = sector_alloc.get(sector, 0.0) + abs(weight)

        return approved
