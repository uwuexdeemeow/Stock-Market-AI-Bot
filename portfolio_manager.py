"""
portfolio_manager.py — Portfolio-level exposure, sector, and correlation controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

import numpy as np

from settings import (
    MAX_GROSS_EXPOSURE, MAX_NET_EXPOSURE, MAX_SECTOR_EXPOSURE,
    MAX_SINGLE_NAME_EXPOSURE, MAX_PAIR_CORRELATION, MAX_DRAWDOWN_HALT_PCT,
    CORRELATION_DOWNSIDE_MIN_OBS, CORRELATION_EWM_HALFLIFE, CORRELATION_LOOKBACK_WINDOWS,
    CORRELATION_STRESS_FLOOR, CORRELATION_STRESS_VIX_THRESHOLD,
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

    def _max_abs_pair_correlation(
        self,
        left: pd.Series,
        right: pd.Series,
        vix_level: float | None = None,
    ) -> float | None:
        joined = pd.concat([left, right], axis=1).dropna()
        stress_floor = None
        if vix_level is not None:
            try:
                if float(vix_level) > CORRELATION_STRESS_VIX_THRESHOLD:
                    stress_floor = CORRELATION_STRESS_FLOOR
            except (TypeError, ValueError):
                stress_floor = None

        if len(joined) < 20:
            if stress_floor is not None:
                return abs(float(stress_floor))
            return None

        candidates: list[float] = []
        for window in CORRELATION_LOOKBACK_WINDOWS:
            sample = joined.tail(int(window))
            if len(sample) >= min(20, int(window)):
                corr = sample.iloc[:, 0].corr(sample.iloc[:, 1])
                if pd.notna(corr) and np.isfinite(corr):
                    candidates.append(abs(float(corr)))

        ewm_sample = joined.tail(60)
        if len(ewm_sample) >= 20:
            cov = ewm_sample.iloc[:, 0].ewm(
                halflife=CORRELATION_EWM_HALFLIFE, min_periods=15
            ).cov(ewm_sample.iloc[:, 1]).iloc[-1]
            std0 = ewm_sample.iloc[:, 0].ewm(
                halflife=CORRELATION_EWM_HALFLIFE, min_periods=15
            ).std().iloc[-1]
            std1 = ewm_sample.iloc[:, 1].ewm(
                halflife=CORRELATION_EWM_HALFLIFE, min_periods=15
            ).std().iloc[-1]
            denom = float(std0) * float(std1)
            if denom > 1e-12 and pd.notna(cov):
                candidates.append(abs(float(np.clip(cov / denom, -1.0, 1.0))))

        downside = joined.tail(120)
        downside = downside[(downside.iloc[:, 0] < 0) | (downside.iloc[:, 1] < 0)]
        if len(downside) >= CORRELATION_DOWNSIDE_MIN_OBS:
            corr = downside.iloc[:, 0].corr(downside.iloc[:, 1])
            if pd.notna(corr) and np.isfinite(corr):
                candidates.append(abs(float(corr)))

        if stress_floor is not None:
            candidates.append(abs(float(stress_floor)))

        return max(candidates) if candidates else None

    def approve_day(
        self,
        candidates: List[ProposedTrade],
        price_history: Dict[str, pd.Series],
        equity_curve: pd.Series,
        current_gross: float = 0.0,   # gross exposure already held in open positions
        current_net: float = 0.0,     # net long/short exposure already held
        open_tickers: set | None = None,  # tickers already held (no double-entries)
        vix_level: float | None = None,  # optional stress proxy; high VIX floors equity correlations
    ) -> List[ProposedTrade]:
        """
        Decide which proposed trades to approve for today.

        current_gross / current_net must include ALL positions still open from
        prior days so the exposure caps are enforced across the full portfolio,
        not just today's new entries.
        """
        approved: List[ProposedTrade] = []

        current_peak = equity_curve.cummax().iloc[-1] if not equity_curve.empty else 1.0
        current_val = equity_curve.iloc[-1] if not equity_curve.empty else 1.0
        drawdown = (current_val / current_peak - 1.0) if current_peak > 0 else 0.0
        if drawdown <= -self.max_drawdown_halt_pct:
            return approved

        # No soft-scaling: regime filter (VIX >= 25 or SPY < MA200) already
        # blocks new entries during bad markets.  Soft-scaling here just creates
        # a "zombie portfolio" that never quite halts but barely trades once the
        # all-time equity peak is passed.  The hard halt above handles extreme
        # drawdown protection.
        soft_gross = self.max_gross_exposure
        soft_net = self.max_net_exposure

        candidates = sorted(candidates, key=lambda x: (x.confidence, abs(x.expected_return)), reverse=True)

        # Start from existing portfolio exposure so new entries don't push us over the cap.
        gross = current_gross
        net = current_net
        # Tickers already in portfolio — don't open a second position in same stock.
        held_tickers: set = set(open_tickers) if open_tickers else set()
        sector_alloc: Dict[str, float] = {}
        chosen_tickers: List[str] = list(held_tickers)
        returns_cache: Dict[str, pd.Series] = {}

        for ticker, series in price_history.items():
            try:
                px = pd.Series(series).dropna()
                returns_cache[ticker] = px.pct_change().dropna().tail(max(CORRELATION_LOOKBACK_WINDOWS))
            except Exception:
                returns_cache[ticker] = pd.Series(dtype=float)

        for trade in candidates:
            # Skip if we already hold this ticker (no doubling up on same name).
            if trade.ticker in held_tickers:
                continue

            weight = min(max(float(trade.requested_position_pct), 0.0), self.max_single_name_exposure)
            if weight <= 0:
                continue
            signed = self._signed(trade.signal, weight)

            # Enforce gross/net caps INCLUDING existing open positions.
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
                if chosen == trade.ticker:
                    continue
                rc = returns_cache.get(chosen, pd.Series(dtype=float))
                corr = self._max_abs_pair_correlation(rt, rc, vix_level=vix_level)
                if corr is not None and corr >= self.max_pair_correlation:
                    too_correlated = True
                    break
            if too_correlated:
                continue

            approved.append(trade)
            chosen_tickers.append(trade.ticker)
            held_tickers.add(trade.ticker)   # mark as now held so it can't be entered again today
            gross += abs(weight)
            net += signed
            sector_alloc[sector] = sector_alloc.get(sector, 0.0) + abs(weight)

        return approved
