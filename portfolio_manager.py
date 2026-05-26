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
    MIN_DIVERSIFICATION_RATIO, DIVERSIFICATION_MIN_OBS,
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
        min_diversification_ratio: float = MIN_DIVERSIFICATION_RATIO,
    ):
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_single_name_exposure = max_single_name_exposure
        self.max_pair_correlation = max_pair_correlation
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self.min_diversification_ratio = min_diversification_ratio
        # Track tickers we've already warned about so the log isn't spammed
        # (same ticker showing up day after day with no SECTOR_MAP entry).
        self._unmapped_warned: set[str] = set()

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

    def _diversification_ratio(
        self,
        weights: Dict[str, float],
        returns_cache: Dict[str, pd.Series],
    ) -> float | None:
        """Return the portfolio's diversification ratio, or None if data is thin.

        PLAIN ENGLISH: DR = Σ(weight_i × vol_i) / portfolio_vol.

        Imagine the weighted sum of single-name vols (treating every name as
        if it moved independently of every other) divided by the portfolio's
        ACTUAL realized vol.  Two cases:

          * DR ≈ 1.0  → portfolio vol equals the sum of single-name vols,
                       meaning everything moves together.  No
                       diversification benefit.
          * DR > 1.3  → portfolio vol is meaningfully below the sum of
                       single-name vols.  Names are cancelling each other —
                       real diversification.

        Returns None when there's not enough overlapping return data to
        compute a stable covariance matrix (we don't want to reject trades
        based on noise).
        """
        if len(weights) <= 1:
            # Single-asset "portfolio" — DR is always 1 by definition; the
            # check is meaningful only with 2+ names.  Skip the gate.
            return None

        # Build a return matrix where columns are tickers and rows are days.
        # Drop dates that don't have data for every name (inner join).
        series_list: list[pd.Series] = []
        weight_list: list[float] = []
        for ticker, weight in weights.items():
            series = returns_cache.get(ticker)
            if series is None or len(series) == 0:
                return None  # missing data for one name → can't compute DR
            series_list.append(series.rename(ticker))
            weight_list.append(float(weight))

        joined = pd.concat(series_list, axis=1).dropna()
        if len(joined) < DIVERSIFICATION_MIN_OBS:
            return None

        w = np.asarray(weight_list, dtype=float)
        # Normalise weights so they sum to 1 — DR is scale-invariant but we
        # need a clean vector to multiply against the covariance matrix.
        w_sum = float(np.abs(w).sum())
        if w_sum <= 1e-12:
            return None
        w = w / w_sum

        # Daily vols and covariance matrix
        vols = joined.std().to_numpy()
        cov = joined.cov().to_numpy()
        port_var = float(w @ cov @ w)
        if port_var <= 1e-12:
            return None
        port_vol = port_var ** 0.5
        weighted_vol = float((np.abs(w) * vols).sum())
        if weighted_vol <= 1e-12:
            return None
        return weighted_vol / port_vol

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
        # Track weights of approved trades so we can compute the
        # diversification ratio incrementally as new picks are considered.
        chosen_weights: Dict[str, float] = {}
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

            signal = str(trade.signal).upper().strip()
            if signal not in {"LONG", "SHORT"}:
                continue
            try:
                requested_weight = float(trade.requested_position_pct)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(requested_weight):
                continue
            weight = min(max(requested_weight, 0.0), self.max_single_name_exposure)
            if weight <= 0:
                continue
            signed = self._signed(signal, weight)

            # Enforce gross/net caps INCLUDING existing open positions.
            if gross + abs(weight) > soft_gross:
                continue
            if abs(net + signed) > soft_net:
                continue

            sector = SECTOR_MAP.get(trade.ticker, "OTHER")
            if sector == "OTHER":
                # Unmapped ticker — log loudly so it doesn't quietly bypass
                # the sector cap.  Five "OTHER"-mapped names can stack to
                # 100% of the portfolio without tripping a 40% sector cap,
                # which defeats the diversification gate entirely.
                if trade.ticker not in self._unmapped_warned:
                    print(
                        f"  ⚠ SECTOR_MAP missing {trade.ticker} — sector cap "
                        f"may be bypassed.  Add to settings.SECTOR_MAP."
                    )
                    self._unmapped_warned.add(trade.ticker)
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

            # ── Diversification ratio gate ───────────────────────────────
            # Pair-correlation only looks at the WORST pair.  But a
            # portfolio of 5 names that are all moderately correlated
            # (no pair above 0.85, all around 0.6-0.7) still has very
            # little real diversification.  The DR gate catches that —
            # it compares the weighted sum of single-name vols against
            # the portfolio's actual vol.  Skipped when there's only
            # one name (DR is meaningless) or when min_ratio is 0.
            if self.min_diversification_ratio > 0 and chosen_weights:
                proposed_weights = dict(chosen_weights)
                proposed_weights[trade.ticker] = weight
                dr = self._diversification_ratio(proposed_weights, returns_cache)
                if dr is not None and dr < self.min_diversification_ratio:
                    # Adding this trade would crash the diversification
                    # ratio below the floor — skip it.
                    continue

            approved.append(trade)
            chosen_tickers.append(trade.ticker)
            chosen_weights[trade.ticker] = weight
            held_tickers.add(trade.ticker)   # mark as now held so it can't be entered again today
            gross += abs(weight)
            net += signed
            sector_alloc[sector] = sector_alloc.get(sector, 0.0) + abs(weight)

        return approved
