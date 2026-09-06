"""Pure paper sizing rules, shared by offline ledger and broker order planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PaperPolicy:
    """A frozen policy is a record of decisions, not permission to trade."""
    etf_drift: float = .03
    stock_drift: float = .01
    minimum_trade: float = 25.
    maximum_gross: float = 1.
    cash_buffer_pct: float = .005
    cash_buffer_min: float = 0.
    trailing_stop: float = .08
    drawdown_halt: float = .12
    halt_recovery_ratio: float = .5
    etfs: tuple[str, ...] = ("SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD")

    def to_dict(self) -> dict:
        return asdict(self)


def whole_share_target(equity: float, weight: float, price: float) -> int:
    """Round to whole shares with the existing paper account's rounding rule."""
    if not all(math.isfinite(x) for x in (equity, weight, price)) or price <= 0 or min(equity, weight) < 0:
        raise ValueError("Invalid equity, weight, or price for whole-share sizing")
    return round(equity * weight / price)


def drift_requires_trade(drift: float, threshold: float, *, force=False) -> bool:
    """A forced plan bypasses the drift threshold, just as paper planning does."""
    return bool(force or drift >= threshold)


def cash_reserve(equity: float, percent: float, dollars: float) -> float:
    """Keep the larger of a fixed dollar buffer and an equity percentage."""
    return max(0., dollars, equity * max(0., percent))


def rebalance_deltas(holdings: dict, targets: dict, prices: dict, equity: float, policy: PaperPolicy) -> list[tuple[str, int]]:
    """Compare actual marked holdings with targets; return sells before buys."""
    if any(not math.isfinite(w) or w < 0 for w in targets.values()) or sum(targets.values()) > policy.maximum_gross + 1e-10:
        raise ValueError("Targets exceed the frozen long-only gross limit")
    orders = []
    for ticker in sorted(set(holdings) | set(targets)):
        if not holdings.get(ticker, 0) and not targets.get(ticker, 0):
            continue
        price = float(prices[ticker])
        held = float(holdings.get(ticker, 0))
        weight = float(targets.get(ticker, 0))
        drift = abs(held * price / equity - weight)
        threshold = policy.etf_drift if ticker in policy.etfs else policy.stock_drift
        quantity = whole_share_target(equity, weight, price) - held
        if quantity != 0 and drift_requires_trade(drift, threshold) and abs(quantity * price) >= policy.minimum_trade:
            orders.append((ticker, quantity))
    return sorted(orders, key=lambda item: (item[1] > 0, item[0]))
