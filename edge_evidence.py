"""Net performance and uncertainty from paired daily ledgers, never raw 'alpha'."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _block_samples(count, block, rng):
    starts = rng.integers(0, count, size=int(np.ceil(count / block)))
    return np.concatenate([(start + np.arange(block)) % count for start in starts])[:count]


def edge_summary(strategy: pd.Series, benchmark: pd.Series, cohort_ic: pd.Series, *, horizon=20, simulations=2000) -> dict:
    """Paired blocks keep market days together and preserve short-run dependence."""
    horizon = max(int(horizon), int(cohort_ic.attrs.get("longest_label_sessions", horizon)), 1)
    paired = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1)
    if paired.empty or paired.isna().any().any() or not np.isfinite(paired).all().all():
        return {"edge_health_status": "advisory", "reason": "paired_daily_returns_missing", "statistical_healthy": False}
    if (paired <= -1).any().any():
        raise ValueError("Return at or below -100% cannot support log growth")
    values = paired.to_numpy()
    net, bench = np.prod(1 + values, axis=0) - 1
    ics = pd.to_numeric(cohort_ic, errors="coerce").dropna().to_numpy()
    ics = ics[np.isfinite(ics)]
    out = {"net_portfolio_return_pct": float(net * 100), "benchmark_return_pct": float(bench * 100),
           "benchmark_excess_return_pct": float((net - bench) * 100), "matured_independent_cohorts": len(ics),
           "edge_health_status": "advisory", "statistical_healthy": False,
           "reason": "insufficient_matured_cohorts", "block_sessions": max(1, horizon),
           "benchmark_definition": "exposure_matched_SPY_QQQ_cash_net_costs"}
    design = np.column_stack([np.ones(len(values)), values[:, 1]])
    if len(values) > 2 and np.linalg.matrix_rank(design) == 2:
        intercept, beta = np.linalg.lstsq(design, values[:, 0], rcond=None)[0]
        out.update(regression_alpha_daily=float(intercept), regression_alpha_annualized_pct=float(intercept * 25200),
                   regression_beta=float(beta), regression_model="daily_net_return = intercept + beta * matched_benchmark; cash_rate=0")
    if len(ics) < 20 or len(values) < 20 * horizon:
        return out
    rng = np.random.default_rng(20260906)
    excess, rank_ic, alphas = [], [], []
    for _ in range(simulations):
        sample = values[_block_samples(len(values), max(1, horizon), rng)]
        compounded = np.prod(1 + sample, axis=0) - 1
        excess.append(float((compounded[0] - compounded[1]) * 100))
        # Cohorts are already non-overlapping; two-cohort blocks retain longer dependence.
        rank_ic.append(float(ics[_block_samples(len(ics), 2, rng)].mean()))
        matrix = np.column_stack([np.ones(len(sample)), sample[:, 1]])
        if np.linalg.matrix_rank(matrix) == 2:
            alphas.append(float(np.linalg.lstsq(matrix, sample[:, 0], rcond=None)[0][0] * 25200))
    excess_ci = np.quantile(excess, [.025, .975]).tolist()
    ic_ci = np.quantile(rank_ic, [.025, .975]).tolist()
    healthy = excess_ci[0] > 0 and ic_ci[0] > 0
    out.update(excess_return_ci95_pct=excess_ci, rank_ic_ci95=ic_ci,
               regression_alpha_ci95_pct=np.quantile(alphas, [.025, .975]).tolist() if alphas else None,
               statistical_healthy=healthy, edge_health_status="ok" if healthy else "advisory",
               reason="positive_lower_confidence_bounds" if healthy else "edge_inconclusive")
    return out


def non_overlapping_ic(panel, *, score="causal_score", label="forward_return_20d", as_of, horizon=20):
    """One cross-sectional IC per matured, non-overlapping decision cohort."""
    end_col = f"{label}_end_date"
    dates = sorted(pd.to_datetime(panel.date).unique())
    result = {}
    longest = horizon
    next_eligible = pd.Timestamp.min
    for date in dates:
        date = pd.Timestamp(date)
        if date < next_eligible:
            continue
        group = panel.loc[pd.to_datetime(panel.date) == date]
        endpoints = pd.to_datetime(group[end_col])
        if endpoints.isna().any() or endpoints.max() > pd.Timestamp(as_of):
            continue
        valid = group[[score, label]].replace([np.inf, -np.inf], np.nan)
        if valid.isna().any().any():
            # An incomplete cohort contributes no statistic; never quietly
            # remove the unavailable stock and improve its apparent rank IC.
            continue
        if len(valid) >= 3 and valid[score].nunique() > 1 and valid[label].nunique() > 1:
            result[date] = valid[score].corr(valid[label], method="spearman")
            from core_satellite_alpha import _nyse_sessions
            longest = max(longest, len(_nyse_sessions(date, endpoints.max())) - 1)
            next_eligible = endpoints.max() + pd.Timedelta(nanoseconds=1)
    output = pd.Series(result, dtype=float)
    output.attrs["longest_label_sessions"] = longest
    return output
