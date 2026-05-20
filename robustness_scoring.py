"""
Shared robustness objective for strategy grid selection.

The original objective stayed in Sharpe units:

    score = Sharpe - drawdown penalty - turnover penalty - instability penalty

It does not reward CAGR or total return. Those metrics can still be reported,
but they should not be the primary optimizer target.

Alpha alternative — set env var ROBUSTNESS_OBJECTIVE=alpha_vs_qqq:

    score = (alpha_vs_qqq_pct / 10) - drawdown penalty - instability penalty
            (NO turnover penalty in this objective)

Rationale: a per-fold audit on the 14-year walkforward CSV found that
`inner_mean_alpha_vs_qqq` was the only metric with positive correlation
to OOS Sharpe (+0.121), while the current Sharpe-stack scored -0.147
(slightly anti-predictive after the best-of-N selection bias).  Also,
`inner_mean_turnover` came in at +0.189 OOS-Sharpe correlation —
hinting that the turnover penalty was actively hurting selection.

Divisor of 10 brings alpha-percent into the same numeric range as
Sharpe (typical OOS aSPY is 5-30 in pct, vs Sharpe 0.5-3), so the
existing penalty magnitudes stay roughly proportional.

Reversible hybrid alternative:

    score = 0.50 * Sharpe + 0.50 * (alpha_vs_qqq_pct / 10)
            - drawdown penalty - 0.25 * turnover penalty - instability penalty

Hybrid was tested on the recent-alpha grid on 2026-05-21 and came in worse
than alpha_vs_qqq: lower compound/CAGR/alpha, higher worst turnover, and
score_predictiveness still FAIL.  Keep it as an opt-in experiment, but make
alpha_vs_qqq the default again.

Switch per-run with env vars, no code rollback needed:
    ROBUSTNESS_OBJECTIVE=hybrid        # opt-in blend experiment
    ROBUSTNESS_OBJECTIVE=sharpe        # original behavior
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable, Mapping, Sequence


# Selection objective.  Hybrid A/B on 2026-05-21 was worse than alpha_vs_qqq,
# so the default stays on the stronger recent walkforward result.  Override
# via env: ROBUSTNESS_OBJECTIVE=hybrid or sharpe for reversible comparisons.
DEFAULT_OBJECTIVE = "alpha_vs_qqq"
_VALID_OBJECTIVES = ("sharpe", "alpha_vs_qqq", "hybrid")
DEFAULT_HYBRID_SHARPE_WEIGHT = 0.50
DEFAULT_HYBRID_ALPHA_WEIGHT = 0.50
DEFAULT_HYBRID_TURNOVER_WEIGHT = 0.25


def _objective_from_env() -> str:
    raw = (os.environ.get("ROBUSTNESS_OBJECTIVE") or DEFAULT_OBJECTIVE).strip().lower()
    if raw not in _VALID_OBJECTIVES:
        raise ValueError(
            f"ROBUSTNESS_OBJECTIVE={raw!r} not in {_VALID_OBJECTIVES}"
        )
    return raw


DEFAULT_DRAWDOWN_TOLERANCE_PCT = 25.0
# Turnover penalty: anything above 400% annual turnover starts getting
# penalized.  Old value was 5000% which was basically no penalty — the
# optimizer had no reason to avoid churn.  400% = ~1.6x annual turnover
# for a 10-day holding period, reasonable for an active strategy.
#
# SPAN was 4000 (gentle: 1000% turnover → only 0.15 Sharpe penalty), but
# selection still chose churny candidates whose OOS turnover spiked above
# the run-wide 600% live-approval gate.  Tightened to 1500 so 1000%
# turnover now costs 0.40 Sharpe units — large enough to push the
# optimizer toward lower-turnover variants of the same strategy family,
# while still letting a clearly-superior high-alpha config win.
DEFAULT_TURNOVER_FREE_PCT = 400.0
DEFAULT_TURNOVER_PENALTY_SPAN_PCT = 1_500.0
DEFAULT_INSTABILITY_FLAG_PENALTY = 0.25

INSTABILITY_FLAGS = (
    "subperiod_stability_pass",
    "year_alpha_concentration_pass",
    "holdout_2023_2026_vs_qqq_pass",
    "holdout_2023_2026_vs_blend_pass",
)


def _get(row: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def robustness_score_components(
    row: Mapping[str, Any] | Any,
    *,
    drawdown_tolerance_pct: float = DEFAULT_DRAWDOWN_TOLERANCE_PCT,
    turnover_free_pct: float = DEFAULT_TURNOVER_FREE_PCT,
    turnover_penalty_span_pct: float = DEFAULT_TURNOVER_PENALTY_SPAN_PCT,
    instability_flag_penalty: float = DEFAULT_INSTABILITY_FLAG_PENALTY,
    instability_flags: tuple[str, ...] = INSTABILITY_FLAGS,
    objective: str | None = None,
) -> dict[str, float]:
    """Return the robustness score and its penalty components.

    `objective` selects the primary metric:
      - "sharpe": the original Sharpe-stack objective
      - "alpha_vs_qqq": experimental — primary metric is alpha vs QQQ
        (the only inner metric that had positive OOS correlation in the
        14-year audit).  Turnover penalty is dropped in this mode
        because inner_mean_turnover came in at +0.19 OOS-Sharpe
        correlation — penalizing it was working against selection.
      - "hybrid": opt-in experiment that balances Sharpe and QQQ alpha,
        then applies a mild turnover penalty.

    `objective=None` falls through to the env var
    `ROBUSTNESS_OBJECTIVE` (default "alpha_vs_qqq").
    """
    if objective is None:
        objective = _objective_from_env()

    sharpe = _as_float(_get(row, "sharpe"), 0.0)

    max_drawdown_pct = _as_float(_get(row, "max_drawdown_pct"), 0.0)
    drawdown_depth_pct = abs(min(max_drawdown_pct, 0.0))
    drawdown_penalty = max(0.0, drawdown_depth_pct - drawdown_tolerance_pct) / max(
        float(drawdown_tolerance_pct),
        1e-9,
    )

    turnover_pct = max(0.0, _as_float(_get(row, "turnover_pct"), 0.0))
    turnover_penalty = max(0.0, turnover_pct - turnover_free_pct) / max(
        float(turnover_penalty_span_pct),
        1e-9,
    )

    instability_penalty = 0.0
    for flag in instability_flags:
        value = _as_bool(_get(row, flag))
        if value is False:
            instability_penalty += float(instability_flag_penalty)

    alpha_vs_qqq_pct = _as_float(_get(row, "alpha_vs_qqq_pct"), 0.0)
    alpha_metric = alpha_vs_qqq_pct / 10.0

    if objective == "alpha_vs_qqq":
        # Use alpha vs QQQ as primary metric.  Divide by 10 to bring %
        # units into the same range as Sharpe so the drawdown / instability
        # penalties retain their relative magnitudes.
        primary_metric = alpha_metric
        # Skip turnover penalty in this mode — historical audit showed
        # higher turnover correlated POSITIVELY with OOS Sharpe.
        score = primary_metric - drawdown_penalty - instability_penalty
        turnover_penalty_applied = 0.0
    elif objective == "hybrid":
        primary_metric = (
            DEFAULT_HYBRID_SHARPE_WEIGHT * sharpe
            + DEFAULT_HYBRID_ALPHA_WEIGHT * alpha_metric
        )
        turnover_penalty_applied = DEFAULT_HYBRID_TURNOVER_WEIGHT * turnover_penalty
        score = (
            primary_metric
            - drawdown_penalty
            - turnover_penalty_applied
            - instability_penalty
        )
    else:  # "sharpe" (default)
        primary_metric = sharpe
        score = sharpe - drawdown_penalty - turnover_penalty - instability_penalty
        turnover_penalty_applied = turnover_penalty

    return {
        "robustness_score": round(float(score), 4),
        "drawdown_penalty": round(float(drawdown_penalty), 4),
        "turnover_penalty": round(float(turnover_penalty_applied), 4),
        "instability_penalty": round(float(instability_penalty), 4),
        "objective": objective,
        "primary_metric": round(float(primary_metric), 4),
        "sharpe_component": round(float(sharpe), 4),
        "alpha_vs_qqq_component": round(float(alpha_metric), 4),
    }


def add_cost_stress_approval_columns(
    grid,
    *,
    key_cols: Sequence[str],
    required_costs: Iterable[float],
    row_gate_col: str,
    cost_col: str = "cost_stress",
    alpha_spy_col: str = "alpha_vs_spy_pct",
    alpha_qqq_col: str = "alpha_vs_qqq_pct",
    alpha_blend_col: str = "alpha_vs_blend_pct",
    holdout_qqq_col: str = "holdout_alpha_vs_qqq_pct",
    holdout_blend_col: str = "holdout_alpha_vs_blend_pct",
):
    """Attach grouped cost-stress approval fields to a grid DataFrame.

    Cost stress is a validation gate, not a config identity.  Rows are grouped
    by the economic config fields in ``key_cols`` and approved only if every
    required cost level exists, every row-level hard gate passes, and worst-case
    alpha remains positive across the stress group.
    """
    import pandas as pd

    out = grid.copy()
    summary_cols = [
        "stress_cost_levels",
        "stress_has_required_costs",
        "stress_all_gates_pass",
        "stress_min_alpha_vs_spy_pct",
        "stress_min_alpha_vs_qqq_pct",
        "stress_min_alpha_vs_blend_pct",
        "stress_min_holdout_alpha_vs_qqq_pct",
        "stress_min_holdout_alpha_vs_blend_pct",
        "robust_cost_stress_pass",
    ]
    if out.empty:
        for col in summary_cols:
            out[col] = False if col.endswith("_pass") or col.startswith("stress_has") or col.startswith("stress_all") else None
        return out

    required = {float(v) for v in required_costs}
    missing_cols = [
        c
        for c in [
            cost_col,
            row_gate_col,
            alpha_spy_col,
            alpha_qqq_col,
            alpha_blend_col,
            holdout_qqq_col,
            holdout_blend_col,
        ]
        if c not in out.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing cost-stress grid columns: {missing_cols}")

    def _levels(series) -> str:
        values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        return ",".join(str(float(v)) for v in sorted(set(values)))

    stress = (
        out.groupby(list(key_cols), dropna=False)
        .agg(
            stress_cost_levels=(cost_col, _levels),
            stress_min_alpha_vs_spy_pct=(alpha_spy_col, "min"),
            stress_min_alpha_vs_qqq_pct=(alpha_qqq_col, "min"),
            stress_min_alpha_vs_blend_pct=(alpha_blend_col, "min"),
            stress_min_holdout_alpha_vs_qqq_pct=(holdout_qqq_col, "min"),
            stress_min_holdout_alpha_vs_blend_pct=(holdout_blend_col, "min"),
            stress_all_gates_pass=(row_gate_col, lambda s: bool(pd.Series(s).map(bool).all())),
        )
        .reset_index()
    )

    stress["stress_has_required_costs"] = stress["stress_cost_levels"].apply(
        lambda value: required.issubset({float(v) for v in str(value).split(",") if v})
    )
    stress["robust_cost_stress_pass"] = (
        stress["stress_has_required_costs"]
        & stress["stress_all_gates_pass"]
        & (pd.to_numeric(stress["stress_min_alpha_vs_spy_pct"], errors="coerce") > 0)
        & (pd.to_numeric(stress["stress_min_alpha_vs_qqq_pct"], errors="coerce") > 0)
        & (pd.to_numeric(stress["stress_min_alpha_vs_blend_pct"], errors="coerce") > 0)
        & (pd.to_numeric(stress["stress_min_holdout_alpha_vs_qqq_pct"], errors="coerce") > 0)
        & (pd.to_numeric(stress["stress_min_holdout_alpha_vs_blend_pct"], errors="coerce") > 0)
    )

    return out.merge(stress[list(key_cols) + summary_cols], on=list(key_cols), how="left")
