"""
factor_decay_monitor.py — recent health checks for the core-satellite overlay.

The production signal remains unchanged. This monitor answers whether the
factor overlay still has recent cross-sectional IC and realized overlay alpha,
so paper/real-capital gates can distinguish a healthy backtest from a decaying
edge.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_backtest import attach_scores, load_factor_panel, load_feature_specs, load_prediction_scores
from backtest import _newey_west_tstat
import core_satellite_alpha as core
from settings import LOG_DIR, SIGNAL_DIR
from safe_io import atomic_write_csv, atomic_write_json
from validation_bundle import add_validation_context


OUT_CSV = Path(SIGNAL_DIR) / "factor_decay_monitor.csv"
OUT_JSON = Path(LOG_DIR) / "factor_decay_monitor.json"
TRADES_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_trades.csv"
METRICS_PATH = Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json"
MIN_WEAK_OVERLAY_ALPHA_PCT = -0.1  # warn if overlay alpha below -0.1% (noise floor)
# Block threshold (negative). The old value of 0.0 fired "block" on any
# negative cumulative overlay alpha, no matter how thin the sample.  With
# holding_days=20 the 60-day window often has only ~2 trades, so one bad
# month flipped the cumulative negative and tripped the gate — a false
# positive.  Require a meaningfully large drawdown PLUS a minimum sample.
OVERLAY_ALPHA_BLOCK_THRESHOLD_PCT = -1.5
# Minimum overlay trade count before "block" can fire on a window.  Set
# to 4 so the 60-day window (which usually has only 2 trades at 20-day
# holding) can still surface a "warning" early indicator but cannot
# escalate to "block" off a single bad trade.  120-day windows naturally
# have ≥4 trades and remain the canonical block source.
MIN_BLOCK_OVERLAY_PERIODS = 4
# Minimum overlay trade count before a *warning* fires on overlay alpha.
# With only 1–2 trades the cumulative is dominated by single-trade noise,
# so we suppress the overlay-alpha-based warning and let the row fall
# through to the IC-based "advisory" check.  This preserves IC as the
# early-warning signal without letting a single trade trigger a gate.
MIN_WARNING_OVERLAY_PERIODS = 3

# ── Notification status ranking ──────────────────────────────────────────
# Higher number = worse health.  Used to compute transitions (did edge
# weaken since last run? recover?) so we only alert on state changes
# instead of spamming the same warning every run.
_STATUS_SEVERITY = {"ok": 0, "advisory": 1, "warning": 2, "block": 3}


def _previous_status() -> str:
    """Return the edge_health_status from the most recent JSON, or 'ok'."""
    try:
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8", errors="replace"))
        return str(prev.get("edge_health_status") or "ok")
    except (FileNotFoundError, ValueError, OSError):
        return "ok"


def _maybe_notify(new_status: str, prev_status: str, payload: dict) -> None:
    """Send a Telegram/desktop alert when factor edge health changes.

    PLAIN ENGLISH: This script used to write the decay report to a JSON
    file with no notification — meaning a "block" state could sit in the
    file for days unnoticed until someone checked the dashboard.  Now any
    status change (or arrival at block) fires send_alert() so the user
    finds out on their phone the same day.
    """
    new_rank = _STATUS_SEVERITY.get(new_status, 0)
    prev_rank = _STATUS_SEVERITY.get(prev_status, 0)

    # Map status → alert priority.  "block" always escalates to critical
    # even if last run was also a block, so a persistent problem is
    # surfaced again (in case the user missed yesterday's alert).
    if new_status == "block":
        priority = "critical"
        title = "Factor decay BLOCK"
        body_prefix = "Real-capital gate is BLOCKED."
    elif new_rank > prev_rank:
        # Health worsened (e.g. ok → warning, advisory → block)
        priority = "warning"
        title = f"Factor decay: {prev_status} → {new_status}"
        body_prefix = "Factor edge health deteriorated."
    elif new_rank < prev_rank:
        # Health recovered (no alert spam if we're already healthy)
        priority = "info"
        title = f"Factor decay: {prev_status} → {new_status} (recovered)"
        body_prefix = "Factor edge recovered."
    else:
        # No change and not a block — skip the alert entirely.
        return

    # Build a compact message body — the user sees this on their phone.
    body_lines = [body_prefix]
    reason = str(payload.get("reason") or "")
    if reason:
        body_lines.append(reason)
    for row in payload.get("rows", []):
        lookback = row.get("lookback_days")
        ic = row.get("daily_ic_mean")
        alpha = row.get("overlay_alpha_sum_pct")
        if lookback is None:
            continue
        try:
            body_lines.append(
                f"{int(lookback)}d: IC={float(ic):+.3f}, overlay_α={float(alpha):+.2f}%"
            )
        except (TypeError, ValueError):
            continue
    body = "\n".join(body_lines)

    try:
        from notifications import send_alert
        send_alert(body, title=title, priority=priority)
    except Exception as exc:
        # Notifications must never crash the monitor — log and move on.
        print(f"⚠ Failed to send factor decay notification: {exc}")


def edge_health_status(row: dict | pd.Series) -> str:
    """Classify live factor edge health without overreacting to full-rank IC alone.

    PLAIN ENGLISH: We have multiple recency lookbacks (60d, 120d, ...).
    Each one returns one of {pass, advisory, warning, block}.  The
    aggregate is the worst.  "block" must require enough evidence —
    otherwise a single bad trade in a thin 60-day window (~2 trades at
    20-day holding) is enough to halt live trading on noise.  The
    minimum-period gate below prevents that.
    """
    rank_ic = pd.to_numeric(pd.Series([row.get("daily_ic_mean")]), errors="coerce").iloc[0]
    top_excess = pd.to_numeric(pd.Series([row.get("top_bucket_excess_return_pct")]), errors="coerce").iloc[0]
    overlay_alpha = pd.to_numeric(pd.Series([row.get("overlay_alpha_sum_pct")]), errors="coerce").iloc[0]
    try:
        overlay_periods = int(float(row.get("overlay_periods", 0) or 0))
    except (TypeError, ValueError):
        overlay_periods = 0
    # Two-stage block check:
    #   1. Sample size must clear MIN_BLOCK_OVERLAY_PERIODS — otherwise the
    #      cumulative overlay alpha is too sample-dependent to act on.
    #   2. Magnitude must be more negative than OVERLAY_ALPHA_BLOCK_THRESHOLD_PCT
    #      so a tiny negative drift doesn't trip a paper-only restriction.
    if (
        pd.notna(overlay_alpha)
        and overlay_periods >= MIN_BLOCK_OVERLAY_PERIODS
        and float(overlay_alpha) < OVERLAY_ALPHA_BLOCK_THRESHOLD_PCT
    ):
        return "block"
    if pd.isna(top_excess) or float(top_excess) <= 0.0:
        return "warning"
    # Only fire an overlay-alpha warning when the sample is large enough
    # for the cumulative to mean something.  With <3 trades in the window
    # the alpha is one-trade dominated and would create false warnings.
    if (
        pd.notna(overlay_alpha)
        and overlay_periods >= MIN_WARNING_OVERLAY_PERIODS
        and float(overlay_alpha) <= MIN_WEAK_OVERLAY_ALPHA_PCT
    ):
        return "warning"
    if pd.notna(rank_ic) and float(rank_ic) < 0.0:
        return "advisory"
    return "pass"


def aggregate_edge_health_status(rows: list[dict]) -> str:
    statuses = [str(row.get("edge_health_status", "warning")) for row in rows]
    for status in ("block", "warning", "advisory"):
        if status in statuses:
            return status
    return "pass"


def _selected_config() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit("Missing signals/core_satellite_alpha_metrics.json. Run core_satellite_alpha.py first.")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8", errors="replace"))
    keys = (
        "regime_mode",
        "score_source",
        # These fields may not change the IC calculation below, but they do
        # identify which live strategy this evidence belongs to.
        "shape",
        "weighting",
        "holding_days",
        "overlay_gross",
        "regime_ma_window",
        "regime_high_vol",
        "high_vol_mode",
        "tqqq_weight",
        "risk_control_mode",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _safe_spearman(group: pd.DataFrame, score_col: str, return_col: str) -> float:
    sub = group[[score_col, return_col]].dropna()
    if len(sub) < 5 or sub[score_col].nunique() < 2 or sub[return_col].nunique() < 2:
        return np.nan
    return float(sub[score_col].corr(sub[return_col], method="spearman"))


def _non_overlapping_rebalance_panel(scored: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    """Keep only the exact non-overlapping dates used by the strategy.

    PLAIN ENGLISH: a 20-day future return observed every day overlaps the next
    19 observations. Treating all of those as independent makes the sample look
    much larger than it is. The strategy rebalances every 20 trading dates, so
    the decay monitor now measures those same independent cohorts.
    """
    if scored.empty:
        return scored.copy()
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(scored["date"], errors="coerce").dropna().unique()))
    step = max(1, int(holding_days))
    cohort_dates = set(pd.Timestamp(value) for value in dates[::step])
    return scored[pd.to_datetime(scored["date"], errors="coerce").isin(cohort_dates)].copy()


def _decile_shape(recent: pd.DataFrame) -> dict:
    """Describe whether higher score deciles generally earn higher returns."""
    decile_returns: dict[int, list[float]] = {decile: [] for decile in range(1, 11)}
    for _date, group in recent.groupby("date"):
        usable = group[["_monitor_score", "_monitor_return"]].dropna().copy()
        if len(usable) < 10 or usable["_monitor_score"].nunique() < 10:
            continue
        usable["_decile"] = pd.qcut(
            usable["_monitor_score"].rank(method="first"),
            10,
            labels=False,
        ) + 1
        for decile, values in usable.groupby("_decile")["_monitor_return"]:
            decile_returns[int(decile)].extend(float(value) for value in values)
    means = {
        str(decile): round(float(np.mean(values)) * 100.0, 4) if values else None
        for decile, values in decile_returns.items()
    }
    populated = [(int(decile), value) for decile, value in means.items() if value is not None]
    monotonicity = (
        pd.Series([value for _decile, value in populated]).corr(
            pd.Series([decile for decile, _value in populated]), method="spearman"
        )
        if len(populated) >= 3
        else np.nan
    )
    return {
        "decile_mean_return_pct": means,
        "decile_monotonicity_spearman": round(float(monotonicity), 4) if pd.notna(monotonicity) else None,
    }


def _regime_score_panel(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = panel.copy()
    holding_days = int(config.get("holding_days", core.HORIZON_DAYS))
    return_col = "forward_return" if holding_days == core.HORIZON_DAYS else f"forward_return_{holding_days}d"
    if return_col not in out.columns:
        return_col = "forward_return"
    dates = pd.DatetimeIndex(sorted(out["date"].unique()))
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days) for dt in dates])
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in core.REGIME_PRESETS:
        regime_indicators = core._load_regime_indicators(dates, exit_dates, config)

    regime_by_date: dict[pd.Timestamp, str] = {}
    score_by_date: dict[pd.Timestamp, str] = {}
    dummy_weights = {"SPY": 0.0, "QQQ": 1.0}
    for dt in dates:
        eval_config = {**config, "core_weights": dummy_weights, "core_gross": 1.0, "overlay_gross": 0.0}
        regime, _core_weights, _core_gross, _overlay_gross = core._resolve_allocation(pd.Timestamp(dt), eval_config, regime_indicators)
        regime_by_date[pd.Timestamp(dt)] = regime
        score_by_date[pd.Timestamp(dt)] = core._score_col_for_regime(str(config.get("score_source", "regime_adaptive")), regime)

    out["_monitor_regime"] = out["date"].map(regime_by_date)
    out["_monitor_score_col"] = out["date"].map(score_by_date)
    score = pd.Series(np.nan, index=out.index, dtype=float)
    for score_col in sorted(set(score_by_date.values())):
        if score_col not in out.columns:
            continue
        mask = out["_monitor_score_col"].eq(score_col)
        score.loc[mask] = pd.to_numeric(out.loc[mask, score_col], errors="coerce")
    out["_monitor_score"] = score
    out["_monitor_return"] = pd.to_numeric(out[return_col], errors="coerce")
    return out.dropna(subset=["_monitor_score", "_monitor_return"])


def _ic_summary(scored: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = scored[(scored["date"] >= start) & (scored["date"] <= as_of)].copy()
    if recent.empty:
        return {
            "lookback_days": lookback_days,
            "daily_ic_mean": np.nan,
            "daily_ic_positive_rate": np.nan,
            "top_bucket_excess_return_pct": np.nan,
            "daily_ic_newey_west_tstat": 0.0,
            "cross_sections": 0,
            "observations": 0,
            **_decile_shape(recent),
        }
    daily_ic = recent.groupby("date").apply(lambda g: _safe_spearman(g, "_monitor_score", "_monitor_return"))
    top_returns = []
    for _dt, group in recent.groupby("date"):
        if len(group) < 10:
            continue
        ranks = group["_monitor_score"].rank(pct=True)
        top = group.loc[ranks >= 0.90, "_monitor_return"]
        rest = group.loc[ranks < 0.90, "_monitor_return"]
        if not top.empty and not rest.empty:
            top_returns.append(float(top.mean() - rest.mean()))
    return {
        "lookback_days": lookback_days,
        "daily_ic_mean": round(float(daily_ic.mean()), 5) if daily_ic.notna().any() else np.nan,
        "daily_ic_positive_rate": round(float((daily_ic.dropna() > 0).mean()), 4) if daily_ic.notna().any() else np.nan,
        "daily_ic_newey_west_tstat": round(_newey_west_tstat(daily_ic.dropna()), 4),
        "top_bucket_excess_return_pct": round(float(np.nanmean(top_returns) * 100.0), 4) if top_returns else np.nan,
        "cross_sections": int(daily_ic.notna().sum()),
        "observations": int(len(recent)),
        **_decile_shape(recent),
    }


def _overlay_summary(trades: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    if trades.empty:
        return {
            "lookback_days": lookback_days,
            "overlay_periods": 0,
            "overlay_return_sum_pct": np.nan,
            "overlay_hit_rate": np.nan,
            "overlay_alpha_sum_pct": np.nan,
        }
    frame = trades.copy()
    frame["exit_date"] = pd.to_datetime(frame.get("exit_date"), errors="coerce")
    frame["factor_overlay_return"] = pd.to_numeric(frame.get("factor_overlay_return"), errors="coerce")
    frame["core_return"] = pd.to_numeric(frame.get("core_return"), errors="coerce")
    frame = frame.dropna(subset=["exit_date", "factor_overlay_return"])
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = frame[(frame["exit_date"] >= start) & (frame["exit_date"] <= as_of)]
    if recent.empty:
        return {
            "lookback_days": lookback_days,
            "overlay_periods": 0,
            "overlay_return_sum_pct": np.nan,
            "overlay_hit_rate": np.nan,
            "overlay_alpha_sum_pct": np.nan,
        }
    overlay = recent["factor_overlay_return"]
    return {
        "lookback_days": lookback_days,
        "overlay_periods": int(len(recent)),
        "overlay_return_sum_pct": round(float(overlay.sum() * 100.0), 4),
        "overlay_hit_rate": round(float((overlay > 0).mean()), 4),
        "overlay_alpha_sum_pct": round(float(overlay.sum() * 100.0), 4),
    }


def _top_contributor(trades: pd.DataFrame, lookback_days: int, as_of: pd.Timestamp) -> dict:
    if trades.empty or "overlay_weights_json" not in trades.columns:
        return {"lookback_days": lookback_days, "top_selected_ticker": "", "top_selected_count": 0}
    frame = trades.copy()
    frame["exit_date"] = pd.to_datetime(frame.get("exit_date"), errors="coerce")
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = frame[(frame["exit_date"] >= start) & (frame["exit_date"] <= as_of)]
    counts: dict[str, int] = {}
    for raw in recent["overlay_tickers"].fillna("").astype(str):
        for ticker in [t for t in raw.split(",") if t]:
            counts[ticker] = counts.get(ticker, 0) + 1
    if not counts:
        return {"lookback_days": lookback_days, "top_selected_ticker": "", "top_selected_count": 0}
    ticker, count = max(counts.items(), key=lambda kv: kv[1])
    return {"lookback_days": lookback_days, "top_selected_ticker": ticker, "top_selected_count": int(count)}


def main() -> None:
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    config = _selected_config()
    specs = load_feature_specs()
    panel = core._ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, load_prediction_scores()))
    scored_all = _regime_score_panel(panel, config)
    holding_days = int(config.get("holding_days", core.HORIZON_DAYS))
    scored = _non_overlapping_rebalance_panel(scored_all, holding_days)
    as_of = pd.Timestamp(scored["date"].max())
    trades = pd.read_csv(TRADES_PATH) if TRADES_PATH.exists() else pd.DataFrame()

    rows = []
    for lookback in (60, 120):
        row = {
            **_ic_summary(scored, lookback, as_of),
            **{k: v for k, v in _overlay_summary(trades, lookback, as_of).items() if k != "lookback_days"},
            **{k: v for k, v in _top_contributor(trades, lookback, as_of).items() if k != "lookback_days"},
        }
        row["as_of"] = str(as_of.date())
        row["rank_ic_warning"] = bool(pd.notna(row["daily_ic_mean"]) and float(row["daily_ic_mean"]) < 0)
        row["overlay_alpha_warning"] = bool(pd.notna(row["overlay_alpha_sum_pct"]) and float(row["overlay_alpha_sum_pct"]) < 0)
        row["edge_health_status"] = edge_health_status(row)
        row["warning"] = bool(row["edge_health_status"] in {"warning", "block"})
        rows.append(row)

    # Capture the previous status BEFORE we overwrite OUT_JSON below — we
    # need it to decide whether to alert on a state transition.
    previous_status = _previous_status()

    out = pd.DataFrame(rows)
    atomic_write_csv(out, OUT_CSV, index=False)
    edge_status = aggregate_edge_health_status(rows)
    real_capital_block = edge_status == "block"
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(),
        "purpose": "factor_decay_monitor",
        "as_of": str(as_of.date()),
        "selected_score_source": config.get("score_source"),
        "score_direction": "higher_is_better",
        "sampling_method": "non_overlapping_rebalance_cohorts",
        "holding_days": holding_days,
        "cohort_dates": int(scored["date"].nunique()),
        "rows": rows,
        "edge_health_status": edge_status,
        "warning": bool(edge_status == "warning"),
        "advisory": bool(edge_status == "advisory"),
        "real_capital_block": real_capital_block,
        "reason": (
            "recent overlay alpha is negative"
            if real_capital_block
            else "top-bucket edge is weak/non-positive"
            if edge_status == "warning"
            else "rank IC is weak/negative but top-bucket edge and overlay alpha remain positive"
            if edge_status == "advisory"
            else "factor decay monitor has no real-capital blocking warning"
        ),
    }
    atomic_write_json(add_validation_context(payload, config=config), OUT_JSON)

    print(f"Factor decay monitor written -> {OUT_CSV}")
    print(f"Detailed report -> {OUT_JSON}")
    print(out.to_string(index=False))

    # Fire a notification IF status changed (or we're currently in block).
    # Compared to previous_status captured before the JSON was overwritten.
    _maybe_notify(edge_status, previous_status, payload)


if __name__ == "__main__":
    main()
