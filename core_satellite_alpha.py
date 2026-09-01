"""
core_satellite_alpha.py — test factor alpha as a satellite around SPY/QQQ.

The prior alpha_factor_backtest asks whether factors can replace ETF beta. This
script asks a more realistic question: can a lower-turnover factor sleeve add
active return on top of a SPY/QQQ core without exceeding 1.25x gross exposure?
It is research-only and does not enable paper trading.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alpha_factor_backtest import (
    HORIZON_DAYS,
    MAX_GROSS_EXPOSURE,
    attach_scores,
    benchmark_equity,
    compare_to_benchmarks,
    gate_metrics,
    load_factor_panel,
    load_feature_specs,
    load_prediction_scores,
    portfolio_stats,
    subperiod_metrics,
)
from backtest import INITIAL_CAPITAL, _load_etf_price_frame
from feature_health import enrich_feature_specs
from robustness_scoring import add_cost_stress_approval_columns, robustness_score_components
from signal_freshness import latest_completed_us_trading_day, live_config_fingerprint
from validation_bundle import (
    current_robustness_evidence,
    strategy_config_fingerprint,
    validate_validation_bundle,
)
from execution_cost_calibration import calibrated_turnover_cost_pct
from settings import (
    DATA_DIR,
    SIGNAL_DIR,
    SLIPPAGE_BASE_PCT,
    SURVIVORSHIP_TRAINING_TICKERS,
    VIX_INVERSION_THRESHOLD,
    WATCHLIST,
    validate_sector_map_coverage,
)
# Atomic signal/report writes — broker readers never see half-written files.
from safe_io import atomic_write_csv, atomic_write_json
from run_evidence import current_run_id


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE QUALITY GATE — only use features graded A/B/C from diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: If the feature quality diagnostic has been run (it produces
# signals/feature_quality_report.json), we load its grades and EXCLUDE features
# rated D or F from the scoring pipeline.  These are features with:
#   - IC too close to zero (no predictive power)
#   - Unstable IC across time (works some years, not others)
#   - Only works in bull markets (regime-dependent)
#
# If the diagnostic hasn't been run yet, we use ALL features (no filtering).
# Run: python3 feature_quality_diagnostic.py --top 48
# This produces the report that this gate reads.

FEATURE_QUALITY_MIN_GRADE = "C"  # drop D and F features
_QUALITY_GRADE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}


def _feature_quality_freshness_inputs() -> list[Path]:
    """Files that should make the live feature-quality report stale."""
    inputs = [Path("logs/feature_ic_shortlist.csv")]
    factor_universe = sorted(set(WATCHLIST) | set(SURVIVORSHIP_TRAINING_TICKERS))
    inputs.extend(Path(DATA_DIR) / f"{ticker.upper()}.parquet" for ticker in factor_universe)
    return inputs


def _load_feature_quality_filter(strict: bool = False) -> set[str] | None:
    """Load feature quality report and return set of features to KEEP.

    Returns None if no report exists in non-strict mode (use all features).
    Returns set of feature names that passed quality gate (grade >= C).
    """
    report_path = Path(SIGNAL_DIR) / "feature_quality_report.json"
    if not report_path.exists():
        if strict:
            raise SystemExit(
                f"Missing live feature quality report: {report_path}. "
                "Run `python3 feature_quality_diagnostic.py --top 48` before live signal generation."
            )
        return None  # no report -> use all features in research/helper mode

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if strict:
            raise SystemExit(
                f"Invalid live feature quality report: {report_path} ({exc}). "
                "Run `python3 feature_quality_diagnostic.py --top 48` to rebuild it."
            )
        return None

    if strict:
        report_mtime = report_path.stat().st_mtime
        freshness_inputs = _feature_quality_freshness_inputs()
        stale_inputs = [
            path
            for path in freshness_inputs
            if path.exists() and path.stat().st_mtime > report_mtime
        ]
        if stale_inputs:
            newest = max(stale_inputs, key=lambda path: path.stat().st_mtime)
            raise SystemExit(
                f"Live feature quality report is older than {newest}. "
                "Run `python3 feature_quality_diagnostic.py --top 48` after refreshing data."
            )

    min_grade_val = _QUALITY_GRADE_ORDER.get(FEATURE_QUALITY_MIN_GRADE, 3)
    keep_features = set()
    seen_features = set()
    dropped_features = []
    usable_grade_count = 0

    for feat_info in report.get("features", []):
        feature = str(feat_info.get("feature", "")).strip() if isinstance(feat_info, dict) else ""
        if not feature:
            continue
        usable_grade_count += 1
        seen_features.add(feature)
        grade = str(feat_info.get("grade", "C")).strip().upper()
        grade_val = _QUALITY_GRADE_ORDER.get(grade, 3)
        if grade_val >= min_grade_val:
            keep_features.add(feature)
        else:
            dropped_features.append((feature, grade))

    if strict and usable_grade_count <= 0:
        raise SystemExit(
            f"Live feature quality report has no usable feature grades: {report_path}. "
            "Run `python3 feature_quality_diagnostic.py --top 48` to rebuild it."
        )
    if strict and not keep_features:
        raise SystemExit(
            "Live feature quality report filters out every graded feature. "
            "Run `python3 feature_quality_diagnostic.py --top 48` and inspect D/F feature grades."
        )

    # ── Sanity check: report shouldn't be drastically smaller than the
    # IC shortlist.  A "3 features only" report (seen May 19 2026) means
    # the diagnostic ran against a partial panel during research rebuild.
    # Loading that stale report would collapse the cluster gate to 2
    # clusters @ 50% weight and block trading.  Refuse and ask for a
    # re-run instead of silently using the bad data.
    if strict:
        shortlist_path_check = Path("logs/feature_ic_shortlist.csv")
        expected_floor = int(os.environ.get("FEATURE_QUALITY_MIN_GRADED", "20"))
        if shortlist_path_check.exists():
            try:
                _shortlist_df = pd.read_csv(shortlist_path_check)
                # Approximate expected size — apply the same target/horizon
                # filter the spec loader uses (sector_excess / 5d horizon).
                if {"target", "horizon"}.issubset(_shortlist_df.columns):
                    _filtered = _shortlist_df[
                        (_shortlist_df["target"] == "sector_excess")
                        & (_shortlist_df["horizon"] == 5)
                    ]
                    expected_floor = max(expected_floor, int(len(_filtered) * 0.5))
            except (OSError, pd.errors.EmptyDataError):
                pass
        if usable_grade_count < expected_floor:
            raise SystemExit(
                f"Live feature quality report only grades {usable_grade_count} features "
                f"but the shortlist has at least {expected_floor} expected. "
                f"This usually means the diagnostic ran against a partial panel mid-research-rebuild. "
                f"Re-run `python3 feature_quality_diagnostic.py --top 48` "
                f"after the parquet rebuild fully completes."
            )

    shortlist_path = Path("logs/feature_ic_shortlist.csv")
    if shortlist_path.exists():
        try:
            shortlist = pd.read_csv(shortlist_path)
            if "feature" in shortlist.columns:
                keep_features.update(str(f) for f in shortlist["feature"].dropna().unique() if str(f) not in seen_features)
        except (OSError, pd.errors.EmptyDataError):
            pass

    if dropped_features:
        print(f"  Feature quality gate: keeping {len(keep_features)} features, "
              f"dropping {len(dropped_features)} (grade < {FEATURE_QUALITY_MIN_GRADE})")
        for feat, grade in dropped_features[:5]:
            print(f"    DROPPED: {feat[:50]} (grade={grade})")
        if len(dropped_features) > 5:
            print(f"    ... and {len(dropped_features) - 5} more")
    else:
        print(f"  Feature quality gate: all {len(keep_features)} features pass")

    return keep_features if keep_features else None


def _apply_live_feature_quality_filter(specs: list[dict], quality_filter: set[str]) -> list[dict]:
    original_count = len(specs)
    filtered = [s for s in specs if str(s.get("feature", "")) in quality_filter]
    if not filtered:
        raise SystemExit(
            "Live feature quality filter removed every active feature spec. "
            "Run `python3 feature_quality_diagnostic.py --top 48` after `python3 feature_research.py`."
        )
    if len(filtered) < original_count:
        print(f"  Feature filter applied: {original_count} -> {len(filtered)} specs")
    enriched, _feature_health_profile = enrich_feature_specs(filtered)
    return enriched


def _validate_live_feature_inputs(specs: list[dict], signal_panel: pd.DataFrame) -> None:
    selected_features = [str(s.get("feature", "")).strip() for s in specs if str(s.get("feature", "")).strip()]
    has_xs_rank = any(
        feature.startswith("xs_rank_market_") or feature.startswith("xs_rank_sector_")
        for feature in selected_features
    )
    if not has_xs_rank:
        raise SystemExit(
            "Live feature set contains no cross-sectional rank features "
            "(xs_rank_market_* or xs_rank_sector_*). Run `python3 research.py --xs-only`, "
            "then `python3 feature_quality_diagnostic.py --top 48`."
        )

    missing = sorted(feature for feature in selected_features if feature not in signal_panel.columns)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f", ... +{len(missing) - 8} more" if len(missing) > 8 else ""
        raise SystemExit(
            f"Live signal panel is missing selected feature columns: {preview}{suffix}. "
            "Run `python3 research.py --xs-only`, then `python3 feature_quality_diagnostic.py --top 48`."
        )

    health = getattr(signal_panel, "attrs", {}).get("feature_health_summary", {}) or {}
    if health and not bool(health.get("feature_health_gate_pass", True)):
        print("  Feature health gate failed in live panel; signal will be written as not tradeable.")


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT VETO — filter out stocks with strongly negative news sentiment
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: Before finalizing the live signal, we check each selected
# overlay stock's recent news sentiment.  If a stock has very negative news
# (e.g., fraud allegations, major lawsuit, earnings disaster), we remove it
# from the signal and let the next-best candidate take its spot.
# This only runs during LIVE signal generation, not during backtesting.

# PLAIN ENGLISH: Keep sentiment veto on by default, but allow offline CI/local
# dry-runs to disable it instead of waiting on news-feed retries.
SENTIMENT_VETO_ENABLED = os.environ.get("CORE_ALPHA_SENTIMENT_VETO", "1").strip().lower() not in {"0", "false", "no"}
SENTIMENT_VETO_THRESHOLD = -0.25   # compound score below this → vetoed
SENTIMENT_BOOST_WEIGHT = 0.10      # multiply factor score by (1 + sentiment * this)


def _fetch_live_sentiment(tickers: list[str], timeout_per_ticker: float = 5.0) -> dict[str, float]:
    """
    Fetch real-time news sentiment for a list of tickers.

    PLAIN ENGLISH: For each ticker, grab recent news headlines from RSS feeds
    (Yahoo Finance, Google News, etc.), run them through FinVADER/FinBERT, and
    return a sentiment score between -1 (very negative) and +1 (very positive).
    Score of 0 means neutral or no news found.

    Returns dict mapping ticker → compound sentiment score.
    """
    try:
        from sentiment_engine import get_sentiment_engine, score_todays_news
    except ImportError:
        # If sentiment engine not available, return empty (no veto)
        return {}

    engine = get_sentiment_engine("vader")  # fast, maintained, no GPU needed
    results: dict[str, float] = {}

    for ticker in tickers:
        try:
            live_news = score_todays_news(ticker, engine, verbose=False)
            results[ticker] = float(live_news.get("composite_score", 0.0) or 0.0)
        except Exception:
            # Network/provider errors degrade to neutral for the veto. The
            # predictor's sentiment health gate handles complete outages.
            results[ticker] = 0.0  # no news = no veto

    return results


def _apply_sentiment_veto(
    selected: pd.DataFrame,
    day: pd.DataFrame,
    *,
    score_col: str,
    shape: str,
    exit_rank_floor: float,
    max_per_sector: int,
    earnings_blackout_days: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Check sentiment for selected overlay stocks and veto strongly negative ones.

    PLAIN ENGLISH: After the factor model picks the top stocks, we ask:
    "Is there terrible news about any of these stocks RIGHT NOW?"
    If yes, we drop that stock and let the next-best candidate in.

    Returns (updated_selected, sentiment_scores_dict).
    """
    if not SENTIMENT_VETO_ENABLED or selected.empty:
        return selected, {}

    tickers = selected["ticker"].tolist()
    # Also fetch sentiment for a few backup candidates in case we need replacements
    backup_tickers = []
    ranked_all = day.dropna(subset=[score_col]).copy()
    ranked_all["_rank"] = ranked_all[score_col].rank(pct=True)
    ranked_all = ranked_all.sort_values("_rank", ascending=False)
    for _, row in ranked_all.iterrows():
        t = str(row["ticker"])
        if t not in tickers and len(backup_tickers) < 3:
            backup_tickers.append(t)

    print(f"  Fetching sentiment for {len(tickers)} selected + {len(backup_tickers)} backup tickers...")
    sentiments = _fetch_live_sentiment(tickers + backup_tickers)

    # Report sentiment scores
    vetoed = []
    for t in tickers:
        score = sentiments.get(t, 0.0)
        status = "✗ VETO" if score < SENTIMENT_VETO_THRESHOLD else "✓ OK"
        print(f"    {t:6s} sentiment={score:+.3f}  {status}")
        if score < SENTIMENT_VETO_THRESHOLD:
            vetoed.append(t)

    if not vetoed:
        return selected, sentiments

    # Remove vetoed stocks and re-select to fill spots
    print(f"  ⚠ Vetoed {len(vetoed)} stocks: {', '.join(vetoed)}")
    # Mark vetoed tickers in day so they won't be re-selected
    day_filtered = day[~day["ticker"].isin(vetoed)].copy()
    new_selected = _select_sticky_holdings(
        day_filtered,
        set(t for t in tickers if t not in vetoed),  # keep non-vetoed as "held"
        score_col=score_col,
        return_col=None,
        shape=shape,
        exit_rank_floor=exit_rank_floor,
        max_per_sector=max_per_sector,
        earnings_blackout_days=earnings_blackout_days,
    )
    return new_selected, sentiments


# ─────────────────────────────────────────────────────────────────────────────
# DATA FRESHNESS GATE
# ─────────────────────────────────────────────────────────────────────────────
# PLAIN ENGLISH: Before we generate any signal, check how old the factor data
# is.  If the newest data is more than 5 trading days old, the signal might be
# stale and we warn loudly.  If it's more than 10 trading days old, we refuse
# to write a signal at all — trading on ancient data is dangerous.

STALE_WARN_TRADING_DAYS = 5    # warn if factor data older than this
STALE_BLOCK_TRADING_DAYS = 10  # refuse to generate signal if older than this


def _count_nyse_sessions(start: object, end: object) -> int:
    """Count real NYSE sessions in an inclusive date window."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        return 0
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        return int(len(calendar.sessions_in_range(start_ts, end_ts)))
    except Exception:
        return int(sum(1 for day in pd.date_range(start_ts, end_ts, freq="D") if day.weekday() < 5))


def check_factor_freshness(
    panel: pd.DataFrame,
    warn_days: int = STALE_WARN_TRADING_DAYS,
    block_days: int = STALE_BLOCK_TRADING_DAYS,
    ignore_stale: bool = False,
    now: datetime | pd.Timestamp | None = None,
) -> dict:
    """
    Check how old the factor panel data is.

    PLAIN ENGLISH: Looks at the most recent date in the factor panel and
    compares it to today.  Returns a dict with the age info and whether
    the data is fresh enough to trade on.

    Returns:
        dict with keys:
            latest_date: str — most recent date in the panel
            age_trading_days: int — how many trading days old the data is
            fresh: bool — True if data is fresh enough (no warning)
            blocked: bool — True if data is too old to generate signal
            message: str — human-readable status
    """
    latest = pd.Timestamp(panel["date"].max())
    today = latest_completed_us_trading_day(now=now)

    # Count real NYSE sessions so weekends/holidays do not create false age.
    # (excludes weekends, but not holidays — close enough for a safety check)
    if latest >= today:
        age = 0
    else:
        age = _count_nyse_sessions(latest + pd.Timedelta(days=1), today)

    result = {
        "latest_date": str(latest.date()),
        "age_trading_days": age,
        "fresh": age <= warn_days,
        "blocked": age > block_days and not ignore_stale,
        "ignore_stale": ignore_stale,
    }

    if age <= warn_days:
        result["message"] = f"Factor data is fresh ({age} trading days old, latest={latest.date()})"
    elif age <= block_days:
        result["message"] = (
            f"⚠ WARNING: Factor data is {age} trading days old (latest={latest.date()}). "
            f"Signal will be generated but gates_all_pass will be set to False."
        )
    else:
        if ignore_stale:
            result["message"] = (
                f"⚠ OVERRIDE: Factor data is {age} trading days old (latest={latest.date()}). "
                f"--ignore-stale flag used — proceeding anyway with gates_all_pass=False."
            )
            result["blocked"] = False
        else:
            result["message"] = (
                f"✗ BLOCKED: Factor data is {age} trading days old (latest={latest.date()}). "
                f"Too stale to generate a signal (limit={block_days} trading days). "
                f"Run your data pipeline to refresh, or use --ignore-stale to override."
            )

    return result


CORE_PRESETS = {
    "qqq_tilt_40_60": {"SPY": 0.40, "QQQ": 0.60},
    "qqq_heavy_25_75": {"SPY": 0.25, "QQQ": 0.75},
}
REGIME_PRESETS = {
    "qqq_trend_switch": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.00,
            "overlay_gross": 0.25,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay35": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.90,
            "overlay_gross": 0.35,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay50": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.75,
            "overlay_gross": 0.50,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_fast_core110_overlay15": {
        "ma_window": 75,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 1.10,
            "overlay_gross": 0.15,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.85,
            "overlay_gross": 0.40,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.75,
            "overlay_gross": 0.25,
        },
    },
    "qqq_trend_switch_overlay70_core55": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.65,
            "overlay_gross": 0.35,
        },
    },
    # --- CASH BUFFER VARIANT ---
    # PLAIN ENGLISH: Same as overlay70_core55 but holds 20% cash in risk_off
    # and 10% cash in neutral.  The idea is that when the market is stressed,
    # you want LESS total exposure, not just different ETFs.  The grid search
    # will compare this variant to the original and pick whichever is better.
    "qqq_trend_switch_overlay70_core55_cashbuffer": {
        "ma_window": 100,
        "high_vol": 0.30,
        "risk_on": {
            # Risk_on stays fully invested — bull market, no reason for cash
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            # Neutral: reduce overlay slightly → ~10% cash buffer
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.50,
            "overlay_gross": 0.60,
        },
        "risk_off": {
            # Risk_off: pull back to 80% gross → 20% cash buffer
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.55,
            "overlay_gross": 0.25,
        },
    },
    # --- ADAPTIVE VOL + CASH BUFFER ---
    # PLAIN ENGLISH: Combines the cash buffer with percentile-based vol
    # detection.  Instead of a fixed 0.30 vol threshold, it uses the top 20%
    # of trailing 1-year vol.  This adapts to changing market conditions —
    # avoids whipsawing between regimes when overall vol shifts higher/lower.
    "qqq_trend_switch_overlay70_core55_adaptive": {
        "ma_window": 100,
        "high_vol": 0.30,           # fallback, but percentile mode overrides
        "high_vol_mode": "percentile",  # ← the new adaptive threshold
        # Blend risk_on and risk_off scores proportionally (no hard flip)
        "score_blend": True,
        # Trigger early rebalance when regime changes mid-holding-period
        "early_rebalance_on_regime_change": True,
        "risk_on": {
            "core_weights": {"SPY": 0.00, "QQQ": 1.00},
            "core_gross": 0.55,
            "overlay_gross": 0.70,
        },
        "neutral": {
            "core_weights": {"SPY": 0.25, "QQQ": 0.75},
            "core_gross": 0.50,
            "overlay_gross": 0.60,
        },
        "risk_off": {
            "core_weights": {"SPY": 0.60, "QQQ": 0.40},
            "core_gross": 0.55,
            "overlay_gross": 0.25,
        },
    },
}
CORE_OVERLAY_COMBOS = (
    (1.00, 0.25),
    (0.75, 0.25),
    (0.75, 0.50),
)
# ── REDUCED SEARCH SPACE (anti-overfitting) ────────────────────────────────
# PLAIN ENGLISH: We deliberately keep the grid SMALL to avoid
# fitting noise.  Each dimension must represent a genuinely different economic
# hypothesis, not a minor numerical tweak.  Parameters that have one correct
# answer (like "don't trade into earnings") are FIXED, not searched.
#
# Old grid: 6,336 configs → massive selection bias.  A random config that
# happens to work 2010-2026 is NOT the same as a robust strategy.
# New grid: a few hundred configs → each config is a meaningfully different bet.
#
# Dimensions that vary (each represents a real economic question):
#   - score_source: "do raw factors work, or does regime-conditioning help?"
#   - shape: "diversify across 5, 10, or 15 overlay names?"
#   - weighting: "score-only or score × inverse-vol across overlay?"
#   - max_per_sector: "allow sector bets or force diversification?"
#   - overlay_gross: "small overlay or moderate overlay?"
#   - holding_days: "10-day alpha or 20-day lower-turnover alpha?"
#   - cost_stress: validation only — "does the same config survive 2x/3x/5x costs?"
#
# Fixed (not searched — one obvious correct answer):
#   - earnings_blackout = 5 days (safety, not tunable)
#   - exit_rank_floor = 0.80 (well-validated, not worth searching)

SCORE_SOURCES = ("factor_walkforward", "regime_adaptive")
# PLAIN ENGLISH: test whether wider baskets reduce concentration risk without
# diluting alpha too much.  top3 added back after full walkforward showed
# it crushed 2021-2025 (+90% in 2024) — concentrated bets work when the
# market rewards conviction and features have strong IC.
SHAPES = ("top3", "top5", "top10", "top15")
# PLAIN ENGLISH: "sticky_score" weights by model confidence. "risk_parity"
# sizes positions inversely to volatility (equal risk contribution).
# "sticky_vol_score" blends both approaches.
# risk_parity added back after full run showed regime_adaptive/top3/risk_parity
# was the dominant config for 2021-2025.
WEIGHTING_MODES = ("sticky_score", "risk_parity", "sticky_vol_score")
EXIT_RANK_FLOORS = (0.80,)
ADAPTIVE_EXIT_MODES = ("fixed",)
# PLAIN ENGLISH: max_per_sector controls how many stocks from the same
# sector can be in the overlay at once.  (2) = up to 2 tech stocks,
# (1) = force cross-sector diversification (one per sector max).
# Genuine question: does concentration in winning sectors help or hurt?
MAX_PER_SECTOR_OPTIONS = (1, 2)
# FIXED: Don't trade into earnings — this is a safety rule, not a parameter.
EARNINGS_BLACKOUT_DAY_OPTIONS = (5,)
# Search only the cost stresses we actually care about for robustness.
COST_STRESS_MULTIPLIERS = (2.0, 3.0, 5.0)
COST_STRESS_VALIDATION = COST_STRESS_MULTIPLIERS
# 10 days captures faster alpha; 20 days checks whether lower turnover helps.
HOLDING_DAY_OPTIONS = (10, 20)
# Keep overlay sizing coarse: small and moderate sleeves only.
OVERLAY_GROSS_OPTIONS = (0.25, 0.50)
# PLAIN ENGLISH: Drawdown circuit breaker.  If portfolio equity drops more than
# this % from its all-time peak, we cut ALL exposure to zero (100% cash) until
# equity recovers above the re-entry threshold.  This protects against catastrophic
# losses during market crashes.
# (0.0) = disabled (no circuit breaker), (0.25) = go to cash if down 25% from peak.
# Grid search tests both to see if the protection is worth the missed recovery.
# PLAIN ENGLISH: Regime confirmation cooldown.  When SPY or QQQ crosses
# above/below its moving average, we require this many CONSECUTIVE days on
# the new side before flipping the regime indicator.  This prevents whipsaw:
# a single noisy day touching the MA line shouldn't trigger a full rebalance.
# 3 days is a good balance — fast enough to react to real regime changes,
# slow enough to ignore 1-2 day noise.  Set to 1 for no confirmation (old behavior).
REGIME_CONFIRM_DAYS = 3
DRAWDOWN_CIRCUIT_BREAKER_OPTIONS = (0.0,)  # DD breaker hurts regime-switching configs (redundant), disabled from grid
# PLAIN ENGLISH: Portfolio volatility targeting.  We track how volatile our
# portfolio returns have been recently (rolling 20-day annualized stdev).
# If realized vol is HIGHER than our target, we shrink exposure (smaller positions).
# If realized vol is LOWER than our target, we increase exposure (bigger positions).
# This keeps risk roughly constant across calm and turbulent markets.
# (0.0) = disabled (no vol scaling).  (0.15) = target 15% annualized vol.
# Cap at 1.5x to prevent excessive leverage during very calm periods.
VOL_TARGET_OPTIONS = (0.0,)  # 0.15 tested — reduces DD but hurts Sharpe/returns, disabled
VOL_TARGET_LOOKBACK = 20        # how many past returns to use for realized vol
VOL_TARGET_MAX_SCALE = 1.5      # never scale exposure above 1.5x base
VOL_TARGET_MIN_SCALE = 0.3      # never scale exposure below 0.3x base
# PLAIN ENGLISH: When True, blend risk_on and risk_off scores proportionally
# based on regime strength (0-1).  When False, hard-switch between score types
# like before.  Grid search tests both and picks whichever works better.
# NOTE: score_blend and early_rebalance_on_regime_change are baked into
# specific preset configs (the "adaptive" variant) rather than grid-searched
# across ALL presets — this keeps the total config count manageable.
# PLAIN ENGLISH: No single overlay stock can be more than this fraction of
# the overlay sleeve.  0.25 = max 25% in one name (was 0.35).  Lower cap
# spreads risk more evenly across overlay picks, reducing blowup risk from
# one stock cratering.
MAX_SINGLE_NAME_WEIGHT = 0.25
# Broker/paper-trading safety: research can evaluate up to MAX_GROSS_EXPOSURE,
# but submitted paper signals default to no leverage unless the paper-trading
# script is deliberately run with --allow-leverage-submit.
PAPER_MAX_GROSS_EXPOSURE = 1.00
PAPER_SIGNAL_TIMEZONE = os.environ.get("PAPER_SIGNAL_TIMEZONE", os.environ.get("TZ", "Asia/Singapore"))
STICKY_STATE_EXCLUDED_TICKERS = {"SPY", "QQQ", "TQQQ", "BIL", "IEF", "GLD"}

MAX_POSITIVE_YEAR_ALPHA_SHARE = 0.35
MAX_TOP_TICKER_CONTRIB_SHARE = 0.50
ROBUST_COST_STRESSES = (2.0, 3.0, 5.0)
PERIODS_PER_YEAR = 252.0 / HORIZON_DAYS
# ── Per-panel day-map cache ────────────────────────────────────────────
# `_panel_day_map(panel)` splits a panel into `{date: day_frame}` so the
# main strategy loop can do O(1) lookups instead of re-filtering by date
# every rebalance.  Building the map is ~50-200 MB depending on panel
# size, which is why we cache it.
#
# The naive cache (`dict[id(panel), mapped]`, unbounded) was responsible
# for the nested-walkforward memory leak: `evaluate_window` creates a
# NEW sliced DataFrame each call (`panel.loc[panel["_date"] <= end_ts]`)
# so every call had a different `id()` and missed the cache, but the
# cache held STRONG references to all prior `mapped` dicts (which in
# turn referenced their slice DataFrames via groupby).  After 960
# evaluate_window calls per outer fold the cache held ~960 day maps,
# each ~150-200 MB → main process working set ballooned past physical
# RAM.
#
# Fix: bounded LRU eviction plus weak identity checks.  Keep only the most
# recent _MAX_PANEL_DAY_ENTRIES panels; anything older falls off and its
# mapped dict becomes eligible for GC.  The weakref matters because Python can
# reuse `id(panel)` after a DataFrame is freed.  If that happens, we rebuild
# instead of accidentally returning a day map for an old, different panel.
import collections as _collections
import weakref as _weakref
try:
    _MAX_PANEL_DAY_ENTRIES = max(1, int(os.environ.get("PANEL_DAY_CACHE_MAX_ENTRIES", "4")))
except ValueError:
    _MAX_PANEL_DAY_ENTRIES = 4
_PANEL_DAY_CACHE: _collections.OrderedDict[
    int,
    tuple[_weakref.ReferenceType[pd.DataFrame], dict[pd.Timestamp, pd.DataFrame]],
] = _collections.OrderedDict()
# (key) -> (cached_dataframe, inserted_at_unix_timestamp).  TTL guards
# against a long-running process (notebook, dashboard, GitHub Actions
# step that loops) seeing stale ETF prices after new parquet writes —
# audit #13.  Default 30 min; env-overridable.
import os as _os, time as _time
_ETF_PRICE_CACHE_TTL_SEC = int(_os.environ.get("ETF_PRICE_CACHE_TTL_SEC", "1800"))
_ETF_PRICE_CACHE: dict[tuple[tuple[pd.Timestamp, ...], tuple[str, ...]], tuple[pd.DataFrame, float]] = {}


def _regime_preset_with_overlay_gross(regime_preset: dict, overlay_gross: float) -> dict:
    """Return a copy of a regime preset with the same overlay gross in each regime."""
    adjusted: dict = {}
    for key, value in regime_preset.items():
        if key in {"risk_on", "neutral", "risk_off"} and isinstance(value, dict):
            regime = dict(value)
            regime["core_weights"] = dict(regime.get("core_weights", {}))
            regime["overlay_gross"] = float(overlay_gross)
            adjusted[key] = regime
        else:
            adjusted[key] = value
    return adjusted


def _panel_day_map(panel: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Return a `{date: day_frame}` view of `panel`, cached per panel.

    Cache is an OrderedDict bounded at `_MAX_PANEL_DAY_ENTRIES` — when
    we exceed that count, the least-recently-used entry is evicted.
    Each entry also stores a weak reference to the exact DataFrame object
    that created it, so a recycled Python object id cannot return stale data.
    """
    cache_key = id(panel)
    cached = _PANEL_DAY_CACHE.get(cache_key)
    if cached is not None:
        cached_panel_ref, cached_map = cached
        if cached_panel_ref() is panel:
            # move_to_end marks this entry as most-recently-used so the
            # next evict-on-overflow drops something staler.
            _PANEL_DAY_CACHE.move_to_end(cache_key)
            return cached_map
        # Same integer id, different DataFrame.  Drop the stale map before
        # rebuilding so the caller never receives rows from an old panel.
        _PANEL_DAY_CACHE.pop(cache_key, None)
    mapped = {pd.Timestamp(dt): day for dt, day in panel.groupby("date", sort=True)}
    _PANEL_DAY_CACHE[cache_key] = (_weakref.ref(panel), mapped)
    # Evict the oldest entry if we're over the bound.  popitem(last=False)
    # drops from the FRONT, which is the LRU since we move_to_end on hit.
    while len(_PANEL_DAY_CACHE) > _MAX_PANEL_DAY_ENTRIES:
        _PANEL_DAY_CACHE.popitem(last=False)
    return mapped


def clear_panel_day_cache() -> None:
    """Release cached day maps after memory-heavy validation batches."""
    _PANEL_DAY_CACHE.clear()


def _cached_etf_prices(price_index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    normalized_dates = tuple(pd.Timestamp(dt) for dt in price_index)
    key = (normalized_dates, tuple(tickers))
    cached = _ETF_PRICE_CACHE.get(key)
    if cached is not None:
        prices, inserted_at = cached
        # TTL check — entries older than ETF_PRICE_CACHE_TTL_SEC are
        # discarded.  Prevents a long-running process (notebook, web
        # dashboard, looping CI step) from returning yesterday's prices
        # after new parquet bars are written.
        if (_time.time() - inserted_at) < _ETF_PRICE_CACHE_TTL_SEC:
            return prices
        # else: fall through and re-fetch
    prices = _load_etf_price_frame(pd.DatetimeIndex(normalized_dates), tickers)
    if len(prices) > 1:
        synthetic = [
            ticker
            for ticker in tickers
            if ticker in prices.columns
            and float(pd.to_numeric(prices[ticker], errors="coerce").diff().abs().fillna(0.0).sum()) == 0.0
        ]
        if synthetic:
            raise RuntimeError(
                f"ETF price data unavailable for {synthetic}; refusing to run core alpha with synthetic flat prices."
            )
    _ETF_PRICE_CACHE[key] = (prices, _time.time())
    return prices


def _low_vol_rank(panel: pd.DataFrame) -> pd.Series:
    for col in (
        "hvol_20d",
        "factor_idio_vol_252_spy",
        "xs_rank_sector_hvol_20d",
        "xs_rank_sector_factor_idio_vol_252_spy",
    ):
        if col not in panel.columns:
            continue
        raw = pd.to_numeric(panel[col], errors="coerce")
        if raw.notna().any():
            return raw.groupby(panel["date"]).rank(pct=True)
    return pd.Series(0.50, index=panel.index)


def _rank_existing_score(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(0.50, index=panel.index)
    raw = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return raw.groupby(panel["date"]).rank(pct=True).fillna(0.50)


def _consensus_score(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    ranks = pd.concat([_rank_existing_score(panel, col).rename(col) for col in cols], axis=1)
    avg_rank = ranks.mean(axis=1, skipna=True)
    floor_rank = ranks.min(axis=1, skipna=True)
    disagreement = ranks.std(axis=1, skipna=True).fillna(0.0)
    raw = 0.70 * avg_rank + 0.25 * floor_rank - 0.10 * disagreement
    return raw.groupby(panel["date"]).rank(pct=True).fillna(avg_rank).fillna(0.50)


def _ensure_robust_score_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    low_vol_bonus = (1.0 - _low_vol_rank(out)).fillna(0.50)
    if "factor_risk_on_low_vol_score" not in out.columns:
        out["factor_risk_on_low_vol_score"] = (
            0.80 * pd.to_numeric(out["factor_risk_on_score"], errors="coerce")
            + 0.20 * low_vol_bonus
        ).fillna(out["factor_risk_on_score"])
    if "factor_walkforward_low_vol_score" not in out.columns:
        out["factor_walkforward_low_vol_score"] = (
            0.80 * pd.to_numeric(out["factor_walkforward_score"], errors="coerce")
            + 0.20 * low_vol_bonus
        ).fillna(out["factor_walkforward_score"])
    if "factor_risk_on_consensus_score" not in out.columns:
        out["factor_risk_on_consensus_score"] = _consensus_score(
            out,
            ["factor_risk_on_score", "factor_walkforward_score", "factor_risk_on_low_vol_score"],
        )
    if "factor_walkforward_consensus_score" not in out.columns:
        out["factor_walkforward_consensus_score"] = _consensus_score(
            out,
            ["factor_walkforward_score", "factor_score", "factor_walkforward_low_vol_score"],
        )
    if "factor_defensive_consensus_score" not in out.columns:
        out["factor_defensive_consensus_score"] = _consensus_score(
            out,
            ["factor_defensive_score", "factor_walkforward_score", "factor_walkforward_low_vol_score"],
        )
    return out


def _load_regime_indicators(rebalance_dates: pd.DatetimeIndex, exit_dates: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    """
    Compute trend and volatility indicators used for regime detection.

    PLAIN ENGLISH: This function decides whether we're in a bull market
    (risk_on), a mixed market (neutral), or a stressed/bear market (risk_off).
    It looks at two things:
      1. TREND: Are SPY and QQQ above their moving averages? (uptrend = good)
      2. VOLATILITY: Is QQQ moving around a lot? (high vol = bad)

    The high_vol_mode config controls how we measure "high" volatility:
      - "fixed" (default): volatility above a hard number (e.g. 0.30 = 30%)
      - "percentile": volatility in the top 20% of its trailing 1-year range
        This adapts to changing market conditions — what's "high" in a calm
        year is different from "high" in a volatile year.
    """
    start = pd.Timestamp(rebalance_dates.min()) - pd.tseries.offsets.BDay(260)
    end = pd.Timestamp(exit_dates.max())
    timeline = pd.date_range(start, end, freq="B")
    prices = _cached_etf_prices(pd.DatetimeIndex(timeline), ["SPY", "QQQ"])
    ma_window = int(config.get("regime_ma_window", 100))
    high_vol_threshold = float(config.get("regime_high_vol", 0.30))
    high_vol_mode = str(config.get("high_vol_mode", "fixed"))
    out = pd.DataFrame(index=prices.index)
    out["SPY"] = prices["SPY"]
    out["QQQ"] = prices["QQQ"]
    out["spy_trend_ok"] = prices["SPY"] >= prices["SPY"].rolling(200, min_periods=50).mean()
    out["qqq_trend_ok"] = prices["QQQ"] >= prices["QQQ"].rolling(ma_window, min_periods=50).mean()
    out["qqq_realized_vol"] = prices["QQQ"].pct_change().rolling(20, min_periods=10).std().mul(np.sqrt(252))
    # PLAIN ENGLISH: Positive values mean QQQ has beaten SPY over roughly the
    # last six months.  That is a simple proxy for a narrow mega-cap-led market.
    out["concentration_qqq_spy_120d"] = (
        prices["QQQ"].pct_change(120) - prices["SPY"].pct_change(120)
    ).fillna(0.0)

    if high_vol_mode == "percentile":
        # PLAIN ENGLISH: Instead of a fixed number, compare today's vol to
        # the last ~1 year of vol.  If today's vol is in the top 20%
        # (i.e. percentile rank > 0.80), that's "high."  This adapts —
        # in a calm year, 20% vol might be high; in a crazy year, 35%
        # might only be moderate.
        vol_pctile = out["qqq_realized_vol"].rolling(252, min_periods=60).rank(pct=True)
        out["high_vol"] = vol_pctile > 0.80
    else:
        # Fixed threshold — original behavior
        out["high_vol"] = out["qqq_realized_vol"] > high_vol_threshold

    # ── VIX term structure inversion ─────────────────────────────────────
    # PLAIN ENGLISH: When VIX > VIX3M (short-term fear exceeds medium-term
    # fear), the term structure is "inverted."  This is one of the strongest
    # short-term crash signals — it means options traders expect imminent
    # turbulence that exceeds what they expect over the next 3 months.
    # If VIX data can't be fetched, we simply skip this check (defensive).
    vix_inversion_threshold = float(config.get("vix_inversion_threshold", VIX_INVERSION_THRESHOLD))
    try:
        vix_prices = _cached_etf_prices(pd.DatetimeIndex(timeline), ["^VIX", "^VIX3M"])
        vix = vix_prices["^VIX"]
        vix3m = vix_prices["^VIX3M"]
        vix_ratio = vix / (vix3m + 1e-9)
        out["vix_inverted"] = (vix_ratio > vix_inversion_threshold).reindex(out.index, fill_value=False)
        out["vix_ratio"] = vix_ratio.reindex(out.index, fill_value=1.0)
    except Exception:
        # VIX data unavailable — don't crash, just skip this signal
        out["vix_inverted"] = False
        out["vix_ratio"] = 1.0

    # ── Regime confirmation cooldown ──────────────────────────────────────
    # PLAIN ENGLISH: Without this, a single day where QQQ dips below its
    # moving average flips the regime from "risk_on" to "neutral" or
    # "risk_off," triggering unnecessary rebalancing and trading costs.
    # The confirmation buffer requires N consecutive days of the new signal
    # before the indicator actually flips.  This prevents whipsaw — rapid
    # back-and-forth regime flips caused by noise around the MA line.
    #
    # Example: if REGIME_CONFIRM_DAYS=3 and QQQ drops below its MA for
    # 1 day then bounces back, the trend still shows "ok."  Only after
    # 3 consecutive days below does it flip to "not ok."
    confirm_days = int(config.get("regime_confirm_days", REGIME_CONFIRM_DAYS))
    if confirm_days > 1:
        for col in ("spy_trend_ok", "qqq_trend_ok"):
            raw = out[col].astype(float)
            # Rolling minimum over confirm_days: only True if ALL recent
            # days agree.  For False→True transitions, use rolling max
            # (only flip back to True if ALL recent days are True).
            # This creates hysteresis: slow to flip in either direction.
            confirmed_off = raw.rolling(confirm_days, min_periods=1).min()  # all 0 → flip off
            confirmed_on = raw.rolling(confirm_days, min_periods=1).max()   # all 1 → flip on
            # Start with previous confirmed state, update only when
            # confirmed_off or confirmed_on gives a unanimous signal.
            prev = raw.iloc[0] if len(raw) > 0 else True
            confirmed = []
            for off_val, on_val in zip(confirmed_off, confirmed_on):
                if off_val == 0.0:
                    # N consecutive days below MA → confirmed downtrend
                    prev = False
                elif on_val == 1.0:
                    # N consecutive days above MA → confirmed uptrend
                    prev = True
                # else: mixed signals → hold previous state (no flip)
                confirmed.append(prev)
            out[col] = confirmed

        # Also smooth the high_vol flag — require N consecutive high-vol
        # days before declaring vol regime shift
        raw_vol = out["high_vol"].astype(float)
        confirmed_high = raw_vol.rolling(confirm_days, min_periods=1).min()
        confirmed_low = (1.0 - raw_vol).rolling(confirm_days, min_periods=1).min()
        prev_vol = bool(raw_vol.iloc[0]) if len(raw_vol) > 0 else False
        confirmed_vol = []
        for hi, lo in zip(confirmed_high, confirmed_low):
            if hi == 1.0:
                prev_vol = True
            elif lo == 1.0:
                prev_vol = False
            confirmed_vol.append(prev_vol)
        out["high_vol"] = confirmed_vol

    return out.ffill().bfill()


def _apply_concentration_overlay_target(
    dt: pd.Timestamp,
    core_gross: float,
    overlay_gross: float,
    regime_indicators: pd.DataFrame | None,
    config: dict,
) -> tuple[float, float, float]:
    """Return overlay gross adjusted for QQQ-led concentration regimes.

    PLAIN ENGLISH: When QQQ is strongly beating SPY, the market is usually led
    by a few mega-cap growth names.  A small, concentrated overlay can work in
    that regime, but a broad overlay can lag the benchmark.  This rule lets a
    config use a smaller overlay in normal/broad markets and a larger overlay
    only when the QQQ-vs-SPY gap says concentration is high.
    """
    if str(config.get("concentration_overlay_mode", "off")) != "qqq_spy_dynamic":
        return float(overlay_gross), float(overlay_gross), 0.0
    if regime_indicators is None or regime_indicators.empty:
        return float(overlay_gross), float(overlay_gross), 0.0

    try:
        row = regime_indicators.loc[pd.Timestamp(dt)]
    except KeyError:
        row = regime_indicators.reindex([pd.Timestamp(dt)], method="ffill").iloc[0]
    gap = float(row.get("concentration_qqq_spy_120d", 0.0) or 0.0)
    low = float(config.get("concentration_overlay_low_gross", 0.30))
    high = float(config.get("concentration_overlay_high_gross", 0.70))
    threshold = float(config.get("concentration_overlay_threshold", 0.05))
    span = max(float(config.get("concentration_overlay_span", 0.05)), 1e-9)
    strength = float(np.clip((gap - threshold) / span, 0.0, 1.0))
    target = low + (high - low) * strength
    max_overlay = max(0.0, float(config.get("max_gross_exposure", MAX_GROSS_EXPOSURE)) - float(core_gross))
    adjusted = float(np.clip(target, 0.0, max_overlay))
    return adjusted, target, gap


def _resolve_allocation(dt: pd.Timestamp, config: dict, regime_indicators: pd.DataFrame | None) -> tuple[str, dict[str, float], float, float]:
    regime_mode = str(config.get("regime_mode", "static"))
    if regime_mode in REGIME_PRESETS and regime_indicators is not None:
        row = regime_indicators.loc[pd.Timestamp(dt)]
        # VIX inversion override — strongest short-term crash signal.
        # PLAIN ENGLISH: If VIX > VIX3M (term structure inverted), it means
        # options traders expect an imminent crash.  Override all other regime
        # signals and go defensive immediately.  This catches events like
        # Feb 2018 volmageddon, March 2020 COVID, where trends were still
        # technically "ok" but volatility was exploding.
        if "vix_inverted" in row.index and bool(row.get("vix_inverted", False)):
            regime = "risk_off"
        elif bool(row["qqq_trend_ok"]) and bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "risk_on"
        elif bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
            regime = "neutral"
        else:
            regime = "risk_off"
        regime_preset = config.get("regime_preset") or REGIME_PRESETS[regime_mode]
        preset = regime_preset[regime]
        return (
            regime,
            dict(preset["core_weights"]),
            float(preset["core_gross"]),
            float(preset["overlay_gross"]),
        )
    forced_regime = str(config.get("current_regime", "") or "")
    if regime_mode in REGIME_PRESETS and forced_regime in REGIME_PRESETS[regime_mode]:
        regime_preset = config.get("regime_preset") or REGIME_PRESETS[regime_mode]
        preset = regime_preset[forced_regime]
        return (
            forced_regime,
            dict(preset["core_weights"]),
            float(preset["core_gross"]),
            float(preset["overlay_gross"]),
        )
    return (
        "static",
        dict(config["core_weights"]),
        float(config["core_gross"]),
        float(config["overlay_gross"]),
    )


def _core_tickers_for_config(config: dict) -> list[str]:
    tickers = {"SPY", "QQQ"}
    if isinstance(config.get("core_weights"), dict):
        tickers.update(str(k).upper() for k in config["core_weights"])
    regime_preset = config.get("regime_preset")
    if isinstance(regime_preset, dict):
        for regime in ("risk_on", "neutral", "risk_off"):
            weights = regime_preset.get(regime, {}).get("core_weights", {})
            tickers.update(str(k).upper() for k in weights)
    return sorted(t for t in tickers if t)


def _top_count(n_names: int, shape: str) -> int:
    if shape == "top3":
        return 3
    if shape == "top5":
        return 5
    if shape == "top10":
        return 10
    if shape == "top15":
        return 15
    return max(1, int(np.ceil(n_names * 0.10)))


def _score_col(source: str) -> str:
    if source == "factor_plus_model":
        return "factor_plus_model_score"
    if source == "factor_walkforward":
        return "factor_walkforward_score"
    if source == "regime_adaptive":
        return "factor_walkforward_score"
    return "factor_score"


def _compute_regime_strength(dt: pd.Timestamp, regime_indicators: pd.DataFrame | None) -> float:
    """
    Compute a continuous 0-to-1 regime strength score.

    PLAIN ENGLISH: Instead of hard-switching between "risk_on" and "risk_off"
    scores, this gives a number between 0 and 1 that says HOW MUCH we're in
    risk_on territory.

    The score is based on 3 binary signals:
      - QQQ above its moving average (+1)
      - SPY above its moving average (+1)
      - Volatility NOT high (+1)

    So: 3/3 = 1.0 (full risk_on), 0/3 = 0.0 (full risk_off),
    1/3 or 2/3 = in between.  This lets us blend offensive and defensive
    scores proportionally rather than flipping all at once.
    """
    if regime_indicators is None:
        return 1.0  # No indicators → assume risk_on (default behavior)
    try:
        row = regime_indicators.loc[pd.Timestamp(dt)]
        signals = [
            bool(row["qqq_trend_ok"]),
            bool(row["spy_trend_ok"]),
            not bool(row["high_vol"]),
        ]
        return sum(signals) / len(signals)
    except (KeyError, IndexError):
        return 1.0


def _blended_score_col(
    day: pd.DataFrame,
    source: str,
    regime: str,
    regime_strength: float,
) -> str:
    """
    Create a blended score column in `day` that mixes risk_on and risk_off scores.

    PLAIN ENGLISH: When regime_strength is 1.0, you get pure risk_on scores
    (momentum).  When it's 0.0, you get pure risk_off scores (defensive).
    In between, you get a weighted average.  This prevents the portfolio from
    flipping overnight from all-momentum to all-defensive.

    Returns the name of the (possibly new) column in `day` that contains
    the blended scores.
    """
    # If strength is all-or-nothing, just use the existing column (no blend needed)
    if regime_strength >= 1.0:
        return _score_col_for_regime(source, "risk_on")
    if regime_strength <= 0.0:
        return _score_col_for_regime(source, "risk_off")

    risk_on_col = _score_col_for_regime(source, "risk_on")
    risk_off_col = _score_col_for_regime(source, "risk_off")

    # If both columns are the same (e.g. non-regime-adaptive source), no blend
    if risk_on_col == risk_off_col:
        return risk_on_col

    # Check both columns exist
    if risk_on_col not in day.columns or risk_off_col not in day.columns:
        return _score_col_for_regime(source, regime)

    # Blend: strength * risk_on + (1-strength) * risk_off
    # NOTE: We use a unique column name per strength level to avoid
    # overwriting the cached day DataFrame when multiple configs share it.
    # Rounding strength to 2 decimals keeps the number of columns manageable.
    strength_key = f"{regime_strength:.2f}".replace(".", "_")
    blend_col = f"_blended_regime_score_{strength_key}"
    if blend_col not in day.columns:
        day[blend_col] = (
            regime_strength * day[risk_on_col].fillna(0)
            + (1 - regime_strength) * day[risk_off_col].fillna(0)
        )
    return blend_col


def _score_col_for_regime(source: str, regime: str) -> str:
    if source == "regime_adaptive_riskoff_guard":
        # PLAIN ENGLISH: Risk-on and neutral keep the established route.  Only
        # risk-off uses a score-health guard that chooses the defensive score
        # or the walk-forward score from trailing, shifted IC history.
        if regime == "risk_on":
            return "factor_risk_on_score"
        if regime == "risk_off":
            return "factor_defensive_guard_score"
        return "factor_walkforward_score"
    if source == "regime_adaptive_consensus":
        if regime == "risk_on":
            return "factor_risk_on_consensus_score"
        if regime == "risk_off":
            return "factor_defensive_consensus_score"
        return "factor_walkforward_consensus_score"
    if source == "regime_adaptive_low_vol":
        if regime == "risk_on":
            return "factor_risk_on_low_vol_score"
        if regime == "risk_off":
            return "factor_defensive_score"
        return "factor_walkforward_low_vol_score"
    if source != "regime_adaptive":
        return _score_col(source)
    if regime == "risk_on":
        return "factor_risk_on_score"
    if regime == "risk_off":
        return "factor_defensive_score"
    return "factor_walkforward_score"


def _select_sticky_holdings(
    day: pd.DataFrame,
    held: set[str],
    *,
    score_col: str,
    return_col: str | None,
    shape: str,
    exit_rank_floor: float,
    max_per_sector: int,
    earnings_blackout_days: int = 0,
    diagnostics: dict | None = None,
) -> pd.DataFrame:
    required_cols = [score_col] if return_col is None else [score_col, return_col]
    ranked = day.dropna(subset=required_cols).copy()
    if ranked.empty:
        return ranked
    ranked["_rank_score"] = ranked[score_col].rank(pct=True)
    if earnings_blackout_days > 0 and "days_to_next_earnings" in ranked.columns:
        days_to_earnings = pd.to_numeric(ranked["days_to_next_earnings"], errors="coerce")
        ranked["_earnings_blackout"] = days_to_earnings.between(0, earnings_blackout_days, inclusive="both")
    else:
        ranked["_earnings_blackout"] = False
    ranked = ranked.sort_values("_rank_score", ascending=False)
    target_n = _top_count(len(ranked), shape)

    keep = ranked[(ranked["ticker"].isin(held)) & (ranked["_rank_score"] >= exit_rank_floor)]
    selected_rows = []
    sector_counts: dict[str, int] = {}

    def _can_add(row: pd.Series) -> bool:
        if max_per_sector <= 0:
            return True
        sector = str(row.get("sector", "OTHER"))
        return sector_counts.get(sector, 0) < max_per_sector

    def _add(row: pd.Series) -> None:
        selected_rows.append(row)
        sector = str(row.get("sector", "OTHER"))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for _idx, row in keep.sort_values("_rank_score", ascending=False).iterrows():
        if len(selected_rows) >= target_n:
            break
        if _can_add(row):
            _add(row)
    selected_tickers = {str(row["ticker"]) for row in selected_rows}

    # Track tickers skipped due to earnings blackout for observability.
    # PLAIN ENGLISH: We want to know exactly which stocks were excluded
    # (and why) so we can audit the overlay selection after each run.
    earnings_blackout_skipped: list[dict] = []

    for _idx, row in ranked.iterrows():
        if len(selected_rows) >= target_n:
            break
        ticker = str(row["ticker"])
        if ticker in selected_tickers:
            continue
        if bool(row.get("_earnings_blackout", False)):
            earnings_blackout_skipped.append({
                "ticker": ticker,
                "days_to_next_earnings": (
                    float(row.get("days_to_next_earnings"))
                    if pd.notna(row.get("days_to_next_earnings", np.nan))
                    else None
                ),
                "rank_score": (
                    float(row.get("_rank_score"))
                    if pd.notna(row.get("_rank_score", np.nan))
                    else None
                ),
            })
            continue
        if _can_add(row):
            _add(row)
            selected_tickers.add(ticker)

    if diagnostics is not None:
        diagnostics["earnings_blackout_skips"] = earnings_blackout_skipped

    if not selected_rows:
        result = ranked.iloc[0:0]
    else:
        result = pd.DataFrame(selected_rows)
    # Attach metadata for signal CSV output
    result.attrs["earnings_blackout_skipped"] = [str(item["ticker"]) for item in earnings_blackout_skipped]
    result.attrs["earnings_blackout_skipped_details"] = earnings_blackout_skipped
    return result


def _cap_and_rescale(weights: pd.Series, gross: float, cap: float) -> pd.Series:
    if weights.empty or cap <= 0:
        return weights
    capped = weights.clip(upper=cap)
    for _ in range(10):
        remaining = gross - float(capped.sum())
        if remaining <= 1e-9:
            break
        free = capped < cap - 1e-12
        if not bool(free.any()):
            break
        free_sum = float(capped[free].sum())
        if free_sum <= 0:
            capped.loc[free] = capped.loc[free] + remaining / int(free.sum())
        else:
            capped.loc[free] = capped.loc[free] + remaining * capped.loc[free] / free_sum
        capped = capped.clip(upper=cap)
    total = float(capped.sum())
    if total > 0 and total < gross - 1e-6:
        capped = capped / total * min(gross, total)
    return capped


def _overlay_weights(
    selected: pd.DataFrame,
    overlay_gross: float,
    weighting: str,
    *,
    max_single_name_weight: float = MAX_SINGLE_NAME_WEIGHT,
) -> pd.Series:
    if selected.empty or float(overlay_gross) <= 0.0:
        return pd.Series(dtype=float)
    if weighting == "score":
        raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
        raw_sum = float(raw.sum())
        weights = raw / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    elif weighting == "vol_score":
        raw = (selected["_rank_score"] - selected["_rank_score"].min() + 0.01).clip(lower=0.01)
        if "hvol_20d" in selected.columns:
            vol = pd.to_numeric(selected["hvol_20d"], errors="coerce")
        elif "factor_idio_vol_252_spy" in selected.columns:
            vol = pd.to_numeric(selected["factor_idio_vol_252_spy"], errors="coerce")
        else:
            vol = pd.Series(0.20, index=selected.index)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median()).clip(lower=0.05, upper=2.0)
        adjusted = raw / vol
        raw_sum = float(adjusted.sum())
        weights = adjusted / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    elif weighting == "risk_parity":
        # PLAIN ENGLISH: Equal Risk Contribution (ERC) approximation.
        # Instead of weighting by score (which ignores risk), we weight
        # inversely by each stock's recent volatility.  A stock with 40%
        # annualized vol gets HALF the position size of one with 20% vol.
        # This way each stock contributes roughly equal risk to the portfolio.
        # With only 3-5 stocks, inverse-vol is the standard ERC approximation
        # (full covariance-based ERC is too noisy with so few names).
        if "hvol_20d" in selected.columns:
            vol = pd.to_numeric(selected["hvol_20d"], errors="coerce")
        elif "factor_idio_vol_252_spy" in selected.columns:
            vol = pd.to_numeric(selected["factor_idio_vol_252_spy"], errors="coerce")
        else:
            vol = pd.Series(0.20, index=selected.index)
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(vol.median()).clip(lower=0.05, upper=2.0)
        inv_vol = 1.0 / vol
        raw_sum = float(inv_vol.sum())
        weights = inv_vol / raw_sum * overlay_gross if raw_sum > 0 else pd.Series(overlay_gross / len(selected), index=selected.index)
    else:
        weights = pd.Series(overlay_gross / len(selected), index=selected.index)
    weights.index = selected["ticker"].astype(str).to_numpy()
    weights = weights.astype(float)
    if max_single_name_weight > 0 and float(weights.max()) > max_single_name_weight:
        weights = _cap_and_rescale(weights, overlay_gross, max_single_name_weight)
    return weights


def _sticky_overlay_weights(
    selected: pd.DataFrame,
    overlay_gross: float,
    weighting: str,
    prev_overlay: pd.Series,
    *,
    max_single_name_weight: float = MAX_SINGLE_NAME_WEIGHT,
    sticky_blend: float = 0.65,
) -> pd.Series:
    # PLAIN ENGLISH: Map the "sticky_X" wrapper name to the base weighting method.
    # "sticky_score" → base "score", "sticky_vol_score" → score × inverse-vol.
    # The stickiness logic below blends the new weights with previous weights to
    # reduce unnecessary turnover (and thus trading costs).
    STICKY_MAP = {
        "sticky_score": "score",
        "sticky_vol_score": "vol_score",
        "sticky_risk_parity": "risk_parity",
    }
    is_sticky = weighting in STICKY_MAP
    current_weighting = STICKY_MAP.get(weighting, weighting)
    current = _overlay_weights(
        selected,
        overlay_gross,
        current_weighting,
        max_single_name_weight=max_single_name_weight,
    )
    if current.empty or prev_overlay.empty or not is_sticky:
        return current
    retained = current.index.intersection(prev_overlay.index)
    if retained.empty:
        return current
    prev = prev_overlay.reindex(current.index).fillna(0.0)
    if float(prev.abs().sum()) > 0:
        prev = prev / float(prev.abs().sum()) * overlay_gross
    blended = (1.0 - sticky_blend) * current + sticky_blend * prev
    missing = current.index.difference(retained)
    if len(missing) > 0:
        blended.loc[missing] = current.loc[missing]
    total = float(blended.abs().sum())
    if total > 0:
        blended = blended / total * overlay_gross
    if max_single_name_weight > 0 and float(blended.max()) > max_single_name_weight:
        blended = _cap_and_rescale(blended, overlay_gross, max_single_name_weight)
    return blended.astype(float)


def _exit_floor_for_regime(config: dict, regime: str) -> float:
    base = float(config.get("exit_rank_floor", 0.80))
    if str(config.get("adaptive_exit_mode", "fixed")) != "regime":
        return base
    if regime == "risk_on":
        return min(base, 0.70)
    if regime == "risk_off":
        return max(base, 0.90)
    return base


def run_core_satellite(
    panel: pd.DataFrame,
    config: dict,
    *,
    evaluation_start: pd.Timestamp | None = None,
    evaluation_end: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Run the strategy, optionally as a flat-start evaluation window.

    PLAIN ENGLISH: research folds may still use old rows to calculate moving
    averages, but they must not inherit a position chosen before the fold.  The
    optional boundaries therefore create a brand-new rebalance schedule inside
    the window and keep only complete holding periods.
    """
    holding_days = int(config.get("holding_days", HORIZON_DAYS))
    entry_delay_days = int(config.get("entry_delay_days", 0))
    if entry_delay_days > 0:
        return_col = f"forward_return_delay{entry_delay_days}_{holding_days}d"
    else:
        return_col = "forward_return" if holding_days == HORIZON_DAYS else f"forward_return_{holding_days}d"
    if return_col not in panel.columns:
        raise ValueError(f"Missing return column for core-satellite run: {return_col}")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    start_ts = pd.Timestamp(evaluation_start) if evaluation_start is not None else None
    end_ts = pd.Timestamp(evaluation_end) if evaluation_end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("evaluation_start must be on or before evaluation_end")

    # PLAIN ENGLISH: keep the full panel above for feature history, while only
    # scheduling new trades on dates that belong to this evaluation fold.
    schedule_dates = dates
    if start_ts is not None:
        schedule_dates = schedule_dates[schedule_dates >= start_ts]
    if end_ts is not None:
        schedule_dates = schedule_dates[schedule_dates <= end_ts]
    rebalance_dates = schedule_dates[::holding_days]

    # Count the old schedule position that would have crossed into this fold.
    # It is evidence of what was deliberately purged, never a scored trade.
    full_schedule = dates[::holding_days]
    purged_leading = 0
    if start_ts is not None:
        full_exits = pd.DatetimeIndex([
            pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
            for dt in full_schedule
        ])
        purged_leading = int(((full_schedule < start_ts) & (full_exits >= start_ts)).sum())

    # A return whose exit lies after the fold would use future evidence.  Drop
    # that rebalance before any portfolio state is created.
    purged_trailing = 0
    if end_ts is not None:
        nominal_exits = pd.DatetimeIndex([
            pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
            for dt in rebalance_dates
        ])
        keep = nominal_exits <= end_ts
        purged_trailing = int((~keep).sum())
        rebalance_dates = rebalance_dates[keep]
    if len(rebalance_dates) == 0:
        raise ValueError("No complete holding periods inside evaluation boundaries")
    entry_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days) for dt in rebalance_dates])
    exit_dates = pd.DatetimeIndex([pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days) for dt in rebalance_dates])
    price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(entry_dates) | set(exit_dates)))
    etf_tickers = _core_tickers_for_config(config)
    etf_prices = _cached_etf_prices(price_index, etf_tickers)
    regime_indicators = None
    if str(config.get("regime_mode", "static")) in REGIME_PRESETS:
        regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)

    # --- EARLY REGIME-CHANGE REBALANCE ---
    # PLAIN ENGLISH: Normally we only rebalance every N days (e.g. every 10
    # trading days).  But if the regime changes in the middle of a holding
    # period (e.g. risk_on → risk_off on day 3), we're stuck with the wrong
    # positioning for 7 more days.
    #
    # When early_rebalance_on_regime_change is True, we check every trading
    # day for regime changes and insert extra rebalance dates when the regime
    # flips.  This means the portfolio adapts faster to market shifts.
    #
    # Risk: More rebalances = more turnover = more cost.  The grid search
    # will test both settings and pick whichever gives better net returns.
    if config.get("early_rebalance_on_regime_change", False) and regime_indicators is not None:
        extra_dates = []
        prev_regime = None
        for dt in schedule_dates:
            ts = pd.Timestamp(dt)
            if ts not in regime_indicators.index:
                continue
            row = regime_indicators.loc[ts]
            if "vix_inverted" in row.index and bool(row.get("vix_inverted", False)):
                cur_regime = "risk_off"
            elif bool(row["qqq_trend_ok"]) and bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
                cur_regime = "risk_on"
            elif bool(row["spy_trend_ok"]) and not bool(row["high_vol"]):
                cur_regime = "neutral"
            else:
                cur_regime = "risk_off"

            if prev_regime is not None and cur_regime != prev_regime:
                # Regime changed — add this date as a rebalance point
                if ts not in rebalance_dates:
                    extra_dates.append(ts)
            prev_regime = cur_regime

        if extra_dates:
            # Merge extra dates into rebalance schedule and re-sort
            combined = sorted(set(rebalance_dates) | set(extra_dates))
            rebalance_dates = pd.DatetimeIndex(combined)
            # Recompute entry/exit for all dates
            entry_dates = pd.DatetimeIndex([
                pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days)
                for dt in rebalance_dates
            ])
            exit_dates = pd.DatetimeIndex([
                pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
                for dt in rebalance_dates
            ])
            price_index = pd.DatetimeIndex(sorted(
                set(rebalance_dates) | set(entry_dates) | set(exit_dates)
            ))
            etf_prices = _cached_etf_prices(price_index, etf_tickers)
            # Reload regime indicators with expanded date range
            regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)

    # Regime changes can add a late rebalance after the regular schedule was
    # filtered.  Apply the same full-period rule again so that shortcut cannot
    # cross the fold end.
    if end_ts is not None:
        final_nominal_exits = pd.DatetimeIndex([
            pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
            for dt in rebalance_dates
        ])
        final_keep = final_nominal_exits <= end_ts
        purged_trailing += int((~final_keep).sum())
        rebalance_dates = rebalance_dates[final_keep]
        if len(rebalance_dates) == 0:
            raise ValueError("No complete holding periods inside evaluation boundaries")
        entry_dates = pd.DatetimeIndex([
            pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days) for dt in rebalance_dates
        ])
        exit_dates = pd.DatetimeIndex([
            pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days) for dt in rebalance_dates
        ])
        price_index = pd.DatetimeIndex(sorted(set(rebalance_dates) | set(entry_dates) | set(exit_dates)))
        etf_prices = _cached_etf_prices(price_index, etf_tickers)
        if regime_indicators is not None:
            regime_indicators = _load_regime_indicators(rebalance_dates, exit_dates, config)
    day_map = _panel_day_map(panel)

    equity = INITIAL_CAPITAL
    held: set[str] = set()
    prev_overlay = pd.Series(dtype=float)
    rows = [{"date": pd.Timestamp(rebalance_dates[0]), "equity": equity, "strategy_ret": 0.0}]
    trade_rows: list[dict] = []
    total_turnover = 0.0
    total_cost = 0.0
    ticker_contrib: dict[str, float] = {}
    concentration_overlay_adjustment_sum = 0.0
    concentration_overlay_active_count = 0

    # ── Drawdown circuit breaker state ─────────────────────────────────
    # PLAIN ENGLISH: Track the highest equity we've ever had.  If current
    # equity drops too far below that peak, go to 100% cash.  We stay in
    # cash until the REGIME turns back to risk_on — because cash doesn't
    # grow, so we can't wait for portfolio equity to "recover" on its own.
    # Instead we wait for the market to show strength again (regime = risk_on)
    # before re-entering.
    dd_threshold = float(config.get("drawdown_circuit_breaker", 0.0))
    peak_equity = equity       # highest equity seen so far
    circuit_breaker_active = False  # True = we're in cash, waiting for risk_on

    # ── Volatility targeting state ─────────────────────────────────────
    # PLAIN ENGLISH: Keep a rolling window of recent portfolio returns so
    # we can compute realized volatility.  If a target is set, we scale
    # exposure each period to keep vol near the target.
    vol_target = float(config.get("vol_target", 0.0))
    recent_returns: list[float] = []  # rolling window of strategy returns

    for dt in rebalance_dates:
        day = day_map[pd.Timestamp(dt)]

        regime, core_weights, core_gross, overlay_gross = _resolve_allocation(pd.Timestamp(dt), config, regime_indicators)

        # ── Check circuit breaker ──────────────────────────────────────
        # PLAIN ENGLISH: If the circuit breaker threshold is set (> 0):
        #   1. If breaker is OFF: check if we've dropped too far from peak.
        #      If yes, go to 100% cash (activate breaker).
        #   2. If breaker is ON: check if regime is back to risk_on.
        #      If yes, deactivate breaker and re-enter the market.
        #      We use regime (not equity recovery) because cash doesn't grow —
        #      we'd be stuck forever waiting for equity to recover while in cash.
        if dd_threshold > 0.0:
            drawdown_from_peak = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
            if circuit_breaker_active:
                # Re-enter when regime turns risk_on (market showing strength)
                if regime == "risk_on":
                    circuit_breaker_active = False
                    # Reset peak to current equity so we don't immediately re-trigger
                    peak_equity = equity
            else:
                # Trigger breaker if drawdown exceeds threshold
                if drawdown_from_peak >= dd_threshold:
                    circuit_breaker_active = True

        # If circuit breaker is active, override to zero exposure (100% cash)
        if circuit_breaker_active:
            core_gross = 0.0
            overlay_gross = 0.0

        # ── Volatility targeting ──────────────────────────────────────────
        # PLAIN ENGLISH: If vol targeting is enabled, compute how volatile
        # our recent returns have been.  If we're running hotter than target,
        # shrink positions.  If calmer, grow them (up to 1.5x cap).
        # Need at least VOL_TARGET_LOOKBACK periods of history before scaling.
        if vol_target > 0.0 and len(recent_returns) >= VOL_TARGET_LOOKBACK:
            recent_arr = np.array(recent_returns[-VOL_TARGET_LOOKBACK:])
            realized_vol = float(np.std(recent_arr, ddof=1)) * np.sqrt(252.0 / holding_days)
            if realized_vol > 1e-6:
                vol_scale = float(np.clip(
                    vol_target / realized_vol,
                    VOL_TARGET_MIN_SCALE,
                    VOL_TARGET_MAX_SCALE,
                ))
                core_gross *= vol_scale
                overlay_gross *= vol_scale

        if "feature_health_overlay_allowed" in day.columns and not bool(day["feature_health_overlay_allowed"].iloc[0]):
            overlay_gross = 0.0

        overlay_before_concentration = overlay_gross
        overlay_gross, concentration_overlay_target, concentration_gap = _apply_concentration_overlay_target(
            pd.Timestamp(dt),
            core_gross,
            overlay_gross,
            regime_indicators,
            config,
        )
        concentration_overlay_adjustment = overlay_gross - overlay_before_concentration
        concentration_overlay_adjustment_sum += concentration_overlay_adjustment
        if abs(concentration_overlay_adjustment) > 1e-12:
            concentration_overlay_active_count += 1

        # PLAIN ENGLISH: If score blending is enabled, we mix risk_on and
        # risk_off scores based on how "risk_on" the market really is (0-1).
        # This avoids the jarring overnight flip from all-momentum to all-defensive.
        if config.get("score_blend", False):
            strength = _compute_regime_strength(pd.Timestamp(dt), regime_indicators)
            score_col = _blended_score_col(day, str(config["score_source"]), regime, strength)
        else:
            score_col = _score_col_for_regime(str(config["score_source"]), regime)
        exit_floor = _exit_floor_for_regime(config, regime)
        selected = _select_sticky_holdings(
            day,
            held,
            score_col=score_col,
            return_col=return_col,
            shape=config["shape"],
            exit_rank_floor=exit_floor,
            max_per_sector=int(config["max_per_sector"]),
            earnings_blackout_days=int(config.get("earnings_blackout_days", 0)),
        )
        overlay = _sticky_overlay_weights(
            selected,
            overlay_gross,
            config["weighting"],
            prev_overlay,
            max_single_name_weight=float(config.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
            # A candidate may ask for more weight persistence.  The approved
            # live configuration omits this key, so its historical 0.65 value
            # remains frozen unless a shadow experiment sets it explicitly.
            sticky_blend=float(config.get("sticky_blend", 0.65)),
        )
        held = set(overlay.index.astype(str))

        aligned = pd.concat([prev_overlay.rename("prev"), overlay.rename("now")], axis=1).fillna(0.0)
        turnover = float((aligned["now"] - aligned["prev"]).abs().sum())
        extra_cost = turnover * float(config.get("extra_turnover_cost_bps", 0.0)) / 10_000.0
        base_turnover_cost = calibrated_turnover_cost_pct()
        cost = turnover * base_turnover_cost * float(config.get("cost_stress", COST_STRESS_MULTIPLIERS[0])) + extra_cost
        total_turnover += turnover
        total_cost += cost

        # ── Compute exit date and holding period scaling ────────────────
        # PLAIN ENGLISH: If early rebalance is enabled, the ACTUAL holding period
        # might be shorter than holding_days (because we rebalance early on regime
        # change).  We use the actual next rebalance date as exit, not the full
        # holding_days.  This prevents double-counting returns across overlapping
        # periods.
        entry_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(entry_delay_days)
        try:
            dt_idx = rebalance_dates.get_loc(dt)
        except KeyError:
            dt_idx = -1
        if isinstance(dt_idx, int) and dt_idx >= 0 and dt_idx + 1 < len(rebalance_dates):
            next_rebal = pd.Timestamp(rebalance_dates[dt_idx + 1])
            exit_dt = next_rebal + pd.tseries.offsets.BDay(entry_delay_days)
        else:
            exit_dt = pd.Timestamp(dt) + pd.tseries.offsets.BDay(holding_days + entry_delay_days)
        # Compute actual days held for scaling factor returns
        actual_bdays = max(1, len(pd.bdate_range(entry_dt, exit_dt)) - 1)
        # PLAIN ENGLISH: factor_scale adjusts pre-computed forward returns
        # (which assume full holding_days) to the actual shorter period.
        # E.g. if expected 10 days but held only 3, multiply returns by 3/10.
        factor_scale = actual_bdays / max(1, holding_days)

        factor_ret = 0.0
        max_sector_weight = 0.0
        effective_overlay_names = 0.0
        if not overlay.empty:
            selected_by_ticker = selected.set_index("ticker")
            ticker_returns = selected_by_ticker.loc[overlay.index, return_col]
            ticker_period_contrib = ticker_returns * overlay * factor_scale
            factor_ret = float(ticker_period_contrib.sum())
            for ticker, value in ticker_period_contrib.items():
                ticker_contrib[str(ticker)] = ticker_contrib.get(str(ticker), 0.0) + float(value)
            sector_weights = selected_by_ticker.loc[overlay.index, "sector"].to_frame().join(overlay.rename("weight"))
            max_sector_weight = float(sector_weights.groupby("sector")["weight"].sum().abs().max())
            norm_w = overlay.abs() / max(float(overlay.abs().sum()), 1e-9)
            effective_overlay_names = float(1.0 / max(float((norm_w ** 2).sum()), 1e-9))

        core_component_ret = 0.0
        for ticker, weight in core_weights.items():
            ticker = str(ticker).upper()
            if ticker not in etf_prices.columns:
                continue
            etf_ret = float(etf_prices.loc[exit_dt, ticker] / etf_prices.loc[entry_dt, ticker] - 1.0)
            core_component_ret += float(weight) * etf_ret
        core_ret = core_gross * core_component_ret
        strategy_ret = core_ret + factor_ret - cost
        equity *= 1.0 + strategy_ret
        # Track returns for vol targeting (add BEFORE updating peak)
        recent_returns.append(strategy_ret)
        # Update peak equity for circuit breaker tracking
        if equity > peak_equity:
            peak_equity = equity

        rows.append({"date": exit_dt, "equity": equity, "strategy_ret": strategy_ret})
        trade_rows.append({
            "date": pd.Timestamp(dt),
            "exit_date": exit_dt,
            "regime": regime,
            "circuit_breaker_active": circuit_breaker_active,
            "score_col": score_col,
            "adaptive_exit_mode": str(config.get("adaptive_exit_mode", "fixed")),
            "exit_rank_floor_used": exit_floor,
            "earnings_blackout_days": int(config.get("earnings_blackout_days", 0)),
            "entry_delay_days": entry_delay_days,
            "extra_turnover_cost_bps": float(config.get("extra_turnover_cost_bps", 0.0)),
            "n_overlay_positions": int(len(overlay)),
            "core_spy_weight": float(core_weights.get("SPY", 0.0)),
            "core_qqq_weight": float(core_weights.get("QQQ", 0.0)),
            "core_tqqq_weight": float(core_weights.get("TQQQ", 0.0)),
            "core_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in core_weights.items()}, sort_keys=True),
            "core_gross": core_gross,
            "overlay_gross": overlay_gross,
            "concentration_overlay_target": concentration_overlay_target,
            "concentration_overlay_adjustment": concentration_overlay_adjustment,
            "concentration_qqq_spy_120d": concentration_gap,
            "gross_exposure": core_gross + float(overlay.abs().sum()),
            "top_overlay_weight": float(overlay.abs().max()) if not overlay.empty else 0.0,
            "effective_overlay_names": effective_overlay_names,
            "max_sector_overlay_weight": max_sector_weight,
            "overlay_tickers": ",".join(overlay.index.astype(str).tolist()),
            "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
            "turnover": turnover,
            "cost": cost,
            "core_return": core_ret,
            "factor_overlay_return": factor_ret,
            "period_return": strategy_ret,
        })
        prev_overlay = overlay

    equity_series = pd.DataFrame(rows).drop_duplicates("date").set_index("date")["equity"].sort_index()
    trades = pd.DataFrame(trade_rows)
    positive_contrib = {k: v for k, v in ticker_contrib.items() if v > 0}
    positive_contrib_sum = sum(positive_contrib.values())
    if positive_contrib and positive_contrib_sum > 0:
        top_ticker, top_value = max(positive_contrib.items(), key=lambda kv: kv[1])
        top_ticker_share = float(top_value / positive_contrib_sum)
    else:
        top_ticker, top_ticker_share = "", 1.0
    extra = {
        "turnover_pct": round(total_turnover * 100.0, 2),
        "estimated_cost_pct": round(total_cost * 100.0, 4),
        "avg_gross_exposure": round(float(trades["gross_exposure"].mean()), 3) if not trades.empty else 0.0,
        "avg_overlay_positions": round(float(trades["n_overlay_positions"].mean()), 2) if not trades.empty else 0.0,
        "max_single_name_weight": round(float(trades["top_overlay_weight"].max()), 4) if not trades.empty else 0.0,
        "avg_effective_overlay_names": round(float(trades["effective_overlay_names"].mean()), 3) if not trades.empty else 0.0,
        "max_sector_overlay_weight": round(float(trades["max_sector_overlay_weight"].max()), 4) if not trades.empty else 0.0,
        "top_ticker_overlay_contributor": top_ticker,
        "top_ticker_overlay_contribution_share": round(top_ticker_share, 3),
        "n_rebalances": int(len(trades)),
        "boundary_mode": "flat_start_full_periods_only" if start_ts is not None or end_ts is not None else "full_history",
        "evaluation_trade_start": str(pd.Timestamp(rebalance_dates[0]).date()) if len(rebalance_dates) else None,
        "evaluation_trade_end": str(pd.Timestamp(trades["exit_date"].max()).date()) if not trades.empty else None,
        "purged_leading_trade_count": purged_leading,
        "purged_trailing_trade_count": purged_trailing,
        "concentration_overlay_active_rebalances": int(concentration_overlay_active_count),
        "avg_concentration_overlay_adjustment": round(
            float(concentration_overlay_adjustment_sum / max(len(trades), 1)),
            4,
        ),
    }
    if "regime" in trades.columns:
        extra["regime_counts"] = {str(k): int(v) for k, v in trades["regime"].value_counts().sort_index().items()}
    return equity_series, trades, extra


def evaluate(panel: pd.DataFrame, config: dict) -> tuple[dict, pd.Series, pd.DataFrame]:
    equity, trades, extra = run_core_satellite(panel, config)
    periods_per_year = 252.0 / int(config.get("holding_days", HORIZON_DAYS))
    stats = portfolio_stats(equity, periods_per_year)
    bench = benchmark_equity(pd.DatetimeIndex(equity.index))
    bench_stats = {symbol: portfolio_stats(bench[symbol], periods_per_year) for symbol in bench.columns}
    comps = compare_to_benchmarks(equity, bench)
    subs = subperiod_metrics(equity, bench)
    holdout = _holdout_comparisons(equity, bench, start="2023-01-01", end="2026-12-31")
    strat_rets = equity.pct_change().fillna(0.0)
    blend_rets = bench["BLEND"].pct_change().reindex(equity.index).fillna(0.0)
    yearly_alpha = (strat_rets - blend_rets).groupby(equity.index.year).sum() * 100.0
    metrics = {
        **config,
        **stats,
        **extra,
        "benchmark_comparisons": comps,
        "benchmark_stats": bench_stats,
        "subperiods": subs,
        "holdout_2023_2026": holdout,
        "yearly_alpha_pct": {str(k): round(float(v), 2) for k, v in yearly_alpha.items()},
    }
    metrics["cost_stress"] = float(config.get("cost_stress", COST_STRESS_MULTIPLIERS[0]))
    gates = gate_metrics(metrics, bench_stats, subs, yearly_alpha)
    gates.update(_core_robust_gate_overrides(metrics, holdout, yearly_alpha))
    health = dict(getattr(panel, "attrs", {}).get("feature_health_summary") or {"feature_health_gate_pass": True})
    gates["feature_health_gate_pass"] = bool(health.get("feature_health_gate_pass", True))
    gates["all_pass"] = all(v for k, v in gates.items() if k.endswith("_pass"))
    metrics["core_satellite_gate_results"] = gates
    metrics.update({
        "feature_health_gate_pass": bool(health.get("feature_health_gate_pass", True)),
        "feature_health_gate_reasons": health.get("feature_health_gate_reasons", []),
        "raw_feature_count": int(health.get("raw_feature_count", 0) or 0),
        "active_cluster_count": int(health.get("active_cluster_count", 0) or 0),
        "effective_cluster_count": int(health.get("effective_cluster_count", 0) or 0),
        "quarantined_features": health.get("quarantined_features", []),
        "watchlist_features": health.get("watchlist_features", []),
        "max_cluster_weight": float(health.get("max_cluster_weight", 0.0) or 0.0),
    })
    metrics["paper_ready"] = False
    return metrics, equity, trades


def _holdout_comparisons(equity: pd.Series, bench: pd.DataFrame, *, start: str, end: str) -> dict:
    eq = equity.loc[(equity.index >= start) & (equity.index <= end)]
    if len(eq) < 3:
        return {"data_available": False}
    out: dict[str, object] = {"data_available": True}
    eq_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    out["strategy_return_pct"] = round(eq_ret * 100.0, 2)
    for symbol in ("SPY", "QQQ", "BLEND"):
        b = bench[symbol].reindex(eq.index).ffill().bfill()
        bm_ret = float(b.iloc[-1] / b.iloc[0] - 1.0)
        out[f"{symbol.lower()}_return_pct"] = round(bm_ret * 100.0, 2)
        out[f"alpha_vs_{symbol.lower()}_pct"] = round((eq_ret - bm_ret) * 100.0, 2)
    return out


def _core_robust_gate_overrides(metrics: dict, holdout: dict, yearly_alpha: pd.Series) -> dict:
    positive_alpha_years = yearly_alpha[yearly_alpha > 0]
    if not positive_alpha_years.empty:
        year_share = float(positive_alpha_years.max() / max(float(positive_alpha_years.sum()), 1e-9))
    else:
        year_share = 1.0
    top_ticker_share = float(metrics.get("top_ticker_overlay_contribution_share", 1.0) or 1.0)
    max_weight = float(metrics.get("max_single_name_weight", 1.0) or 1.0)
    holdout_available = bool(holdout.get("data_available", False))
    return {
        "year_alpha_concentration_pass": bool(year_share <= MAX_POSITIVE_YEAR_ALPHA_SHARE),
        "holdout_2023_2026_vs_qqq_pass": bool(holdout_available and float(holdout.get("alpha_vs_qqq_pct", -999.0)) > 0),
        "holdout_2023_2026_vs_blend_pass": bool(holdout_available and float(holdout.get("alpha_vs_blend_pct", -999.0)) > 0),
        "single_name_weight_cap_pass": bool(max_weight <= MAX_SINGLE_NAME_WEIGHT + 1e-9),
        "top_ticker_contribution_pass": bool(top_ticker_share <= MAX_TOP_TICKER_CONTRIB_SHARE),
        "max_positive_year_alpha_share": round(year_share, 3),
    }



def _scale_paper_targets_to_gross(
    *,
    target_spy: float,
    target_qqq: float,
    target_tqqq: float = 0.0,
    overlay: pd.Series,
    max_gross: float = PAPER_MAX_GROSS_EXPOSURE,
) -> tuple[float, float, float, pd.Series, float, float, bool]:
    """
    Convert a research allocation into a broker-safe paper allocation.

    Backtests may intentionally test 1.25x gross exposure, but paper order
    submission should default to <= 1.00x gross so the broker does not
    reject buy orders because of insufficient buying power.

    Returns: (target_spy, target_qqq, target_tqqq, overlay, raw_gross, scale, scaled)
    """
    raw_gross = float(abs(target_spy) + abs(target_qqq) + abs(target_tqqq) + float(overlay.abs().sum()))
    if raw_gross <= 0 or raw_gross <= float(max_gross) + 1e-9:
        return target_spy, target_qqq, target_tqqq, overlay.copy(), raw_gross, 1.0, False
    scale = float(max_gross) / raw_gross
    return target_spy * scale, target_qqq * scale, target_tqqq * scale, overlay * scale, raw_gross, scale, True


def _paper_signal_timestamp() -> str:
    try:
        tz = ZoneInfo(str(PAPER_SIGNAL_TIMEZONE))
    except Exception:
        tz = datetime.now().astimezone().tzinfo
    return datetime.now(tz).isoformat(timespec="minutes")


def _core_satellite_signal_path() -> Path:
    return Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"


def _alpaca_daily_status_path() -> Path:
    """Return the Alpaca status snapshot used by sticky live holdings."""
    return Path(SIGNAL_DIR) / "alpaca_daily_status.json"


def _select_live_status_path() -> tuple[Path, str]:
    """Use Alpaca account state as the sticky live-holdings source."""
    path = _alpaca_daily_status_path()
    return path, "alpaca_status_present" if path.exists() else "alpaca_status_missing"


def _empty_live_sticky_state(reason: str, source: str = "none") -> dict:
    return {
        "source": source,
        "used": False,
        "held_tickers": set(),
        "prev_overlay": pd.Series(dtype=float),
        "reason": reason,
    }


def _normalise_sticky_ticker(ticker: object) -> str:
    value = str(ticker).strip().upper()
    return value.split(".", 1)[1] if "." in value else value


def _overlay_series_from_values(values: dict, equity: float) -> pd.Series:
    if not np.isfinite(float(equity)) or float(equity) <= 0:
        return pd.Series(dtype=float)
    weights: dict[str, float] = {}
    for raw_ticker, raw_value in values.items():
        ticker = _normalise_sticky_ticker(raw_ticker)
        if not ticker or ticker in STICKY_STATE_EXCLUDED_TICKERS:
            continue
        try:
            value = abs(float(raw_value or 0.0))
        except Exception:
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        weights[ticker] = value / float(equity)
    if not weights:
        return pd.Series(dtype=float)
    return pd.Series(weights, dtype=float).sort_index()


def _overlay_series_from_prior_signal(signal_path: Path) -> pd.Series:
    if not signal_path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(signal_path)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    try:
        raw = json.loads(str(df.iloc[0].get("overlay_weights_json", "{}") or "{}"))
    except Exception:
        return pd.Series(dtype=float)
    weights: dict[str, float] = {}
    for raw_ticker, raw_weight in dict(raw).items():
        ticker = _normalise_sticky_ticker(raw_ticker)
        if not ticker or ticker in STICKY_STATE_EXCLUDED_TICKERS:
            continue
        try:
            weight = abs(float(raw_weight or 0.0))
        except Exception:
            continue
        if np.isfinite(weight) and weight > 0:
            weights[ticker] = weight
    if not weights:
        return pd.Series(dtype=float)
    return pd.Series(weights, dtype=float).sort_index()


def _load_live_sticky_overlay_state(
    *,
    status_path: Path | None = None,
    signal_path: Path | None = None,
) -> dict:
    """Return live overlay holdings for sticky live selection.

    Reads Alpaca's daily status first.  If that file has no usable
    holdings, it falls back to the prior unified signal overlay.
    """
    if status_path is None:
        status_path, _selection_reason = _select_live_status_path()
    else:
        status_path = Path(status_path)
        _selection_reason = "explicit_override"
    signal_path = Path(signal_path or _core_satellite_signal_path())

    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            status = None
        if isinstance(status, dict):
            try:
                equity = float(status.get("account_equity", 0.0) or 0.0)
            except Exception:
                equity = 0.0
            position_values = status.get("position_values", {})
            broker = str(status.get("broker", "alpaca"))
            if isinstance(position_values, dict) and equity > 0 and np.isfinite(equity):
                prev_overlay = _overlay_series_from_values(position_values, equity)
                source_tag = f"{broker}_daily_status"
                reason_tag = (
                    f"loaded_from_{broker}_status ({_selection_reason})"
                    if not prev_overlay.empty
                    else f"no_overlay_positions_in_{broker}_status ({_selection_reason})"
                )
                return {
                    "source": source_tag,
                    "used": not prev_overlay.empty,
                    "held_tickers": set(prev_overlay.index.astype(str)),
                    "prev_overlay": prev_overlay,
                    "reason": reason_tag,
                }

    prev_overlay = _overlay_series_from_prior_signal(signal_path)
    if not prev_overlay.empty:
        return {
            "source": "previous_signal",
            "used": True,
            "held_tickers": set(prev_overlay.index.astype(str)),
            "prev_overlay": prev_overlay,
            "reason": "loaded_from_previous_signal",
        }
    return _empty_live_sticky_state("no_live_or_prior_overlay_state")


def _mark_live_regime_failure(metrics: dict, exc: Exception) -> dict:
    error = str(exc).strip() or exc.__class__.__name__
    if len(error) > 300:
        error = error[:297] + "..."
    metrics["paper_ready"] = False
    gates = dict(metrics.get("core_satellite_gate_results", {}) or {})
    gates["all_pass"] = False
    metrics["core_satellite_gate_results"] = gates
    metrics["live_regime_refresh_failed"] = True
    metrics["live_regime_refresh_error"] = error
    reasons = [str(x) for x in metrics.get("live_gate_reasons", []) if str(x)]
    if "regime_refresh_failed" not in reasons:
        reasons.append("regime_refresh_failed")
    metrics["live_gate_reasons"] = reasons
    return metrics


def write_paper_signal(panel: pd.DataFrame, metrics: dict) -> Path:
    holding_days = int(metrics.get("holding_days", HORIZON_DAYS))
    latest_date = pd.Timestamp(panel["date"].max())
    regime_indicators = None
    regime_refresh_failed = False
    current_regime = str(metrics.get("current_regime", "static"))
    core_weights: dict[str, float] = {}
    core_gross = 0.0
    overlay_gross = 0.0
    if str(metrics.get("regime_mode", "static")) in REGIME_PRESETS:
        try:
            regime_indicators = _load_regime_indicators(
                pd.DatetimeIndex([latest_date]),
                pd.DatetimeIndex([latest_date]),
                metrics,
            )
        except Exception as exc:
            metrics = _mark_live_regime_failure(metrics, exc)
            regime_refresh_failed = True
            preset = REGIME_PRESETS[str(metrics.get("regime_mode", "static"))]
            if current_regime not in preset:
                last_known = str(metrics.get("last_known_regime", ""))
                current_regime = last_known if last_known in preset else "unknown"
            print(f"Live regime refresh failed; writing non-tradeable signal: {exc}")
    if not regime_refresh_failed:
        current_regime, core_weights, core_gross, overlay_gross = _resolve_allocation(latest_date, metrics, regime_indicators)
    feature_health_gate_pass = bool(metrics.get("feature_health_gate_pass", True))
    feature_health_reason = ""
    if not feature_health_gate_pass:
        overlay_gross = 0.0
        feature_health_reason = "feature_health_gate_failed"
    overlay_before_concentration = overlay_gross
    overlay_gross, concentration_overlay_target, concentration_gap = _apply_concentration_overlay_target(
        latest_date,
        core_gross,
        overlay_gross,
        regime_indicators,
        metrics,
    )
    metrics["concentration_overlay_target"] = round(float(concentration_overlay_target), 6)
    metrics["concentration_overlay_adjustment"] = round(
        float(overlay_gross - overlay_before_concentration),
        6,
    )
    metrics["concentration_qqq_spy_120d"] = round(float(concentration_gap), 6)
    score_col = _score_col_for_regime(str(metrics["score_source"]), current_regime)
    day = panel[panel["date"] == latest_date]
    sticky_state = _load_live_sticky_overlay_state()
    held_tickers = set(sticky_state["held_tickers"])
    prev_overlay = sticky_state["prev_overlay"]
    metrics["sticky_holdings_source"] = str(sticky_state["source"])
    metrics["sticky_holdings_used"] = bool(sticky_state["used"])
    metrics["sticky_held_tickers"] = sorted(held_tickers)
    metrics["sticky_prev_overlay_json"] = json.dumps(
        {str(k): round(float(v), 6) for k, v in prev_overlay.items()}, sort_keys=True
    )
    metrics["sticky_holdings_reason"] = str(sticky_state["reason"])
    selection_diagnostics: dict = {}
    selected = _select_sticky_holdings(
        day,
        held_tickers,
        score_col=score_col,
        return_col=None,
        shape=str(metrics["shape"]),
        exit_rank_floor=float(metrics["exit_rank_floor"]),
        max_per_sector=int(metrics["max_per_sector"]),
        earnings_blackout_days=int(metrics.get("earnings_blackout_days", 0)),
        diagnostics=selection_diagnostics,
    )
    blackout_skips = list(selection_diagnostics.get("earnings_blackout_skips", []))
    blackout_tickers = [str(item.get("ticker", "")) for item in blackout_skips if item.get("ticker")]
    metrics["earnings_blackout_skipped_count"] = int(len(blackout_skips))
    metrics["earnings_blackout_skipped_tickers"] = ",".join(blackout_tickers)
    metrics["earnings_blackout_skipped_json"] = json.dumps(blackout_skips, sort_keys=True)
    if blackout_skips:
        print(
            f"  ⚠ Earnings blackout skipped {len(blackout_skips)} tickers: "
            f"{', '.join(blackout_tickers)}"
        )

    # ── Sentiment veto: remove stocks with very negative recent news ──────
    # PLAIN ENGLISH: Check if any selected stock has terrible news right now.
    # If so, drop it and pick the next-best candidate from the factor rankings.
    sentiment_scores = {}
    if SENTIMENT_VETO_ENABLED and not selected.empty:
        try:
            selected, sentiment_scores = _apply_sentiment_veto(
                selected, day,
                score_col=score_col,
                shape=str(metrics["shape"]),
                exit_rank_floor=float(metrics["exit_rank_floor"]),
                max_per_sector=int(metrics["max_per_sector"]),
                earnings_blackout_days=int(metrics.get("earnings_blackout_days", 0)),
            )
        except Exception as exc:
            print(f"  ⚠ Sentiment veto failed (proceeding without): {exc}")

    overlay = _sticky_overlay_weights(
        selected,
        overlay_gross,
        str(metrics["weighting"]),
        prev_overlay,
        max_single_name_weight=float(metrics.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)),
    )
    raw_target_spy = core_gross * float(core_weights.get("SPY", 0.0))
    raw_target_qqq = core_gross * float(core_weights.get("QQQ", 0.0))
    # TQQQ weight comes from the unified grid — 0.0 when the winning config
    # has no TQQQ, positive when the data decided TQQQ helps.
    raw_target_tqqq = core_gross * float(core_weights.get("TQQQ", 0.0))
    raw_gross = core_gross + float(overlay.abs().sum())
    target_spy, target_qqq, target_tqqq, paper_overlay, raw_paper_gross, paper_scale, paper_scaled = _scale_paper_targets_to_gross(
        target_spy=raw_target_spy,
        target_qqq=raw_target_qqq,
        target_tqqq=raw_target_tqqq,
        overlay=overlay,
        max_gross=PAPER_MAX_GROSS_EXPOSURE,
    )
    gross = abs(target_spy) + abs(target_qqq) + abs(target_tqqq) + float(paper_overlay.abs().sum())
    row = {
        "paper_signal_type": "core_satellite_alpha",
        "paper_ready": bool(metrics.get("paper_ready", False)),
        "core_preset": metrics["core_preset"],
        "risk_control_mode": str(metrics.get("risk_control_mode", "off")),
        "regime_mode": str(metrics.get("regime_mode", "static")),
        "current_regime": current_regime,
        "score_source": metrics["score_source"],
        "live_config_hash": str(metrics.get("live_config_hash", "")),
        "live_config_created_at": str(metrics.get("live_config_created_at", "")),
        "live_config_source_json": str(metrics.get("live_config_source_json", "")),
        "approved_family_signature": str(metrics.get("approved_family_signature", "")),
        "approved_exact_config": str(metrics.get("approved_exact_config", "")),
        "target_spy_weight": round(target_spy, 4),
        "target_qqq_weight": round(target_qqq, 4),
        "target_tqqq_weight": round(target_tqqq, 4),
        "target_cash_weight": round(1.0 - gross, 4),
        "gross_exposure": round(gross, 4),
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "paper_max_gross_exposure": PAPER_MAX_GROSS_EXPOSURE,
        "raw_research_gross_exposure": round(raw_gross, 4),
        "paper_weight_scale": round(paper_scale, 8),
        "paper_weights_scaled": bool(paper_scaled),
        "overlay_gross": round(float(paper_overlay.abs().sum()), 4),
        "raw_overlay_gross": round(float(overlay.abs().sum()), 4),
        "core_gross": round(abs(target_spy) + abs(target_qqq) + abs(target_tqqq), 4),
        "raw_core_gross": round(core_gross, 4),
        "max_single_name_weight": round(float(metrics.get("max_single_name_weight", MAX_SINGLE_NAME_WEIGHT)), 4),
        "robust_cost_stress_pass": bool(metrics.get("robust_cost_stress_pass", False)),
        "adaptive_exit_mode": str(metrics.get("adaptive_exit_mode", "fixed")),
        "earnings_blackout_days": int(metrics.get("earnings_blackout_days", 0)),
        "earnings_blackout_skipped_count": int(metrics.get("earnings_blackout_skipped_count", 0)),
        "earnings_blackout_skipped_tickers": str(metrics.get("earnings_blackout_skipped_tickers", "")),
        "earnings_blackout_skipped_json": str(metrics.get("earnings_blackout_skipped_json", "[]")),
        "holding_days": holding_days,
        "overlay_tickers": ",".join(paper_overlay.index.astype(str).tolist()),
        "overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in paper_overlay.items()}, sort_keys=True),
        "raw_overlay_weights_json": json.dumps({str(k): round(float(v), 6) for k, v in overlay.items()}, sort_keys=True),
        "single_name_stock_picker_enabled": False,
        "ml_overlay_enabled": bool(metrics["score_source"] == "factor_plus_model"),
        "factor_overlay_enabled": True,
        "latest_factor_date": str(latest_date.date()),
        "cost_stress": float(metrics.get("cost_stress", COST_STRESS_MULTIPLIERS[0])),
        "gates_all_pass": bool(metrics.get("core_satellite_gate_results", {}).get("all_pass", False)),
        "reason": (
            "nested walk-forward live gates pass"
            if metrics.get("paper_ready")
            else "; ".join(str(x) for x in metrics.get("live_gate_reasons", []) if str(x))
            or feature_health_reason
            or "nested walk-forward live gates have not passed"
        ),
        "live_gate_source": str(metrics.get("live_gate_source", "")),
        "live_gate_reasons": ";".join(str(x) for x in metrics.get("live_gate_reasons", [])),
        "walkforward_approval_pass": bool(metrics.get("walkforward_approval_pass", False)),
        "nested_cost_stress_approval_pass": bool(metrics.get("nested_cost_stress_approval_pass", False)),
        "medium_risk_review_pass": bool(metrics.get("medium_risk_review_pass", False)),
        "medium_risk_review_reasons": ";".join(str(x) for x in metrics.get("medium_risk_review_reasons", [])),
        "survivorship_review": json.dumps(metrics.get("survivorship_review", {}), sort_keys=True),
        "execution_stress_review": json.dumps(metrics.get("execution_stress_review", {}), sort_keys=True),
        "factor_decay_review": json.dumps(metrics.get("factor_decay_review", {}), sort_keys=True),
        "feature_health_gate_pass": feature_health_gate_pass,
        "live_regime_refresh_failed": bool(metrics.get("live_regime_refresh_failed", False)),
        "live_regime_refresh_error": str(metrics.get("live_regime_refresh_error", "")),
        "active_cluster_count": int(metrics.get("active_cluster_count", 0) or 0),
        "effective_cluster_count": int(metrics.get("effective_cluster_count", 0) or 0),
        "max_cluster_weight": round(float(metrics.get("max_cluster_weight", 0.0) or 0.0), 6),
        "quarantined_features": ",".join(str(x) for x in metrics.get("quarantined_features", [])),
        "watchlist_features": ",".join(str(x) for x in metrics.get("watchlist_features", [])),
        "predicted_at": _paper_signal_timestamp(),
        # Metadata only: this links the signal to reports and orders without
        # changing any model score, target weight, or trade rule.
        "run_id": current_run_id(),
        "sticky_holdings_source": str(sticky_state["source"]),
        "sticky_holdings_used": bool(sticky_state["used"]),
        "sticky_held_tickers": ",".join(sorted(held_tickers)),
        "sticky_prev_overlay_json": metrics["sticky_prev_overlay_json"],
        "sticky_holdings_reason": str(sticky_state["reason"]),
        "sentiment_veto_enabled": SENTIMENT_VETO_ENABLED,
        "sentiment_scores_json": json.dumps(
            {k: round(v, 4) for k, v in sentiment_scores.items()}, sort_keys=True
        ) if sentiment_scores else "{}",
    }
    out = Path(SIGNAL_DIR) / "core_satellite_alpha_signal.csv"
    # Atomic write — broker scripts polling this CSV never see a torn read.
    atomic_write_csv(pd.DataFrame([row]), out, index=False)
    # Preserve the exact latest feature rows that produced this target. This is
    # evidence only: it is written after the target calculation and never read
    # back by live trading.
    input_snapshot = day.copy()
    input_snapshot["run_id"] = current_run_id()
    atomic_write_csv(
        input_snapshot,
        Path(SIGNAL_DIR) / "core_satellite_alpha_input_snapshot.csv",
        index=False,
    )
    return out


def _notify_earnings_blackout_if_needed(metrics: dict, notifier=None) -> bool:
    blackout_count = int(metrics.get("earnings_blackout_skipped_count", 0) or 0)
    if blackout_count <= 0:
        return False
    blackout_tickers = str(metrics.get("earnings_blackout_skipped_tickers", "") or "")
    msg = (
        f"Earnings blackout skipped {blackout_count} overlay candidate(s)"
        + (f": {blackout_tickers}" if blackout_tickers else "")
    )
    if notifier is None:
        from notifications import notify_warning as notifier
    notifier(msg, title="Earnings Blackout")
    return True


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "n/a"


def _fmt_num(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "n/a"


def print_run_summary(grid: pd.DataFrame, metrics: dict, signal_path: Path) -> None:
    comps = metrics["benchmark_comparisons"]
    gates = metrics["core_satellite_gate_results"]
    subperiods = metrics["subperiods"]

    print("\nCore-Satellite Alpha Summary")
    print("=" * 30)
    print(f"Status:        {'PAPER READY' if metrics.get('paper_ready') else 'NOT READY'}")
    print(f"Selected:      {metrics.get('core_preset')} ({metrics.get('regime_mode', 'static')})")
    print(f"Holding days:  {metrics.get('holding_days')}")
    print(f"Cost stress:   {_fmt_num(metrics.get('cost_stress'), 1)}x")
    vt = metrics.get("vol_target", 0.0)
    if vt and float(vt) > 0:
        print(f"Vol target:    {float(vt)*100:.0f}%")
    print(f"Regime counts: {metrics.get('regime_counts', {})}")

    print("\nPerformance (backtest — expect significant degradation live)")
    cagr = float(metrics.get("cagr_pct", 0) or 0)
    print(f"  Total return: {_fmt_pct(metrics.get('total_return_pct'))}")
    print(f"  CAGR:         {_fmt_pct(metrics.get('cagr_pct'))}")
    if cagr > 30:
        print(f"    ⚠ CAGR > 30% is likely overstated. Realistic live expectation: 15-25%")
    print(f"  Sharpe:       {_fmt_num(metrics.get('sharpe'))}")
    print(f"  Max DD:       {_fmt_pct(metrics.get('max_drawdown_pct'))}")
    print(f"  Turnover:     {_fmt_pct(metrics.get('turnover_pct'))}")
    print(f"  Est. costs:   {_fmt_pct(metrics.get('estimated_cost_pct'))}")
    print(f"  Max name wt:  {_fmt_pct(float(metrics.get('max_single_name_weight', 0.0)) * 100.0)}")
    print(f"  Eff. names:   {_fmt_num(metrics.get('avg_effective_overlay_names'))}")
    # Robustness score from grid selection
    rob_score = metrics.get("robustness_score", None)
    if rob_score is not None:
        print(f"  Robustness:   {float(rob_score):.3f}")
        print(
            f"    penalties: DD={float(metrics.get('drawdown_penalty', 0.0)):.3f}, "
            f"turnover={float(metrics.get('turnover_penalty', 0.0)):.3f}, "
            f"instability={float(metrics.get('instability_penalty', 0.0)):.3f}"
        )

    print("\nBenchmark Alpha")
    for symbol in ("SPY", "QQQ", "BLEND"):
        comp = comps.get(symbol, {})
        print(
            f"  vs {symbol:<5} alpha {_fmt_pct(comp.get('alpha_pct'))}"
            f" | benchmark {_fmt_pct(comp.get('benchmark_return_pct'))}"
            f" | t-stat {_fmt_num(comp.get('nw_tstat_vs_benchmark'))}"
        )

    holdout = metrics.get("holdout_2023_2026", {})
    if holdout.get("data_available"):
        print("\nHoldout 2023-2026")
        print(
            f"  strategy {_fmt_pct(holdout.get('strategy_return_pct'))}"
            f" | alpha vs QQQ {_fmt_pct(holdout.get('alpha_vs_qqq_pct'))}"
            f" | alpha vs BLEND {_fmt_pct(holdout.get('alpha_vs_blend_pct'))}"
        )

    stress = metrics.get("robust_cost_stress_summary", {})
    if stress:
        print("\nCost Stress")
        print(
            f"  levels {stress.get('cost_levels', 'n/a')}"
            f" | min alpha vs QQQ {_fmt_pct(stress.get('min_alpha_vs_qqq_pct'))}"
            f" | pass {'YES' if metrics.get('robust_cost_stress_pass') else 'NO'}"
        )

    print("\nSubperiod Alpha vs 60/40")
    for name, row in subperiods.items():
        if not row.get("data_available"):
            print(f"  {name:<9} n/a")
            continue
        print(
            f"  {name:<9} alpha {_fmt_pct(row.get('alpha_pct'))}"
            f" | strategy {_fmt_pct(row.get('return_pct'))}"
            f" | benchmark {_fmt_pct(row.get('benchmark_return_pct'))}"
        )

    failed = [name for name, ok in gates.items() if name.endswith("_pass") and not bool(ok)]
    print("\nGates")
    print(f"  Result: {'PASS' if gates.get('all_pass') else 'FAIL'}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  Failed: none")

    display_cols = [
        "paper_ready",
        "core_preset",
        "regime_mode",
        "holding_days",
        "earnings_blackout_days",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "alpha_vs_qqq_pct",
        "alpha_vs_blend_pct",
        "robust_cost_stress_pass",
        "subperiod_stability_pass",
    ]
    top = grid.head(5).copy()
    print("\nTop Grid Rows")
    print(top[[c for c in display_cols if c in top.columns]].to_string(index=False))

    print("\nSaved Outputs")
    print(f"  metrics: signals/core_satellite_alpha_metrics.json")
    print(f"  grid:    signals/core_satellite_alpha_grid.csv")
    print(f"  signal:  {signal_path}")


# ══════════════════════════════════════════════════════════════════════════════
# NESTED WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
# PLAIN ENGLISH: This is the MOST IMPORTANT part for avoiding overfitting.
#
# Normal grid search: fit parameters on ALL data 2010-2026, pick best config.
# Problem: the "best" config on the full sample is OVERFIT to that sample.
# A 6,336-config search over 16 years of data will ALWAYS find something
# that looks amazing — even on random noise.
#
# Nested walk-forward fixes this by separating parameter tuning from evaluation:
#
#   OUTER LOOP (true out-of-sample evaluation — NEVER seen during tuning):
#     For each test year (e.g., 2015, 2016, ..., 2025):
#       - Training window = all data BEFORE the test year
#       - The grid search runs on the training window only
#       - The winning config is then evaluated on the test year
#       - The test year result is TRULY out-of-sample
#
#   The final OOS metrics are the AVERAGE across all test years.
#   This tells you what performance you'd have gotten IF you had been running
#   this system live since 2015, re-tuning parameters annually.
#
# Why this matters:
#   - A strategy with 35,000% backtest return but 8% OOS return is USELESS
#   - A strategy with 300% backtest return but 15% OOS CAGR is GOLD
#   - Nested walk-forward reveals the difference
#
# The primary metric we report is: OOS Sharpe (average across all test folds)
# Secondary: OOS CAGR, OOS max DD, stability (variance across folds)


def run_nested_walkforward(panel: pd.DataFrame, min_train_years: int = 4) -> dict:
    """Run the proper core-alpha nested walk-forward validator.

    This wrapper exists for backwards compatibility with the signal script.
    The real implementation lives in core_satellite_nested_walkforward.py and
    uses inner train/validation folds inside each outer train period before
    evaluating the selected config once on the unseen outer test year.
    """
    from core_satellite_nested_walkforward import run_nested_walkforward as _proper_nested_walkforward
    from core_satellite_nested_walkforward import write_outputs as _write_nested_outputs

    result = _proper_nested_walkforward(panel, strategy="core-alpha", min_train_years=min_train_years)
    _write_nested_outputs(result, output_prefix="core_satellite_alpha_walkforward")
    return result


LIVE_CONFIG_PATH = Path(SIGNAL_DIR) / "core_satellite_live_configs.json"

# ── Config expiry ───────────────────────────────────────────────────────
# PLAIN ENGLISH: If the approved walkforward config is older than this many
# days, the daily run will treat it as expired and refuse to trade with it.
# This prevents a stale config (approved under a different market regime)
# from running indefinitely without re-validation.  The walkforward must
# be re-run to produce a fresh approval.
LIVE_CONFIG_MAX_AGE_DAYS = int(os.environ.get("LIVE_CONFIG_MAX_AGE_DAYS", "45"))


def _portable_artifact_path(path_text: str, *, base_dir: Path | None = None) -> Path:
    """Turn a stored Windows or Linux artifact path into a local path.

    PLAIN ENGLISH: Research evidence may be created on Windows and consumed by
    GitHub's Linux runner. Linux treats a backslash as a normal character, so
    normalizing both slash styles prevents a valid bundle from looking absent.
    """
    normalized = str(path_text).strip().replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path


def _load_approved_live_config(strategy: str = "core-alpha") -> dict:
    if not LIVE_CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing approved live config file: {LIVE_CONFIG_PATH}. "
            "Run `python3 core_satellite_nested_walkforward.py --strategy core-alpha` first."
        )
    try:
        payload = json.loads(LIVE_CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid approved live config file {LIVE_CONFIG_PATH}: {exc}") from exc

    approvals = payload.get("approvals", {})
    approval = approvals.get(strategy, {})
    if not bool(approval.get("approved", False)):
        reasons = approval.get("reasons", ["not approved"])
        # Return rejection info instead of raising — let caller write the rejection signal first
        return {
            "approved": False,
            "reasons": reasons,
            "approval": approval,
        }

    # ── Config expiry check ────────────────────────────────────────────────
    # PLAIN ENGLISH: Check how old the live config is.  If it was created more
    # than LIVE_CONFIG_MAX_AGE_DAYS ago, reject it.  Market conditions change —
    # a config approved 2 months ago may no longer be valid.  Re-run the nested
    # walkforward to get a fresh approval.
    created_at_str = payload.get("created_at", "")
    if created_at_str:
        try:
            from datetime import datetime, timezone
            created_at = datetime.fromisoformat(created_at_str)
            age_days = (datetime.now(timezone.utc) - created_at).days
            if age_days > LIVE_CONFIG_MAX_AGE_DAYS:
                return {
                    "approved": False,
                    "reasons": [
                        f"config_expired: approved {age_days} days ago "
                        f"(max={LIVE_CONFIG_MAX_AGE_DAYS}). "
                        "Re-run nested walkforward to refresh."
                    ],
                    "approval": approval,
                    "expired": True,
                    "age_days": age_days,
                }
        except (ValueError, TypeError):
            pass  # If created_at can't be parsed, skip expiry check

    approved = payload.get("approved_live_configs", {}).get(strategy)
    config = approved.get("config") if isinstance(approved, dict) else None
    if not isinstance(config, dict):
        raise SystemExit(f"Approved live config for {strategy} is missing from {LIVE_CONFIG_PATH}.")

    # PLAIN ENGLISH: The live config is only an index card. The validation
    # bundle is the signed evidence behind it. Refuse to trade when the card
    # points at missing, damaged, or different evidence.
    bundle_path_text = str(payload.get("validation_bundle_path", "")).strip()
    expected_bundle_hash = str(payload.get("validation_bundle_hash", "")).strip()
    if not bundle_path_text or not expected_bundle_hash:
        return {
            "approved": False,
            "reasons": ["validation_bundle_reference_missing"],
            "approval": approval,
        }
    bundle_path = _portable_artifact_path(bundle_path_text)
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "approved": False,
            "reasons": [f"validation_bundle_unreadable:{exc.__class__.__name__}"],
            "approval": approval,
        }
    bundle_ok, bundle_issues = validate_validation_bundle(bundle)
    if expected_bundle_hash != str(bundle.get("validation_bundle_hash", "")):
        bundle_issues.append("live_config_bundle_hash_mismatch")
    if strategy_config_fingerprint(config) != str(bundle.get("config_fingerprint", "")):
        bundle_issues.append("live_config_strategy_fingerprint_mismatch")
    deployment = bundle.get("deployment", {}) or {}
    if not bool(deployment.get("paper_approved", False)):
        bundle_issues.append("validation_bundle_not_paper_approved")

    # PLAIN ENGLISH: the bundle records what passed when it was created.  Read
    # today's files too, compare their checksums, and refuse to trust a copied
    # approval when a report has since become stale or changed to warning.
    current_robustness = current_robustness_evidence(
        expected_config_fingerprint=str(bundle.get("config_fingerprint", "")),
        expected_dataset_fingerprint=str((bundle.get("dataset", {}) or {}).get("dataset_fingerprint", "")),
    )
    bundled_reports = bundle.get("robustness_reports", {}) or {}
    for name, current_record in (current_robustness.get("reports", {}) or {}).items():
        if str(current_record.get("sha256", "")) != str((bundled_reports.get(name, {}) or {}).get("sha256", "")):
            bundle_issues.append(f"robustness_report_changed:{name}")
    if not bool(current_robustness.get("pass", False)):
        bundle_issues.extend(
            f"current_robustness_failed:{reason}"
            for reason in current_robustness.get("reasons", []) or ["unknown"]
        )
    if not bundle_ok or bundle_issues:
        return {
            "approved": False,
            "reasons": sorted(set(bundle_issues)),
            "approval": approval,
        }
    # PLAIN ENGLISH: This short hash is the approved config's ID card.  The
    # broker compares it against the signal so an old signal cannot be traded
    # after a new walkforward config is published.
    config_hash = live_config_fingerprint(payload, strategy=strategy)
    return {
        "config": config,
        "approval": approval,
        "approved_config_family": approved.get("approved_config_family"),
        "approved_family_signature": approved.get("approved_family_signature"),
        "approved_exact_config": approved.get("approved_exact_config"),
        "source_metrics": approved.get("source_metrics", {}),
        # This current review, not the embedded historical copy, controls the
        # signal gate below.
        "medium_risk_review": current_robustness.get("medium_risk_review", {}),
        "source_json": payload.get("source_json"),
        "created_at": payload.get("created_at"),
        "live_config_hash": config_hash,
        "validation_bundle_path": str(bundle_path),
        "validation_bundle_hash": expected_bundle_hash,
        "deployment_status": deployment.get("status", "paper_provisional"),
        "real_capital_approved": False,
    }


def _write_rejection_signal(reasons: list[str]) -> None:
    """Overwrite the signal file with paper_ready=False so stale approvals can't persist."""
    signal_path = _core_satellite_signal_path()
    reason_str = "nested_walkforward_rejected: " + "; ".join(reasons)
    if signal_path.exists():
        try:
            df = pd.read_csv(signal_path)
            df["paper_ready"] = False
            df["gates_all_pass"] = False
            df["reason"] = reason_str
            atomic_write_csv(df, signal_path, index=False)
            return
        except Exception:
            pass
    atomic_write_csv(
        pd.DataFrame([{
            "paper_signal_type": "core_satellite_alpha",
            "paper_ready": False,
            "gates_all_pass": False,
            "reason": reason_str,
            "predicted_at": datetime.now().isoformat(),
        }]),
        signal_path,
        index=False,
    )


def _apply_nested_live_approval_gates(metrics: dict, live: dict, freshness: dict) -> dict:
    """Use nested approval as the live gate while preserving full-sample diagnostics."""
    full_sample_gates = dict(metrics.get("core_satellite_gate_results", {}) or {})
    metrics["full_sample_core_satellite_gate_results"] = full_sample_gates
    metrics["full_sample_gate_all_pass"] = bool(full_sample_gates.get("all_pass", False))

    approval = live.get("approval", {}) or {}
    source_metrics = live.get("source_metrics", {}) or {}
    walkforward_pass = bool(approval.get("approved", False))
    cost_pass = bool(source_metrics.get("cost_stress_approval_pass", False))
    medium_review = live.get("medium_risk_review", {}) or {}
    medium_review_pass = bool(medium_review.get("pass", False))
    feature_health_pass = bool(metrics.get("feature_health_gate_pass", True))
    freshness_pass = bool(freshness.get("fresh", False))

    reasons: list[str] = []
    if not walkforward_pass:
        approval_reasons = approval.get("reasons") or ["not approved"]
        reasons.append("nested_walkforward_approval_failed:" + ",".join(str(x) for x in approval_reasons))
    if not cost_pass:
        reasons.append("nested_cost_stress_approval_failed")
    if not medium_review_pass:
        review_reasons = medium_review.get("reasons") or ["missing_medium_risk_review"]
        reasons.append("medium_risk_review_failed:" + ",".join(str(x) for x in review_reasons))
    if not feature_health_pass:
        health_reasons = metrics.get("feature_health_gate_reasons") or ["feature_health_gate_failed"]
        reasons.append("feature_health_gate_failed:" + ",".join(str(x) for x in health_reasons))
    if not freshness_pass:
        reasons.append("factor_data_stale")
        metrics["factor_data_stale"] = True
        metrics["factor_data_freshness"] = freshness

    live_ready = not reasons
    metrics["live_gate_source"] = "nested_walkforward_approval"
    metrics["walkforward_approval_pass"] = walkforward_pass
    metrics["walkforward_approval_reasons"] = list(approval.get("reasons") or [])
    metrics["nested_cost_stress_approval_pass"] = cost_pass
    metrics["medium_risk_review_pass"] = medium_review_pass
    metrics["medium_risk_review_reasons"] = list(medium_review.get("reasons") or [])
    metrics["survivorship_review"] = medium_review.get("survivorship_review", {})
    metrics["execution_stress_review"] = medium_review.get("execution_stress_review", {})
    metrics["factor_decay_review"] = medium_review.get("factor_decay_review", {})
    metrics["robust_cost_stress_pass"] = cost_pass
    metrics["live_gate_reasons"] = reasons
    metrics["paper_ready"] = live_ready
    metrics["core_satellite_gate_results"] = {
        "walkforward_approval_pass": walkforward_pass,
        "nested_cost_stress_approval_pass": cost_pass,
        "medium_risk_review_pass": medium_review_pass,
        "feature_health_gate_pass": feature_health_pass,
        "factor_data_freshness_pass": freshness_pass,
        "all_pass": live_ready,
    }
    return metrics


def write_core_alpha_metrics(metrics: dict, output_path: Path | None = None) -> Path:
    """Write the core-alpha metrics JSON atomically.

    PLAIN ENGLISH: The dashboard and broker checks read this report. Atomic
    writing keeps them from seeing a half-written metrics file.
    """
    path = output_path or (Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json")
    atomic_write_json(metrics, path)
    return path


def write_core_alpha_backtest_artifacts(
    metrics: dict,
    equity: pd.Series,
    trades: pd.DataFrame,
) -> dict[str, Path]:
    """Write core-alpha equity, trades, and metrics through safe writers."""
    paths = {
        "equity": Path(SIGNAL_DIR) / "core_satellite_alpha_equity.csv",
        "trades": Path(SIGNAL_DIR) / "core_satellite_alpha_trades.csv",
        "metrics": Path(SIGNAL_DIR) / "core_satellite_alpha_metrics.json",
    }
    # Preserve the old equity CSV shape: pandas used to include the Series index.
    atomic_write_csv(pd.DataFrame({"equity": equity}), paths["equity"], index=True)
    atomic_write_csv(trades, paths["trades"], index=False)
    write_core_alpha_metrics(metrics, paths["metrics"])
    return paths


def _generate_signal_from_approved_config(
    *,
    panel: pd.DataFrame,
    signal_panel: pd.DataFrame,
    specs: list[dict],
    freshness: dict,
) -> tuple[pd.DataFrame, dict, Path]:
    live = _load_approved_live_config("core-alpha")
    if live.get("approved") == False:
        reasons = live.get("reasons", ["not approved"])
        _write_rejection_signal(reasons)
        rejection_metrics = {
            "paper_ready": False,
            "nested_walkforward_rejected": True,
            "nested_walkforward_rejection_reasons": reasons,
        }
        write_core_alpha_metrics(rejection_metrics)
        # ── Notify on config rejection/expiry ─────────────────────────
        # PLAIN ENGLISH: If the live config was rejected (e.g. expired or
        # failed approval), send a Telegram alert so you know the daily
        # run won't trade today and the walkforward needs re-running.
        try:
            from notifications import send_alert as _rejection_notify
            is_expired = live.get("expired", False)
            age_str = f" (age: {live.get('age_days', '?')} days)" if is_expired else ""
            _rejection_notify(
                f"Daily signal BLOCKED — live config rejected{age_str}\n"
                f"Reasons: {'; '.join(reasons)}\n"
                f"Action: re-run nested walkforward.",
                title="Config Rejected",
                priority="warning",
            )
        except Exception:
            pass
        raise SystemExit(
            f"Nested walk-forward has not approved a live core-alpha config: {reasons}. "
            "Run `python3 core_satellite_nested_walkforward.py --strategy core-alpha` and inspect the OOS report."
        )
    selected_config = dict(live["config"])
    selected_config["live_config_source"] = "nested_walkforward"
    selected_config["live_config_source_json"] = live.get("source_json")

    best_metrics, best_equity, best_trades = evaluate(panel, selected_config)
    best_metrics["selected_features"] = specs
    best_metrics["grid_rows"] = 0
    best_metrics["best_config_source"] = "nested_walkforward_approved_live_config"
    best_metrics["full_sample_grid_used_for_selection"] = False
    best_metrics["live_config_hash"] = live.get("live_config_hash")
    best_metrics["live_config_created_at"] = live.get("created_at")
    best_metrics["live_config_source_json"] = live.get("source_json")
    best_metrics["approved_family_signature"] = live.get("approved_family_signature") or live.get("approved_config_family")
    best_metrics["approved_exact_config"] = live.get("approved_exact_config")
    best_metrics["walkforward_approval"] = {
        "approved_config_family": live.get("approved_config_family"),
        "approved_family_signature": live.get("approved_family_signature"),
        "approved_exact_config": live.get("approved_exact_config"),
        "approval": live.get("approval", {}),
        "source_metrics": live.get("source_metrics", {}),
        "source_json": live.get("source_json"),
        "created_at": live.get("created_at"),
        "live_config_hash": live.get("live_config_hash"),
    }
    best_metrics = _apply_nested_live_approval_gates(best_metrics, live, freshness)
    if not bool(freshness.get("fresh", False)):
        print("  ⚠ gates_all_pass forced to False because factor data is stale")

    write_core_alpha_backtest_artifacts(best_metrics, best_equity, best_trades)
    signal_path = write_paper_signal(signal_panel, best_metrics)

    row = {
        "paper_ready": best_metrics.get("paper_ready"),
        "core_preset": best_metrics.get("core_preset"),
        "risk_control_mode": best_metrics.get("risk_control_mode", "off"),
        "regime_mode": best_metrics.get("regime_mode"),
        "holding_days": best_metrics.get("holding_days"),
        "overlay_gross": best_metrics.get("overlay_gross"),
        "score_source": best_metrics.get("score_source"),
        "shape": best_metrics.get("shape"),
        "weighting": best_metrics.get("weighting"),
        "max_per_sector": best_metrics.get("max_per_sector"),
        "sharpe": best_metrics.get("sharpe"),
        "max_drawdown_pct": best_metrics.get("max_drawdown_pct"),
        "full_sample_grid_used_for_selection": False,
    }
    return pd.DataFrame([row]), best_metrics, signal_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Core-satellite alpha signal generator")
    parser.add_argument("--ignore-stale", action="store_true",
                        help="Override stale data block — generate signal even if factor data is very old")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run proper nested core-alpha validation before generating the signal. "
                             "For normal validation, prefer core_satellite_nested_walkforward.py.")
    parser.add_argument("--no-walkforward", action="store_true",
                        help="Legacy no-op. Nested validation is skipped by default; use --walkforward to run it.")
    parser.add_argument("--min-train-years", type=int, default=4,
                        help="Minimum training years before first test fold (default: 4)")
    args = parser.parse_args()

    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)

    # Sector-map coverage check — prints a clear warning if any
    # alpha-universe ticker is missing from SECTOR_MAP.  Unmapped tickers
    # would fall to the "OTHER" bucket and silently bypass sector caps.
    validate_sector_map_coverage()

    # ── FEATURE QUALITY FILTER ────────────────────────────────────────────────
    # Live signal generation is strict: diagnostics must exist, parse cleanly,
    # and leave usable cross-sectional rank features in the final panel.
    quality_filter = _load_feature_quality_filter(strict=True)
    specs = load_feature_specs()
    specs = _apply_live_feature_quality_filter(specs, quality_filter)
    ml_scores = load_prediction_scores()

    # Backtest/training panel: requires forward returns, so its newest usable row
    # naturally lags the raw data by the forward-return horizon. Keep this panel
    # for grid/backtest/walk-forward evaluation only.
    panel = _ensure_robust_score_columns(attach_scores(load_factor_panel(specs), specs, ml_scores))
    print(f"  Backtest panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates, latest={pd.Timestamp(panel['date'].max()).date()}")

    # Live signal panel: does NOT require future returns, so it includes the
    # freshest feature rows. Use this panel for freshness checks and final signals.
    signal_panel = _ensure_robust_score_columns(
        attach_scores(load_factor_panel(specs, require_forward_returns=False), specs, ml_scores)
    )
    print(f"  Live signal panel: {len(signal_panel)} rows, {signal_panel['ticker'].nunique()} tickers, "
          f"{signal_panel['date'].nunique()} dates, latest={pd.Timestamp(signal_panel['date'].max()).date()}")
    _validate_live_feature_inputs(specs, signal_panel)

    # ── OPTIONAL: NESTED WALK-FORWARD VALIDATION ────────────────────────────
    # PLAIN ENGLISH: Daily signal generation should stay fast and focused.
    # Run `python3 core_satellite_nested_walkforward.py --strategy core-alpha` as the research trust
    # gate for the unified live signal.  This flag remains useful when you
    # want a one-off core-alpha validation before generating today's signal.
    if args.walkforward:
        print("\n  Running nested walk-forward validation (--walkforward)...")
        print("  (This tests TRUE out-of-sample performance, year by year)")
        print("  (For full validation, run: python3 core_satellite_nested_walkforward.py --strategy core-alpha)")
        wf_results = run_nested_walkforward(panel, min_train_years=args.min_train_years)

        print("\n  Now generating today's signal from nested-approved live config...")
    else:
        print("\n  Skipping nested validation for daily signal generation.")
        print("  Trust check lives in: python3 core_satellite_nested_walkforward.py --strategy core-alpha")

    # ── Data freshness gate ────────────────────────────────────────────
    # PLAIN ENGLISH: Before generating the signal, check if the factor data
    # is fresh.  If it's too old, refuse to generate a signal because stale
    # data leads to stale stock picks → bad trades.
    freshness = check_factor_freshness(signal_panel, ignore_stale=args.ignore_stale)
    print(f"\n  {freshness['message']}")
    if freshness["blocked"]:
        raise SystemExit(f"Aborting: {freshness['message']}")
    # We'll pass this flag down so write_paper_signal can set gates_all_pass=False
    _FACTOR_DATA_FRESH = freshness["fresh"]

    print(f"\n  Loading approved live config from nested walk-forward: {LIVE_CONFIG_PATH}")

    # ── Config change detection ───────────────────────────────────────
    # PLAIN ENGLISH: Before generating the signal, check if the approved
    # config family changed since the last run.  If it did, that means
    # the walkforward picked a different winning strategy — send an alert
    # so you know the portfolio will shift.
    _prev_config_family = None
    _prev_signal = _core_satellite_signal_path()
    if _prev_signal.exists():
        try:
            _prev_df = pd.read_csv(_prev_signal)
            _prev_config_family = str(_prev_df["core_preset"].iloc[0]) if "core_preset" in _prev_df.columns else None
        except Exception:
            pass

    summary_grid, best_metrics, signal_path = _generate_signal_from_approved_config(
        panel=panel,
        signal_panel=signal_panel,
        specs=specs,
        freshness=freshness,
    )
    try:
        if _notify_earnings_blackout_if_needed(best_metrics):
            print(f"\n  ⚠ Earnings blackout alert sent for {best_metrics.get('earnings_blackout_skipped_tickers', '')}")
    except Exception:
        pass

    # Check if the config changed and notify
    _new_config_family = best_metrics.get("core_preset")
    if _prev_config_family and _new_config_family and _prev_config_family != _new_config_family:
        _change_msg = (
            f"Live config rotated!\n"
            f"Old: {_prev_config_family}\n"
            f"New: {_new_config_family}"
        )
        print(f"\n  🔄 CONFIG CHANGE DETECTED: {_change_msg}")
        try:
            from notifications import send_alert as _notify
            _notify(_change_msg, title="Config Rotation", priority="warning")
        except Exception:
            pass
    elif _prev_config_family is None and _new_config_family:
        print(f"\n  First signal generation — config: {_new_config_family}")

    print_run_summary(summary_grid, best_metrics, signal_path)


# ── DEAD CODE REMOVED ─────────────────────────────────────────────────────
# The old --grid-only code path (full-sample grid search on all data) was
# removed.  It trained and tested on the same data → overfitting.  Use
# core_satellite_nested_walkforward.py instead — it does the same grid
# search but with proper train/test splits so results are trustworthy.


if __name__ == "__main__":
    # PID lock — prevents two copies of core_satellite_alpha.py from racing
    # to write the same signal CSV (e.g. GitHub Actions firing while the
    # dashboard runs the script manually).  Atomic writes already prevent
    # half-written files, but two concurrent runs would still waste compute
    # and "last writer wins" could publish stale data over fresh data.
    from safe_io import PidLock, PidLockTaken
    from pathlib import Path as _Path
    _lock_path = _Path("logs") / "core_satellite_alpha.lock"
    try:
        with PidLock(_lock_path):
            main()
    except PidLockTaken as exc:
        print(
            f"⛔ Another core_satellite_alpha.py is already running.\n"
            f"   {exc}\n"
            f"   Refusing to start a second copy — exiting to prevent a race "
            f"on signals/core_satellite_alpha_signal.csv."
        )
        import sys as _sys
        _sys.exit(1)
