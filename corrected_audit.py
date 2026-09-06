"""Offline corrected research entry point; never connects to a broker or cuts over paper."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from causal_research import fit_features, score_features, fingerprint, nested_evaluate, FoldArtifact, CAUSAL_VERSION
from corrected_data import validate_sources, load_raw_panel, eligible_candidates, build_raw_features
from edge_evidence import edge_summary, non_overlapping_ic
from paper_policy import PaperPolicy
from portfolio_ledger import simulate_daily, LEDGER_VERSION
from safe_io import atomic_write_json, atomic_write_csv


def build_daily_targets(scored, config, membership_path, *, bars):
    """Use frozen scores and prior-session prices; keep actual holdings sticky."""
    import core_satellite_alpha as core
    candidates = eligible_candidates(scored, membership_path)
    day_map = {pd.Timestamp(date): group for date, group in candidates.groupby("date")}
    prices = bars.pivot(index="date", columns="ticker", values="Close").sort_index()
    # Historical regime inputs must be causal, split-aware signal prices. They
    # are distinct from the raw dollars used for order fills and cash accounting.
    regime = None
    if str(config.get("regime_mode", "static")) in core.REGIME_PRESETS:
        if "signal_close" not in bars:
            raise ValueError("Causal corporate-action-aware signal_close required for regime replay")
        signal = bars.pivot(index="date", columns="ticker", values="signal_close").sort_index()
        spy, qqq = signal["SPY"], signal["QQQ"]
        regime = pd.DataFrame(index=signal.index)
        regime["spy_trend_ok"] = spy >= spy.rolling(200, min_periods=50).mean()
        regime["qqq_trend_ok"] = qqq >= qqq.rolling(int(config.get("regime_ma_window", 100)), min_periods=50).mean()
        vol = qqq.pct_change(fill_method=None).rolling(20, min_periods=10).std() * np.sqrt(252)
        regime["high_vol"] = vol.rolling(252, min_periods=60).rank(pct=True).gt(.8) if config.get("high_vol_mode") == "percentile" else vol.gt(float(config.get("regime_high_vol", .3)))
        regime["concentration_qqq_spy_120d"] = qqq.pct_change(120, fill_method=None) - spy.pct_change(120, fill_method=None)
        for flag in ("spy_trend_ok", "qqq_trend_ok", "high_vol"):
            regime[flag] = core._confirm_regime_flag(regime[flag], int(config.get("regime_confirm_days", core.REGIME_CONFIRM_DAYS)))
        if "vix_inverted" not in scored:
            raise ValueError("Dated VIX inversion observations required for corrected regime policy")
        regime["vix_inverted"] = scored.groupby("date").vix_inverted.first().reindex(regime.index)
    def target(date, shares, equity):
        if date not in day_map or date not in prices.index:
            raise ValueError(f"Missing prior-session candidates/prices: {date}")
        day = day_map[date]
        blackout = int(config.get("earnings_blackout_days", 0))
        if blackout and ("days_to_next_earnings" not in day or day.days_to_next_earnings.isna().any()):
            raise ValueError(f"Dated earnings calendar required at {date}")
        selected = core._select_sticky_holdings(day, {t for t, q in shares.items() if q},
            score_col="causal_score", return_col=None, shape=config.get("shape", "top5"),
            exit_rank_floor=float(config.get("exit_rank_floor", .8)), max_per_sector=int(config.get("max_per_sector", 2)),
            earnings_blackout_days=blackout)
        allocation = {"core_weights": {"SPY": .5, "QQQ": .5}, "core_gross": .7, "overlay_gross": .3, **config}
        if regime is not None and pd.isna(regime.loc[date, "vix_inverted"]):
            raise ValueError(f"Missing VIX observation at {date}")
        _, core_weights, core_gross, overlay_gross = core._resolve_allocation(date, allocation, regime)
        overlay_gross, _, _ = core._apply_concentration_overlay_target(date, core_gross, overlay_gross, regime, allocation)
        previous = pd.Series({t: q * float(prices.loc[date, t]) / equity for t, q in shares.items() if q and t not in PaperPolicy().etfs})
        overlay = core._sticky_overlay_weights(selected, overlay_gross, config.get("weighting", "score"), previous,
                                               max_single_name_weight=float(config.get("max_single_name_weight", .25)))
        result = {ticker: float(weight) * core_gross for ticker, weight in core_weights.items()}
        result.update({ticker: float(weight) for ticker, weight in overlay.items()})
        # Match the approved paper gross ceiling after sticky weights and caps.
        gross = sum(result.values())
        if gross > 1:
            result = {ticker: weight / gross for ticker, weight in result.items()}
        result = {ticker: weight for ticker, weight in result.items() if weight > 0}
        return result
    return target


def evaluate_corrected(panel, config, *, start, end, bars, actions, provenance, membership_path, policy=PaperPolicy()):
    """Adapter preserves the existing equity/trades/metrics research tuple."""
    targets = build_daily_targets(panel, config, membership_path, bars=bars)
    from execution_cost_calibration import causal_cost_parameters
    costs = causal_cost_parameters(config.get("cost_evidence", {}), pd.Timestamp(start) - pd.Timedelta(days=1))
    result = simulate_daily(bars, targets, start=start, end=end, provenance=provenance, policy=policy,
                            base_slippage_pct=costs["base_slippage_pct"],
                            actions=actions, cost_stress=float(config.get("cost_stress", 1.)))
    result.metrics["cost_calibration"] = costs
    return result


def code_fingerprints():
    """Include preprocessing, costs and reporting in the frozen implementation."""
    import hashlib
    names = ("portfolio_ledger.py", "paper_policy.py", "causal_research.py", "corrected_audit.py", "corrected_data.py",
             "execution_model.py", "execution_cost_calibration.py", "settings.py", "edge_evidence.py", "core_satellite_alpha.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def strategy_identity(config, artifact, policy):
    """New observations do not change the strategy; changed implementation does."""
    return fingerprint({"configuration": config, "artifact": asdict(artifact), "policy": asdict(policy), "code": code_fingerprints()})


def observation_identity(bars, panel, actions, through):
    """Detect rewritten past observations without rejecting newly matured labels."""
    import hashlib
    cutoff = pd.Timestamp(through)
    def digest(frame):
        return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
    old_bars = bars.loc[pd.to_datetime(bars.date) <= cutoff].sort_values(["date", "ticker"])
    old_panel = panel.loc[pd.to_datetime(panel.date) <= cutoff, [c for c in panel if not c.startswith("forward_return_")]].sort_values(["date", "ticker"])
    old_actions = actions.loc[pd.to_datetime(actions.date) <= cutoff].sort_values("event_id")
    return fingerprint({"bars": digest(old_bars), "decision_inputs": digest(old_panel), "actions": digest(old_actions)})


def paired_benchmark(result, bars, actions, provenance, *, start, end, policy):
    """Match the strategy's opening exposure; use identical costs and terminal sales."""
    from core_satellite_alpha import _session_offset
    exposures = {_session_offset(date, -1): float(row.gross_exposure) for date, row in result.equity.iloc[1:].iterrows()}
    target = lambda date, shares, equity: {"SPY": exposures[date] * .5, "QQQ": exposures[date] * .5}
    # The alternative follows strategy exposure, not its own independent halt.
    # Whole shares and the common cash reserve can leave small exposure residuals.
    matched_policy = replace(policy, drawdown_halt=1., etf_drift=0., minimum_trade=0.)
    result_benchmark = simulate_daily(bars, target, start=start, end=end, provenance=provenance, policy=matched_policy, actions=actions,
                          base_slippage_pct=result.metrics["cost_calibration"]["base_slippage_pct"], cost_stress=result.metrics.get("cost_stress", 1.),
                          terminal_liquidation=result.metrics["terminal_liquidation"])
    result_benchmark.metrics["maximum_exposure_rounding_gap"] = float((result_benchmark.equity.gross_exposure - result.equity.gross_exposure).abs().max())
    return result_benchmark


def prospective_status(frozen: dict, *, now, current_fingerprint, observed_sessions) -> dict:
    """Count observed sessions, not elapsed days; a changed strategy needs a new freeze."""
    from core_satellite_alpha import _nyse_sessions
    freeze_time = pd.Timestamp(frozen["frozen_at"])
    freeze_day = freeze_time.tz_convert("America/New_York").tz_localize(None).normalize()
    end = pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None).normalize()
    sessions = _nyse_sessions(freeze_day + pd.Timedelta(days=1), end) if end > freeze_day else pd.DatetimeIndex([])
    observed = pd.DatetimeIndex(pd.to_datetime(observed_sessions)).normalize().unique()
    completed = len(sessions.intersection(observed))
    changed = frozen.get("strategy_fingerprint") != current_fingerprint
    return {"status": "restart_required" if changed else "ready_for_final_review" if completed >= 252 else "collecting",
            "observed_sessions": completed, "required_sessions": 252, "automatic_cutover": False,
            "final_review_is_profitability_proof": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--membership", type=Path, default=Path("data/universe_membership.csv"))
    parser.add_argument("--output", type=Path, default=Path("signals/corrected_audit"))
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--spec", type=Path, help="Frozen JSON with features, configurations, folds and label")
    parser.add_argument("--freeze", action="store_true", help="Freeze only a fully validated corrected specification")
    parser.add_argument("--observe", action="store_true", help="Record a completed-session shadow observation under an existing freeze")
    parser.add_argument("--replay-events", type=Path, help="Verified individual fill/action events CSV")
    parser.add_argument("--opening-balances", type=Path)
    parser.add_argument("--closing-balances", type=Path)
    parser.add_argument("--marks", type=Path, help="Daily timestamp,ticker,price CSV for recorded replay")
    parser.add_argument("--history-evidence", type=Path, help="Broker retrieval report with history_complete for replay certification")
    parser.add_argument("--import-membership", type=Path, help="Import an attributed free-source membership CSV")
    args = parser.parse_args(argv)
    if args.import_membership:
        from corrected_data import import_membership
        print(json.dumps(import_membership(args.import_membership, args.membership)))
        return
    if args.replay_events:
        from portfolio_ledger import replay_events
        opening = json.loads(args.opening_balances.read_text()) if args.opening_balances else {}
        closing = json.loads(args.closing_balances.read_text()) if args.closing_balances else {}
        result = replay_events(pd.read_csv(args.replay_events), opening_cash=opening.get("cash"), opening_holdings=opening.get("holdings"),
                               expected_cash=closing.get("cash"), expected_holdings=closing.get("holdings"),
                               marks=pd.read_csv(args.marks) if args.marks else None)
        args.output.mkdir(parents=True, exist_ok=True)
        history = json.loads(args.history_evidence.read_text(encoding="utf-8")) if args.history_evidence else {}
        atomic_write_json({**result.metrics, "gaps": result.data_quality, "source_history_complete": history.get("history_complete") is True}, args.output / "replay_reconciliation.json")
        atomic_write_csv(result.events, args.output / "replay_events.csv")
        atomic_write_csv(result.equity, args.output / "replay_daily_equity.csv")
        return
    report = validate_sources(args.data_dir, args.membership, start=args.start, end=args.end)
    report.update(ledger_version=LEDGER_VERSION, causal_version=CAUSAL_VERSION, generated_at=datetime.now(timezone.utc).isoformat(),
                  active_paper_unchanged=True, shadow_only=True, historical_claim="blocked" if not report["complete"] else "not_yet_evaluated")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, args.output / "data_quality.json")
    if not report["complete"] or args.audit_only:
        print(json.dumps({"complete": report["complete"], "gaps": len(report["gaps"]), "report": str(args.output / "data_quality.json")}))
        if args.freeze:
            raise SystemExit("Cannot freeze: verified data and completed corrected validation are required")
        return
    if args.spec is None:
        raise SystemExit("--spec is required for corrected evaluation")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not spec.get("folds") or not spec.get("configurations"):
        raise SystemExit("At least one outer fold and configuration required")
    if any(float(config.get("cost_stress", 1.)) != 1. for config in spec["configurations"]):
        raise SystemExit("Select configurations at baseline costs; run stresses separately on the selected configuration")
    bars, provenance = load_raw_panel(args.data_dir, args.membership, start=args.start, end=args.end)
    actions = pd.read_csv(args.data_dir / "raw" / "actions.csv")
    # Feature input includes every historical member, with as-of source metadata.
    generated_panel, bars = build_raw_features(bars, actions, horizon=int(spec.get("horizon", 20)))
    if spec.get("feature_panel"):
        panel_path = Path(spec["feature_panel"])
        panel = pd.read_parquet(panel_path)
        if spec.get("feature_provenance_verified") is not True:
            raise SystemExit("Feature publication/cutoff provenance is required")
        from corrected_data import validate_dated_inputs
        validate_dated_inputs(panel)
    else:
        panel = generated_panel
    # VIX and sector facts are dated source observations, never today's state
    # copied into history. Optional inputs join on exact date/ticker keys.
    if spec.get("dated_context"):
        context = pd.read_parquet(spec["dated_context"])
        from corrected_data import validate_dated_inputs
        validate_dated_inputs(context)
        panel = panel.merge(context, on=["date", "ticker"], how="left", validate="one_to_one", suffixes=("", "_context"))
        if "sector_context" in panel:
            panel["sector"] = panel.pop("sector_context")
    panel = eligible_candidates(panel, args.membership)
    policy = PaperPolicy(**spec.get("policy", {}))
    if args.observe:
        frozen = json.loads((args.output / "prospective_freeze.json").read_text(encoding="utf-8"))
        if fingerprint(spec) != fingerprint(frozen["specification"]):
            raise SystemExit("Frozen protocol changed: use a new prospective cohort")
        artifact = FoldArtifact(**frozen["artifact"])
        current_id = strategy_identity(frozen["configuration"], artifact, policy)
        if current_id != frozen["strategy_fingerprint"]:
            raise SystemExit("Material strategy change: a new prospective freeze is required")
        # Rechecking training identity detects historical data revisions without fitting on new outcomes.
        rechecked = fit_features(panel, spec["features"], cutoff=artifact.cutoff, label=artifact.label)
        if rechecked.identity != artifact.identity:
            raise SystemExit("Frozen training data changed: prospective restart required")
        from core_satellite_alpha import _nyse_sessions
        freeze_day = pd.Timestamp(frozen["frozen_at"]).tz_convert("America/New_York").tz_localize(None).normalize()
        from signal_freshness import latest_completed_us_trading_day
        available = min(pd.Timestamp(panel.date.max()).normalize(), latest_completed_us_trading_day(), pd.Timestamp(args.end))
        sessions = _nyse_sessions(freeze_day + pd.Timedelta(days=1), available) if available > freeze_day else pd.DatetimeIndex([])
        if sessions.empty:
            raise SystemExit("No completed post-freeze session yet")
        scored = score_features(panel.loc[pd.to_datetime(panel.date) <= available], artifact)
        observation_path = args.output / "prospective_observation_identity.json"
        if observation_path.exists():
            prior = json.loads(observation_path.read_text(encoding="utf-8"))
            if observation_identity(bars, panel, actions, prior["through"]) != prior["identity"]:
                raise SystemExit("Past observed data changed: preserve evidence and restart the prospective cohort")
        targets = build_daily_targets(scored, frozen["configuration"], args.membership, bars=bars)
        result = simulate_daily(bars, targets, start=sessions[0], end=sessions[-1], provenance=provenance, actions=actions,
                                policy=policy, terminal_liquidation=False, base_slippage_pct=frozen["cost_parameters"]["base_slippage_pct"],
                                cost_stress=float(frozen["configuration"].get("cost_stress", 1.)))
        result.metrics["cost_calibration"] = frozen["cost_parameters"]
        atomic_write_csv(result.equity.reset_index(), args.output / "prospective_daily_equity.csv")
        atomic_write_csv(result.events, args.output / "prospective_events.csv")
        atomic_write_json({"through": str(available), "identity": observation_identity(bars, panel, actions, available)}, observation_path)
        status = prospective_status(frozen, now=datetime.now(timezone.utc), current_fingerprint=current_id, observed_sessions=sessions)
        atomic_write_json(status, args.output / "prospective_status.json")
        benchmark = paired_benchmark(result, bars, actions, provenance, start=sessions[0], end=sessions[-1], policy=policy)
        ics = non_overlapping_ic(scored.loc[pd.to_datetime(scored.date) >= sessions[0]], label=artifact.label, as_of=available)
        edge = edge_summary(result.equity.equity.pct_change().dropna(), benchmark.equity.equity.pct_change().dropna(), ics)
        atomic_write_json({**edge, "ledger_version": LEDGER_VERSION, "as_of": str(available.date()),
                           "strategy_fingerprint": current_id, "shadow_only": True}, args.output / "edge_monitor.json")
        return
    policy_id = fingerprint({"spec": spec, "prices": report["price_fingerprints"], "ledger": LEDGER_VERSION, "code": code_fingerprints(),
                             "actions": __import__("hashlib").sha256((args.data_dir / "raw/actions.csv").read_bytes()).hexdigest(),
                             "membership": __import__("hashlib").sha256(args.membership.read_bytes()).hexdigest(),
                             "causal": CAUSAL_VERSION, "policy": asdict(policy), "feature_panel": __import__("hashlib").sha256(pd.util.hash_pandas_object(panel, index=False).values.tobytes()).hexdigest()})
    run_output = args.output / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + policy_id[:12])
    run_output.mkdir(parents=True, exist_ok=False)
    counter = 0
    def evaluate(scored, config, start, end, artifact):
        nonlocal counter
        counter += 1
        result = evaluate_corrected(scored, config, start=start, end=end, bars=bars, actions=actions,
                                    provenance=provenance, membership_path=args.membership, policy=policy)
        benchmark = paired_benchmark(result, bars, actions, provenance, start=start, end=end, policy=policy)
        ics = non_overlapping_ic(scored.loc[(pd.to_datetime(scored.date) >= pd.Timestamp(start))], label=spec["label"], as_of=end,
                                horizon=int(config.get("holding_days", 20)))
        edge = edge_summary(result.equity.equity.pct_change().dropna(), benchmark.equity.equity.pct_change().dropna(), ics,
                            horizon=int(config.get("holding_days", 20)))
        folder = run_output / f"trial_{counter:05d}"
        folder.mkdir(exist_ok=True)
        atomic_write_csv(result.events, folder / "events.csv")
        atomic_write_csv(result.holdings, folder / "holdings.csv")
        atomic_write_csv(result.equity.reset_index(), folder / "daily_equity.csv")
        atomic_write_csv(benchmark.equity.reset_index(), folder / "benchmark_daily_equity.csv")
        atomic_write_csv(benchmark.events, folder / "benchmark_events.csv")
        atomic_write_json({**result.metrics, **edge, "feature_artifact": asdict(artifact), "benchmark_metrics": benchmark.metrics}, folder / "metrics.json")
        return {**result.metrics, **edge}
    def record_trial(config, artifact, fold, metrics):
        # Dedicated shadow trial journal leaves old experiment evidence untouched.
        from causal_research import checkpoint_identity
        trial_identity = checkpoint_identity(code=code_fingerprints(), data=policy_id, policy=asdict(policy),
                                              artifact=asdict(artifact), costs=metrics.get("cost_calibration"), config=config)
        row = {"config": config, "artifact": asdict(artifact), "fold": fold, "metrics": metrics, "strategy_fingerprint": policy_id,
               "checkpoint_identity": trial_identity}
        with (run_output / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    outputs = nested_evaluate(panel, spec["features"], spec["configurations"], spec["folds"], evaluate,
                              label=spec["label"], record_trial=record_trial)
    for output in outputs:
        artifact = FoldArtifact(**output["artifact"])
        fold = output["fold"]
        scored = score_features(panel.loc[pd.to_datetime(panel.date) <= pd.Timestamp(fold["end"])], artifact)
        output["stress_checks"] = []
        for multiplier in spec.get("stress_multipliers", [2., 3., 5.]):
            if not np.isfinite(multiplier) or multiplier < 1:
                raise ValueError("Stress multipliers must be finite and at least one")
            stressed = {**output["configuration"], "cost_stress": float(multiplier)}
            metrics = evaluate(scored, stressed, fold["start"], fold["end"], artifact)
            record_trial(stressed, artifact, {**fold, "trial_kind": "selected_configuration_stress"}, metrics)
            output["stress_checks"].append({"cost_stress": multiplier, "metrics": metrics})
    atomic_write_json({"ledger_version": LEDGER_VERSION, "folds": outputs, "shadow_only": True, "strategy_fingerprint": policy_id,
                       "trial_count": counter, "historical_interpretation": "retrospective_diagnostic",
                       "selection_trials": len(spec["configurations"]) * sum(len(fold["inner"]) for fold in spec["folds"]),
                       "historical_intervals_adjusted_for_selection": False,
                       "multiple_testing_interpretation": "Trial counts disclosed; inspected history cannot establish prospective significance"}, run_output / "validation.json")
    if args.freeze:
        replay_path = args.output / "replay_reconciliation.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
        if replay.get("reconciled") is not True or replay.get("ledger_version") != LEDGER_VERSION or replay.get("source_history_complete") is not True:
            raise SystemExit("Freeze blocked: recorded-event accounting must reconcile first")
        artifact = fit_features(panel, spec["features"], cutoff=args.end, label=spec["label"])
        selected = outputs[-1]["configuration"]
        from execution_cost_calibration import causal_cost_parameters
        cost_parameters = causal_cost_parameters(selected.get("cost_evidence", {}), args.end)
        freeze_path = args.output / "prospective_freeze.json"
        if freeze_path.exists():
            raise SystemExit("Existing freeze preserved; use a new output directory for a restarted cohort")
        atomic_write_json({"frozen_at": datetime.now(timezone.utc).isoformat(), "strategy_fingerprint": strategy_identity(selected, artifact, policy),
                           "required_sessions": 252, "automatic_cutover": False, "configuration": selected,
                           "artifact": asdict(artifact), "cost_parameters": cost_parameters, "specification": spec}, freeze_path)


if __name__ == "__main__":
    main()
