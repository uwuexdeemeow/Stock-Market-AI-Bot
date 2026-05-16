"""Quick diagnostic: why are folds 2016-2026 failing cost stress?"""
import pandas as pd
import numpy as np
from core_satellite_nested_walkforward import (
    iter_candidate_configs, config_with_cost_stress, config_signature,
    evaluate_window, build_fold_splits, build_inner_folds,
    BASE_COST_STRESS,
)
from alpha_factor_backtest import (
    load_factor_panel, load_feature_specs, load_prediction_scores,
    attach_scores,
)
from core_satellite_alpha import (
    _ensure_robust_score_columns, COST_STRESS_MULTIPLIERS,
)

# Load panel
specs = load_feature_specs(max_specs=48)
panel = load_factor_panel(specs)
panel = attach_scores(panel, specs, load_prediction_scores())
panel = _ensure_robust_score_columns(panel)
panel["_date"] = pd.to_datetime(panel["date"], errors="coerce")

# One config to test
configs = iter_candidate_configs(
    strategy="core-alpha",
    holding_days=(10,),
    overlay_gross=(0.50,),
    ma_windows=(100,),
    high_vol_values=(0.30,),
    high_vol_modes=("fixed",),
    shapes=("top10",),
    weightings=("sticky_score",),
    tqqq_weights=(0.0,),
)
config = configs[0]
print(f"Config: {config_signature(config)}")
print(f"Cost stress levels: {COST_STRESS_MULTIPLIERS}")
print()

# Test a few failing folds
splits = build_fold_splits(panel, min_train_years=4)
for test_year in [2016, 2020, 2023, 2025]:
    matching = [s for s in splits if s.outer_year == test_year]
    if not matching:
        print(f"=== {test_year}: no fold ===")
        continue
    split = matching[0]
    inner_years = list(range(split.train_start.year, split.train_end.year + 1))
    inner_folds = build_inner_folds(inner_years, min_inner_train_years=3)[-5:]

    print(f"=== {test_year} outer fold ===")
    print(f"  Inner folds: {[f.validation_year for f in inner_folds]}")

    # Test first inner fold at each cost level
    fold = inner_folds[-1]  # most recent inner fold
    print(f"  Testing inner fold {fold.validation_year}:")
    for cost in COST_STRESS_MULTIPLIERS:
        cost_config = config_with_cost_stress(config, float(cost))
        try:
            m = evaluate_window(panel, cost_config, fold.validation_start, fold.validation_end)
            spy = m.get("alpha_vs_spy_pct", 0) or 0
            qqq = m.get("alpha_vs_qqq_pct", 0) or 0
            blend = m.get("alpha_vs_blend_pct", 0) or 0
            gate = float(m.get("max_drawdown_pct", -999) or -999) > -50.0
            print(f"    cost={cost}x: alpha_spy={spy:+.1f}% alpha_qqq={qqq:+.1f}% "
                  f"alpha_blend={blend:+.1f}% dd={m.get('max_drawdown_pct',0):.1f}% "
                  f"gate={'PASS' if gate else 'FAIL'}"
                  f"{'  ← NEGATIVE ALPHA' if spy <= 0 or qqq <= 0 or blend <= 0 else ''}")
        except Exception as e:
            print(f"    cost={cost}x: ERROR {e}")
    print()
