# Production Roadmap — Phases 1 through 5

Beginner-friendly companion to the new modules added during the production
hardening pass. Every file listed here is a real, runnable Python script at
the project root. Commands assume your terminal is already in the project
directory and a venv with `requirements.txt` is active.

---

## Phase 1 — Prove there's an edge

### `leakage_audit.py`
**What:** Checks that no feature accidentally uses future information.
**Run:** `python leakage_audit.py`
**Output:** `logs/leakage_audit.json` with one entry per ticker; status `pass` or `FAIL`.
**Key terms:**
- *Feature* — one input column the model sees (e.g., 14-day RSI).
- *Leakage* — feature value at time `t` depends on data from time `t+1` or later. Fatal bug.

### `universe.py`
**What:** Loads training/backtest universe (S&P 500) and live universe (4 approved).
**Run:** `python universe.py` (prints counts)
**Use in code:** `from universe import sp500_universe, live_universe`
**Key terms:**
- *Survivorship bias* — using only today's S&P 500 in a 2015 backtest omits the losers, flattering results.

### `labels.py`
**What:** Three target definitions — plain forward return, vol-normalized return, triple-barrier.
**Use:** `from labels import forward_return, vol_normalized_return, triple_barrier`
**Key terms:**
- *Triple barrier* — label a trade by which of three events fires first: profit target, stop, time-out.

---

## Phase 2 — Make the model more robust

### `nested_cv.py`
**What:** Honest hyperparameter search. Outer loop tests; inner loop tunes.
**Use:** `from nested_cv import nested_walk_forward_search`
**Key terms:**
- *Cross-validation* — repeatedly splitting data into train/test to measure generalization.
- *Walk-forward* — time-ordered splits only. You never train on the future.

### `shap_feature_reducer.py`
**What:** Drops weak features, keeps top ~15 by SHAP importance.
**Use:** `from shap_feature_reducer import reduce_features`
**Output:** `models/selected_features.json`.
**Key terms:**
- *SHAP value* — how much each feature changed one specific prediction.

### `calibration_stability.py`
**What:** Verifies the confidence→probability mapping doesn't drift between folds.
**Use:** `from calibration_stability import calibration_stability`
**Key terms:**
- *Calibration* — if the model says "70%", it should be right 70% of the time.
- *KS statistic* — distance between two distributions, 0 identical to 1 disjoint.

### `settings.py` change
`ENSEMBLE_WEIGHTS` now carries a `"neural": 0.0` slot. Flip to e.g. `0.3` after
offline validation confirms the neural branch adds out-of-sample edge.

---

## Phase 3 — Risk & execution realism

### `risk_sizing.py`
**What:** Three sizing tools.
- `vol_target_size(equity, asset_vol)` — size so each position contributes the same vol.
- `fractional_kelly(edge)` — bankroll-safe Kelly bet at 25% default.
- `atr_stop(entry, atr)` / `position_size_with_stop(...)` — stop-loss + shares from stop distance.
**Key terms:**
- *Kelly criterion* — math-optimal bet for a repeated edge. Full Kelly is too aggressive in practice.
- *ATR* — average true range, a short-window volatility measure.

### `execution_model.py`
**What:** Realistic fill price = mid + half-spread + sqrt-impact + baseline slippage.
**Use:** `from execution_model import realistic_fill_price, commission, capacity_warning`
**Key terms:**
- *Spread* — difference between bid and ask.
- *Market impact* — your order itself moves price; grows roughly with sqrt(size/ADV).

### `portfolio_manager.py` change
Added a soft de-risking band: between -8% and -15% drawdown, the manager
linearly cuts gross/net exposure budgets toward zero before the hard halt.

---

## Phase 4 — Production plumbing

### `orchestration.py`
**What:** Single entry point chaining scanner → research → predict → paper trade.
**Run:** `python orchestration.py daily`
**Cron:** `30 16 * * 1-5 cd /path/to/project && python orchestration.py daily >> logs/cron.log 2>&1`
**Output:** `logs/pipeline_runs.jsonl`.

### `model_registry.py`
**What:** Versioned storage for trained models + metrics + git SHA.
**Use:** `from model_registry import register, latest`
**Notes:** Uses MLflow if installed, otherwise writes `models/registry.json`.

### `data_validation.py`
**What:** Schema + freshness checks before data enters train/infer.
**Use:** `from data_validation import validate_price_frame, validate_feature_frame`

### `drift_monitor.py`
**What:** PSI + KS checks comparing today's features/scores to a saved baseline.
**Use:** snapshot once via `snapshot_baseline(...)` at train time; call `check_drift(...)` daily.
**Key terms:**
- *PSI* — Population Stability Index. <0.1 stable, 0.1–0.25 caution, >0.25 drift.

### `broker_interface.py`
**What:** Abstract `Broker` base class + `DryRunBroker`. New venues just add an adapter.

### `tests/test_sanity.py`
**Run:** `python -m pytest tests/`
**What:** 6 smoke tests (≈0.4s total) covering labels, sizing, execution, validation.

---

## Phase 5 — Paper gauntlet & go-live gate

### `alpaca_paper_gauntlet.py`
**What:** Reads `signals/alpaca_paper_equity.csv` + `signals/alpaca_paper_log.csv`, benchmarks vs SPY, computes:
- net Sharpe, max drawdown, Newey-West t-stat vs SPY, realized vs modeled slippage ratio.
**Run:** `python alpaca_paper_gauntlet.py`
**Gate criteria:** see `GO_LIVE_CHECKLIST.md`.

### `GO_LIVE_CHECKLIST.md`
**What:** Eight-section checklist you must complete before risking real capital.

---

## End-to-end workflow (updated)

```
scanner.py         → shortlist candidates
  ↓
research.py        → pull features, news, macro
  ↓  (validate with data_validation.validate_price_frame)
train.py           → fit model, register with model_registry.register
  ↓
leakage_audit.py   → sanity-check feature pipeline
  ↓
backtest.py        → walk-forward, costs on, universe=sp500_universe()
  ↓ (gate: Sharpe ≥ 1.0, t-stat > 2)
shap_feature_reducer.py  →  nested_cv.py  →  calibration_stability.py
  ↓
predict.py + risk_sizing + execution_model  → daily signals
  ↓
portfolio_manager.py  → risk approval (soft DD band, hard halt)
  ↓
alpaca_paper_trading.py via broker_interface abstraction
  ↓  (daily: drift_monitor.check_drift)
alpaca_paper_gauntlet.py (≥60 days)  →  GO_LIVE_CHECKLIST.md
  ↓ (only if every box checked)
LIVE (start at 10% target size)
```

Every step writes structured logs into `logs/` so a monitoring stack can
alert on failures.
