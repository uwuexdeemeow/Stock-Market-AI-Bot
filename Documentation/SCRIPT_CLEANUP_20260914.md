# Script cleanup — September 14, 2026

## Scope and method

Reviewed the 112 root Python files, dashboard/page modules, test references,
GitHub workflows, tracked notebooks, and shell/Windows launchers. Checked Python
imports, script names in subprocess commands, dashboard discovery, documentation,
and the paper release lock. An unimported command-line tool is not automatically
unused: monthly research and incident recovery have separate entry points.

Comparing parsed Python syntax found no identical complete Python files among
the 195 tracked Python files (excluding empty package initializers). The overlap
below was obsolete functionality, rather than identical copies of whole files.

## Removed

| File | Evidence for removal | Current route |
|---|---|---|
| `concentration_overlay.py` | No caller imported its multiplier helpers. Its executable section printed sample values, despite old runbooks calling it a position audit. The deployed configuration uses a different integrated calculation. | `core_satellite_alpha.py` owns `_apply_concentration_overlay_target`, used by live signals and the research engines. Its calculation is unchanged. |
| `memprofile_walkforward.py` | Isolated reproducer for the old walk-forward memory leak; no workflow, notebook, or other script calls it. The weekly runbook identifies that leak as fixed, and the main engine now copies the input panel and recycles workers. | Keep the engine's memory controls and `run_walkforward_batched.py` for process isolation on smaller machines. |
| `sync_from_remote.sh` | Unreferenced older sync helper copied only four evidence files and duplicated ETF/research refresh commands. It omitted the current research wrapper's downstream checks. | Existing `pull_daily.sh` / `pull_daily.bat` for evidence download, and `refresh_local_research_data.py` for the checked research refresh. |

Removed the prototype's standalone documentation and its dashboard category and
runbook entries. Removed the old profiler entry. Git history retains all removed
source; no account data, research results, checkpoints, or incident history was
deleted.

## Standalone tools deliberately retained

| Tool | Why it remains needed |
|---|---|
| `prepare_colab_walkforward.py` | Documented Colab packaging route in the project-wide beginner guide. |
| `run_walkforward_batched.py` | Documented monthly wrapper that restarts between folds; different purpose from the removed leak reproducer. |
| `refresh_local_research_data.py` | Local research refresh with artifact checks. |
| `membership_reconciliation.py` | Offline review of historical membership sources needed by the corrected audit track. |
| `quant_performance_audit.py` | Independent daily accounting audit; not equivalent to the original periodic backtest. |
| `score_predictiveness_audit.py` | Specific audit of inner selection scores against later held-out results. |
| `shap_feature_reducer.py` | Separate feature-selection method with its own regression coverage. |
| `validate_fixed_live_config.py` | Tests a fixed candidate across outer years, unlike nested selection. |
| `paper_scorecard.py` | Documented weekly performance comparison; execution scorecards and operational health answer different questions. |
| `config_health.py` | Manual configuration and dependency diagnosis, with safety tests. |
| `core_satellite_tqqq.py` | Live generation is retired, but nested walk-forward and independent performance audit still import its research backtest. |
| `ci_check_feature_report.py` | Called by daily workflow cache repair; not obsolete despite its narrow purpose. |
| `pull_daily.sh`, `pull_daily.bat` | Bash and Windows entry points; platform equivalents remain useful. |

The training, prediction, sentiment, and other model modules remain part of the
documented research/shadow path or are imported by retained scripts. Removing
them merely because the factor strategy currently owns paper trading would
break that path.

## Verification

Run `python3 -m pytest -q` and `python3 paper_validation_epoch.py --check-lock`
from the project root. The first checks retained behavior; the second confirms
this cleanup did not change the frozen paper release. No trading workflow is
triggered to test a source cleanup.

Local result: 726 tests passed, 37 skipped; the release lock passed. All 193
retained Python files parsed successfully. A reference scan found no retained
code, workflow, notebook, or launcher calling the removed files, and dashboard
discovery still found the retained daily, corrected-audit, Colab, batched
walk-forward, and paper-scorecard entry points.
