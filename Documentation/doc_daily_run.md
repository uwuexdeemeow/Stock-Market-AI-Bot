# daily_run.py — Daily Pipeline Orchestrator

The runner gives all child scripts one run ID. After monitoring and formal
validation finish, it refreshes canonical readiness and publishes a same-run
evidence manifest. An incomplete manifest blocks `signals/latest`. Health-only
mode refreshes the same report without enforcing failure.

## What it does (plain English)

This is the "one command that runs everything" for daily paper trading.
Instead of running 14+ scripts in the right order, you run `daily_run.py`
and it handles the orchestration.

It runs the steps below in sequence.  If a CRITICAL step fails,
downstream trading steps are skipped so you don't trade on stale data.  A few
watchdog/housekeeping steps still run so their output files stay fresh.

## The pipeline

| # | Step | Critical? | What it does |
|---|------|-----------|--------------|
| 1 | `refresh_etf_data --strict` | ✓ | Download/validate SPY/QQQ/TQQQ etc.; block the run if ETF data remains unhealthy. |
| 2 | `refresh_factor_data` | ✓ | `research.py --incremental` — refresh per-ticker factor panel |
| 3 | `refresh_feature_quality` | ✓ | `feature_quality_diagnostic.py` — re-rank features |
| 4 | `fill_monitor` | ✓, always-run | Check yesterday's orders for stuck fills |
| 5 | `broker_health` | — | Ping Alpaca, alert if down |
| 6 | `core_satellite_signal` | ✓ | `core_satellite_alpha.py` — generate today's signal |
| 7 | `alpaca_submit` | — | Send orders to Alpaca paper |
| 8 | `alpaca_reconcile` | — | Check what filled vs cancelled |
| 9 | `alpaca_execution_guard` | — | Repair ETF stops, cancel stale orders |
| 10 | `alpaca_paper_health` | — | Build deep health summary + drift detection |
| 11 | `alpaca_execution_scorecard` | — | Grade fill quality + execution-risk throttle |
| 12 | `alpaca_gauntlet` | — | Run go-live gauntlet check |
| 13 | `regime_monitor` | — | Detect regime change + alert |
| 14 | `monitor_heartbeat` | always-run | Watchdog over all monitors |

In `--health-only` mode, the watchdog checks the matching
`logs/local_health_YYYYMMDD.json` startup stub. This keeps a local dashboard
refresh separate from the real scheduled `daily_run` evidence.
| 15 | `log_cleanup` | always-run | Disk usage check |

Before signal generation, `drift_monitor` also compares recent ML inputs with
the pooled model's training baseline. It is advisory: a missing baseline is
reported as `no_data`, and the next successful training run creates one.

## How to run

```bash
# Default — run everything
python3 daily_run.py

# Explicit Alpaca run
python3 daily_run.py --alpaca

# Skip data refresh (use existing parquets — useful if you already ran research.py manually)
python3 daily_run.py --alpaca --skip-refresh

# Per-step timeout in seconds (default 300, increase if research.py needs longer)
python3 daily_run.py --alpaca --timeout 600

# Dry-run — show what would run without executing
python3 daily_run.py --dry-run

# Local dashboard refresh only — pull GitHub Actions signal files, then run
# Alpaca health checks without submitting new orders
python3 daily_run.py --health-only

# Same health-only mode, but use whatever files already exist locally
python3 daily_run.py --health-only --no-github-sync

# Force-run on weekends/holidays
python3 daily_run.py --force

# Also run stress tests (factor decay, drawdown throttle, execution, survivorship)
python3 daily_run.py --alpaca --stress

```

## Outputs

| File | What's in it |
|------|--------------|
| `logs/daily_run_YYYYMMDD.json` | Per-step results, timings, errors |
| `logs/local_health_YYYYMMDD.json` | Local `--health-only` results |
| All artifacts from each step | See individual script docs |

## Schedule

In production, this is scheduled via GitHub Actions:

```yaml
cron: "35 14 * * 1-5"  # 14:35 UTC = 9:35 AM ET, weekdays only
```

The workflow file `.github/workflows/daily_paper_trading.yml` invokes
`python3 daily_run.py --alpaca --timeout 600`.

## Safety features

- **PID lock** — refuses to run if another `daily_run.py` is already
  running (prevents cron + manual overlap from double-submitting).
- **Weekend/holiday guard** — skips automatically on weekends and US
  market holidays.  Use `--force` to override (e.g., for testing).
- **Critical-step short-circuit** — if a step marked `critical=True`
  fails, downstream trading steps don't run.  Better to skip than to trade on
  broken state.
- **Always-run monitors** — `fill_monitor`, `monitor_heartbeat`, and
  `log_cleanup` still run after an upstream failure.  This keeps watchdog files
  fresh even on no-trade or blocked days.
- **Startup stubs** — before any step runs, `daily_run.py` writes a fresh
  `logs/daily_run_YYYYMMDD.json` and `signals/fill_monitor.json` placeholder.
  If the run crashes early, `monitor_heartbeat.py` still sees where the pipeline
  got to instead of reporting a misleading missing file.
- **Consolidated Telegram in GitHub Actions** — the workflow sets
  `STOCKBOT_SCRIPT_TELEGRAM_ENABLED=0`, so scripts do not send their own
  Telegram alerts.  The workflow sends one final summary instead.
- **Health-only mode** — `python3 daily_run.py --health-only` is for your
  laptop after GitHub Actions already traded.  It first fetches the
  `signals/latest` branch and copies signal/dashboard files into local
  `signals/` and `logs/`, then runs health checks without `alpaca_submit` or
  `core_satellite_signal`. The synced files include
  `alpaca_daily_status.json`, so local target-weight checks use the same saved
  Alpaca positions and equity as the automated run.
- **Per-step timeout** — default 5 min per step (10 min recommended
  for research.py).  The timeout is enforced even when a child script is
  silent and prints no progress.
- **Internal Telegram alert** — sends a warning when any step fails,
  regardless of the workflow exit code.
- **No automatic trading retry** — GitHub runs the trading pipeline once. A
  failed attempt may already have submitted some orders, so the workflow stays
  failed and waits for broker reconciliation instead of rerunning the full plan
  and possibly hiding a partial failure behind duplicate-order protection.

## Key concepts

- **Critical step** — a step whose failure means downstream steps can't
  safely run.  Marked with `critical=True` in the Step dataclass.
  Example: if data refresh fails, signal generation can't trust prices.
- **Always-run step** — a safety or housekeeping step that still runs after an
  earlier blocker because its output is useful even when trading is skipped.
- **Skipped step** — a step that ran but didn't execute (e.g., in
  dry-run mode, or because a critical predecessor failed).  Skipped is
  not the same as failed — it counts as "didn't run" not "broken".
- **PID lock** — a file (`logs/daily_run.lock`) that the script
  exclusive-locks while running.  If a second copy can't acquire the
  lock, it exits cleanly.  Auto-released when the process ends.

## Recent additions

- `broker_truth` step (June 6, 2026) runs after `alpaca_execution_guard` and
  before `alpaca_paper_health`.  It reconciles signal targets, planned orders,
  the paper log, live Alpaca positions, and trailing stops into
  `signals/broker_truth.csv` and `signals/broker_truth.json`. Normal trading
  runs now require settled alignment after a bounded 90-second wait; a failed
  verdict fails the workflow while later always-run health reports still run.
  Local `--health-only` mode remains report-only.
  A settled failure also publishes `signals/alignment_recovery_plan.csv` for
  manual review. It updates `signals/alignment_incident_ledger.csv` until a
  later live pass records resolution. The daily runner never executes either
  file.
- `alpaca_paper_health` step (May 19, 2026) runs drift detection against
  walkforward results on the Alpaca pipeline.
- `--health-only` mode (May 20, 2026) — added for local dashboard refresh
  when GitHub Actions is responsible for the real daily trading run.

## When something goes wrong

1. Check the Telegram alert — it tells you which step failed.
2. Read `logs/daily_run_YYYYMMDD.json` for per-step error tails.
3. Common fixes:
   - **Refresh timeout** — increase `--timeout` to 600.
   - **Feature quality stale** — manually run
     `python3 feature_quality_diagnostic.py --top 48`.
   - **Market closed** — the Alpaca submit step prompts in interactive
     mode but auto-proceeds in CI.  Orders queue for next open.

## Reliability Order

Broker health and submission are critical. After a critical failure,
reconciliation, broker truth, health reports, cost calibration, the gauntlet,
and paper-epoch status still run. Telegram therefore reports the real account
condition. `no_action` is healthy when the market is closed, the portfolio is
aligned, or a deliberate safety rule correctly prevents trading.
