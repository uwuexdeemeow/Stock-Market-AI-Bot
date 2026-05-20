#!/usr/bin/env bash
# pull_daily.sh — Pull yesterday's Actions outputs into the local repo.
#
# PLAIN ENGLISH:
# The trading bot runs on GitHub Actions every weekday.  Actions's
# outputs (orders, fills, equity, dashboard files, daily-run logs)
# land on a dedicated `signals/latest` branch, NOT on `main`.  Your
# local repo doesn't see them unless you pull them explicitly.
#
# This script does the three pulls you'd otherwise type by hand:
#
#   1. fetch new refs from origin
#   2. fast-forward main if you pushed code from another machine
#   3. copy yesterday's signal/log files into your working tree
#
# Then it un-stages the files so they don't accidentally get
# committed back to main (signals/latest is the workflow's branch —
# manual commits there would be force-pushed away anyway).
#
# Usage:
#   bash pull_daily.sh
#   # or on macOS/Linux: ./pull_daily.sh   (after `chmod +x pull_daily.sh`)
#   # or Windows: run via Git Bash, MSYS2, or WSL.
#
# Exit codes:
#   0 — everything synced
#   1 — git operation failed (most commonly: uncommitted local changes
#       block the main fast-forward; commit or stash them first)

set -e   # Fail fast on any git error so we don't proceed with partial state.

echo "[pull_daily] fetching origin..."
# `git fetch origin` downloads refs for ALL branches we track (main +
# signals/latest), but does NOT touch working-tree files.  Safe to run
# any time.
git fetch origin

echo "[pull_daily] fast-forwarding main..."
# Switch to main and fast-forward.  --ff-only refuses to merge — if
# you have local commits that diverged from origin/main, this errors
# out so you don't accidentally create a merge commit you didn't want.
git checkout main
git pull --ff-only origin main

echo "[pull_daily] copying signals/latest into working tree..."
# `git checkout <ref> -- <path>...` overwrites those paths in the
# working tree with the version from <ref>.  We use it to land the
# latest signals/ and logs/ files without changing our branch.
# `signals/` and `logs/` are TWO SEPARATE paths; the space matters.
git checkout origin/signals/latest -- signals/ logs/

echo "[pull_daily] un-staging signal/log files..."
# The checkout above stages those files in the index because they
# differ from main's tree.  We don't want them committed to main
# (signals/latest is the workflow's branch).  `git reset HEAD <paths>`
# moves them back to "modified, unstaged" without touching the files
# themselves — the dashboard still reads them as fresh.  The 2>/dev/null
# silences "no changes" warnings when there's nothing to reset.
git reset HEAD signals/ logs/ 2>/dev/null || true

echo ""
echo "[pull_daily] done. Latest dashboard inputs:"
# `ls -la ... 2>/dev/null` swallows "file not found" errors if a
# particular dashboard input doesn't exist yet (e.g. brand-new repo).
ls -la \
    signals/alpaca_paper_log.csv \
    signals/monitor_heartbeat.json \
    signals/factor_data_health.json \
    logs/daily_run_*.json \
    2>/dev/null | tail -5 || true
