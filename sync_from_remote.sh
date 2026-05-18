#!/bin/bash
# sync_from_remote.sh — pull today's signals + refresh local research data
#
# PLAIN ENGLISH: After the GitHub Actions bot runs each morning, this
# script syncs what it produced to your local machine.
#   - Signal CSVs come from the signals/latest git branch.
#   - Research data (parquet files) is regenerated locally because it's
#     too large to push to git.
#
# Usage:
#   ./sync_from_remote.sh             # pull signals + refresh data
#   ./sync_from_remote.sh --signals   # only pull signals
#   ./sync_from_remote.sh --data      # only refresh data

set -e  # stop on first error
cd "$(dirname "$0")"

# Defaults: do both
SYNC_SIGNALS=1
SYNC_DATA=1

# Parse flags — restrict to one or the other if requested
case "${1:-both}" in
    --signals) SYNC_DATA=0 ;;
    --data)    SYNC_SIGNALS=0 ;;
    both|"")   ;;
    *) echo "Usage: $0 [--signals|--data]"; exit 1 ;;
esac

# ── Sync signals from signals/latest branch ─────────────────────────────
if [ "$SYNC_SIGNALS" = "1" ]; then
    echo "→ Fetching signals/latest from GitHub..."
    git fetch origin signals/latest 2>&1 | tail -3

    mkdir -p signals
    # Files the workflow pushes — copy each one if present remotely
    for f in \
        signals/core_satellite_alpha_signal.csv \
        signals/core_satellite_alpha_orders.csv \
        signals/core_satellite_alpha_metrics.json \
        signals/alpaca_paper_equity.csv ; do
        if git show "origin/signals/latest:$f" > "$f.tmp" 2>/dev/null; then
            mv "$f.tmp" "$f"
            echo "  ✓ $f"
        else
            rm -f "$f.tmp"
            echo "  ⚠ skipped (not in remote): $f"
        fi
    done

    # Last commit info on signals/latest
    echo ""
    echo "Latest signal commit:"
    git log origin/signals/latest --oneline -1 2>/dev/null || echo "  (no commits yet)"
fi

# ── Refresh research data (re-runs locally — not stored in git) ─────────
if [ "$SYNC_DATA" = "1" ]; then
    echo ""
    echo "→ Refreshing research data (this may take 2-3 minutes)..."
    python3 refresh_etf_data.py --refresh --force 2>&1 | tail -3
    python3 research.py --incremental 2>&1 | tail -5
    echo "  ✓ Research data refreshed"
fi

echo ""
echo "Done. To inspect today's picks:"
echo "  python3 -c \"import pandas as pd, json; r = pd.read_csv('signals/core_satellite_alpha_signal.csv').iloc[0]; w = json.loads(r['overlay_weights_json']); print(json.dumps(w, indent=2))\""
