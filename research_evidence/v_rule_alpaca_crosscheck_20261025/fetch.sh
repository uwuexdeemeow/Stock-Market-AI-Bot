#!/bin/bash
# fetch.sh — download QQQ + BIL daily bars and the NYSE calendar with the
# Alpaca CLI into raw/.  Every page of the answer is kept as its own file.
# Usage: ALP=<path to alpaca wrapper or alpaca binary> bash fetch.sh
set -euo pipefail
cd "$(dirname "$0")"
ALP="${ALP:-alpaca}"
START=2015-01-01
END=2026-10-23
for ADJ in all; do
  page=1; token=""
  while :; do
    out="raw/bars_QQQ_BIL_${ADJ}_p${page}.json"
    args=(data multi-bars --symbols QQQ,BIL --start $START --end $END --timeframe 1Day
          --feed sip --adjustment $ADJ --limit 10000 --quiet)
    [ -n "$token" ] && args+=(--page-token "$token")
    "$ALP" "${args[@]}" > "$out"
    token=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('next_page_token') or '')" "$out")
    [ -z "$token" ] && break
    page=$((page+1))
  done
done
"$ALP" calendar --start $START --end 2026-12-31 --quiet > raw/calendar.json
echo "pages: $(ls raw/bars_* | wc -l)"
