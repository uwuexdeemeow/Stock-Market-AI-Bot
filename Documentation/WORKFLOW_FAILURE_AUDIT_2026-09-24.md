# Workflow failure audit — September 24, 2026

This audit separates software/data defects from a genuine strategy rejection.
It covers the recent failed runs relevant to the current paper workflow, not
every historical run ever created in the repository.

| Failure | Evidence | Current position |
| --- | --- | --- |
| Feature-quality report older than a refreshed ticker parquet | Daily Paper Trading runs [#288](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35624652155) and [#289](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35639815009) | Prior daily rebuild/order fixes are in main; the later [#295](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35876592636) succeeded. The local refresh wrapper now also rebuilds its feature-health profile after quality grading. |
| Stale factor-decay evidence | Daily run [#287](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35606546349) | Daily workflow now refreshes this short-lived report before signaling; later run #295 succeeded. |
| Survivorship review rejection | Daily run [#291](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35734424806) | Subsequent survivorship input/validation repairs landed and later run #295 succeeded. This does not guarantee all future survivorship tests will pass. |
| Factor-decay refresh error | Factor Data Refresh [#235](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35854896060) | ETF price repair was added and later factor runs reached the final robustness step, so this earlier step was no longer the immediate failure. |
| Invalid latest NEE OHLC bar | Local strict data-health check: September 23 Yahoo Open 79.04 exceeded High 79.00 | Yahoo subsequently supplied a valid corrected High 79.04. The local NEE parquet and manifest were rebuilt; code now rejects structurally invalid cache/provider bars, can use a narrowly cross-checked Alpaca IEX replacement, and fails closed if sources disagree. After regrading all 42 features and rebuilding feature health, **local strict factor-data health passed** with all 62 required tickers current. GitHub's next run remains to be verified. |
| Final execution-stress rejection | Factor Data Refresh [#237](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35866225766), [#238](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35876534316), and [#239](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35905249186) all failed `verify_robustness_evidence` | **Still unresolved.** One-day-delayed execution scenarios on the previously viewed 2023–2026 period underperformed QQQ. The bounded pre-2023 stable-grid walk-forward also rejected its leading diversified candidate for turnover, selection, and fallback failures. A green CI run or data repair cannot make a strategy pass this test. |

The latest successful daily workflow shows that previously failing software
paths can complete, but it does not prove the current strategy is robust.
Never bypass the delayed-fill gate, late-order cutoff, data-health check, or
paper-version lock just to obtain a green workflow. A new research hypothesis
needs independent validation and prospective paper evidence before promotion.
