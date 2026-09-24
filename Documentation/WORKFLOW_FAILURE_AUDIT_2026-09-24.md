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
| Final execution-stress rejection | Factor Data Refresh [#237](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35866225766), [#238](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35876534316), and [#239](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35905249186) all failed `verify_robustness_evidence` | Those runs genuinely rejected their then-current data snapshot. After the September 23 data and all dependent reports were refreshed locally, the **same incumbent configuration passed** the workflow-order stress, survivorship, factor-decay, strict review, and identity checks. One-day-delay holdout alpha versus QQQ was +21.16%; with 10/25 bps extra cost it was +17.61%/+12.35%. This is a later snapshot, not proof that the earlier result was a software bug or that future runs cannot fail. GitHub's next run must verify it independently. |
| Latest strategy rejection | Factor Data Refresh [#240](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/35993380391) | Its artifact and a local replay of its exact 72-Parquet cache agree: the incumbent's one-day-delay 2023–2026 alpha versus QQQ was **-13.01%**, falling to -16.36% with 10 bps and -21.30% with 25 bps additional cost. The `holdout_2023_2026_vs_qqq_pass` gate correctly rejected all three delayed scenarios. This differs materially from the earlier local 159-file snapshot, so that local pass did not predict this run. No safety threshold was relaxed. |
| Mixed research and robustness cache | Daily Paper Trading [#296](https://github.com/uwuexdeemeow/Stock-Market-AI-Bot/actions/runs/36006649752) | Factor #240 saved its data cache despite the strategy rejection and correctly skipped its validated state cache. Daily #296 then restored the newest data and older state using independent prefix keys. Its signal refused to combine the three mismatched execution, survivorship, and factor-decay fingerprints. Order submission was blocked. The workflow now restores a single price-and-report snapshot saved only after the strict factor gate passes, validates it, refreshes ETF prices once, records the final data fingerprint, and regenerates all three robustness reports before the signal. |
| Candidate configuration not approved | Ten-fold, pre-2023 stable-grid nested walkforward replayed on Factor #240's exact 72-Parquet snapshot | Approval was **false**: worst out-of-sample turnover was 654% versus the 600% ceiling, selector alpha correlation was -0.151, selector Sharpe uplift was -0.177, 2022 required a relaxed fallback, and the execution-stress review failed. This candidate was not published to the live config or used for orders. |

The latest successful daily workflow shows that previously failing software
paths can complete. The refreshed local robustness pass is better evidence
than the stale local report, but still does not guarantee future robustness.
The exact input change responsible for the flip between the 159-file local
snapshot and GitHub #240's 72-file snapshot has not been isolated; do not
attribute it solely to NEE's corrected one-day bar. GitHub #240's research
manifest was also written before its final ETF refresh; six ETF Parquet hashes
changed afterward. Both workflows now write the manifest after their final ETF
refresh. No new configuration was promoted. Never bypass the delayed-fill
gate, late-order cutoff, data-health check, or paper-version lock just to
obtain a green workflow. A new research hypothesis needs independent
validation and prospective paper evidence before promotion.
