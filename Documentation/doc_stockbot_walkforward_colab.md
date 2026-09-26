# Google Colab Walk-Forward Notebook

## What It Does

`Colab/stockbot_walkforward.ipynb` runs the lower-turnover nested walk-forward
on a Colab CPU machine. It mounts Google Drive, clones the exact Git commit,
verifies the snapshot checksum and data manifest, and saves checkpoints after
small fold batches. It cannot place orders and receives no Alpaca credentials.

## Beginner Steps

1. Commit and push the project to GitHub.
2. Run `python3 prepare_colab_walkforward.py` on the Mac.
3. Create `StockBotWalkforward` in Google Drive.
4. Upload the generated `.tar.gz` and `.manifest.json` files.
5. Open `Colab/stockbot_walkforward.ipynb` in Google Colab.
6. Use a High-RAM CPU runtime when available; a GPU is not needed.
7. Run every cell from top to bottom and approve Drive access.
8. If Colab disconnects, reconnect and run the cells again. The Drive
   checkpoint resumes completed folds.
9. Download the final packaged validation results for local review.

## Choosing What To Run

The settings cell has two switches:

- `RUN_WALKFORWARD = True` runs the nested walk-forward (the default).
- `RUN_PHASE_LUCK = False` controls the optional research cell. Set it to
  `True` to run the rebalance-day luck diagnostic and the H-tranche preview
  from `research_evidence/phase_luck_20260926/` (about 6 minutes each). To
  run only the research, set `RUN_WALKFORWARD = False`.

- `RUN_BAKEOFF = False` controls the signal bake-off cell (Hypothesis
  H-bakeoff in `DELAY_STRESS_PAPER_ADVISORY.md`). Set it to `True` to run
  `research_evidence/signal_bakeoff_20260926/signal_bakeoff.py` (240 engine
  runs, about 20–30 minutes). Its idea C needs the 11 sector ETF files, so
  **before making the snapshot** run on the project computer:
  `python refresh_etf_data.py --symbols XLK XLY XLF XLV XLE XLI XLP XLU XLRE XLB XLC --refresh`.
  The cell stops with a clear message if any of them is missing.

The snapshot must be made from a commit that contains those scripts;
otherwise Colab checks out code without them. Merge the work into `main`
first, then run `python3 prepare_colab_walkforward.py`.

## Outputs

Drive keeps the checkpoint, detailed JSON, yearly CSV, analyzer report, and a
compressed validation result bundle (`stockbot_colab_result.tar.gz`) in
`StockBotWalkforward/`.

The research cell saves its own bundle, `stockbot_phase_luck_result.tar.gz`,
holding `phase_luck.json` and `tranche_preview.json`. The bake-off cell saves
`stockbot_bakeoff_result.tar.gz`, holding `signal_bakeoff.json`, and prints the
winner (or "no winner") and the reason. None of them overwrites the
walk-forward bundle. All are research only and approve nothing.

## Key Terms

- **Walk-forward:** repeatedly choose using older data and test on later data.
- **Out of sample:** a period not used by the selector.
- **Worker:** one CPU process evaluating candidates.
- **Resume:** continue from the last saved checkpoint after interruption.
- **Bake-off:** a fixed comparison of several signal ideas under one rule
  written down before the run.
