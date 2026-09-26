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

The snapshot must be made from a commit that contains those scripts;
otherwise Colab checks out code without them.

## Outputs

Drive keeps the checkpoint, detailed JSON, yearly CSV, analyzer report, and a
compressed validation result bundle (`stockbot_colab_result.tar.gz`) in
`StockBotWalkforward/`.

The research cell saves its own bundle, `stockbot_phase_luck_result.tar.gz`,
holding `phase_luck.json` and `tranche_preview.json`. It never overwrites the
walk-forward bundle. Both are research only and approve nothing.

## Key Terms

- **Walk-forward:** repeatedly choose using older data and test on later data.
- **Out of sample:** a period not used by the selector.
- **Worker:** one CPU process evaluating candidates.
- **Resume:** continue from the last saved checkpoint after interruption.
