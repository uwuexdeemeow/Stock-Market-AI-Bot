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

## Outputs

Drive keeps the checkpoint, detailed JSON, yearly CSV, analyzer report, and a
compressed validation result bundle under `StockBotWalkforward/results/`.

## Key Terms

- **Walk-forward:** repeatedly choose using older data and test on later data.
- **Out of sample:** a period not used by the selector.
- **Worker:** one CPU process evaluating candidates.
- **Resume:** continue from the last saved checkpoint after interruption.
