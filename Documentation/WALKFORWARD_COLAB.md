# Run The Next Walk-Forward In Google Colab

## Before Opening Colab

Run this on the project computer:

```bash
python3 prepare_colab_walkforward.py
```

Create a Google Drive folder named `StockBotWalkforward`. Upload the new
snapshot `.tar.gz` and matching `.manifest.json` files from `colab/`.

## In Colab

Open `colab/stockbot_walkforward.ipynb` in Google Colab. Choose a high-RAM CPU
runtime when one is available. Run the cells from top to bottom.

The notebook will:

1. Mount Google Drive.
2. Clone the exact Git commit recorded in the notebook.
3. Verify the snapshot checksum.
4. Install project dependencies.
5. Run two outer folds at a time and copy each checkpoint to Drive.
6. Resume automatically until all folds finish.
7. Run the walk-forward analyzer and package the result for local review.

The notebook never receives Alpaca keys and uses `--no-publish-live-config`, so
it cannot submit orders or change the paper strategy.

## After Colab Finishes

Download `stockbot_colab_result.tar.gz` from the Drive folder into the local
project. Review the JSON, CSV, and analyzer report before any publication.
Matching survivorship, execution-stress, and factor-decay reviews must then be
generated locally for that exact config and dataset fingerprint.
