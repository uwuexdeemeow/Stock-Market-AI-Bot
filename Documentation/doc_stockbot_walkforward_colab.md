# Google Colab Walk-Forward Notebook

## What it does

`Colab/stockbot_walkforward.ipynb` runs the project's nested walk-forward on a
Google Colab computer. The notebook copies the large project bundle to Colab's
fast temporary disk, while keeping the small `signals/` folder in Google Drive.
That gives the calculation faster local file access and keeps checkpoints and
final results safe if the Colab session stops.

The notebook defaults to the next lower-turnover research test. It does not
publish a live configuration and cannot place broker orders.

## How to use it

1. Run `python3 prepare_colab_walkforward.py` on the Mac.
2. In Google Drive, create a folder named `StockBotColab`.
3. Upload `Stock_Market_AI_Bot_Colab.zip` into that folder.
4. Upload `Colab/stockbot_walkforward.ipynb` to Drive or open it from the Mac
   at [colab.research.google.com](https://colab.research.google.com/).
5. In Colab, choose **Runtime > Change runtime type**.
6. Choose **High-RAM** when your Colab plan offers it. A GPU is not needed.
7. Run every notebook cell from top to bottom.
8. Approve Google Drive access when Google asks.
9. Leave the final walk-forward cell running. Colab writes one checkpoint after
   each completed yearly fold.
10. Find results in `My Drive/StockBotColab/runs/low_turnover_next/signals/`.

If Colab disconnects, reconnect and run the cells again. Keep `RESET_RUN` set to
`False`; the notebook will continue from the saved checkpoint.

## Important settings

- `GRID_MODE = "low-turnover"`: tests the lower-turnover candidate grid.
- `WORKERS = 2`: safe default for normal Colab memory.
- `LOW_MEMORY = True`: reduces the risk of a memory crash.
- `RESET_RUN = False`: resumes an existing run.
- `RUN_NAME = "low_turnover_next"`: names the persistent Drive results folder.

For a paid High-RAM CPU runtime, try `WORKERS = 4` and
`LOW_MEMORY = False`. More workers need more memory. Do not increase the number
just because a GPU is attached; this walk-forward mainly uses CPU and memory.

## Expected outputs

- A checkpoint while work is incomplete:
  `signals/walkforward_checkpoint_core_alpha.json`
- Final detailed result:
  `signals/colab_low_turnover_next.json`
- Final yearly table:
  `signals/colab_low_turnover_next.csv`

The script can add a timestamp to the final filename if a result with the same
name already exists.

## What happens after it finishes

Bring the JSON and CSV result back into the local project before analysis. The
result still has to pass the same quant checks: out-of-sample return, drawdown,
turnover, trading-cost stress, stability across years, and leakage review. A
fast Colab run does not make a weak strategy acceptable.

## Key terms

- **Colab runtime**: a temporary Google computer used to run the notebook.
- **CPU**: the part doing most of this walk-forward calculation.
- **High-RAM**: a runtime with more working memory, useful for large datasets.
- **Research-only**: results are measured but are not approved for trading.
- **Out of sample**: a period the strategy selector did not see beforehand.
