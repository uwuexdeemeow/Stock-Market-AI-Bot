# prepare_colab_walkforward.py - Colab Bundle Builder

## What it does

This script makes one compressed copy of the current project for Google Colab.
It includes the code, saved market data, models, and research signals needed by
the walk-forward. It excludes passwords, `.env` files, logs, Git history,
virtual environments, and temporary Python files.

This matters because cloning GitHub alone may not reproduce the exact local
project. Local code changes and ignored market-data files can be newer than the
copy on GitHub.

## How to run it

From the project folder:

```bash
python3 prepare_colab_walkforward.py
```

Expected output:

```text
../Stock_Market_AI_Bot_Colab.zip
```

To choose a different location:

```bash
python3 prepare_colab_walkforward.py --output /path/to/my_bundle.zip
```

Upload the finished zip file to this Google Drive folder:

```text
My Drive/StockBotColab/Stock_Market_AI_Bot_Colab.zip
```

Then open `Colab/stockbot_walkforward.ipynb` in Google Colab and run its cells
from top to bottom.

## Inputs

- Current project code
- `data/` market and factor files
- `models/` saved model files
- `signals/` research inputs and previous results

## Outputs

- `Stock_Market_AI_Bot_Colab.zip`, ready to upload to Google Drive

## Key terms

- **Bundle**: one compressed file containing many project files.
- **Secret**: private information such as an API key, broker password, or token.
- **Checkpoint**: a small progress file that lets a stopped walk-forward resume.
- **Walk-forward**: a historical test that repeatedly trains on earlier periods
  and tests on a later period without using future knowledge.
