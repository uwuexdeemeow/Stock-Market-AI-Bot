# prepare_colab_walkforward.py - Colab Snapshot Builder

## What It Does

This script makes a small, secret-free research snapshot for Google Colab. It
includes parquet prices, their provenance manifests, and approved research
reports. Colab clones the code separately at the exact Git commit recorded in
the snapshot manifest. Alpaca credentials and broker account logs are excluded.

## How To Run It

First commit and push the project. Then run:

```bash
python3 prepare_colab_walkforward.py
```

The command refuses a dirty working tree because uncommitted code cannot be
reproduced in Colab. `--allow-dirty` exists for disposable research only. Use
`--output-dir /path/to/folder` to choose another destination.

## Outputs

Two files appear in `colab/`: a compressed `.tar.gz` snapshot and a matching
`.manifest.json`. The manifest records the archive checksum, exact Git commit,
file list, and whether the working tree was clean. Upload both files to
`My Drive/StockBotWalkforward/`.

## Key Terms

- **Snapshot:** a frozen copy of research inputs.
- **Checksum:** a fingerprint proving the upload was not damaged.
- **Git commit:** the exact saved version of the code.
- **Checkpoint:** saved progress used to resume a stopped walk-forward.
