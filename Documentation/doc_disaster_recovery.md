# `disaster_recovery.py`

## What it does

This script restores local paper evidence from a complete GitHub Actions
artifact. It verifies every checksum, backs up current local files, and restores
only listed files below `signals/` or `logs/`. It never loads broker code or
submits, cancels, or repairs an order.

## How to run it

```bash
python disaster_recovery.py --artifact path/to/artifact --dry-run
python disaster_recovery.py --artifact path/to/artifact
python disaster_recovery.py --from-github --dry-run
python disaster_recovery.py --from-github
```

`--from-github` requires an authenticated GitHub CLI. Expected output includes
the source run, file count, backup directory, and `orders_submitted=false`.

## Key terms

- **Artifact:** files retained by one GitHub Actions run.
- **Checksum:** a fingerprint used to detect corruption.
- **Dry run:** validation without restoring files.
- **Recovery backup:** replaced files saved below `archive/recovery_backups/`.

