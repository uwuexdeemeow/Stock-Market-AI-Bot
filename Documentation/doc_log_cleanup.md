# Log Cleanup

## What it does

`log_cleanup.py` reports old logs and can remove files beyond a retention limit.
Preview is the default, making cleanup reviewable before any deletion.

## How to run it

Run `python3 log_cleanup.py` for a dry preview. Use `--retention DAYS` to change
the age limit, `--check-disk` to show disk use, and `--execute` only after
reviewing the exact files. Output lists candidates, bytes, and completed work.

## Key terms

- **Retention:** how long logs are kept.
- **Dry run:** show intended changes without making them.
- **Disk usage:** storage currently occupied.
