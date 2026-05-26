# safe_io.py - What It Does and How to Run It

## What This Script Does

`safe_io.py` contains shared helpers for writing important files safely. The bot
uses it for signal CSVs, health JSON files, paper logs, and other files that
should not be half-written if a script crashes.

The main idea is simple: write the new content to a temporary file first, then
rename that temporary file over the real file in one operation.

It also:

- Checks free disk space before writing.
- Uses unique temporary filenames so stale temp files are not overwritten.
- Provides UTF-8 subprocess helpers for Windows-safe output capture.
- Provides a PID lock so two copies of a script do not run at the same time.

## How To Run It

This file is usually imported by other scripts:

```bash
python -m pytest tests/test_safe_io.py -q
```

Expected output is a passing pytest run.

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| Atomic write | A write that appears all at once, not half-finished. |
| Temporary file | A short-lived file used while preparing the final output. |
| Disk-space guard | A check that refuses to write when the disk is too full. |
| PID lock | A lock file that prevents duplicate script runs. |
| UTF-8 | A text encoding that can safely carry most symbols and characters. |
