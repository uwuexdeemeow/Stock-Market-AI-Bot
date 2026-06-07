# research_output_guard.py - Research Output Guard

## What It Does

`research_output_guard.py` checks generated research files before they are
trusted, committed, or uploaded by GitHub Actions.

Plain meaning: it makes sure the JSON and CSV files in `signals/` are real
machine-readable files, not broken merge-conflict leftovers.

It checks for:

- Git conflict markers such as `<<<<<<<`, `=======`, and `>>>>>>>`
- invalid JSON
- non-standard JSON values such as `NaN` and `Infinity`
- CSV files with no rows or no columns
- missing expected research artifacts

It can also write `signals/research_run_manifest.json`, which records:

- git commit and branch
- operating system and Python version
- important package versions
- selected non-secret environment variables
- random seed environment values
- checksums for generated outputs
- checksums for `data/*.parquet` input files

That manifest explains why a Mac run and Windows run may differ.

## How To Run It

Validate the default tracked research files:

```bash
python research_output_guard.py
```

Validate and write a reproducibility manifest:

```bash
python research_output_guard.py --write-manifest
```

Record the command that produced the files:

```bash
python research_output_guard.py --write-manifest --command "python refresh_local_research_data.py"
```

Validate specific files:

```bash
python research_output_guard.py --path signals/feature_health_profile.json --path signals/feature_health_profile.csv
```

## Inputs

By default the guard checks these files:

| File | Why It Matters |
|---|---|
| `signals/adaptive_factor_weights.json` | live adaptive factor weights |
| `signals/feature_quality_report.json` | full feature-quality diagnostic |
| `signals/feature_quality_summary.csv` | compact feature-quality table |
| `signals/feature_health_profile.json` | live feature quarantine/cluster profile |
| `signals/feature_health_profile.csv` | feature-health table |
| `signals/feature_research_report.json` | deeper feature research report |
| `signals/feature_research_summary.csv` | IC decay data used by feature health |

## Outputs

When `--write-manifest` is used:

| File | Contents |
|---|---|
| `signals/research_run_manifest.json` | machine, git, package, environment, validation, output checksum, and data checksum metadata |

## Key Concepts

- **Conflict marker** - text Git writes into a file when it cannot merge two
  versions automatically. These markers must be resolved before committing.
- **JSON** - structured machine-readable data. The bot expects JSON reports to
  parse cleanly.
- **CSV** - spreadsheet-style table. The bot expects CSV reports to have headers
  and rows.
- **Checksum** - a file fingerprint. If two machines have different checksums
  for the same path, their file contents differ.
- **Manifest** - a small metadata report that says exactly how a run was made.

## Mac / Windows Workflow

Before running research on a second machine:

```bash
git fetch origin
git switch main
git pull --ff-only
```

After running research:

```bash
python research_output_guard.py --write-manifest --command "your research command"
git diff --check
rg "<<<<<<<|=======|>>>>>>>" .
```

Only commit research outputs if the guard passes.
