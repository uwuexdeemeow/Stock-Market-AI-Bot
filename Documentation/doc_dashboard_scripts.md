# dashboard/scripts.py — Discover runnable tools

## What it does

Reads root Python files and finds their descriptions and command-line options.
The dashboard uses this information to build its Run Scripts page. It reads the
source without executing the scripts. Removed files disappear from discovery;
helper modules and the dashboard launcher itself are hidden.

## How to use it

From the project root, run:

```bash
streamlit run dashboard.py
```

Open **Run Scripts**. Each discovered tool appears with its description and
available inputs. Discovery itself does not run a trading or research command.
This module has no standalone command-line interface.

## Key terms

- **Discovery:** Finding available scripts by reading the project directory.
- **Parser:** Code that reads the structure of a Python file or command options.
- **Flag:** An option such as `--dry-run` passed when starting a script.
- **Category:** A dashboard group such as research or paper trading.
