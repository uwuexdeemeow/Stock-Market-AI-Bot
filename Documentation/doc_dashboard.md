# dashboard.py - Project Dashboard

## What this script does

`dashboard.py` opens the beginner-friendly Streamlit dashboard. It summarizes
the active Alpaca paper account, strategy state, workflow health, execution
quality, and shadow evidence.

The home page includes a clearly labeled **$400 fractional shadow** panel. It
shows pretend equity, cash, largest allocation gap, and modeled daily costs. A
safety badge appears only when evidence confirms shadow-only mode and zero
broker orders.

## How to run it

```bash
streamlit run dashboard.py
```

Open the local address printed in the terminal, normally
`http://localhost:8501`. The fractional panel appears after the first successful
workflow or local `fractional_shadow_paper.py` run.

## Key concepts

**Dashboard:** A visual summary of project evidence files.

**Paper account:** Alpaca's simulated brokerage account.

**Fractional shadow:** A separate $400 pretend ledger for small-capital testing.

**Freshness:** How recently evidence was updated versus its normal schedule.

