# setup.py — What It Does and How to Run It

## What This Script Does (Plain English)

`setup.py` is the **first thing you run** on a fresh machine. It:
1. Checks your Python version (needs 3.10+)
2. Installs all required packages from `requirements.txt`
3. Downloads the FinBERT model weights (~440 MB, only once)
4. Creates the folder structure (`data/`, `models/`, `signals/`, `logs/`)
5. Verifies every import works
6. Checks your API keys and tells you which features are active
7. Detects your hardware (GPU/Apple Silicon/CPU)
8. Prints a colour-coded summary: ✓ green = ready, ⚠ yellow = optional, ✗ red = broken

**First run:** 5–10 minutes (downloading FinBERT).  
**Subsequent runs:** ~30 seconds.

---

## How to Run It

```bash
# Run once before anything else
python setup.py
```

**What you should see at the end:**
```
✓  Python 3.12 — OK
✓  numpy, pandas, xgboost, ... all imports OK
✓  FinBERT model ready
✓  FINNHUB_API_KEY present
⚠  OPENAI_API_KEY missing — GPT-4 sentiment disabled (FinBERT used instead)
✓  Folders: data/ models/ signals/ logs/ — all exist
```

---

## Key Concepts

| Term | Plain-English Meaning |
|---|---|
| **FinBERT** | A deep learning model pre-trained on financial news. Downloads to your local machine once; used by `sentiment_engine.py` to score headlines. |
| **CUDA / MPS** | GPU acceleration. CUDA = NVIDIA cards. MPS = Apple Silicon (M1/M2/M3). If available, neural models train much faster. |
| **API key** | A private password for a data service. Finnhub = market news data. OpenAI = GPT-4 sentiment. Both are optional but improve signal quality. |
| **venv** | A virtual environment — an isolated Python installation for this project. Keeps dependencies from conflicting with other projects. |

---

## Recommended Setup Flow (First Time)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# open .env and fill in your keys

# 4. Run setup check
python setup.py
```
