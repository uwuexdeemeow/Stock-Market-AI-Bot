"""
Test setup for the legacy ML tests.

PLAIN ENGLISH: the old machine-learning scripts (train.py, labels.py, ...)
now live in legacy/.  This puts that folder on Python's search path so the
tests below can still `import labels` and friends.  These tests run in their
own non-blocking CI job (see .github/workflows/ci.yml) because the daily
trading install no longer includes xgboost or scikit-learn.
"""

import sys
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parents[2] / "legacy"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))
