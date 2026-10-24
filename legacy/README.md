# legacy/ — the retired machine-learning layer

These are the scripts from the project's first stage, an XGBoost stock
picker. Its model scored an AUC of about 0.51, no better than a coin flip,
so it was switched off for picks. They were moved here from the project root
on 2026-10-25 (`Documentation/LEGACY_ML_CLEANUP_PLAN.md`).

- **Live trading doesn't use them.** `tests/test_no_legacy_imports.py`
  fails if a live script imports one again.
- **To run one**, install the extra packages first, then call it from the
  project root:

  ```bash
  pip install -r requirements-legacy.txt
  python legacy/train.py --help
  ```

  Each file adds the project root and this folder to Python's search path,
  so `import settings` and `import labels` still work.
- **Docs:** `Documentation/legacy/doc_<name>.md`.
- **Tests:** `tests/legacy/`, run by the non-blocking `legacy-tests` CI job
  (`pytest -q tests/legacy`).
- **Outputs** (`models/`, `research_snapshots/`, ...) stay where they were.
  They are local-only and never committed.

Don't move these files back to the root or import them from live code.
Deleting them needs the owner's agreement.
