"""Keep the beginner documentation promise in AGENTS.md enforceable."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_scoped_root_script_has_a_documentation_file():
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    scoped_scripts = sorted({name for name in re.findall(r"\b[a-z][a-z0-9_]*\.py\b", instructions)})
    documentation_names = [path.stem.lower() for path in (ROOT / "Documentation").iterdir() if path.is_file()]

    missing = []
    for script_name in scoped_scripts:
        script_stem = Path(script_name).stem.lower()
        if (ROOT / script_name).exists() and not any(script_stem in name for name in documentation_names):
            missing.append(script_name)

    assert missing == [], f"Missing beginner documentation for: {', '.join(missing)}"


def test_project_wide_workflow_guide_exists():
    guide = ROOT / "Documentation" / "RELIABILITY_FIRST_WORKFLOW.md"
    text = guide.read_text(encoding="utf-8")
    assert "End-To-End Flow" in text
    assert "Beginner Use" in text
    assert "Why It Is Designed This Way" in text
