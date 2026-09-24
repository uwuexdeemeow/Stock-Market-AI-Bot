"""Guard against the diagnostic analyzer contradicting walk-forward gates."""

import json

from walkforward_analyzer import deployment_readiness_message, load_walkforward_approval


def test_rejected_walkforward_never_reads_as_deployable(tmp_path):
    csv_path = tmp_path / "research.csv"
    csv_path.write_text("fold_year\n2022\n", encoding="utf-8")
    csv_path.with_suffix(".json").write_text(
        json.dumps({"live_config_approval": {"approved": False, "reasons": ["turnover_too_high"]}}),
        encoding="utf-8",
    )
    approval = load_walkforward_approval(str(csv_path))
    assert approval == {"approved": False, "reasons": ["turnover_too_high"]}
    assert "Do not deploy" in deployment_readiness_message(0, 0, approval)


def test_missing_walkforward_approval_fails_closed(tmp_path):
    csv_path = tmp_path / "research.csv"
    csv_path.write_text("fold_year\n2022\n", encoding="utf-8")
    approval = load_walkforward_approval(str(csv_path))
    assert approval["approved"] is None
    assert "cannot authorize" in deployment_readiness_message(0, 0, approval)


def test_approval_still_mentions_other_live_gates():
    message = deployment_readiness_message(0, 0, {"approved": True, "reasons": []})
    assert "Current data, stress, and live gates still apply" in message
