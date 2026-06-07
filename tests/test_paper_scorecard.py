from __future__ import annotations

import paper_scorecard


def test_write_paper_scorecard_uses_atomic_writer(monkeypatch, tmp_path):
    calls = []

    def fake_write_json(data, path, **_kwargs):
        calls.append((path.name, data["status"]))

    monkeypatch.setattr(paper_scorecard, "atomic_write_json", fake_write_json)

    out_path = paper_scorecard.write_paper_scorecard(
        {"status": "ok"},
        output_path=tmp_path / "paper_scorecard.json",
    )

    assert out_path == tmp_path / "paper_scorecard.json"
    assert calls == [("paper_scorecard.json", "ok")]
