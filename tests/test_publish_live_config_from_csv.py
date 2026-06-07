from __future__ import annotations

import json

import publish_live_config_from_csv as publisher


def test_publish_payloads_use_atomic_writer(monkeypatch, tmp_path):
    calls = []
    backups = []

    live_path = tmp_path / "core_satellite_live_configs.json"
    wf_path = tmp_path / "core_satellite_nested_walkforward.json"
    monkeypatch.setattr(publisher, "LIVE_CFG_JSON", live_path)
    monkeypatch.setattr(publisher, "WF_JSON", wf_path)
    monkeypatch.setattr(publisher, "backup_if_exists", lambda path: backups.append(path.name))

    def fake_write_text(path, content, **_kwargs):
        calls.append((path.name, json.loads(content)["kind"]))

    monkeypatch.setattr(publisher, "atomic_write_text", fake_write_text)

    out_live, out_wf = publisher.write_publish_payloads(
        {"kind": "live"},
        {"kind": "walkforward"},
    )

    assert out_live == live_path
    assert out_wf == wf_path
    assert backups == ["core_satellite_live_configs.json", "core_satellite_nested_walkforward.json"]
    assert calls == [
        ("core_satellite_live_configs.json", "live"),
        ("core_satellite_nested_walkforward.json", "walkforward"),
    ]
