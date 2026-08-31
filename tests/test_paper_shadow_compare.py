from __future__ import annotations

import json

import pandas as pd

import paper_shadow_compare as psc


def test_build_comparison_uses_common_start_and_latest_values(tmp_path):
    alpaca_path = tmp_path / "alpaca_paper_equity.csv"
    shadow_path = tmp_path / "shadow_paper_equity.csv"
    alpaca_path.write_text(
        "\n".join([
            "date,timestamp,equity,cash,invested",
            "2026-05-28,2026-05-28 21:35:00,107308.02,0,107308.02",
            "2026-06-03,2026-06-03 21:35:00,110216.08,0,110216.08",
        ]),
        encoding="utf-8",
    )
    shadow_path.write_text(
        "\n".join([
            "date,timestamp,equity,total_return_pct",
            "2026-05-27,2026-05-27T17:00:00+00:00,100000.00,0",
            "2026-05-28,2026-05-28T17:00:00+00:00,100418.51,0.4185",
            "2026-06-02,2026-06-02T17:00:00+00:00,101837.04,1.837",
        ]),
        encoding="utf-8",
    )

    summary, table = psc.build_comparison_payload(
        alpaca_path=alpaca_path,
        shadow_path=shadow_path,
    )

    assert summary["status"] == "ok"
    assert summary["common_start_date"] == "2026-05-28"
    assert summary["latest_dates_match"] is False
    assert summary["alpaca"]["return_pct_since_common_start"] == 2.71
    assert summary["shadow"]["return_pct_since_common_start"] == 1.4126
    assert summary["spread"]["leader"] == "alpaca"
    assert not table.empty


def test_write_comparison_outputs_json_and_csv(tmp_path):
    alpaca_path = tmp_path / "alpaca.csv"
    shadow_path = tmp_path / "shadow.csv"
    csv_out = tmp_path / "compare.csv"
    json_out = tmp_path / "compare.json"
    pd.DataFrame({"date": ["2026-01-02"], "equity": [101.0]}).to_csv(alpaca_path, index=False)
    pd.DataFrame({"date": ["2026-01-02"], "equity": [100.0]}).to_csv(shadow_path, index=False)

    summary = psc.write_comparison(
        alpaca_path=alpaca_path,
        shadow_path=shadow_path,
        csv_out=csv_out,
        json_out=json_out,
    )

    assert summary["status"] == "collecting"
    assert json.loads(json_out.read_text())["aligned_days"] == 1
    assert pd.read_csv(csv_out).iloc[0]["alpaca_equity"] == 101.0


def test_non_overlapping_first_observations_are_collecting(tmp_path):
    alpaca_path = tmp_path / "alpaca.csv"
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame({"date": ["2026-01-02"], "equity": [101.0]}).to_csv(alpaca_path, index=False)
    pd.DataFrame({"date": ["2026-01-05"], "equity": [100.0]}).to_csv(shadow_path, index=False)

    summary, table = psc.build_comparison_payload(alpaca_path=alpaca_path, shadow_path=shadow_path)

    assert summary["status"] == "collecting"
    assert summary["reason"] == "awaiting_overlapping_observation"
    assert summary["aligned_days"] == 0
    assert table.empty
