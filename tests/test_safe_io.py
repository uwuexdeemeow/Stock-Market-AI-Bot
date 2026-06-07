from __future__ import annotations

import pandas as pd

from safe_io import atomic_write_csv, atomic_write_parquet, atomic_write_text


def test_atomic_write_text_preserves_stale_neighbor_tmp(tmp_path):
    target = tmp_path / "state.json"
    stale_tmp = target.with_suffix(target.suffix + ".tmp")
    stale_tmp.write_text("stale", encoding="utf-8")

    atomic_write_text(target, "fresh")

    assert target.read_text(encoding="utf-8") == "fresh"
    assert stale_tmp.read_text(encoding="utf-8") == "stale"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_csv_preserves_stale_neighbor_tmp(tmp_path):
    target = tmp_path / "orders.csv"
    stale_tmp = target.with_suffix(target.suffix + ".tmp")
    stale_tmp.write_text("stale", encoding="utf-8")

    atomic_write_csv(pd.DataFrame([{"ticker": "QQQ", "weight": 0.5}]), target)

    assert "QQQ" in target.read_text(encoding="utf-8")
    assert stale_tmp.read_text(encoding="utf-8") == "stale"
    assert list(tmp_path.glob(".orders.csv.*.tmp")) == []


def test_atomic_write_parquet_preserves_stale_neighbor_tmp(tmp_path):
    target = tmp_path / "prices.parquet"
    stale_tmp = target.with_suffix(target.suffix + ".tmp")
    stale_tmp.write_text("stale", encoding="utf-8")

    atomic_write_parquet(pd.DataFrame([{"Close": 100.0}]), target, index=False)

    assert pd.read_parquet(target)["Close"].iloc[0] == 100.0
    assert stale_tmp.read_text(encoding="utf-8") == "stale"
    assert list(tmp_path.glob(".prices.parquet.*.tmp")) == []
