"""Safety checks for isolated challenger training."""

from __future__ import annotations

import train


def test_shadow_registry_name_cannot_replace_production_lookup(monkeypatch):
    monkeypatch.setattr(train, "TRAINING_DEPLOYMENT_ROLE", "shadow")

    assert train._registry_run_name("pooled") == "shadow/pooled"


def test_production_registry_name_remains_backward_compatible(monkeypatch):
    monkeypatch.setattr(train, "TRAINING_DEPLOYMENT_ROLE", "production")

    assert train._registry_run_name("pooled") == "pooled"
