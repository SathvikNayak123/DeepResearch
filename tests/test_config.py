from __future__ import annotations

from deepresearch.config import RunConfig


def test_from_overrides_applies_top_level_and_budget_fields():
    cfg = RunConfig.from_overrides({"max_workers": 2, "budget": {"max_usd": 0.5}})
    assert cfg.max_workers == 2
    assert cfg.budget.max_usd == 0.5
    # unrelated fields keep their defaults
    assert cfg.budget.max_replans == 2


def test_from_overrides_none_returns_defaults():
    cfg = RunConfig.from_overrides(None)
    assert cfg.max_workers == 4
    assert cfg.budget.max_usd == 5.0


def test_rerank_can_be_toggled_via_overrides():
    cfg = RunConfig.from_overrides({"rerank_enabled": False})
    assert cfg.rerank_enabled is False

    cfg = RunConfig.from_overrides({"rerank_backend": "cohere"})
    assert cfg.rerank_backend == "cohere"
