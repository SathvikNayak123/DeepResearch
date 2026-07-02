from __future__ import annotations

import time

import pytest

from deepresearch.agent.budget import BudgetExceeded, BudgetTracker
from deepresearch.config import BudgetConfig


def test_max_total_tokens_triggers():
    budget = BudgetTracker(BudgetConfig(max_total_tokens=100, max_usd=1000, max_wall_clock_seconds=1000))
    budget.record(tokens_in=60, tokens_out=60, cost_usd=0.0)
    with pytest.raises(BudgetExceeded) as exc:
        budget.check()
    assert "total_tokens_exceeded" in str(exc.value)


def test_max_usd_triggers():
    budget = BudgetTracker(BudgetConfig(max_total_tokens=10_000_000, max_usd=0.01, max_wall_clock_seconds=1000))
    budget.record(tokens_in=1000, tokens_out=1000, cost_usd=0.02)
    with pytest.raises(BudgetExceeded) as exc:
        budget.check()
    assert "cost_exceeded" in str(exc.value)


def test_max_wall_clock_triggers():
    budget = BudgetTracker(BudgetConfig(max_total_tokens=10_000_000, max_usd=1000, max_wall_clock_seconds=0.01))
    time.sleep(0.05)
    with pytest.raises(BudgetExceeded) as exc:
        budget.check()
    assert "wall_clock_exceeded" in str(exc.value)


def test_within_budget_does_not_raise():
    budget = BudgetTracker(BudgetConfig(max_total_tokens=10_000, max_usd=10, max_wall_clock_seconds=60))
    budget.record(tokens_in=10, tokens_out=10, cost_usd=0.001)
    budget.check()


def test_replan_allowed_respects_max_replans():
    budget = BudgetTracker(BudgetConfig(max_replans=1))
    assert budget.replan_allowed() is True
    budget.register_replan()
    assert budget.replan_allowed() is False
