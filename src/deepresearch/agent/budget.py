from __future__ import annotations

import time

from deepresearch.config import BudgetConfig


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetTracker:
    """Hard budget enforcement: max iterations (replans), max total tokens,
    max wall-clock. Checked at every stage boundary — "the agent decides"
    is never the stopping criterion."""

    def __init__(self, budget: BudgetConfig) -> None:
        self._budget = budget
        self._start = time.monotonic()
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.replans = 0

    def record(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd

    def check(self) -> None:
        elapsed = time.monotonic() - self._start
        if elapsed > self._budget.max_wall_clock_seconds:
            raise BudgetExceeded(f"wall_clock_exceeded: {elapsed:.1f}s > {self._budget.max_wall_clock_seconds}s")
        total_tokens = self.tokens_in + self.tokens_out
        if total_tokens > self._budget.max_total_tokens:
            raise BudgetExceeded(f"total_tokens_exceeded: {total_tokens} > {self._budget.max_total_tokens}")
        if self.cost_usd > self._budget.max_usd:
            raise BudgetExceeded(f"cost_exceeded: ${self.cost_usd:.4f} > ${self._budget.max_usd}")

    def replan_allowed(self) -> bool:
        return self.replans < self._budget.max_replans

    def register_replan(self) -> None:
        self.replans += 1


def budget_status(state: dict, budget: BudgetConfig) -> str | None:
    """Pure translation of BudgetTracker.check() — reads accumulated state
    instead of a live tracker object, so it can be evaluated freely from
    routers without mutating anything. Lives here (not in agent/graph.py)
    specifically so agent/graph.py and agent/react_agent.py can both import
    it without a circular import between them (graph.py needs react_agent's
    _record; react_agent's _agent_route needs this). Used against both
    MultihopState and agent/subagent.py's SubagentState (structurally
    compatible: both carry started_monotonic/tokens_in/tokens_out/cost_usd)."""
    elapsed = time.monotonic() - state["started_monotonic"]
    if elapsed > budget.max_wall_clock_seconds:
        return f"wall_clock_exceeded: {elapsed:.1f}s > {budget.max_wall_clock_seconds}s"
    total_tokens = state.get("tokens_in", 0) + state.get("tokens_out", 0)
    if total_tokens > budget.max_total_tokens:
        return f"total_tokens_exceeded: {total_tokens} > {budget.max_total_tokens}"
    cost_usd = state.get("cost_usd", 0.0)
    if cost_usd > budget.max_usd:
        return f"cost_exceeded: ${cost_usd:.4f} > ${budget.max_usd}"
    return None
