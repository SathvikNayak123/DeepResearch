"""Dependency-graph validation and ordering shared by the planner (validates
the DAG it just produced) and synthesis (Phase 4: dependency-ordered write-up).
"""

from __future__ import annotations

from deepresearch.schemas import SubQuestion


class PlanValidationError(ValueError):
    pass


def topological_order(nodes: list[SubQuestion]) -> list[str]:
    """Node ids ordered so every id appears after everything in its
    depends_on. Raises PlanValidationError on an unknown dependency id or a
    cycle — a supervisor computing readiness against a graph like that can
    deadlock (no node ever becomes ready), so this must be checked up front."""
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        for dep in n.depends_on:
            if dep not in by_id:
                raise PlanValidationError(f"node {n.id!r} depends_on unknown node {dep!r}")

    order: list[str] = []
    state: dict[str, int] = {}  # 0=unvisited (absent), 1=visiting, 2=done

    def visit(node_id: str) -> None:
        mark = state.get(node_id, 0)
        if mark == 2:
            return
        if mark == 1:
            raise PlanValidationError(f"dependency cycle detected involving node {node_id!r}")
        state[node_id] = 1
        for dep in by_id[node_id].depends_on:
            visit(dep)
        state[node_id] = 2
        order.append(node_id)

    for n in nodes:
        visit(n.id)
    return order


def truncate_plan(nodes: list[SubQuestion], *, max_nodes: int) -> list[SubQuestion]:
    """If the planner over-generated, keep only the first max_nodes nodes and
    strip any depends_on reference to a node that got dropped.

    Planners tend to emit foundational/independent nodes first and derived/
    dependent nodes later, so keeping the prefix (not a random subset or the
    suffix) is more likely to preserve a coherent, still-answerable sub-plan.
    Without this, an over-eager real model (confirmed live: a real FRAMES
    question produced 7 nodes against max_nodes=6) makes validate_plan raise
    PlanValidationError, which propagates uncaught all the way to the top of
    run_research() -- discarding the whole question's run over a
    decomposition-count technicality, not a genuine structural problem."""
    if len(nodes) <= max_nodes:
        return nodes
    kept = nodes[:max_nodes]
    kept_ids = {n.id for n in kept}
    return [n.model_copy(update={"depends_on": [d for d in n.depends_on if d in kept_ids]}) for n in kept]


def validate_plan(nodes: list[SubQuestion], *, max_nodes: int) -> None:
    if not nodes:
        raise PlanValidationError("plan must contain at least one node")
    ids = [n.id for n in nodes]
    if len(ids) != len(set(ids)):
        raise PlanValidationError(f"duplicate node ids in plan: {ids}")
    if len(nodes) > max_nodes:
        raise PlanValidationError(f"plan has {len(nodes)} nodes, exceeds max_nodes={max_nodes}")
    topological_order(nodes)  # raises on unknown deps / cycles
