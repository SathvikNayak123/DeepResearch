"""Multi-hop planner: DAG parsing + validation (agent/dag.py) and the
node-granularity rule (a node is only a retrieval question; compare/aggregate
steps are never nodes — that's synthesis's job, see docs/DESIGN.md decision
row 1/2 and the approved plan)."""

from __future__ import annotations

import pytest

from deepresearch.agent.dag import PlanValidationError, topological_order, truncate_plan, validate_plan
from deepresearch.agent.planner import plan
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import Plan, SubQuestion


class StubPlannerLLM:
    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    async def complete_structured(self, *, model, system, user_content, response_model, max_tokens=4096):
        usage = LLMUsage(input_tokens=10, output_tokens=10, cost_usd=0.0)
        assert response_model is Plan
        return Plan(sub_questions=[SubQuestion(**n) for n in self._nodes]), usage


@pytest.mark.asyncio
async def test_single_lookup_yields_one_node():
    llm = StubPlannerLLM([{"id": "n1", "question": "What is the capital of France?", "depends_on": []}])
    result, _ = await plan("What is the capital of France?", RunConfig(), llm)
    assert len(result.sub_questions) == 1
    assert result.sub_questions[0].depends_on == []


@pytest.mark.asyncio
async def test_two_hop_question_yields_dependent_node():
    llm = StubPlannerLLM(
        [
            {"id": "n1", "question": "Who is the lead singer of Ratata?", "depends_on": []},
            {"id": "n2", "question": "What year was <n1 singer> born?", "depends_on": ["n1"]},
        ]
    )
    result, _ = await plan("When was the lead singer of Ratata born?", RunConfig(), llm)
    by_id = {n.id: n for n in result.sub_questions}
    assert by_id["n2"].depends_on == ["n1"]


@pytest.mark.asyncio
async def test_compare_query_yields_two_lookup_nodes_no_compare_node():
    """The planner must never emit a node whose job is to compare/aggregate —
    that's synthesis's job. This asserts the contract the prompt encodes: two
    independent lookups, nothing else."""
    llm = StubPlannerLLM(
        [
            {"id": "n1", "question": "Birth year of Ratata's lead singer", "depends_on": []},
            {"id": "n2", "question": "Birth year of Kent's lead singer", "depends_on": []},
        ]
    )
    result, _ = await plan("Compare the birth years of the lead singers of Ratata and Kent", RunConfig(), llm)
    assert len(result.sub_questions) == 2
    assert all(sq.depends_on == [] for sq in result.sub_questions)


@pytest.mark.asyncio
async def test_planner_truncates_over_generated_plan_instead_of_raising():
    """A real model overshooting max_nodes must not crash the whole question
    (confirmed live: a real FRAMES run produced 7 nodes against max_nodes=6,
    and the old raise-on-overflow behavior propagated uncaught all the way
    to the top of run_research(), killing an entire eval batch over this).
    plan() truncates gracefully; only a direct validate_plan() call on an
    over-sized list (below) still raises -- that's the defensive invariant
    check for callers who didn't already truncate."""
    nodes = [{"id": f"n{i}", "question": f"q{i}", "depends_on": []} for i in range(10)]
    llm = StubPlannerLLM(nodes)
    result, _ = await plan("q", RunConfig(max_nodes=6), llm)
    assert len(result.sub_questions) == 6


def test_truncate_plan_keeps_prefix_and_strips_dangling_dependencies():
    nodes = [
        SubQuestion(id="n1", question="q1", depends_on=[]),
        SubQuestion(id="n2", question="q2", depends_on=["n1"]),
        SubQuestion(id="n3", question="q3", depends_on=["n1", "n2"]),
    ]
    truncated = truncate_plan(nodes, max_nodes=2)
    assert [n.id for n in truncated] == ["n1", "n2"]
    # n2 depended on n1 (kept) -- untouched. A hypothetical dependency on the
    # dropped n3 would be the thing this strips; simulate that case directly:
    nodes_with_forward_ref = [
        SubQuestion(id="n1", question="q1", depends_on=["n2"]),  # depends on a node about to be dropped
        SubQuestion(id="n2", question="q2", depends_on=[]),
        SubQuestion(id="n3", question="q3", depends_on=[]),
    ]
    truncated2 = truncate_plan(nodes_with_forward_ref, max_nodes=1)
    assert truncated2[0].depends_on == []  # "n2" reference stripped, n2 was dropped


def test_truncate_plan_is_a_noop_within_the_limit():
    nodes = [SubQuestion(id="n1", question="q1", depends_on=[])]
    assert truncate_plan(nodes, max_nodes=6) == nodes


def test_topological_order_respects_dependencies():
    nodes = [
        SubQuestion(id="n2", question="q2", depends_on=["n1"]),
        SubQuestion(id="n1", question="q1", depends_on=[]),
        SubQuestion(id="n3", question="q3", depends_on=["n1", "n2"]),
    ]
    order = topological_order(nodes)
    assert order.index("n1") < order.index("n2") < order.index("n3")


def test_validate_plan_rejects_duplicate_ids():
    nodes = [
        SubQuestion(id="n1", question="q1", depends_on=[]),
        SubQuestion(id="n1", question="q2", depends_on=[]),
    ]
    with pytest.raises(PlanValidationError):
        validate_plan(nodes, max_nodes=6)


def test_validate_plan_rejects_unknown_dependency():
    nodes = [SubQuestion(id="n1", question="q1", depends_on=["ghost"])]
    with pytest.raises(PlanValidationError):
        validate_plan(nodes, max_nodes=6)


def test_validate_plan_rejects_cycle():
    nodes = [
        SubQuestion(id="n1", question="q1", depends_on=["n2"]),
        SubQuestion(id="n2", question="q2", depends_on=["n1"]),
    ]
    with pytest.raises(PlanValidationError):
        validate_plan(nodes, max_nodes=6)


def test_validate_plan_rejects_empty_plan():
    with pytest.raises(PlanValidationError):
        validate_plan([], max_nodes=6)
