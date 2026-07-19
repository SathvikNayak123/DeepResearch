from __future__ import annotations

import sys

from deepresearch.agent.dag import truncate_plan, validate_plan
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Plan


async def plan(question: str, config: RunConfig, llm: LLMClient) -> tuple[Plan, LLMUsage]:
    """Decomposes a question into a dependency graph of retrieval nodes (a
    single node for a simple lookup). An over-generated plan is truncated to
    config.max_nodes (not rejected — see agent/dag.py's truncate_plan), then
    validated for genuine structural problems (unique ids, known
    dependencies, acyclic) before it ever reaches the supervisor's readiness
    computation."""
    system = load_prompt("planner_v1.txt")
    result, usage = await llm.complete_structured(
        model=config.planner_model,
        system=system,
        user_content=f"Research question: {question}",
        response_model=Plan,
        max_tokens=1024,
    )
    sub_questions = truncate_plan(result.sub_questions, max_nodes=config.max_nodes)
    if len(sub_questions) < len(result.sub_questions):
        # stderr, not stdout: the CLI emits the run result as JSON on stdout
        # (deepresearch/cli.py), so a diagnostic line there would corrupt it.
        print(
            f"[planner] truncated over-generated plan: {len(result.sub_questions)} nodes "
            f"-> {len(sub_questions)} (max_nodes={config.max_nodes})",
            file=sys.stderr,
        )
    validate_plan(sub_questions, max_nodes=config.max_nodes)
    return Plan(sub_questions=sub_questions), usage
