from __future__ import annotations

import uuid

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Plan, ReflectionResult, SubQuestion, WorkerNotes

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "coverage_score": {"type": "number"},
        "rationale": {"type": "string"},
        "should_replan": {"type": "boolean"},
        "new_sub_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["coverage_score", "rationale", "should_replan", "new_sub_questions"],
    "additionalProperties": False,
}


async def reflect(
    question: str,
    plan: Plan,
    notes: list[WorkerNotes],
    *,
    config: RunConfig,
    llm: LLMClient,
) -> tuple[ReflectionResult, LLMUsage]:
    notes_block = "\n\n".join(
        f"Sub-question: {n.sub_question}\n"
        + "\n".join(f"- {c.text} [{c.source_id}]" for c in n.claims)
        + ("\nGaps: " + "; ".join(n.open_gaps) if n.open_gaps else "")
        for n in notes
    )
    system = load_prompt("reflection_v1.txt")
    user_content = (
        f"Research question: {question}\n\n"
        f"Plan sub-questions: {[sq.question for sq in plan.sub_questions]}\n\n"
        f"Gathered notes:\n{notes_block}"
    )
    data, usage = await llm.complete_json(
        model=config.reflection_model,
        system=system,
        user_content=user_content,
        schema=REFLECTION_SCHEMA,
        max_tokens=1024,
    )
    result = ReflectionResult(
        coverage_score=data["coverage_score"],
        rationale=data["rationale"],
        should_replan=data["should_replan"],
        new_sub_questions=[
            SubQuestion(id=uuid.uuid4().hex[:8], question=q) for q in data["new_sub_questions"]
        ],
    )
    return result, usage
