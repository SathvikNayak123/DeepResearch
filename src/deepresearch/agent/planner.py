from __future__ import annotations

import uuid

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Plan, SubQuestion

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["sub_questions"],
    "additionalProperties": False,
}


async def plan(question: str, config: RunConfig, llm: LLMClient) -> tuple[Plan, LLMUsage]:
    system = load_prompt("planner_v1.txt")
    data, usage = await llm.complete_json(
        model=config.planner_model,
        system=system,
        user_content=f"Research question: {question}",
        schema=PLAN_SCHEMA,
        max_tokens=1024,
    )
    sub_questions = [SubQuestion(id=uuid.uuid4().hex[:8], question=q) for q in data["sub_questions"]]
    return Plan(sub_questions=sub_questions), usage
