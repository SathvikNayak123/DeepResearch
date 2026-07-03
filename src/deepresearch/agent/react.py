from __future__ import annotations

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import WorkerNotes

NEXT_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {"type": "boolean"},
        "next_query": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["done", "next_query", "rationale"],
    "additionalProperties": False,
}


async def next_action(
    question: str, notes: list[WorkerNotes], *, config: RunConfig, llm: LLMClient
) -> tuple[dict, LLMUsage]:
    """One interleaved-ReAct step (docs/DESIGN.md decision row 2, alternative
    to plan-first): decide the next single search query from the claims
    gathered so far, or signal done. No upfront decomposition — this is the
    orchestrator.run_research "react" branch's only planning-equivalent call,
    invoked once per step rather than once per run."""
    notes_block = (
        "\n\n".join(
            f"Query: {n.sub_question}\n" + "\n".join(f"- {c.text} [{c.source_id}]" for c in n.claims)
            for n in notes
        )
        or "(none yet)"
    )
    system = load_prompt("react_v1.txt")
    user_content = f"Research question: {question}\n\nClaims gathered so far:\n{notes_block}"
    data, usage = await llm.complete_json(
        model=config.planner_model,
        system=system,
        user_content=user_content,
        schema=NEXT_ACTION_SCHEMA,
        max_tokens=512,
    )
    return data, usage
