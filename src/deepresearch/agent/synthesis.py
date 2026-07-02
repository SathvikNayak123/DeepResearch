from __future__ import annotations

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Report, SourceRegistryEntry, WorkerNotes

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "cited_source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "cited_source_ids"],
    "additionalProperties": False,
}


async def synthesize(
    question: str,
    notes: list[WorkerNotes],
    source_registry: dict[str, SourceRegistryEntry],
    *,
    config: RunConfig,
    llm: LLMClient,
) -> tuple[Report, LLMUsage]:
    notes_block = "\n\n".join(
        f"Sub-question: {n.sub_question}\n" + "\n".join(f"- {c.text} [{c.source_id}]" for c in n.claims)
        for n in notes
    )
    system = load_prompt("synthesis_v1.txt")
    user_content = (
        f"Research question: {question}\n\nNotes:\n{notes_block}\n\n"
        "Cite claims inline using [source_id]."
    )
    data, usage = await llm.complete_json(
        model=config.synthesis_model,
        system=system,
        user_content=user_content,
        schema=SYNTHESIS_SCHEMA,
        max_tokens=4096,
    )
    citations = [source_registry[sid] for sid in data["cited_source_ids"] if sid in source_registry]
    report = Report(text=data["text"], citations=citations)
    return report, usage
