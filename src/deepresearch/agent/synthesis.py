from __future__ import annotations

from pydantic import BaseModel, Field

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Report, SourceRegistryEntry, WorkerNotes


class SynthesisDraft(BaseModel):
    text: str
    cited_source_ids: list[str] = Field(default_factory=list)


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
    data, usage = await llm.complete_structured(
        model=config.synthesis_model,
        system=system,
        user_content=user_content,
        response_model=SynthesisDraft,
        max_tokens=4096,
    )
    citations = [source_registry[sid] for sid in data.cited_source_ids if sid in source_registry]
    report = Report(text=data.text, citations=citations)
    return report, usage
