from __future__ import annotations

import time

from deepresearch.backends.base import SearchBackend
from deepresearch.chunking import chunk_text
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.rerank.base import RerankBackend
from deepresearch.schemas import SourceRegistryEntry, SubQuestion, WorkerNotes
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, stage_span

WORKER_NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "source_id", "quote", "confidence"],
                "additionalProperties": False,
            },
        },
        "open_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims", "open_gaps"],
    "additionalProperties": False,
}


def _cache_hit(backend, misses_before: int | None) -> bool:
    """True if this call didn't register a new miss on the backend's
    CacheStats — i.e. it was served from cache. False (not "unknown") for
    an uncached backend, since there's nothing to hit."""
    stats = getattr(backend, "stats", None)
    if stats is None or misses_before is None:
        return False
    return stats.search_misses + stats.fetch_misses == misses_before


async def run_worker(
    sub_question: SubQuestion,
    *,
    search_backend: SearchBackend,
    config: RunConfig,
    llm: LLMClient,
    source_registry: dict[str, SourceRegistryEntry],
    rerank_backend: RerankBackend | None = None,
    recorder: RunRecorder | None = None,
) -> tuple[WorkerNotes, LLMUsage]:
    # This worker runs inside a "worker:<id>" OTel span opened by the caller
    # (agent/orchestrator.py's _call_stage) — tool_calls attach to that span.
    parent_span_id = current_span_id_hex()

    stats = getattr(search_backend, "stats", None)
    misses_before = (stats.search_misses + stats.fetch_misses) if stats else None

    search_start = time.perf_counter()
    try:
        results = await search_backend.search(sub_question.question, max_results=config.candidate_pool_size)
        search_success = True
    except Exception:
        results = []
        search_success = False
    search_latency_ms = (time.perf_counter() - search_start) * 1000
    if recorder is not None:
        recorder.record_tool_call(
            span_id=parent_span_id,
            tool_name="search",
            args={"query": sub_question.question, "max_results": config.candidate_pool_size},
            result_summary={"n_results": len(results)},
            success=search_success,
            cache_hit=_cache_hit(search_backend, misses_before),
            latency_ms=search_latency_ms,
        )

    # (source_id, title, chunk_text) — one entry per chunk, several chunks per source
    candidates: list[tuple[str, str, str]] = []
    for result in results:
        source_id = f"src_{len(source_registry) + 1}"
        source_registry[source_id] = SourceRegistryEntry(source_id=source_id, url=result.url, title=result.title)

        fetch_misses_before = (stats.search_misses + stats.fetch_misses) if stats else None
        fetch_start = time.perf_counter()
        try:
            fetch_result = await search_backend.fetch(result.url)
            content = fetch_result.content or result.snippet
            fetch_success = True
        except Exception:
            content = result.snippet
            fetch_success = False
        fetch_latency_ms = (time.perf_counter() - fetch_start) * 1000
        if recorder is not None:
            recorder.record_tool_call(
                span_id=parent_span_id,
                tool_name="fetch",
                args={"url": result.url},
                result_summary={"content_length": len(content)},
                success=fetch_success,
                cache_hit=_cache_hit(search_backend, fetch_misses_before),
                latency_ms=fetch_latency_ms,
            )

        for chunk in chunk_text(content) or [content]:
            candidates.append((source_id, result.title, chunk))

    with stage_span(
        "rerank",
        **{
            "rerank.enabled": bool(config.rerank_enabled and rerank_backend is not None),
            "rerank.candidate_count": len(candidates),
        },
    ) as rerank_span:
        rerank_start = time.perf_counter()
        if config.rerank_enabled and rerank_backend is not None and candidates:
            ranked = await rerank_backend.rerank(sub_question.question, [c[2] for c in candidates])
            selected = [candidates[rc.index] for rc in ranked[: config.rerank_top_k]]
        else:
            selected = candidates[: config.rerank_top_k]
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000
        rerank_span.set_attribute("rerank.selected_count", len(selected))

    if recorder is not None:
        recorder.record_tool_call(
            span_id=parent_span_id,
            tool_name="rerank",
            args={"candidate_count": len(candidates), "top_k": config.rerank_top_k},
            result_summary={"selected_count": len(selected)},
            success=True,
            cache_hit=False,  # reranking is never cached (docs/DESIGN.md row 7)
            latency_ms=rerank_latency_ms,
        )

    sources_block = "\n\n".join(f"[{sid}] {title}\n{chunk[:2000]}" for sid, title, chunk in selected)
    system = load_prompt("worker_v1.txt")
    user_content = f"Sub-question: {sub_question.question}\n\nSources:\n{sources_block}"

    data, usage = await llm.complete_json(
        model=config.worker_model,
        system=system,
        user_content=user_content,
        schema=WORKER_NOTES_SCHEMA,
        max_tokens=2048,
    )
    notes = WorkerNotes(
        sub_question_id=sub_question.id,
        sub_question=sub_question.question,
        claims=data["claims"],
        open_gaps=data["open_gaps"],
    )
    return notes, usage
