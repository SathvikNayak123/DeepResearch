from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Awaitable, Callable

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepresearch.agent.budget import budget_status
from deepresearch.agent.graph import (
    MultihopContext,
    MultihopState,
    _findings_to_notes,
    build_multihop_graph,
    recursion_limit_for,
)
from deepresearch.agent.react_agent import TOOLS
from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig, current_git_sha
from deepresearch.llm.chat_model import build_chat_model
from deepresearch.llm.client import LLMClient
from deepresearch.rerank import build_rerank_backend
from deepresearch.schemas import CacheStats, Plan, RunResult, RunStatus
from deepresearch.store import db
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, run_root_context, stage_span

# Fired after each stage's trajectory is recorded — the one hook the SSE
# streaming endpoint (api/streaming.py) needs to surface live progress
# without restructuring the (already tested) blocking stage-by-stage flow.
# None everywhere else (CLI, eval harness, benchmarks) — a no-op.
OnEvent = Callable[[dict], Awaitable[None]]


def _initial_state(question: str, run_started: float) -> MultihopState:
    return {
        "question": question,
        "plan": Plan(sub_questions=[]),
        "findings": [],
        "source_registry": {},
        "verified_ids": {},
        "failed_ids": {},
        "node_corrections": {},
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "started_monotonic": run_started,
        "report": None,
        "reflection_iters": 0,
        "should_continue": False,
    }


async def run_research(
    question: str,
    *,
    config: RunConfig,
    search_backend: SearchBackend,
    llm: LLMClient | None = None,
    chat_model: object | None = None,
    benchmark_name: str | None = None,
    on_event: OnEvent | None = None,
    resume_run_id: str | None = None,
) -> RunResult:
    """Runs one research question end-to-end on the unified
    planner -> supervisor -> subagent(ReAct, +inline per-hop verify) ->
    synthesis -> reflection StateGraph (agent/graph.py). Every call writes a
    `runs` row (plus trajectories/tool_calls) to config.database_url —
    unconditional, not opt-in.

    Persists via an `AsyncSqliteSaver` checkpointer (config.checkpoint_db_path)
    keyed by `thread_id=run_id`, so an interrupted run can resume from its
    last completed wave instead of restarting from scratch: pass
    `resume_run_id` (the original run's run_id) to continue it — this reuses
    the existing `runs` row (no new one is created) and resumes the graph
    from its checkpointed state instead of a fresh initial state.

    Every subagent node is a tool-calling ReAct loop (agent/subagent.py),
    which needs a LangChain chat model bound to [search, calculate] — built
    from config via llm/chat_model.py (OpenRouter) unless `chat_model` is
    passed explicitly. Tests that stub `llm` (the structured-JSON calls: plan/
    finalize/verify/synthesis/reflection) must also pass a stub `chat_model`
    here, or this still reaches for a real OpenRouter-backed one.
    """
    llm = llm or LLMClient()
    chat_model = chat_model.bind_tools(TOOLS) if chat_model is not None else build_chat_model(config).bind_tools(TOOLS)
    rerank_backend = build_rerank_backend(config)
    run_id = resume_run_id or uuid.uuid4().hex
    root_ctx = run_root_context(run_id)
    recorder = RunRecorder(run_id=run_id)
    run_started = time.monotonic()

    if on_event is not None:
        # Fired before any DB/LLM work so a live stream can show run_id
        # (Langfuse trace correlation, docs/DESIGN.md: run_id = trace_id)
        # from the very first byte, not only once the run finishes.
        await on_event({"type": "run_started", "run_id": run_id, "question": question})

    await db.ensure_schema(config.database_url)
    if resume_run_id is None:
        await db.create_run(
            config.database_url,
            run_id=run_id,
            benchmark_name=benchmark_name,
            config=dataclasses.asdict(config),
            git_sha=current_git_sha(),
            status="running",
        )

    # None resumes from the checkpointed state for this thread_id (LangGraph
    # applies no new input, continuing from wherever the run left off); a
    # fresh run passes the real initial state.
    graph_input = None if resume_run_id is not None else _initial_state(question, run_started)

    with stage_span("run", context=root_ctx, **{"run.id": run_id, "run.question": question}) as run_span:
        run_span_id = current_span_id_hex()
        run_context = MultihopContext(
            config=config,
            llm=llm,
            chat_model=chat_model,
            search_backend=search_backend,
            rerank_backend=rerank_backend,
            recorder=recorder,
            run_span_id=run_span_id,
        )

        final_state: MultihopState = {}
        async with AsyncSqliteSaver.from_conn_string(config.checkpoint_db_path) as checkpointer:
            await checkpointer.setup()
            graph = build_multihop_graph(checkpointer=checkpointer)

            # Genuine LLM/tool exceptions propagate uncaught here — only
            # budget ceilings are handled specially, and the graph itself
            # routes around those (never raises), so there is nothing
            # budget-shaped left to catch at this level.
            async for mode, chunk in graph.astream(
                graph_input,
                config={
                    "configurable": {"thread_id": run_id},
                    "max_concurrency": config.max_workers,
                    "recursion_limit": recursion_limit_for(config),
                },
                context=run_context,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if on_event is not None:
                        await on_event(chunk)
                else:  # mode == "values"
                    final_state = chunk

        iterations = final_state.get("reflection_iters", 0) + 1
        budget_hit = budget_status(final_state, config.budget)
        status = RunStatus.BUDGET_EXCEEDED if budget_hit is not None else RunStatus.COMPLETED

        run_span.set_attribute("run.status", status.value)
        run_span.set_attribute("run.tokens_in", final_state.get("tokens_in", 0))
        run_span.set_attribute("run.tokens_out", final_state.get("tokens_out", 0))
        run_span.set_attribute("run.cost_usd", final_state.get("cost_usd", 0.0))
        run_span.set_attribute("run.iterations", iterations)
        if budget_hit:
            run_span.set_attribute("run.budget_hit", budget_hit)

        cache_stats: CacheStats = getattr(search_backend, "stats", CacheStats())
        run_span.set_attribute("cache.hit_rate", cache_stats.hit_rate)
        run_span.set_attribute("cache.dollars_saved", cache_stats.estimated_dollars_saved)

    total_latency_ms = int((time.monotonic() - run_started) * 1000)
    await db.bulk_insert_trajectories(config.database_url, recorder.trajectories)
    await db.bulk_insert_tool_calls(config.database_url, recorder.tool_calls)
    await db.finish_run(
        config.database_url,
        run_id=run_id,
        status=status.value,
        total_cost_usd=final_state.get("cost_usd", 0.0),
        total_latency_ms=total_latency_ms,
    )

    findings = final_state.get("findings", [])
    return RunResult(
        run_id=run_id,
        status=status,
        question=question,
        plan=final_state.get("plan") or Plan(sub_questions=[]),
        worker_notes=_findings_to_notes(findings),
        findings=findings,
        reflections=[],
        report=final_state.get("report"),
        budget_hit=budget_hit,
        total_tokens_in=final_state.get("tokens_in", 0),
        total_tokens_out=final_state.get("tokens_out", 0),
        total_cost_usd=final_state.get("cost_usd", 0.0),
        iterations=iterations,
        cache_stats=cache_stats,
    )
