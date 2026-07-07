from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from deepresearch.agent import planner, react, reflection, synthesis, worker
from deepresearch.agent.budget import BudgetExceeded, BudgetTracker
from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig, current_git_sha
from deepresearch.llm.client import LLMClient
from deepresearch.rerank import build_rerank_backend
from deepresearch.schemas import (
    CacheStats,
    Plan,
    Report,
    RunResult,
    RunStatus,
    SourceRegistryEntry,
    SubQuestion,
    WorkerNotes,
)
from deepresearch.store import db
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, run_root_context, stage_span

logger = logging.getLogger(__name__)

# Fired after each stage's trajectory is recorded — the one hook the SSE
# streaming endpoint (api/streaming.py) needs to surface live progress
# without restructuring the (already tested) blocking stage-by-stage flow
# below. None everywhere else (CLI, eval harness, benchmarks) — a no-op.
OnEvent = Callable[[dict], Awaitable[None]]


async def _call_stage(
    name: str,
    coro,
    *,
    budget: BudgetTracker,
    recorder: RunRecorder,
    stage: str,
    parent_span_id: str,
    input_summary: dict | None = None,
    on_event: OnEvent | None = None,
    **attrs,
):
    start_dt = datetime.now(timezone.utc)
    start = time.monotonic()
    with stage_span(name, **attrs) as span:
        span_id = current_span_id_hex()
        result, usage = await coro
        latency_ms = (time.monotonic() - start) * 1000
        budget.record(usage.input_tokens, usage.output_tokens, usage.cost_usd)
        span.set_attribute("llm.tokens_in", usage.input_tokens)
        span.set_attribute("llm.tokens_out", usage.output_tokens)
        span.set_attribute("llm.cost_usd", usage.cost_usd)
        span.set_attribute("latency_ms", latency_ms)
    end_dt = datetime.now(timezone.utc)

    recorder.record_trajectory(
        span_id=span_id,
        parent_span_id=parent_span_id,
        stage=stage,
        name=name,
        input=input_summary,
        output=result.model_dump() if hasattr(result, "model_dump") else None,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=latency_ms,
        started_at=start_dt,
        ended_at=end_dt,
    )
    if on_event is not None:
        await on_event(
            {
                "type": "stage_complete",
                "stage": stage,
                "name": name,
                "tokens_in": usage.input_tokens,
                "tokens_out": usage.output_tokens,
                "cost_usd": usage.cost_usd,
                "latency_ms": latency_ms,
            }
        )
    return result


async def run_research(
    question: str,
    *,
    config: RunConfig,
    search_backend: SearchBackend,
    llm: LLMClient | None = None,
    benchmark_name: str | None = None,
    on_event: OnEvent | None = None,
) -> RunResult:
    """Runs one research question end-to-end. Every call writes a `runs` row
    (plus trajectories/tool_calls) to config.database_url — "every agent run
    and eval run writes here from now on" (this session's brief), not opt-in.
    """
    llm = llm or LLMClient()
    budget = BudgetTracker(config.budget)
    rerank_backend = build_rerank_backend(config)
    run_id = uuid.uuid4().hex
    root_ctx = run_root_context(run_id)
    recorder = RunRecorder(run_id=run_id)
    run_started = time.monotonic()

    if on_event is not None:
        # Fired before any DB/LLM work so a live stream can show run_id
        # (Langfuse trace correlation, docs/DESIGN.md: run_id = trace_id)
        # from the very first byte, not only once the run finishes.
        await on_event({"type": "run_started", "run_id": run_id, "question": question})

    await db.ensure_schema(config.database_url)
    await db.create_run(
        config.database_url,
        run_id=run_id,
        benchmark_name=benchmark_name,
        config=dataclasses.asdict(config),
        git_sha=current_git_sha(),
        status="running",
    )

    source_registry: dict[str, SourceRegistryEntry] = {}
    all_notes: list[WorkerNotes] = []
    reflections = []
    status = RunStatus.RUNNING
    budget_hit: str | None = None
    report: Report | None = None
    current_plan = Plan(sub_questions=[])
    step = 0  # react-mode step counter; visible after the try/except too, in case of BudgetExceeded mid-loop

    with stage_span("run", context=root_ctx, **{"run.id": run_id, "run.question": question}) as run_span:
        run_span_id = current_span_id_hex()

        is_react = config.planning_style == "react"

        if not is_react:
            current_plan = await _call_stage(
                "plan",
                planner.plan(question, config, llm),
                budget=budget,
                recorder=recorder,
                stage="plan",
                parent_span_id=run_span_id,
                input_summary={"question": question},
                on_event=on_event,
            )

        try:
            if is_react:
                # Interleaved ReAct (docs/DESIGN.md decision row 2
                # alternative, measured head-to-head in the dated addendum):
                # no upfront plan — one query decided at a time from the
                # claims gathered so far, sequential (no worker pool,
                # parallelism doesn't apply when there's only one query in
                # flight at a time). Stopping is still explicit and bounded:
                # max_react_steps ceiling AND the budget ceiling, ANDed with
                # the step's own "done" signal — never "the agent decides"
                # alone.
                while True:
                    budget.check()
                    if step >= config.max_react_steps:
                        break
                    action = await _call_stage(
                        f"react_step:{step}",
                        react.next_action(question, all_notes, config=config, llm=llm),
                        budget=budget,
                        recorder=recorder,
                        stage="react_step",
                        parent_span_id=run_span_id,
                        input_summary={"step": step, "n_notes": len(all_notes)},
                        on_event=on_event,
                    )
                    if action["done"]:
                        break

                    sub_question = SubQuestion(id=f"react-{step}", question=action["next_query"])
                    note = await _call_stage(
                        f"worker:{sub_question.id}",
                        worker.run_worker(
                            sub_question,
                            search_backend=search_backend,
                            config=config,
                            llm=llm,
                            source_registry=source_registry,
                            rerank_backend=rerank_backend,
                            recorder=recorder,
                        ),
                        budget=budget,
                        recorder=recorder,
                        stage="worker",
                        parent_span_id=run_span_id,
                        input_summary={"sub_question": sub_question.question},
                        on_event=on_event,
                        **{"worker.sub_question": sub_question.question},
                    )
                    all_notes.append(note)
                    current_plan.sub_questions.append(sub_question)
                    step += 1
            else:
                while True:
                    budget.check()
                    semaphore = asyncio.Semaphore(config.max_workers)

                    async def _bounded_worker(sub_question):
                        async with semaphore:
                            return await _call_stage(
                                f"worker:{sub_question.id}",
                                worker.run_worker(
                                    sub_question,
                                    search_backend=search_backend,
                                    config=config,
                                    llm=llm,
                                    source_registry=source_registry,
                                    rerank_backend=rerank_backend,
                                    recorder=recorder,
                                ),
                                budget=budget,
                                recorder=recorder,
                                stage="worker",
                                parent_span_id=run_span_id,
                                input_summary={"sub_question": sub_question.question},
                                on_event=on_event,
                                **{"worker.sub_question": sub_question.question},
                            )

                    all_notes.extend(
                        await asyncio.gather(*(_bounded_worker(sq) for sq in current_plan.sub_questions))
                    )

                    budget.check()
                    reflection_result = await _call_stage(
                        "reflection",
                        reflection.reflect(question, current_plan, all_notes, config=config, llm=llm),
                        budget=budget,
                        recorder=recorder,
                        stage="reflection",
                        parent_span_id=run_span_id,
                        input_summary={
                            "question": question,
                            "plan": [sq.question for sq in current_plan.sub_questions],
                        },
                        on_event=on_event,
                    )
                    reflections.append(reflection_result)

                    coverage_met = reflection_result.coverage_score >= config.coverage_threshold
                    if coverage_met or not reflection_result.should_replan or not budget.replan_allowed():
                        break

                    budget.register_replan()
                    current_plan = Plan(sub_questions=reflection_result.new_sub_questions)

            budget.check()
            report = await _call_stage(
                "synthesis",
                synthesis.synthesize(question, all_notes, source_registry, config=config, llm=llm),
                budget=budget,
                recorder=recorder,
                stage="synthesis",
                parent_span_id=run_span_id,
                input_summary={"question": question},
                on_event=on_event,
            )
            status = RunStatus.COMPLETED

        except BudgetExceeded as exc:
            budget_hit = exc.reason
            status = RunStatus.BUDGET_EXCEEDED
            logger.warning("run %s stopped: %s", run_id, exc.reason)
            if all_notes:
                try:
                    report = await _call_stage(
                        "synthesis",
                        synthesis.synthesize(question, all_notes, source_registry, config=config, llm=llm),
                        budget=budget,
                        recorder=recorder,
                        stage="synthesis",
                        parent_span_id=run_span_id,
                        input_summary={"question": question, "budget_exceeded": True},
                        on_event=on_event,
                    )
                except Exception:
                    report = None

        # Computed post-try/except (not inline in the loops) so a
        # BudgetExceeded raised mid-loop still reports the steps actually
        # taken, not a stale initial value.
        iterations = step if is_react else budget.replans + 1

        run_span.set_attribute("run.status", status.value)
        run_span.set_attribute("run.tokens_in", budget.tokens_in)
        run_span.set_attribute("run.tokens_out", budget.tokens_out)
        run_span.set_attribute("run.cost_usd", budget.cost_usd)
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
        total_cost_usd=budget.cost_usd,
        total_latency_ms=total_latency_ms,
    )

    return RunResult(
        run_id=run_id,
        status=status,
        question=question,
        plan=current_plan,
        worker_notes=all_notes,
        reflections=reflections,
        report=report,
        budget_hit=budget_hit,
        total_tokens_in=budget.tokens_in,
        total_tokens_out=budget.tokens_out,
        total_cost_usd=budget.cost_usd,
        iterations=iterations,
        cache_stats=cache_stats,
    )
