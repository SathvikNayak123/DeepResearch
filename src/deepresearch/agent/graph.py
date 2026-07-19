"""The unified planner -> supervisor -> subagent(ReAct, +inline per-hop
verify) -> synthesis -> reflection StateGraph — the single orchestration
engine (docs/DESIGN.md decision rows 1/2/14 now describe this graph; the
old parallel-only plan-first graph, the thin sequential react_step variant,
the standalone react_agent driver, and the pre-LangGraph hand-rolled asyncio
loop have all been superseded and removed).

Node topology differs from the original 3-node sketch (subagent -> verify ->
supervisor) in one deliberate way, confirmed by direct experiment against
this repo's installed langgraph version: a static edge from a
`Send`-fanned-out node coalesces into ONE downstream call per wave, seeing
every branch's merged state -- there is no way for a separate "verify" node
to know which finding belongs to which parallel branch. So each hop's verify
check runs INSIDE the same Send-dispatched node that ran the ReAct subagent
(`subagent_node`), before that node returns its state update. This preserves
per-branch correctness under real parallel fan-out; functionally it is still
"research this hop, then verify it before trusting it downstream," just
implemented as one node instead of two.

Readiness ("ready = depends_on subset of verified, not yet verified/failed")
is recomputed by `supervisor_router` every wave -- a node whose verify failed
but has correction budget left is simply absent from both verified_ids and
failed_ids, so it is ready again next wave with no separate "requeue" event.

Reuses: agent/planner.py (plan), agent/subagent.py (run_subagent, the ReAct
loop), agent/synthesis.py (synthesize), agent/react_agent.py's `_record`
(verify call recording) and its tool-calling primitives (via subagent.py).

Non-serializable per-run dependencies (llm, chat_model, backends, recorder,
config) live in LangGraph's runtime *context* (`MultihopContext`), not graph
state -- state carries only plain/pydantic domain data, which is what makes
it checkpoint-safe for the `AsyncSqliteSaver` wired in agent/orchestrator.py.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from deepresearch.agent import planner, synthesis
from deepresearch.agent.budget import budget_status as _budget_status
from deepresearch.agent.dag import topological_order
from deepresearch.agent.react_agent import _record
from deepresearch.agent.subagent import run_subagent
from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.rerank.base import RerankBackend
from deepresearch.schemas import Finding, Plan, Report, SourceRegistryEntry, SubQuestion, WorkerNotes
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, stage_span

class HopVerdict(BaseModel):
    grounded: bool
    reason: str


class GapCheck(BaseModel):
    has_gaps: bool
    followup_questions: list[str] = Field(default_factory=list)
    rationale: str


def _merge_source_registry(
    left: dict[str, SourceRegistryEntry], right: dict[str, SourceRegistryEntry]
) -> dict[str, SourceRegistryEntry]:
    merged = dict(left)
    merged.update(right)
    return merged


def _dict_union(a: dict, b: dict) -> dict:
    merged = dict(a)
    merged.update(b)
    return merged


def _merge_corrections(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    merged = dict(a)
    for k, v in b.items():
        merged[k] = merged.get(k, 0) + v
    return merged


async def _run_stage(
    name: str,
    stage: str,
    coro,
    *,
    runtime: Runtime[Any],
    input_summary: dict | None = None,
    **attrs,
) -> tuple[Any, Any]:
    """Opens the stage span (nested under the ambient "run" span kept open
    for the whole graph execution in orchestrator.py), records the
    trajectory row, and emits the stage_complete event via the graph's
    stream writer."""
    recorder = runtime.context.recorder
    start_dt = datetime.now(timezone.utc)
    start = time.monotonic()
    with stage_span(name, **attrs) as span:
        span_id = current_span_id_hex()
        result, usage = await coro
        latency_ms = (time.monotonic() - start) * 1000
        span.set_attribute("llm.tokens_in", usage.input_tokens)
        span.set_attribute("llm.tokens_out", usage.output_tokens)
        span.set_attribute("llm.cost_usd", usage.cost_usd)
        span.set_attribute("latency_ms", latency_ms)
    end_dt = datetime.now(timezone.utc)

    recorder.record_trajectory(
        span_id=span_id,
        parent_span_id=runtime.context.run_span_id,
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
    runtime.stream_writer(
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
    return result, usage


class MultihopState(TypedDict, total=False):
    question: str
    plan: Plan
    findings: Annotated[list[Finding], operator.add]
    source_registry: Annotated[dict[str, SourceRegistryEntry], _merge_source_registry]
    # dict[node_id, True] rather than set[str] -- kept plain-dict/JSON-shaped
    # so the whole state stays checkpoint-safe (a set is not a type this
    # project's checkpoint serializer is relied on to round-trip; dicts
    # already are, via _merge_source_registry).
    verified_ids: Annotated[dict[str, bool], _dict_union]
    failed_ids: Annotated[dict[str, bool], _dict_union]
    node_corrections: Annotated[dict[str, int], _merge_corrections]
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    started_monotonic: float
    report: Report | None
    # Annotated w/ operator.add, not a plain int -- a plain int field would
    # silently overwrite to the literal value a node returns ("add 1 more")
    # instead of accumulating, so the ceiling never actually trips (the exact
    # bug class tests/test_graph.py's reflection-ceiling test guards against).
    # reflection_node always returns {"reflection_iters": 1} to mean "one
    # more pass taken", never the running total.
    reflection_iters: Annotated[int, operator.add]
    # Plain flag (no reducer) -- reflection_node is never Send-fanned, only
    # one writer per superstep, so overwrite-on-write is exactly right here.
    should_continue: bool


class SubagentInput(TypedDict):
    node: SubQuestion
    context_facts: str
    attempt: int
    feeds_hop: bool
    started_monotonic: float


@dataclass
class MultihopContext:
    config: RunConfig
    llm: LLMClient
    chat_model: object  # ChatOpenAI already .bind_tools([search, calculate]) -- built once per run
    search_backend: SearchBackend
    rerank_backend: RerankBackend | None
    recorder: RunRecorder
    run_span_id: str


def facts_for(node: SubQuestion, state: MultihopState) -> str:
    """Resolved upstream entities/answers for node's depends_on, formatted for
    injection into its brief. A retried node's earlier (failed-verify)
    attempts also live in `findings` under the same node_id -- the LAST
    entry for a given node_id is always its verified one, since a node stops
    being re-dispatched the moment its own verify passes."""
    if not node.depends_on:
        return ""
    findings = state.get("findings", [])
    lines = []
    for dep_id in node.depends_on:
        candidates = [f for f in findings if f.node_id == dep_id]
        if not candidates:
            continue
        f = candidates[-1]
        entities = ", ".join(f"{k}={v}" for k, v in f.entities_extracted.items())
        lines.append(f"- {f.question} -> {f.answer}" + (f" ({entities})" if entities else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


async def plan_node(state: MultihopState, runtime: Runtime[MultihopContext]) -> dict:
    ctx = runtime.context
    result, usage = await _run_stage(
        "plan", "plan", planner.plan(state["question"], ctx.config, ctx.llm),
        runtime=runtime, input_summary={"question": state["question"]},
    )
    return {
        "plan": result,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


async def _verify_finding(node: SubQuestion, finding: Finding, runtime: Runtime[MultihopContext]) -> tuple[bool, LLMUsage]:
    ctx = runtime.context
    claims_block = (
        "\n".join(f"- {c.text} (source {c.source_id}, quote: {c.quote!r})" for c in finding.claims) or "(no claims)"
    )
    user_content = f"Sub-question: {node.question}\nAnswer given: {finding.answer}\nClaims:\n{claims_block}"
    start_dt = datetime.now(timezone.utc)
    start = time.monotonic()
    with stage_span(f"verify:{node.id}"):
        span_id = current_span_id_hex()
        data, usage = await ctx.llm.complete_structured(
            model=ctx.config.reflection_model, system=load_prompt("hop_verify_v1.txt"),
            user_content=user_content, response_model=HopVerdict, max_tokens=512,
        )
    latency_ms = (time.monotonic() - start) * 1000
    _record(
        runtime, "verify", f"verify:{node.id}",
        input_summary={"sub_question": node.question}, output=data.model_dump(),
        usage=usage, latency_ms=latency_ms, start_dt=start_dt, end_dt=datetime.now(timezone.utc), span_id=span_id,
    )
    return data.grounded, usage


async def subagent_node(state: SubagentInput, runtime: Runtime[MultihopContext]) -> dict:
    ctx = runtime.context
    node, context_facts, attempt, feeds_hop = (
        state["node"], state["context_facts"], state["attempt"], state["feeds_hop"],
    )
    own_registry: dict[str, SourceRegistryEntry] = {}

    finding, usage = await _run_stage(
        f"subagent:{node.id}", "subagent",
        run_subagent(
            node, context_facts, config=ctx.config, chat_model=ctx.chat_model, llm=ctx.llm,
            search_backend=ctx.search_backend, rerank_backend=ctx.rerank_backend, recorder=ctx.recorder,
            run_span_id=ctx.run_span_id, source_registry=own_registry,
            started_monotonic=state.get("started_monotonic", 0.0), source_id_prefix=f"{node.id}_r{attempt}",
        ),
        runtime=runtime, input_summary={"question": node.question},
    )

    update: dict = {
        "findings": [finding],
        "source_registry": own_registry,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }

    if not feeds_hop:
        # No later hop depends on this one -- its own citation correctness is
        # caught post-hoc by eval citation-precision, not gated live (that
        # would double the paid verify-LLM-call count for every leaf node).
        finding.verified = True
        update["verified_ids"] = {node.id: True}
        return update

    grounded, verify_usage = await _verify_finding(node, finding, runtime)
    update["tokens_in"] += verify_usage.input_tokens
    update["tokens_out"] += verify_usage.output_tokens
    update["cost_usd"] += verify_usage.cost_usd

    if grounded:
        finding.verified = True
        update["verified_ids"] = {node.id: True}
    else:
        update["node_corrections"] = {node.id: 1}
        # attempt is 0-indexed (0 = the first try); max_corrections retries
        # are allowed BEYOND that first try (same convention as
        # react_agent.py's max_corrections), so give up once the attempt
        # that just failed was itself attempt number max_corrections.
        if attempt >= ctx.config.max_corrections:
            update["failed_ids"] = {node.id: True}
        # else: absent from both verified_ids and failed_ids -> naturally
        # ready again next wave, no explicit "requeue" needed.
    return update


def _ordered_findings(plan: Plan, findings: list[Finding]) -> list[Finding]:
    """Findings in dependency (topological) order, so synthesis reads a hop's
    upstream facts before the hop that consumed them -- not just whatever
    order waves happened to complete in. A node with more than one finding
    (a retried node) contributes only its LAST one -- always the verified
    attempt, since a node stops being re-dispatched the moment its own verify
    passes (same reasoning as facts_for())."""
    order = topological_order(plan.sub_questions)
    by_id: dict[str, Finding] = {}
    for f in findings:
        by_id[f.node_id] = f  # last write wins == latest attempt
    return [by_id[node_id] for node_id in order if node_id in by_id]


def _findings_to_notes(findings: list[Finding]) -> list[WorkerNotes]:
    """Adapter onto the existing synthesis.synthesize(notes=...) contract --
    findings must already be in the order synthesis should read them in
    (see _ordered_findings)."""
    return [
        WorkerNotes(sub_question_id=f.node_id, sub_question=f.question, claims=f.claims, open_gaps=f.open_gaps)
        for f in findings
    ]


async def synthesis_node(state: MultihopState, runtime: Runtime[MultihopContext]) -> dict:
    ctx = runtime.context
    budget_tripped = _budget_status(state, ctx.config.budget) is not None
    ordered = _ordered_findings(state["plan"], state.get("findings", []))
    notes = _findings_to_notes(ordered)
    try:
        result, usage = await _run_stage(
            "synthesis", "synthesis",
            synthesis.synthesize(state["question"], notes, state.get("source_registry", {}), config=ctx.config, llm=ctx.llm),
            runtime=runtime,
            input_summary={"question": state["question"], "budget_exceeded": True}
            if budget_tripped
            else {"question": state["question"]},
        )
    except Exception:
        if not budget_tripped:
            raise
        return {"report": None}
    return {
        "report": result,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


async def reflection_node(state: MultihopState, runtime: Runtime[MultihopContext]) -> dict:
    """Post-synthesis gap check: scans the FINAL report for
    gaps/contradictions/low-confidence claims. If it finds a real, fixable
    one and the reflection ceiling hasn't been hit, appends independent
    (depends_on: []) follow-up nodes to the plan and signals the supervisor
    to dispatch them; otherwise signals END."""
    ctx = runtime.context
    report = state.get("report")
    reflection_iters = state.get("reflection_iters", 0)

    if report is None or reflection_iters >= ctx.config.max_reflect or _budget_status(state, ctx.config.budget) is not None:
        return {"should_continue": False}

    findings = state.get("findings", [])
    gaps_block = "\n".join(f"- {f.question}: {'; '.join(f.open_gaps)}" for f in findings if f.open_gaps) or "(none noted during research)"
    low_conf = [f.question for f in findings if f.confidence < 0.5]
    user_content = (
        f"Research question: {state['question']}\n\nFinal report:\n{report.text}\n\n"
        f"Open gaps noted during research:\n{gaps_block}\n\nLow-confidence findings: {low_conf}"
    )
    data, usage = await _run_stage(
        "reflection", "reflection",
        ctx.llm.complete_structured(
            model=ctx.config.reflection_model, system=load_prompt("reflection_post_synthesis_v1.txt"),
            user_content=user_content, response_model=GapCheck, max_tokens=1024,
        ),
        runtime=runtime, input_summary={"question": state["question"]},
    )

    update: dict = {
        "reflection_iters": 1,  # reducer: operator.add -- "one more pass taken"
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }
    if data.has_gaps and data.followup_questions:
        new_nodes = [
            SubQuestion(id=f"followup_{reflection_iters}_{i}", question=q, depends_on=[])
            for i, q in enumerate(data.followup_questions)
        ]
        update["plan"] = Plan(sub_questions=state["plan"].sub_questions + new_nodes)
        update["should_continue"] = True
    else:
        update["should_continue"] = False
    return update


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def supervisor_router(state: MultihopState, runtime: Runtime[MultihopContext]):
    """The single dispatch point: used after "plan" (initial wave) and after
    "subagent" (every subsequent wave). Computes which plan nodes are ready
    (all depends_on verified, not already verified/failed) and fans them out
    via Send; empty ready set (fully resolved, or stuck behind a failed
    upstream) or a tripped budget both route to synthesis."""
    if _budget_status(state, runtime.context.config.budget) is not None:
        return "synthesis"

    plan_nodes = state["plan"].sub_questions
    verified = state.get("verified_ids", {})
    failed = state.get("failed_ids", {})
    ready = [
        n for n in plan_nodes
        if n.id not in verified and n.id not in failed and all(d in verified for d in n.depends_on)
    ]
    if not ready:
        return "synthesis"

    corrections = state.get("node_corrections", {})
    depended_on = {dep for n in plan_nodes for dep in n.depends_on}
    return [
        Send("subagent", {
            "node": n,
            "context_facts": facts_for(n, state),
            "attempt": corrections.get(n.id, 0),
            "feeds_hop": n.id in depended_on,
            "started_monotonic": state.get("started_monotonic", 0.0),
        })
        for n in ready
    ]


def _reflection_route(state: MultihopState, runtime: Runtime[MultihopContext]):
    """reflection's conditional edge. If reflection_node appended follow-up
    nodes, they are by construction ready right now (depends_on: [], not yet
    in verified_ids/failed_ids) -- delegating to supervisor_router reuses its
    Send-building logic instead of duplicating it, and can only take the
    "ready nodes exist" branch here, never the "no ready nodes -> synthesis"
    one (that would loop synthesis -> reflection -> synthesis forever with
    nothing new to justify a second pass)."""
    if not state.get("should_continue"):
        return END
    return supervisor_router(state, runtime)


# --------------------------------------------------------------------------
# Graph builder
# --------------------------------------------------------------------------


def build_multihop_graph(checkpointer=None, interrupt_after=None):
    """`interrupt_after` (e.g. ["plan"]) pauses the graph durably right after
    the named node(s) complete -- only meaningful with a real checkpointer;
    a later call with input=None and the same thread_id resumes from there.
    Used by tests/test_checkpointer.py to prove interrupt-then-resume works;
    production runs (agent/orchestrator.py) leave this None."""
    builder = StateGraph(MultihopState, context_schema=MultihopContext)
    builder.add_node("plan", plan_node)
    builder.add_node("subagent", subagent_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("reflection", reflection_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", supervisor_router, ["subagent", "synthesis"])
    builder.add_conditional_edges("subagent", supervisor_router, ["subagent", "synthesis"])
    builder.add_edge("synthesis", "reflection")
    builder.add_conditional_edges("reflection", _reflection_route, ["subagent", "synthesis", END])
    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def recursion_limit_for(config: RunConfig) -> int:
    """Backstop above the real gates (budget ceilings, max_corrections per
    node, max_nodes plan size, max_reflect passes) -- those decide stopping,
    never LangGraph's recursion counter (docs/DESIGN.md decision row 3)."""
    return ((config.max_nodes * (config.max_corrections + 1)) + 10) * (config.max_reflect + 1)
