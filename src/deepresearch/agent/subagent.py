"""ReAct subagent: the per-node worker of the unified planner -> supervisor ->
subagent -> verify -> synthesis graph (replaces agent/worker.py's single-shot
retrieve+extract worker). Each plan node gets its own instance of this
compiled subgraph, invoked once per Send dispatch -- context isolation is
structural, not a convention: only the Finding this module returns crosses
back into the outer graph's state, the raw tool-call transcript (retrieved
pages, intermediate search calls) never leaves this function call.

Reuses agent/react_agent.py's tool-calling primitives verbatim: `agent_node`,
`TOOLS` (search + calculate), `_agent_route`, `_record`, `_evidence_block`,
`AgentContext` -- same tools, same OpenRouter tool-calling LLM split, same
ReAct step bound. Only the exit differs: instead of react_agent's own
finalize/verify/self-correct loop, this ends at one finalize_finding call and
returns control to the OUTER graph's verify node, which decides (with
knowledge of the whole plan) whether to requeue this node.
"""

from __future__ import annotations

import operator
import time
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from deepresearch.agent.react_agent import TOOLS, AgentContext, _agent_route, _evidence_block, _record, agent_node
from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.rerank.base import RerankBackend
from deepresearch.schemas import Claim, Finding, SourceRegistryEntry, SubQuestion
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, stage_span


class FindingDraft(BaseModel):
    """What the LLM actually produces for one node -- everything else on
    Finding (node_id, verified) is set by this module, not the model."""

    answer: str
    claims: list[Claim]
    entities_extracted: dict[str, str] = Field(default_factory=dict)
    confidence: float
    open_gaps: list[str] = Field(default_factory=list)


class SubagentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    step: Annotated[int, operator.add]
    finding: FindingDraft | None
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    started_monotonic: float


async def finalize_finding_node(state: SubagentState, runtime) -> dict:
    ctx: AgentContext = runtime.context
    evidence = _evidence_block(state["messages"]) or "(no passages retrieved)"
    system = load_prompt("finding_v1.txt")
    user_content = f"Research question: {state['question']}\n\nRetrieved passages:\n{evidence}"
    start_dt = datetime.now(timezone.utc)
    start = time.monotonic()
    with stage_span("finalize_finding"):
        span_id = current_span_id_hex()
        data, usage = await ctx.llm.complete_structured(
            model=ctx.config.worker_model, system=system, user_content=user_content,
            response_model=FindingDraft, max_tokens=2048,
        )
    latency_ms = (time.monotonic() - start) * 1000
    _record(
        runtime, "finalize_finding", "finalize_finding",
        input_summary={"question": state["question"]}, output=data.model_dump(),
        usage=usage, latency_ms=latency_ms, start_dt=start_dt, end_dt=datetime.now(timezone.utc), span_id=span_id,
    )
    return {
        "finding": data,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def build_subagent_graph():
    builder = StateGraph(SubagentState, context_schema=AgentContext)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("finalize_finding", finalize_finding_node)

    builder.add_edge(START, "agent")
    # _agent_route (agent/react_agent.py) returns the literal strings
    # "tools"/"finalize" -- reused unmodified via path_map, just renaming
    # this graph's terminal node.
    builder.add_conditional_edges("agent", _agent_route, {"tools": "tools", "finalize": "finalize_finding"})
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize_finding", END)
    return builder.compile()


def _recursion_limit(config: RunConfig) -> int:
    return config.max_react_steps * 2 + 6


async def run_subagent(
    node: SubQuestion,
    context_facts: str,
    *,
    config: RunConfig,
    chat_model,
    llm: LLMClient,
    search_backend: SearchBackend,
    rerank_backend: RerankBackend | None,
    recorder: RunRecorder,
    run_span_id: str,
    source_registry: dict[str, SourceRegistryEntry],
    started_monotonic: float,
    source_id_prefix: str | None = None,
) -> tuple[Finding, LLMUsage]:
    """Runs one plan node's ReAct research loop to a Finding. `context_facts`
    is the upstream entities/answers this node's brief should substitute in
    (built by the supervisor's facts_for() in Phase 3) -- empty string for an
    independent node. `source_registry` must be a fresh per-call dict (the
    caller merges it back via the shared dict-union reducer, namespaced by
    `source_id_prefix`, same discipline as the old parallel worker fan-out).
    `source_id_prefix` defaults to node.id; the caller overrides it (e.g. to
    "{node.id}_r{attempt}") when a node is retried after a failed verify, so a
    superseded attempt's source ids can never collide with the retry's."""
    brief = node.question
    if context_facts:
        brief = f"{node.question}\n\nResolved facts from earlier research steps:\n{context_facts}"

    ctx = AgentContext(
        config=config, chat_model=chat_model, llm=llm, search_backend=search_backend,
        rerank_backend=rerank_backend, recorder=recorder, run_span_id=run_span_id,
        source_registry=source_registry, source_id_prefix=source_id_prefix or node.id,
    )
    initial: SubagentState = {
        "messages": [SystemMessage(content=load_prompt("react_agent_v1.txt")), HumanMessage(content=brief)],
        "question": node.question,
        "step": 0,
        "finding": None,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "started_monotonic": started_monotonic,
    }
    graph = build_subagent_graph()
    final_state = await graph.ainvoke(initial, config={"recursion_limit": _recursion_limit(config)}, context=ctx)

    data = final_state.get("finding") or FindingDraft(
        answer="", claims=[], entities_extracted={}, confidence=0.0, open_gaps=["no finding produced"],
    )
    finding = Finding(
        node_id=node.id,
        question=node.question,
        answer=data.answer,
        claims=data.claims,
        entities_extracted=data.entities_extracted,
        confidence=data.confidence,
        open_gaps=data.open_gaps,
        verified=False,
    )
    usage = LLMUsage(
        input_tokens=final_state.get("tokens_in", 0),
        output_tokens=final_state.get("tokens_out", 0),
        cost_usd=final_state.get("cost_usd", 0.0),
    )
    return finding, usage
