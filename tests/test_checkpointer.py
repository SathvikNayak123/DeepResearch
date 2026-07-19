"""LangGraph checkpointer (agent/orchestrator.py wires an AsyncSqliteSaver
keyed by thread_id=run_id): proves a run interrupted mid-graph resumes from
its last completed node instead of restarting from scratch -- the spec's
"multi-session run persistence" (docs §4.0), tested directly against the
unified graph (agent/graph.py) rather than through the full orchestrator
driver, to isolate the LangGraph mechanic itself from run-store/DB concerns.
"""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepresearch.agent.graph import (
    GapCheck,
    HopVerdict,
    MultihopContext,
    MultihopState,
    build_multihop_graph,
    recursion_limit_for,
)
from deepresearch.agent.subagent import FindingDraft
from deepresearch.agent.synthesis import SynthesisDraft
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import Plan, SubQuestion
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import init_telemetry

DOCS = [{"doc_id": "d1", "title": "Filler", "text": "Irrelevant filler content for the local corpus backend."}]


class RecordingChatModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        msg = AIMessage(content="done", tool_calls=[])
        msg.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
        return msg


class CountingStubLLM:
    """Counts plan_node calls -- the thing a genuine resume (vs. a silent
    restart) must keep at 1 across both graph invocations."""

    def __init__(self) -> None:
        self.plan_calls = 0

    async def complete_structured(self, *, model, system, user_content, response_model, max_tokens=4096):
        usage = LLMUsage(input_tokens=10, output_tokens=10, cost_usd=0.0)
        if response_model is Plan:
            self.plan_calls += 1
            return Plan(sub_questions=[SubQuestion(id="n1", question="q1", depends_on=[])]), usage
        if response_model is HopVerdict:
            return HopVerdict(grounded=True, reason=""), usage
        if response_model is FindingDraft:
            return FindingDraft(answer="a1", claims=[], entities_extracted={}, confidence=0.9, open_gaps=[]), usage
        if response_model is GapCheck:
            return GapCheck(has_gaps=False, followup_questions=[], rationale="stub: no gaps"), usage
        if response_model is SynthesisDraft:
            return SynthesisDraft(text="stub report", cited_source_ids=[]), usage
        raise ValueError(f"unexpected response_model: {response_model}")


def _initial_state(started: float) -> MultihopState:
    return {
        "question": "test question", "plan": Plan(sub_questions=[]), "findings": [],
        "source_registry": {}, "verified_ids": {}, "failed_ids": {}, "node_corrections": {},
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "started_monotonic": started,
        "report": None, "reflection_iters": 0, "should_continue": False,
    }


@pytest.mark.asyncio
async def test_interrupted_run_resumes_from_checkpoint_without_replanning(tmp_path):
    init_telemetry()
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = CountingStubLLM()
    ctx = MultihopContext(
        config=config, llm=llm, chat_model=RecordingChatModel(), search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="test-run"), run_span_id="span-root",
    )
    db_path = str(tmp_path / "checkpoints.sqlite")
    thread_config = {"configurable": {"thread_id": "thread-1"}, "recursion_limit": recursion_limit_for(config)}

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()

        # First "attempt": interrupted right after plan -- durably paused,
        # nothing researched yet.
        paused_graph = build_multihop_graph(checkpointer=checkpointer, interrupt_after=["plan"])
        paused_state = await paused_graph.ainvoke(_initial_state(time.monotonic()), config=thread_config, context=ctx)

        assert paused_state["plan"].sub_questions  # plan ran
        assert paused_state.get("findings", []) == []  # nothing researched yet -- genuinely paused
        assert llm.plan_calls == 1

        # "Resume" in a fresh graph object (simulating a new process), same
        # thread_id, input=None -- must continue from the checkpointed plan,
        # not re-plan or restart.
        resumed_graph = build_multihop_graph(checkpointer=checkpointer)
        final_state = await resumed_graph.ainvoke(None, config=thread_config, context=ctx)

    assert llm.plan_calls == 1  # plan_node did NOT re-run on resume
    assert final_state["report"] is not None
    assert {f.node_id for f in final_state["findings"]} == {"n1"}


@pytest.mark.asyncio
async def test_resume_with_a_fresh_thread_id_starts_over(tmp_path):
    """Sanity check on the test above: a DIFFERENT thread_id against the same
    checkpointer DB gets a genuinely fresh run, not someone else's state --
    confirms thread_id is really the isolation key, not just a label."""
    init_telemetry()
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = CountingStubLLM()
    ctx = MultihopContext(
        config=config, llm=llm, chat_model=RecordingChatModel(), search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="test-run-2"), run_span_id="span-root-2",
    )
    db_path = str(tmp_path / "checkpoints.sqlite")

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        await checkpointer.setup()

        graph_a = build_multihop_graph(checkpointer=checkpointer, interrupt_after=["plan"])
        await graph_a.ainvoke(
            _initial_state(time.monotonic()),
            config={"configurable": {"thread_id": "thread-a"}, "recursion_limit": recursion_limit_for(config)},
            context=ctx,
        )
        assert llm.plan_calls == 1

        graph_b = build_multihop_graph(checkpointer=checkpointer)
        final_state_b = await graph_b.ainvoke(
            _initial_state(time.monotonic()),
            config={"configurable": {"thread_id": "thread-b"}, "recursion_limit": recursion_limit_for(config)},
            context=ctx,
        )

    assert llm.plan_calls == 2  # thread-b planned independently, not resumed from thread-a
    assert final_state_b["report"] is not None
