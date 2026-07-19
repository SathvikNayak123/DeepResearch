"""Unified StateGraph (agent/graph.py): supervisor readiness/wave-dispatch,
facts_for upstream-entity injection, the inline per-hop verify gate (requeue
on fail, give up at max_corrections), dependency-ordered synthesis, and the
bounded post-synthesis reflection loop -- offline, stubbed chat model +
stubbed structured-JSON LLM, no network.
"""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepresearch.agent.graph import (
    GapCheck,
    HopVerdict,
    MultihopContext,
    MultihopState,
    _ordered_findings,
    build_multihop_graph,
    recursion_limit_for,
)
from deepresearch.agent.subagent import FindingDraft
from deepresearch.agent.synthesis import SynthesisDraft
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import Finding, Plan, SubQuestion
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import init_telemetry

DOCS = [{"doc_id": "d1", "title": "Filler", "text": "Irrelevant filler content for the local corpus backend."}]


class RecordingChatModel:
    """Every subagent call finishes in one turn (no tool_calls) -- finalize
    output is fully controlled by StubMultihopLLM below, so exercising the
    real search/calculate tool loop isn't needed here (tests/test_subagent.py
    already covers that). Records every brief it was given so tests can
    confirm facts_for() actually reached a dependent node's message."""

    def __init__(self) -> None:
        self.briefs: list[str] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        human = next(m for m in messages if isinstance(m, HumanMessage))
        self.briefs.append(human.content)
        msg = AIMessage(content="done", tool_calls=[])
        msg.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
        return msg


class StubMultihopLLM:
    """Structured stub covering all four response models the unified graph
    requests: Plan (planner), FindingDraft (per-hop finalize), HopVerdict
    (per-hop verify), GapCheck (reflection), SynthesisDraft (synthesis) --
    dispatched on response_model identity, matching what Instructor's real
    validation would accept. Findings and verify verdicts are looked up by
    matching the node's question text against the caller's user_content,
    since that's the only thing threaded through complete_structured's plain
    string argument."""

    def __init__(
        self,
        nodes: list[dict],
        findings_by_question: dict[str, dict],
        verify_by_question: dict[str, list[bool]] | None = None,
        reflection_sequence: list[dict] | None = None,
    ) -> None:
        self._nodes = nodes
        self._findings_by_question = findings_by_question
        self._verify_by_question = verify_by_question or {}
        self._verify_calls: dict[str, int] = {}
        self.grounded_call_count = 0
        # Defaults to "no gaps" every call, so tests that don't care about
        # reflection sail through the synthesis->reflection edge unchanged
        # and land on END after exactly one reflection pass.
        self._reflection_sequence = reflection_sequence or [{"has_gaps": False, "followup_questions": [], "rationale": "stub: no gaps"}]
        self.reflection_call_count = 0

    async def complete_structured(self, *, model, system, user_content, response_model, max_tokens=4096):
        usage = LLMUsage(input_tokens=10, output_tokens=10, cost_usd=0.0)
        if response_model is Plan:
            return Plan(sub_questions=[SubQuestion(**n) for n in self._nodes]), usage
        if response_model is HopVerdict:
            self.grounded_call_count += 1
            for q, seq in self._verify_by_question.items():
                if q in user_content:
                    i = self._verify_calls.get(q, 0)
                    v = seq[min(i, len(seq) - 1)]
                    self._verify_calls[q] = i + 1
                    return HopVerdict(grounded=v, reason="" if v else "stub: forcing a retry"), usage
            return HopVerdict(grounded=True, reason=""), usage
        if response_model is FindingDraft:
            for q, data in self._findings_by_question.items():
                if q in user_content:
                    return FindingDraft(**data), usage
            raise ValueError(f"StubMultihopLLM: no finding stub matches {user_content[:200]!r}")
        if response_model is GapCheck:
            i = self.reflection_call_count
            self.reflection_call_count += 1
            return GapCheck(**self._reflection_sequence[min(i, len(self._reflection_sequence) - 1)]), usage
        if response_model is SynthesisDraft:
            return SynthesisDraft(text="stub synthesized report", cited_source_ids=[]), usage
        raise ValueError(f"StubMultihopLLM doesn't recognize response_model: {response_model}")


async def _run_graph(nodes, findings_by_question, *, verify_by_question=None, config=None, reflection_sequence=None):
    init_telemetry()
    config = config or RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = StubMultihopLLM(nodes, findings_by_question, verify_by_question, reflection_sequence)
    chat_model = RecordingChatModel()
    ctx = MultihopContext(
        config=config, llm=llm, chat_model=chat_model, search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="test-run"), run_span_id="span-root",
    )
    initial: MultihopState = {
        "question": "test question", "plan": Plan(sub_questions=[]), "findings": [],
        "source_registry": {}, "verified_ids": {}, "failed_ids": {}, "node_corrections": {},
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "started_monotonic": time.monotonic(),
        "report": None, "reflection_iters": 0, "should_continue": False,
    }
    graph = build_multihop_graph()
    final_state = await graph.ainvoke(initial, config={"recursion_limit": recursion_limit_for(config)}, context=ctx)
    return final_state, chat_model, llm


@pytest.mark.asyncio
async def test_dependent_hop_waits_for_both_upstreams_and_gets_injected_facts():
    nodes = [
        {"id": "n1", "question": "Who is the lead singer of Ratata?", "depends_on": []},
        {"id": "n2", "question": "Who is the lead singer of Kent?", "depends_on": []},
        {"id": "n3", "question": "What is the age difference between the two singers?", "depends_on": ["n1", "n2"]},
    ]
    findings = {
        "Who is the lead singer of Ratata?": {
            "answer": "Mauro Scocco", "claims": [], "entities_extracted": {"singer": "Mauro Scocco"},
            "confidence": 0.9, "open_gaps": [],
        },
        "Who is the lead singer of Kent?": {
            "answer": "Joakim Berg", "claims": [], "entities_extracted": {"singer": "Joakim Berg"},
            "confidence": 0.9, "open_gaps": [],
        },
        "What is the age difference between the two singers?": {
            "answer": "unknown", "claims": [], "entities_extracted": {}, "confidence": 0.3, "open_gaps": ["need birth years"],
        },
    }

    final_state, chat_model, _ = await _run_graph(nodes, findings)

    assert final_state["report"] is not None
    node_ids = {f.node_id for f in final_state["findings"]}
    assert node_ids == {"n1", "n2", "n3"}
    # n3's brief must contain BOTH upstream answers -- proof facts_for()
    # injected them, and proof n3 only ran after n1 AND n2 were verified.
    assert any("Mauro Scocco" in b and "Joakim Berg" in b for b in chat_model.briefs)


@pytest.mark.asyncio
async def test_leaf_node_skips_verify_but_feeding_node_is_gated():
    """n1 feeds n2 (n2 depends_on n1) so n1 must be verified; n2 is a leaf
    (nothing depends on it) so it gets a free pass with no verify call."""
    nodes = [
        {"id": "n1", "question": "Who is the lead singer of Ratata?", "depends_on": []},
        {"id": "n2", "question": "What year was the singer born?", "depends_on": ["n1"]},
    ]
    findings = {
        "Who is the lead singer of Ratata?": {
            "answer": "Mauro Scocco", "claims": [], "entities_extracted": {"singer": "Mauro Scocco"},
            "confidence": 0.9, "open_gaps": [],
        },
        "What year was the singer born?": {
            "answer": "1962", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": [],
        },
    }

    final_state, _, llm = await _run_graph(nodes, findings)

    # Both end up in verified_ids -- required so the supervisor never
    # re-dispatches either node -- but n2 (the leaf) gets there for free,
    # with no verify LLM call, unlike n1 which is actually gated.
    assert final_state["verified_ids"].get("n1") is True
    assert final_state["verified_ids"].get("n2") is True
    assert llm.grounded_call_count == 1  # only n1 (which feeds n2) is ever paid-verified
    node_ids = [f.node_id for f in final_state["findings"]]
    assert node_ids.count("n1") == 1
    assert node_ids.count("n2") == 1


@pytest.mark.asyncio
async def test_failed_verify_requeues_node_then_succeeds():
    nodes = [
        {"id": "n1", "question": "Who is the lead singer of Ratata?", "depends_on": []},
        {"id": "n2", "question": "What year was the singer born?", "depends_on": ["n1"]},
    ]
    findings = {
        "Who is the lead singer of Ratata?": {
            "answer": "Mauro Scocco", "claims": [], "entities_extracted": {"singer": "Mauro Scocco"},
            "confidence": 0.9, "open_gaps": [],
        },
        "What year was the singer born?": {
            "answer": "1962", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": [],
        },
    }
    verify = {"Who is the lead singer of Ratata?": [False, True]}

    final_state, _, _ = await _run_graph(nodes, findings, verify_by_question=verify)

    assert final_state["verified_ids"].get("n1") is True
    assert final_state["node_corrections"].get("n1") == 1
    node_ids = [f.node_id for f in final_state["findings"]]
    assert node_ids.count("n1") == 2  # one failed attempt + one successful retry
    assert "n2" in node_ids  # only became ready once n1 was verified


@pytest.mark.asyncio
async def test_verify_never_passes_gives_up_at_max_corrections_and_still_synthesizes():
    nodes = [
        {"id": "n1", "question": "Who is the lead singer of Ratata?", "depends_on": []},
        {"id": "n2", "question": "What year was the singer born?", "depends_on": ["n1"]},
    ]
    findings = {
        "Who is the lead singer of Ratata?": {
            "answer": "guess", "claims": [], "entities_extracted": {}, "confidence": 0.2, "open_gaps": [],
        },
        "What year was the singer born?": {
            "answer": "1962", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": [],
        },
    }
    verify = {"Who is the lead singer of Ratata?": [False, False, False, False, False]}
    config = RunConfig(cache_enabled=False, rerank_enabled=False, max_corrections=2)

    final_state, _, _ = await _run_graph(nodes, findings, verify_by_question=verify, config=config)

    assert final_state["failed_ids"].get("n1") is True
    assert "n1" not in final_state["verified_ids"]
    # n2 depends on n1, which never verified -> n2 never became ready.
    assert "n2" not in {f.node_id for f in final_state["findings"]}
    # the run still terminates and reaches synthesis over whatever exists.
    assert final_state["report"] is not None


# --------------------------------------------------------------------------
# Dependency-ordered synthesis
# --------------------------------------------------------------------------


def test_ordered_findings_respects_dependency_order_regardless_of_completion_order():
    plan = Plan(sub_questions=[
        SubQuestion(id="n1", question="q1", depends_on=[]),
        SubQuestion(id="n2", question="q2", depends_on=["n1"]),
        SubQuestion(id="n3", question="q3", depends_on=["n2"]),
    ])
    f1 = Finding(node_id="n1", question="q1", answer="a1", claims=[], confidence=0.9)
    f2 = Finding(node_id="n2", question="q2", answer="a2", claims=[], confidence=0.9)
    f3 = Finding(node_id="n3", question="q3", answer="a3", claims=[], confidence=0.9)
    # Deliberately out of dependency order -- simulates findings having
    # appended in whatever order async wave-completion happened to produce.
    out_of_order = [f3, f1, f2]

    ordered = _ordered_findings(plan, out_of_order)

    assert [f.node_id for f in ordered] == ["n1", "n2", "n3"]


def test_ordered_findings_uses_latest_attempt_for_retried_node():
    plan = Plan(sub_questions=[SubQuestion(id="n1", question="q1", depends_on=[])])
    attempt1 = Finding(node_id="n1", question="q1", answer="wrong guess", claims=[], confidence=0.1)
    attempt2 = Finding(node_id="n1", question="q1", answer="correct", claims=[], confidence=0.9)

    ordered = _ordered_findings(plan, [attempt1, attempt2])

    assert len(ordered) == 1
    assert ordered[0].answer == "correct"


# --------------------------------------------------------------------------
# Post-synthesis reflection loop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_spawns_followup_then_stops():
    nodes = [{"id": "n1", "question": "Q1", "depends_on": []}]
    findings = {
        "Q1": {"answer": "a1", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": []},
        "Followup question?": {
            "answer": "followup answer", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": [],
        },
    }
    reflection_seq = [
        {"has_gaps": True, "followup_questions": ["Followup question?"], "rationale": "stub: found a gap"},
        {"has_gaps": False, "followup_questions": [], "rationale": "stub: resolved"},
    ]

    final_state, _, _ = await _run_graph(nodes, findings, reflection_sequence=reflection_seq)

    assert final_state["reflection_iters"] == 2  # gap found (pass 1) + confirmed clean (pass 2)
    node_ids = {f.node_id for f in final_state["findings"]}
    assert "n1" in node_ids
    assert any(nid.startswith("followup_") for nid in node_ids)  # the follow-up node actually ran
    assert final_state["report"] is not None


@pytest.mark.asyncio
async def test_reflection_ceiling_stops_even_when_llm_never_converges():
    """reflection reports a gap on every call -- only max_reflect should stop
    it, never an unbounded loop."""
    nodes = [{"id": "n1", "question": "Q1", "depends_on": []}]
    findings = {
        "Q1": {"answer": "a1", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": []},
        "Followup question?": {
            "answer": "followup answer", "claims": [], "entities_extracted": {}, "confidence": 0.9, "open_gaps": [],
        },
    }
    reflection_seq = [{"has_gaps": True, "followup_questions": ["Followup question?"], "rationale": "stub: always finds a gap"}]
    config = RunConfig(cache_enabled=False, rerank_enabled=False, max_reflect=1)

    final_state, _, llm = await _run_graph(nodes, findings, reflection_sequence=reflection_seq, config=config)

    assert final_state["reflection_iters"] == 1  # ceiling hit after exactly one real pass
    assert llm.reflection_call_count == 1  # the 2nd would-be pass short-circuits before calling the LLM
    assert final_state["report"] is not None
