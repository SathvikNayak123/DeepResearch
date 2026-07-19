"""ReAct subagent (agent/subagent.py): one plan node in -> one cited Finding
out, offline (stubbed chat model + stub finalize LLM, mirroring
test_react_agent.py's discipline -- no network, no real provider).
"""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage

from deepresearch.agent.subagent import FindingDraft, run_subagent
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import Claim, SubQuestion
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import init_telemetry

DOCS = [
    {"doc_id": "d1", "title": "Ratata", "text": "Ratata is a Swedish pop group fronted by Mauro Scocco."},
    {"doc_id": "d2", "title": "Mauro Scocco", "text": "Mauro Scocco was born on 11 September 1962 in Sweden."},
]


def _ai(tool_query: str | None = None, content: str = "") -> AIMessage:
    tool_calls = (
        [{"name": "search", "args": {"query": tool_query}, "id": "call_1", "type": "tool_call"}] if tool_query else []
    )
    msg = AIMessage(content=content, tool_calls=tool_calls)
    msg.usage_metadata = {"input_tokens": 12, "output_tokens": 6}
    return msg


class StubChatModel:
    def __init__(self, scripted: list[AIMessage]) -> None:
        self._scripted = scripted
        self._i = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        msg = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return msg


class StubFindingLLM:
    async def complete_structured(self, *, model, system, user_content, response_model, max_tokens=4096):
        usage = LLMUsage(input_tokens=8, output_tokens=8, cost_usd=0.0)
        assert response_model is FindingDraft
        return FindingDraft(
            answer="11 September 1962",
            claims=[
                Claim(
                    text="Mauro Scocco born 11 September 1962",
                    source_id="n1_1",
                    quote="born on 11 September 1962",
                    confidence=0.9,
                )
            ],
            entities_extracted={"date_of_birth": "1962-09-11"},
            confidence=0.9,
            open_gaps=[],
        ), usage


@pytest.mark.asyncio
async def test_run_subagent_returns_one_finding_with_entities_and_claims():
    init_telemetry()
    scripted = [_ai(tool_query="Mauro Scocco date of birth"), _ai(content="Found the date of birth")]
    chat_model = StubChatModel(scripted)
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    node = SubQuestion(id="n1", question="What is Mauro Scocco's date of birth?", depends_on=[])

    finding, usage = await run_subagent(
        node, "",
        config=config, chat_model=chat_model, llm=StubFindingLLM(), search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="r1"), run_span_id="span1",
        source_registry={}, started_monotonic=time.monotonic(),
    )

    assert finding.node_id == "n1"
    assert finding.answer == "11 September 1962"
    assert finding.entities_extracted == {"date_of_birth": "1962-09-11"}
    assert finding.confidence == 0.9
    assert finding.open_gaps == []
    assert finding.verified is False
    assert len(finding.claims) == 1
    assert finding.claims[0].source_id == "n1_1"
    assert usage.input_tokens > 0


@pytest.mark.asyncio
async def test_run_subagent_namespaces_source_ids_by_node_id():
    """Each subagent call gets a fresh registry namespaced by node.id, so the
    outer graph's dict-union reducer can merge multiple nodes' registries
    without id collisions (same discipline as the old parallel worker
    fan-out, agent/retrieve.py's source_id_prefix)."""
    init_telemetry()
    scripted = [_ai(tool_query="q"), _ai(content="answer")]
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    node = SubQuestion(id="n7", question="q?", depends_on=[])
    registry: dict = {}

    await run_subagent(
        node, "",
        config=config, chat_model=StubChatModel(scripted), llm=StubFindingLLM(), search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="r2"), run_span_id="span2",
        source_registry=registry, started_monotonic=time.monotonic(),
    )

    assert all(sid.startswith("src_n7_") for sid in registry)


@pytest.mark.asyncio
async def test_run_subagent_injects_context_facts_into_brief():
    """context_facts (built by the supervisor's facts_for() in Phase 3) must
    reach the agent's first message -- this is how a dependent hop's brief
    substitutes an upstream entity."""
    init_telemetry()

    class RecordingChatModel(StubChatModel):
        def __init__(self, scripted):
            super().__init__(scripted)
            self.seen_messages = None

        async def ainvoke(self, messages):
            self.seen_messages = messages
            return await super().ainvoke(messages)

    scripted = [_ai(content="answer")]
    chat_model = RecordingChatModel(scripted)
    config = RunConfig(cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    node = SubQuestion(id="n2", question="What year was <the singer> born?", depends_on=["n1"])

    await run_subagent(
        node, "singer = Mauro Scocco",
        config=config, chat_model=chat_model, llm=StubFindingLLM(), search_backend=backend,
        rerank_backend=None, recorder=RunRecorder(run_id="r3"), run_span_id="span3",
        source_registry={}, started_monotonic=time.monotonic(),
    )

    human_content = chat_model.seen_messages[1].content
    assert "singer = Mauro Scocco" in human_content
