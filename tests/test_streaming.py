from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from deepresearch.api.main import app
from deepresearch.schemas import Plan, Report, RunResult, RunStatus


@pytest.fixture
def client():
    return TestClient(app)


def _read_sse_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_research_stream_bad_config_yields_run_error_not_a_500(client):
    """A bad local_corpus_dir path used to 500 before StreamingResponse ever
    started (config parsing/backend construction ran outside the generator).
    Confirmed live, fixed by moving both inside it - this pins the fix."""
    config = json.dumps({"search_backend": "local_corpus", "local_corpus_dir": "data/does/not/exist.json"})
    with client.stream("GET", "/research/stream", params={"question": "test", "config": config}) as response:
        assert response.status_code == 200  # the SSE connection itself succeeds
        events = _read_sse_events(response)

    assert len(events) == 1
    assert events[0]["type"] == "run_error"
    assert "does" in events[0]["message"] or "exist" in events[0]["message"]


def test_research_stream_malformed_config_json_yields_run_error(client):
    with client.stream("GET", "/research/stream", params={"question": "test", "config": "{not valid json"}) as response:
        assert response.status_code == 200
        events = _read_sse_events(response)

    assert len(events) == 1
    assert events[0]["type"] == "run_error"


def test_research_stream_success_path_emits_stage_events_then_done(client, monkeypatch):
    """Mocks run_research itself - this test is about streaming.py's own
    queue/generator plumbing (ordering, sentinel handling, SSE formatting),
    not the agent's internal behavior (covered separately in
    tests/test_orchestrator_persistence.py) - and needs no LLM/API key."""

    async def fake_run_research(question, *, config, search_backend, on_event=None, **kwargs):
        await on_event({"type": "run_started", "run_id": "fake-run-id", "question": question})
        await on_event({"type": "stage_complete", "stage": "plan", "name": "plan", "latency_ms": 1.0, "cost_usd": 0.0})
        await on_event(
            {"type": "stage_complete", "stage": "synthesis", "name": "synthesis", "latency_ms": 2.0, "cost_usd": 0.0}
        )
        return RunResult(
            run_id="fake-run-id",
            status=RunStatus.COMPLETED,
            question=question,
            plan=Plan(sub_questions=[]),
            worker_notes=[],
            reflections=[],
            report=Report(text="fake report", citations=[]),
        )

    monkeypatch.setattr("deepresearch.api.streaming.run_research", fake_run_research)

    with client.stream("GET", "/research/stream", params={"question": "test question"}) as response:
        assert response.status_code == 200
        events = _read_sse_events(response)

    types = [e["type"] for e in events]
    assert types == ["run_started", "stage_complete", "stage_complete", "done"]
    assert events[-1]["result"]["status"] == "completed"
    assert events[-1]["result"]["report"]["text"] == "fake report"


def test_research_stream_agent_exception_yields_run_error(client, monkeypatch):
    async def failing_run_research(*args, **kwargs):
        raise RuntimeError("simulated agent failure")

    monkeypatch.setattr("deepresearch.api.streaming.run_research", failing_run_research)

    with client.stream("GET", "/research/stream", params={"question": "test"}) as response:
        assert response.status_code == 200
        events = _read_sse_events(response)

    assert events[-1] == {"type": "run_error", "message": "simulated agent failure"}
