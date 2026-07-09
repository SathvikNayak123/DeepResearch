from __future__ import annotations

import pytest

from deepresearch.agent.orchestrator import run_research
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.schemas import RunStatus
from deepresearch.store import db
from deepresearch.telemetry.otel_setup import init_telemetry

from eval.fake_llm import FakeLLMClient

DOCS = [
    {"doc_id": "d1", "title": "Paris", "text": "Paris is the capital of France and its largest city."},
    {"doc_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany, known for its history."},
]


@pytest.mark.asyncio
async def test_run_research_persists_runs_trajectories_and_tool_calls(tmp_path):
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_test.db'}"

    config = RunConfig(
        database_url=db_url,
        cache_enabled=False,
        rerank_enabled=False,  # keep it fast — reranker model isn't the point of this test
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = FakeLLMClient(seed=1)

    result = await run_research("What is the capital of France?", config=config, search_backend=backend, llm=llm, benchmark_name="test")

    assert result.status == RunStatus.COMPLETED

    from sqlalchemy import select
    from deepresearch.store.models import runs, trajectories, tool_calls

    engine = db.get_engine(db_url)
    async with engine.begin() as conn:
        run_rows = (await conn.execute(select(runs).where(runs.c.run_id == result.run_id))).fetchall()
        traj_rows = (await conn.execute(select(trajectories).where(trajectories.c.run_id == result.run_id))).fetchall()
        tool_rows = (await conn.execute(select(tool_calls).where(tool_calls.c.run_id == result.run_id))).fetchall()

    assert len(run_rows) == 1
    assert run_rows[0]._mapping["status"] == "completed"
    assert run_rows[0]._mapping["benchmark_name"] == "test"

    stages = {row._mapping["stage"] for row in traj_rows}
    assert stages == {"plan", "worker", "reflection", "synthesis"}

    tool_names = {row._mapping["tool_name"] for row in tool_rows}
    assert "search" in tool_names
    assert "fetch" in tool_names
    # every tool_call's span_id must reference a real trajectory span_id (FK)
    traj_span_ids = {row._mapping["span_id"] for row in traj_rows}
    assert all(row._mapping["span_id"] in traj_span_ids for row in tool_rows)


@pytest.mark.asyncio
async def test_run_research_react_mode_persists_react_steps(tmp_path):
    """docs/DESIGN.md decision row 2 alternative: no upfront "plan" stage,
    "react_step" stages instead, bounded by max_react_steps."""
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_react_test.db'}"

    config = RunConfig(
        database_url=db_url,
        cache_enabled=False,
        rerank_enabled=False,
        planning_style="react",
        max_react_steps=3,
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = FakeLLMClient(seed=1)

    result = await run_research(
        "What is the capital of France?", config=config, search_backend=backend, llm=llm, benchmark_name="test"
    )

    assert result.status == RunStatus.COMPLETED
    assert result.iterations <= config.max_react_steps

    from sqlalchemy import select
    from deepresearch.store.models import trajectories

    engine = db.get_engine(db_url)
    async with engine.begin() as conn:
        traj_rows = (await conn.execute(select(trajectories).where(trajectories.c.run_id == result.run_id))).fetchall()

    stages = {row._mapping["stage"] for row in traj_rows}
    assert "plan" not in stages
    assert "reflection" not in stages
    assert "react_step" in stages
    assert "worker" in stages
    assert "synthesis" in stages


@pytest.mark.asyncio
async def test_run_research_on_event_fires_for_every_stage(tmp_path):
    """The SSE streaming endpoint (api/streaming.py) has no other hook into
    run_research()'s progress — on_event must fire once per _call_stage,
    in order, with the stages a plan-first run actually produces."""
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_events_test.db'}"
    config = RunConfig(database_url=db_url, cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = FakeLLMClient(seed=1)

    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    result = await run_research(
        "What is the capital of France?", config=config, search_backend=backend, llm=llm, on_event=on_event
    )

    assert result.status == RunStatus.COMPLETED
    assert events[0] == {"type": "run_started", "run_id": result.run_id, "question": "What is the capital of France?"}

    stage_events = [e for e in events[1:] if e["type"] == "stage_complete"]
    assert len(stage_events) >= 3  # plan, >=1 worker, reflection, synthesis
    stages_in_order = [e["stage"] for e in stage_events]
    assert stages_in_order[0] == "plan"
    assert stages_in_order[-1] == "synthesis"
    assert "worker" in stages_in_order
    assert all(isinstance(e["latency_ms"], (int, float)) for e in stage_events)


@pytest.mark.asyncio
async def test_run_research_bypass_cache_flag_means_no_stats(tmp_path):
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_test2.db'}"
    config = RunConfig(database_url=db_url, cache_enabled=False, rerank_enabled=False)
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = FakeLLMClient(seed=2)

    result = await run_research("Capital of Germany?", config=config, search_backend=backend, llm=llm)

    assert result.cache_stats.hit_rate == 0.0
    assert result.cache_stats.estimated_dollars_saved == 0.0
