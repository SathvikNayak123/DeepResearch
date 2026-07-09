from __future__ import annotations

import pytest

from deepresearch.store import db


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


@pytest.mark.asyncio
async def test_create_and_finish_run(db_url):
    await db.init_schema(db_url)
    run_id = "a" * 32
    await db.create_run(db_url, run_id=run_id, benchmark_name="musique", config={"x": 1}, git_sha="abc", status="running")
    await db.finish_run(db_url, run_id=run_id, status="completed", total_cost_usd=0.05, total_latency_ms=100)

    scores = await db.get_eval_scores_for_run(db_url, run_id)
    assert scores == []  # nothing inserted yet, but the query itself must not error


@pytest.mark.asyncio
async def test_bulk_insert_and_query_eval_scores(db_url):
    await db.init_schema(db_url)
    run_id = "b" * 32
    await db.create_run(db_url, run_id=run_id, benchmark_name="frames", config={}, git_sha="abc", status="running")
    await db.bulk_insert_eval_scores(
        db_url,
        [
            {
                "run_id": run_id,
                "benchmark_name": "frames",
                "question_id": "q1",
                "metric_name": "accuracy",
                "value": 1.0,
                "judge_model": "claude-haiku-4-5",
                "rubric_version": "v1",
                "raw_judge_output": {"rationale": "ok"},
            }
        ],
    )
    scores = await db.get_eval_scores_for_run(db_url, run_id)
    assert len(scores) == 1
    assert scores[0]["metric_name"] == "accuracy"
    assert float(scores[0]["value"]) == 1.0


@pytest.mark.asyncio
async def test_bulk_insert_empty_list_is_a_noop(db_url):
    await db.init_schema(db_url)
    await db.bulk_insert_trajectories(db_url, [])
    await db.bulk_insert_tool_calls(db_url, [])
    await db.bulk_insert_eval_scores(db_url, [])  # must not raise


@pytest.mark.asyncio
async def test_ci_baseline_roundtrip_gets_latest(db_url):
    await db.init_schema(db_url)
    await db.upsert_ci_baseline(
        db_url, benchmark_name="musique", metric_name="answer_f1", baseline_value=0.5, config={}, git_sha="v1"
    )
    await db.upsert_ci_baseline(
        db_url, benchmark_name="musique", metric_name="answer_f1", baseline_value=0.6, config={}, git_sha="v2"
    )
    latest = await db.get_latest_ci_baseline(db_url, benchmark_name="musique", metric_name="answer_f1")
    assert float(latest["baseline_value"]) == 0.6
    assert latest["git_sha"] == "v2"


@pytest.mark.asyncio
async def test_get_run_returns_none_for_missing_and_data_for_existing(db_url):
    await db.init_schema(db_url)
    assert await db.get_run(db_url, "does-not-exist") is None

    run_id = "c" * 32
    await db.create_run(db_url, run_id=run_id, benchmark_name="musique", config={"x": 1}, git_sha="abc", status="running")
    row = await db.get_run(db_url, run_id)
    assert row["run_id"] == run_id
    assert row["status"] == "running"
    assert row["benchmark_name"] == "musique"


@pytest.mark.asyncio
async def test_get_trajectories_for_run_orders_by_started_at(db_url):
    from datetime import datetime, timedelta, timezone

    await db.init_schema(db_url)
    run_id = "d" * 32
    await db.create_run(db_url, run_id=run_id, benchmark_name="musique", config={}, git_sha="abc", status="running")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db.bulk_insert_trajectories(
        db_url,
        [
            {
                "run_id": run_id, "span_id": "s2", "parent_span_id": None, "stage": "synthesis", "name": "synthesis",
                "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_ms": 1,
                "started_at": base + timedelta(seconds=10), "ended_at": base + timedelta(seconds=11),
            },
            {
                "run_id": run_id, "span_id": "s1", "parent_span_id": None, "stage": "plan", "name": "plan",
                "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_ms": 1,
                "started_at": base, "ended_at": base + timedelta(seconds=1),
            },
        ],
    )
    rows = await db.get_trajectories_for_run(db_url, run_id)
    assert [r["stage"] for r in rows] == ["plan", "synthesis"]  # ordered by started_at, not insert order

    assert await db.get_trajectories_for_run(db_url, "does-not-exist") == []


@pytest.mark.asyncio
async def test_judge_cache_is_idempotent_on_race(db_url):
    await db.init_schema(db_url)
    await db.set_judge_cache(db_url, cache_key="k1", verdict={"correct": True}, judge_model="m", rubric_version="v1")
    await db.set_judge_cache(db_url, cache_key="k1", verdict={"correct": False}, judge_model="m", rubric_version="v1")
    cached = await db.get_judge_cache(db_url, "k1")
    assert cached["verdict"]["correct"] is True  # first write wins, second is a no-op

    missing = await db.get_judge_cache(db_url, "does-not-exist")
    assert missing is None
