"""Async CRUD helpers over the run-store schema (store/models.py).

Every function takes database_url explicitly rather than a global connection
— keeps this importable/testable without a process-wide singleton, and lets
eval runs and the live agent point at different databases if needed.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from deepresearch.store.models import ci_baselines, eval_scores, judge_cache, metadata, runs, tool_calls, trajectories

_engines: dict[str, AsyncEngine] = {}
_schema_initialized: set[str] = set()


def get_engine(database_url: str) -> AsyncEngine:
    if database_url not in _engines:
        _engines[database_url] = create_async_engine(database_url)
    return _engines[database_url]


async def init_schema(database_url: str) -> None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


async def ensure_schema(database_url: str) -> None:
    """init_schema, but only once per URL per process — every run calling
    this on every request would otherwise re-issue "does table exist"
    checks on every single run."""
    if database_url in _schema_initialized:
        return
    await init_schema(database_url)
    _schema_initialized.add(database_url)


async def create_run(database_url: str, *, run_id: str, benchmark_name: str | None, config: dict, git_sha: str, status: str) -> None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(
            runs.insert().values(
                run_id=run_id, benchmark_name=benchmark_name, config=config, git_sha=git_sha, status=status
            )
        )


async def finish_run(database_url: str, *, run_id: str, status: str, total_cost_usd: float, total_latency_ms: int) -> None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(
            runs.update()
            .where(runs.c.run_id == run_id)
            .values(status=status, total_cost_usd=total_cost_usd, total_latency_ms=total_latency_ms)
        )


async def bulk_insert_trajectories(database_url: str, rows: list[dict]) -> None:
    if not rows:
        return
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(trajectories.insert(), rows)


async def bulk_insert_tool_calls(database_url: str, rows: list[dict]) -> None:
    if not rows:
        return
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(tool_calls.insert(), rows)


async def bulk_insert_eval_scores(database_url: str, rows: list[dict]) -> None:
    if not rows:
        return
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(eval_scores.insert(), rows)


async def upsert_ci_baseline(
    database_url: str, *, benchmark_name: str, metric_name: str, baseline_value: float, config: dict, git_sha: str
) -> None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        await conn.execute(
            ci_baselines.insert().values(
                benchmark_name=benchmark_name,
                metric_name=metric_name,
                baseline_value=baseline_value,
                config=config,
                git_sha=git_sha,
            )
        )


async def get_latest_ci_baseline(database_url: str, *, benchmark_name: str, metric_name: str) -> dict | None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(
            select(ci_baselines)
            .where(ci_baselines.c.benchmark_name == benchmark_name, ci_baselines.c.metric_name == metric_name)
            # id (monotonic autoincrement), not created_at — two baselines
            # created within the same second would otherwise tie and return
            # an arbitrary row instead of the actually-latest one.
            .order_by(ci_baselines.c.id.desc())
            .limit(1)
        )
        row = result.first()
        return dict(row._mapping) if row else None


async def get_judge_cache(database_url: str, cache_key: str) -> dict | None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(judge_cache).where(judge_cache.c.cache_key == cache_key))
        row = result.first()
        return dict(row._mapping) if row else None


async def set_judge_cache(
    database_url: str, *, cache_key: str, verdict: dict, judge_model: str, rubric_version: str
) -> None:
    """Idempotent: a concurrent judge call racing to cache the same
    (example, answer) pair is a no-op, not an error."""
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert

        stmt = (
            dialect_insert(judge_cache)
            .values(cache_key=cache_key, verdict=verdict, judge_model=judge_model, rubric_version=rubric_version)
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        await conn.execute(stmt)


async def get_run(database_url: str, run_id: str) -> dict | None:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(runs).where(runs.c.run_id == run_id))
        row = result.first()
        return dict(row._mapping) if row else None


async def get_trajectories_for_run(database_url: str, run_id: str) -> list[dict]:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(
            select(trajectories).where(trajectories.c.run_id == run_id).order_by(trajectories.c.started_at)
        )
        return [dict(row._mapping) for row in result.fetchall()]


async def get_eval_scores_for_run(database_url: str, run_id: str) -> list[dict]:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(eval_scores).where(eval_scores.c.run_id == run_id))
        return [dict(row._mapping) for row in result.fetchall()]


async def get_tool_calls_for_run(database_url: str, run_id: str) -> list[dict]:
    engine = get_engine(database_url)
    async with engine.begin() as conn:
        result = await conn.execute(select(tool_calls).where(tool_calls.c.run_id == run_id))
        return [dict(row._mapping) for row in result.fetchall()]
