from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from deepresearch.api.auth import require_api_key
from deepresearch.config import RunConfig
from deepresearch.store import db

router = APIRouter()


@router.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
async def get_run(run_id: str) -> dict:
    """A run's Postgres row + its full trajectory + tool-call history —
    the queryable counterpart to a live SSE stream (streaming.py) or a
    Langfuse trace by the same run_id (docs/DESIGN.md: run_id = trace_id)."""
    config = RunConfig()
    run = await db.get_run(config.database_url, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    trajectories = await db.get_trajectories_for_run(config.database_url, run_id)
    tool_calls = await db.get_tool_calls_for_run(config.database_url, run_id)
    return {"run": run, "trajectories": trajectories, "tool_calls": tool_calls}
