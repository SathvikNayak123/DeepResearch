from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from deepresearch.agent.orchestrator import run_research
from deepresearch.api.auth import require_api_key
from deepresearch.backends import build_search_backend
from deepresearch.config import RunConfig
from deepresearch.schemas import RunResult
from deepresearch.telemetry.otel_setup import init_telemetry

app = FastAPI(title="DeepResearch")
init_telemetry()


class ResearchRequest(BaseModel):
    question: str
    config: dict | None = None


@app.post("/research", response_model=RunResult, dependencies=[Depends(require_api_key)])
async def research(request: ResearchRequest) -> RunResult:
    config = RunConfig.from_overrides(request.config)
    backend = build_search_backend(config)
    return await run_research(request.question, config=config, search_backend=backend)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Imported after `app` exists so these routers can register on it - both
# depend on api.auth.require_api_key, not on anything else in this module.
from deepresearch.api import routes_runs, streaming  # noqa: E402

app.include_router(streaming.router)
app.include_router(routes_runs.router)

# Resolved relative to this file, not cwd - works whether the process was
# launched from the repo root (local dev), /app (Docker, Dockerfile COPYs
# ui/ alongside src/), or a test runner's own directory. check_dir=False so
# a deployment that never shipped ui/ 404s on /ui/* instead of crashing the
# whole app at import time.
_UI_DIR = Path(__file__).resolve().parents[3] / "ui"
app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True, check_dir=False), name="ui")
