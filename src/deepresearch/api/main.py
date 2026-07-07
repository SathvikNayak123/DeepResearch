from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from deepresearch.agent.orchestrator import run_research
from deepresearch.backends import build_search_backend
from deepresearch.config import RunConfig
from deepresearch.schemas import RunResult
from deepresearch.telemetry.otel_setup import init_telemetry

app = FastAPI(title="DeepResearch")
init_telemetry()


class ResearchRequest(BaseModel):
    question: str
    config: dict | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Single shared demo key, not multi-tenant auth (docs/DESIGN.md non-goals).
    No-op if DEEPRESEARCH_API_KEY is unset, so local/dev/CI need no header."""
    expected = os.getenv("DEEPRESEARCH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


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
