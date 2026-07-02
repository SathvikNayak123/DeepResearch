from __future__ import annotations

from fastapi import FastAPI, Response
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


@app.post("/research", response_model=RunResult)
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
