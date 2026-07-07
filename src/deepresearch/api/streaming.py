from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query
from starlette.responses import StreamingResponse

from deepresearch.agent.orchestrator import run_research
from deepresearch.api.auth import require_api_key
from deepresearch.backends import build_search_backend
from deepresearch.config import RunConfig

router = APIRouter()


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/research/stream", dependencies=[Depends(require_api_key)])
async def research_stream(question: str = Query(...), config: str | None = Query(default=None)):
    """SSE progress on a live research run — GET, not POST, deliberately: the
    browser's native EventSource API only supports GET with no custom
    headers, so this is the one endpoint DEEPRESEARCH_API_KEY (if set) can't
    protect via header the way /research does. A no-op by default (unset),
    same as everywhere else in this API - not a new gap for the demo's
    single-shared-key threat model (docs/DESIGN.md non-goals: no
    multi-tenant auth), but worth stating plainly rather than silently.

    Emits one `stage_complete` event per orchestrator stage (plan, each
    worker, reflection, synthesis) as they finish, then one final `done`
    event with the full RunResult, or `run_error` if the run raised.
    Deliberately not `error` — that SSE event name is reserved by the
    browser's native EventSource, which routes it to the connection-level
    onerror handler instead of a custom event listener, not to application
    code.
    """
    async def event_generator():
        # Config parsing and backend construction happen inside the
        # generator, not before it — both can raise synchronously (a
        # malformed config JSON, a missing local_corpus_dir file), and if
        # they did outside this generator, FastAPI would turn that into a
        # raw 500 before StreamingResponse ever started, instead of the
        # graceful run_error event a client expecting SSE is actually
        # listening for. Confirmed live: a bad local_corpus_dir path 500'd
        # before this fix.
        try:
            run_config = RunConfig.from_overrides(json.loads(config) if config else None)
            backend = build_search_backend(run_config)
        except Exception as exc:
            yield _sse("run_error", {"type": "run_error", "message": str(exc)})
            return

        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await queue.put(event)

        async def _runner() -> None:
            try:
                result = await run_research(question, config=run_config, search_backend=backend, on_event=on_event)
                await queue.put({"type": "done", "result": result.model_dump()})
            except Exception as exc:  # the stream itself must not crash on an agent-side error
                await queue.put({"type": "run_error", "message": str(exc)})
            finally:
                await queue.put(None)  # sentinel: closes the generator below

        task = asyncio.create_task(_runner())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event.get("type", "message"), event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
