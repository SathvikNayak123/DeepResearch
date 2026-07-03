from __future__ import annotations

import base64
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

_initialized = False


def _default_otlp_endpoint() -> str | None:
    host = os.environ.get("LANGFUSE_HOST")
    return f"{host.rstrip('/')}/api/public/otel/v1/traces" if host else None


def _default_otlp_auth_header() -> str | None:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


def init_telemetry(service_name: str = "deepresearch") -> None:
    """Point OTel spans at Langfuse's OTLP endpoint, if configured.

    Safe to call more than once — only wires up the exporter the first time.
    If LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST aren't set,
    spans are still created (just not exported anywhere) so the agent works
    without Langfuse configured.
    """
    global _initialized
    if _initialized:
        return
    endpoint = os.environ.get("LANGFUSE_OTLP_ENDPOINT") or _default_otlp_endpoint()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        auth_header = os.environ.get("LANGFUSE_OTLP_AUTH_HEADER") or _default_otlp_auth_header()
        headers = {"Authorization": auth_header} if auth_header else {}
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer():
    return trace.get_tracer("deepresearch")


def run_root_context(run_id_hex: str):
    """Context whose OTel trace_id equals run_id_hex (32 hex chars).

    Lets a future Postgres run row and a Langfuse trace join on one id
    (docs/DESIGN.md sections 4 and 6). NOTE: propagation across truly
    concurrent asyncio tasks relies on Python copying contextvars into each
    Task at creation time — this is the behavior DESIGN.md flagged as an
    open risk to verify under real parallel load, not yet stress-tested
    beyond this thin slice.
    """
    trace_id = int(run_id_hex, 16)
    span_id = RandomIdGenerator().generate_span_id()
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(span_context))


@contextmanager
def stage_span(name: str, context=None, **attributes):
    tracer = get_tracer()
    with tracer.start_as_current_span(name, context=context) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def current_span_id_hex() -> str:
    """16-hex-char OTel span id of whatever span is currently active — used
    to tag run-store rows (trajectories.span_id, tool_calls.span_id) with
    the same id Langfuse shows, so a DB row and a trace span are the same
    lookup either direction."""
    span_id = trace.get_current_span().get_span_context().span_id
    return format(span_id, "016x")
