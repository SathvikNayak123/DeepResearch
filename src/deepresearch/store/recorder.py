"""In-memory accumulator for a single run's trajectory/tool-call rows.

Stages and tool calls append to this during the run; the orchestrator flushes
it to the store in one batch at the end (db.bulk_insert_trajectories /
bulk_insert_tool_calls), so no DB round-trip sits on the hot path of every
LLM or tool call.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class RunRecorder:
    run_id: str
    trajectories: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)

    def record_trajectory(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        stage: str,
        name: str,
        input: dict | None,
        output: dict | None,
        tokens_in: int | None,
        tokens_out: int | None,
        cost_usd: float | None,
        latency_ms: float | None,
        started_at: dt.datetime | None,
        ended_at: dt.datetime | None,
    ) -> None:
        self.trajectories.append(
            {
                "run_id": self.run_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "stage": stage,
                "name": name,
                "input": input,
                "output": output,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )

    def record_tool_call(
        self,
        *,
        span_id: str,
        tool_name: str,
        args: dict | None,
        result_summary: dict | None,
        success: bool,
        cache_hit: bool,
        latency_ms: float | None,
    ) -> None:
        self.tool_calls.append(
            {
                "run_id": self.run_id,
                "span_id": span_id,
                "tool_name": tool_name,
                "args": args,
                "result_summary": result_summary,
                "success": success,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
            }
        )
