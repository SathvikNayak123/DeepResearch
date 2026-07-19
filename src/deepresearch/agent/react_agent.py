"""Shared ReAct tool-calling primitives, reused by agent/subagent.py's
per-node research loop (agent ⇄ tools(ToolNode[search, calculate])).

This module used to also own a standalone driver (a whole separate
plan-free "react_agent" topology with its own finalize/verify/self-correct
loop) — that has been superseded by the unified
planner→supervisor→subagent(this loop)→verify→synthesis graph
(agent/graph.py) and removed. What remains here is the reusable core: the
`search`/`calculate` tools, the tool-calling
`agent_node`, its router, and the LLM/context split that made it possible
(a LangChain ChatOpenAI at OpenRouter for tool-calling, because LangGraph's
prebuilt ToolNode needs LangChain-format tool_calls — llm/chat_model.py).
"""

from __future__ import annotations

import ast
import dataclasses
import operator
import time
from datetime import datetime, timezone

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from deepresearch.agent.budget import budget_status
from deepresearch.agent.retrieve import retrieve_chunks
from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig
from deepresearch.llm.chat_model import usage_from_message
from deepresearch.llm.client import LLMClient
from deepresearch.rerank.base import RerankBackend
from deepresearch.schemas import SourceRegistryEntry
from deepresearch.store.recorder import RunRecorder
from deepresearch.telemetry.otel_setup import current_span_id_hex, stage_span


@dataclasses.dataclass
class AgentContext:
    config: RunConfig
    chat_model: object  # ChatOpenAI already .bind_tools([search, calculate])
    llm: LLMClient
    search_backend: SearchBackend
    rerank_backend: RerankBackend | None
    recorder: RunRecorder
    run_span_id: str
    source_registry: dict[str, SourceRegistryEntry]
    # Namespaces source_id per subagent instance (agent/subagent.py: each plan
    # node gets a fresh source_registry fragment merged back into the shared
    # registry by a dict-union reducer) -- "" (default) preserves this
    # module's own single-shared-registry behavior unchanged.
    source_id_prefix: str = ""


# --------------------------------------------------------------------------
# Tools — read per-run deps from the injected ToolRuntime.context
# --------------------------------------------------------------------------


@tool
async def search(query: str, runtime: ToolRuntime) -> str:
    """Search the document corpus and return the most relevant passages, each
    labeled with a [src_id] you can cite. Call this once per fact you need;
    refine the query with entities you have already resolved."""
    ctx: AgentContext = runtime.context
    selected = await retrieve_chunks(
        query,
        search_backend=ctx.search_backend,
        config=ctx.config,
        source_registry=ctx.source_registry,
        rerank_backend=ctx.rerank_backend,
        recorder=ctx.recorder,
        # The currently active OTel span, not ctx.run_span_id (a static value
        # captured once at graph-construction time) -- tool_calls.span_id
        # must reference whichever stage span this call actually nests under
        # (agent/graph.py's "subagent:{node.id}" stage), or the trajectories
        # FK relationship (every tool_call's span_id names a real trajectory
        # row) breaks.
        parent_span_id=current_span_id_hex(),
        source_id_prefix=ctx.source_id_prefix,
    )
    if not selected:
        return "No results found for that query."
    return "\n\n".join(f"[{sid}] {title}\n{chunk[:1500]}" for sid, title, chunk in selected)


# Safe arithmetic/comparison evaluator for the `calculate` tool — an AST
# whitelist, never eval(). Covers what FRAMES numerical-reasoning questions
# need (differences, counts, comparisons over values gathered by search):
# 82% of FRAMES misses are computed answers no source states verbatim, so a
# search-only agent structurally can't produce them (docs/RESULTS.md Phase-2).
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_CMP_OPS = {
    ast.Lt: operator.lt, ast.Gt: operator.gt, ast.LtE: operator.le, ast.GtE: operator.ge,
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "int": int, "float": float}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_OPS:
        return _CMP_OPS[type(node.ops[0])](_safe_eval(node.left), _safe_eval(node.comparators[0]))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def evaluate_expression(expression: str) -> tuple[str, bool]:
    """Pure safe-eval (testable without the tool wrapper): returns (output, ok)."""
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
        return f"{expression} = {result}", True
    except Exception:
        return (
            f"Could not evaluate {expression!r}. Provide a plain arithmetic or comparison "
            "expression over numeric literals, e.g. '1985 - 1962' or 'max(3, 5)'."
        ), False


@tool
def calculate(expression: str, runtime: ToolRuntime) -> str:
    """Evaluate an exact arithmetic or comparison expression over numbers you
    have gathered (e.g. a difference of years, a count, which value is larger).
    Supports + - * / // % ** , comparisons (<, >, ==), and abs/round/min/max.
    Use this instead of computing multi-digit results yourself — e.g.
    calculate("1985 - 1962") or calculate("max(1972, 1968)")."""
    start = time.perf_counter()
    out, success = evaluate_expression(expression)
    ctx: AgentContext = runtime.context
    if ctx.recorder is not None:
        ctx.recorder.record_tool_call(
            span_id=current_span_id_hex(), tool_name="calculate", args={"expression": expression},
            result_summary={"result": out[:200]}, success=success, cache_hit=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    return out


# The agent's toolset — one source of truth so bind_tools (what the model
# sees) and ToolNode (what executes) can never drift apart. `search` gathers
# facts (multi-hop = repeated calls); `calculate` computes exact answers over
# them (FRAMES numerical reasoning). Both are general capabilities, not
# benchmark-specific.
TOOLS = [search, calculate]


# --------------------------------------------------------------------------
# Instrumentation helper (records a trajectory row + emits a stage_complete event)
# --------------------------------------------------------------------------


def _record(runtime, stage, name, *, input_summary, output, usage, latency_ms, start_dt, end_dt, span_id):
    ctx: AgentContext = runtime.context
    ctx.recorder.record_trajectory(
        span_id=span_id,
        parent_span_id=ctx.run_span_id,
        stage=stage,
        name=name,
        input=input_summary,
        output=output,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=latency_ms,
        started_at=start_dt,
        ended_at=end_dt,
    )
    runtime.stream_writer(
        {
            "type": "stage_complete",
            "stage": stage,
            "name": name,
            "tokens_in": usage.input_tokens,
            "tokens_out": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "latency_ms": latency_ms,
        }
    )


# --------------------------------------------------------------------------
# Node + router (the tool-calling ReAct step agent/subagent.py builds on)
# --------------------------------------------------------------------------


async def agent_node(state: dict, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    start_dt = datetime.now(timezone.utc)
    start = time.monotonic()
    with stage_span(f"agent_step:{state.get('step', 0)}"):
        span_id = current_span_id_hex()
        response = await ctx.chat_model.ainvoke(state["messages"])
    latency_ms = (time.monotonic() - start) * 1000
    usage = usage_from_message(response, ctx.config.worker_model)
    n_tools = len(getattr(response, "tool_calls", []) or [])
    _record(
        runtime, "agent_step", f"agent_step:{state.get('step', 0)}",
        input_summary={"n_messages": len(state["messages"])},
        output={"tool_calls": n_tools, "content": (response.content or "")[:500]},
        usage=usage, latency_ms=latency_ms, start_dt=start_dt, end_dt=datetime.now(timezone.utc), span_id=span_id,
    )
    return {
        "messages": [response],
        "step": 1,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def _evidence_block(messages: list) -> str:
    """The retrieved passages the agent saw — every ToolMessage's content,
    which already carries the [src_id] labels the finalize step must cite."""
    return "\n\n".join(m.content for m in messages if isinstance(m, ToolMessage) and m.content)


def _agent_route(state: dict, runtime: Runtime[AgentContext]) -> str:
    """ReAct branch: run the search/calculate tools while the model requests
    them, bounded by max_react_steps and the budget ceiling — else compile
    the answer. Returns the literal strings "tools"/"finalize"; callers map
    "finalize" to their own terminal node name via add_conditional_edges'
    path_map (agent/subagent.py maps it to "finalize_finding")."""
    last = state["messages"][-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))
    within_step_budget = state.get("step", 0) < runtime.context.config.max_react_steps
    budget_ok = budget_status(state, runtime.context.config.budget) is None
    if has_tool_calls and within_step_budget and budget_ok:
        return "tools"
    return "finalize"
