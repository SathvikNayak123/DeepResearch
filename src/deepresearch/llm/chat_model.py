"""LangChain chat-model adapter for the tool-calling react_agent path only.

The rest of the system uses the httpx `LLMClient` (client.py) for structured
JSON completions with per-call cost accounting. The react_agent
(agent/react_agent.py) needs a `BaseChatModel` that emits LangChain-format
tool_calls so LangGraph's prebuilt `ToolNode` can execute them — which
`LLMClient` doesn't produce. This wires `ChatOpenAI` at OpenRouter (which is
OpenAI-compatible) for exactly that node, and maps its `usage_metadata` back
through the same `_cost()` table so run-store cost columns stay consistent
across both LLM paths.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from deepresearch.config import RunConfig
from deepresearch.llm.client import _cost, LLMUsage


def build_chat_model(config: RunConfig, *, model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """A ChatOpenAI bound to the configured provider endpoint. Defaults to
    OpenRouter (the project's real-run provider); the model defaults to the
    worker model unless overridden. Raises early with a clear message if the
    provider isn't the OpenAI-compatible one this adapter supports."""
    provider = os.getenv("DEEPRESEARCH_LLM_PROVIDER", "anthropic")
    if provider != "openrouter":
        raise RuntimeError(
            f"react_agent's ChatOpenAI adapter needs DEEPRESEARCH_LLM_PROVIDER=openrouter "
            f"(OpenAI-compatible tool-calling); got {provider!r}. Point the provider at "
            "OpenRouter, or extend chat_model.py for another OpenAI-compatible endpoint."
        )
    return ChatOpenAI(
        model=model or config.worker_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
    )


def usage_from_message(message, model: str) -> LLMUsage:
    """Map a LangChain AIMessage's usage_metadata to the project's LLMUsage +
    cost (same _cost table as client.py), so an agent-node LLM call is
    accounted identically to an LLMClient call. Missing usage (some providers
    omit it on tool-call turns) counts as zero rather than crashing."""
    meta = getattr(message, "usage_metadata", None) or {}
    tokens_in = int(meta.get("input_tokens", 0) or 0)
    tokens_out = int(meta.get("output_tokens", 0) or 0)
    return LLMUsage(input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=_cost(model, tokens_in, tokens_out))
