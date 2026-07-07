from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic
import httpx

# $ per million tokens (input, output). Verified current pricing —
# keep in sync with shared/models.md if these change.
COST_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenRouter model IDs (provider/model) - non-Anthropic, priced per
    # openrouter.ai/<id>, verified 2026-07 (docs/RESULTS.md if this is
    # promoted to a real ablation branch rather than a one-off cost swap).
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
}

# claude-haiku-4-5 rejects output_config.effort outright ("This model does
# not support the effort parameter") - confirmed live on the deployed stack,
# 2026-07-04. Everything else in COST_PER_MTOK_USD supports it.
MODELS_WITHOUT_EFFORT_SUPPORT = {"claude-haiku-4-5"}


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = COST_PER_MTOK_USD.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


class LLMClient:
    """Thin wrapper around the Anthropic SDK with token/cost accounting.

    Every call returns structured JSON via output_config.format, since the
    agent's claim->source citation mapping needs to be machine-checkable,
    not parsed out of free-form prose.
    """

    def __init__(self) -> None:
        # DEEPRESEARCH_LLM_PROVIDER selects the backend; model family/pricing
        # is orthogonal (set via DEEPRESEARCH_*_MODEL env vars either way):
        #   - "anthropic" (default): direct Anthropic Messages API, ANTHROPIC_API_KEY.
        #   - "bedrock": same Messages API shape, routed through AWS (SigV4 via
        #     boto3, AWS_REGION/credential chain) instead of a stored API key.
        #     Requires `pip install "anthropic[bedrock]"` and DEEPRESEARCH_*_MODEL
        #     set to the Bedrock model ID from the Bedrock model catalog (not the
        #     direct-API name), once that model is invokable in that account/region.
        #   - "openrouter": OpenAI-compatible chat-completions API (not Anthropic's
        #     Messages API - genuinely different request/response shape, handled
        #     separately in complete_json), OPENROUTER_API_KEY. Lets DEEPRESEARCH_*_MODEL
        #     point at any OpenRouter-hosted model (e.g. "google/gemini-2.5-flash"),
        #     not just Claude - a real provider swap, not just a billing-route swap.
        self._provider = os.getenv("DEEPRESEARCH_LLM_PROVIDER", "anthropic")
        if self._provider == "bedrock":
            self._client = anthropic.AsyncAnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
        elif self._provider == "openrouter":
            self._client = httpx.AsyncClient(
                base_url="https://openrouter.ai/api/v1",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                timeout=120.0,
            )
        else:
            self._client = anthropic.AsyncAnthropic()

    async def complete_json(
        self,
        *,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        max_tokens: int = 4096,
        effort: str = "medium",
    ) -> tuple[dict, LLMUsage]:
        if self._provider == "openrouter":
            return await self._complete_json_openrouter(
                model=model, system=system, user_content=user_content, schema=schema, max_tokens=max_tokens
            )

        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if model not in MODELS_WITHOUT_EFFORT_SUPPORT:
            output_config["effort"] = effort

        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            output_config=output_config,
            messages=[{"role": "user", "content": user_content}],
        )
        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=_cost(model, response.usage.input_tokens, response.usage.output_tokens),
        )
        return data, usage

    async def _complete_json_openrouter(
        self, *, model: str, system: str, user_content: str, schema: dict, max_tokens: int
    ) -> tuple[dict, LLMUsage]:
        """OpenAI-compatible chat-completions shape - no "effort" knob (that's
        an Anthropic-specific parameter with no OpenRouter/OpenAI equivalent)."""
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        data = json.loads(text)
        usage_in = body["usage"]["prompt_tokens"]
        usage_out = body["usage"]["completion_tokens"]
        usage = LLMUsage(input_tokens=usage_in, output_tokens=usage_out, cost_usd=_cost(model, usage_in, usage_out))
        return data, usage
