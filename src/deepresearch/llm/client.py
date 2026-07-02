from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

# $ per million tokens (input, output). Verified current pricing —
# keep in sync with shared/models.md if these change.
COST_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


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
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
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
