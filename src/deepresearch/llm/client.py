from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

import anthropic
import instructor
import openai
from pydantic import BaseModel

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

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = COST_PER_MTOK_USD.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


class LLMClient:
    """Thin wrapper around Instructor for structured-output completions with
    token/cost accounting.

    Every call validates its response directly against a Pydantic model
    (`response_model=...`), not a hand-maintained JSON-schema dict parsed
    into a raw dict afterward — the Pydantic model is the single source of
    truth for both what's requested and what's returned, and Instructor
    retries automatically (re-prompting the model with the validation error)
    if the response doesn't validate on the first attempt.
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
        #   - "openrouter": OpenAI-compatible chat-completions API (genuinely
        #     different request/response shape from Anthropic's Messages API,
        #     handled via a separate Instructor client below), OPENROUTER_API_KEY.
        #     Lets DEEPRESEARCH_*_MODEL point at any OpenRouter-hosted model
        #     (e.g. "google/gemini-2.5-flash"), not just Claude.
        # Mode is pinned explicitly, not left at Instructor's default
        # (Mode.TOOLS): live-tested against google/gemini-2.5-flash via
        # OpenRouter on this project's actual nested Plan{sub_questions:
        # list[SubQuestion]} schema, Mode.TOOLS (and TOOLS_STRICT)
        # consistently flattened each SubQuestion into a plain string
        # instead of a nested object, failing validation across all retries
        # -- Mode.JSON_SCHEMA (the API's native structured-output/constrained
        # decoding, matching what this client used natively before Instructor)
        # produced correct nested objects every time. ANTHROPIC_JSON is the
        # equivalent choice for the Anthropic path, for the same reason, but
        # is not live-verified this session (Anthropic credit exhausted --
        # docs/RESULTS.md).
        self._provider = os.getenv("DEEPRESEARCH_LLM_PROVIDER", "anthropic")
        if self._provider == "bedrock":
            raw = anthropic.AsyncAnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
            self._client = instructor.from_anthropic(raw, mode=instructor.Mode.ANTHROPIC_JSON)
            self._openai_style = False
        elif self._provider == "openrouter":
            raw = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
            )
            self._client = instructor.from_openai(raw, mode=instructor.Mode.JSON_SCHEMA)
            self._openai_style = True
        else:
            self._client = instructor.from_anthropic(anthropic.AsyncAnthropic(), mode=instructor.Mode.ANTHROPIC_JSON)
            self._openai_style = False

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        user_content: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> tuple[T, LLMUsage]:
        if self._openai_style:
            result, completion = await self._client.chat.completions.create_with_completion(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
                response_model=response_model,
            )
            usage = LLMUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                cost_usd=_cost(model, completion.usage.prompt_tokens, completion.usage.completion_tokens),
            )
            return result, usage

        result, completion = await self._client.chat.completions.create_with_completion(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            response_model=response_model,
        )
        usage = LLMUsage(
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=_cost(model, completion.usage.input_tokens, completion.usage.output_tokens),
        )
        return result, usage
