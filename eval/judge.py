"""Judge economics — docs/DESIGN.md §5.5.

Cheap model by default (RunConfig.judge_model, "claude-haiku-4-5" unless
overridden — a deliberate exception to "always use the strongest model",
made for a task, grading, that doesn't need frontier capability). Verdicts
are cached keyed on (example, produced answer) so a repeat run against
unchanged content never re-pays judge cost — this session's brief,
task 5.
"""

from __future__ import annotations

import hashlib

from deepresearch.config import RunConfig
from deepresearch.llm.client import COST_PER_MTOK_USD, LLMClient
from deepresearch.store import db

from eval.prompts.loader import load_eval_prompt

ACCURACY_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["correct", "rationale"],
    "additionalProperties": False,
}

CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["supported", "rationale"],
    "additionalProperties": False,
}

# Planning estimate for cost-before-running printouts (task 5) — not a
# measured average. Refine once eval_scores/judge_cache rows give real usage
# to average over; this is deliberately conservative (rounds up).
ESTIMATED_TOKENS_PER_JUDGE_CALL = (400, 80)  # (input, output)


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def estimate_judge_cost_usd(n_calls: int, model: str) -> float:
    price_in, price_out = COST_PER_MTOK_USD.get(model, (0.0, 0.0))
    tokens_in, tokens_out = ESTIMATED_TOKENS_PER_JUDGE_CALL
    return n_calls * ((tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out)


class Judge:
    """One instance per eval run — tracks how many calls actually hit the
    LLM vs. the cache, and the real cost incurred, so run_eval.py can print
    both the up-front estimate and the actual figure at the end."""

    def __init__(self, llm: LLMClient, config: RunConfig) -> None:
        self._llm = llm
        self._config = config
        self.calls_made = 0
        self.cache_hits = 0
        self.total_cost_usd = 0.0

    async def judge_accuracy(self, *, question: str, gold_answer: str, predicted_answer: str) -> tuple[bool, str, bool]:
        """Returns (correct, rationale, was_cache_hit)."""
        cache_key = _cache_key(
            "accuracy", self._config.judge_rubric_version, question, gold_answer, predicted_answer
        )
        cached = await db.get_judge_cache(self._config.database_url, cache_key)
        if cached is not None:
            self.cache_hits += 1
            verdict = cached["verdict"]
            return verdict["correct"], verdict["rationale"], True

        system = load_eval_prompt("judge_accuracy_v1.txt")
        user_content = f"Question: {question}\nGold answer: {gold_answer}\nPredicted answer: {predicted_answer}"
        data, usage = await self._llm.complete_json(
            model=self._config.judge_model,
            system=system,
            user_content=user_content,
            schema=ACCURACY_SCHEMA,
            max_tokens=256,
        )
        self.calls_made += 1
        self.total_cost_usd += usage.cost_usd
        await db.set_judge_cache(
            self._config.database_url,
            cache_key=cache_key,
            verdict=data,
            judge_model=self._config.judge_model,
            rubric_version=self._config.judge_rubric_version,
        )
        return data["correct"], data["rationale"], False

    async def judge_citation(self, *, claim: str, quote: str) -> tuple[bool, str, bool]:
        """Returns (supported, rationale, was_cache_hit)."""
        cache_key = _cache_key("citation", self._config.judge_rubric_version, claim, quote)
        cached = await db.get_judge_cache(self._config.database_url, cache_key)
        if cached is not None:
            self.cache_hits += 1
            verdict = cached["verdict"]
            return verdict["supported"], verdict["rationale"], True

        system = load_eval_prompt("judge_citation_v1.txt")
        user_content = f"Claim: {claim}\nQuote: {quote}"
        data, usage = await self._llm.complete_json(
            model=self._config.judge_model,
            system=system,
            user_content=user_content,
            schema=CITATION_SCHEMA,
            max_tokens=256,
        )
        self.calls_made += 1
        self.total_cost_usd += usage.cost_usd
        await db.set_judge_cache(
            self._config.database_url,
            cache_key=cache_key,
            verdict=data,
            judge_model=self._config.judge_model,
            rubric_version=self._config.judge_rubric_version,
        )
        return data["supported"], data["rationale"], False
