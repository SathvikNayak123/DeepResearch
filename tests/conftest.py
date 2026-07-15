from __future__ import annotations

import random

import pytest

from deepresearch.llm.client import LLMUsage


class StubLLM:
    """Deterministic in-process LLM double for unit tests only.

    Not used by eval.run_eval or any production/eval code path — those
    require a real LLMClient and a real provider key (no fake fallback).
    This stub exists purely so orchestrator control-flow tests (budget
    enforcement, persistence, event ordering) run fast, free, and offline,
    without exercising anything about model quality.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def _snippet_after(self, text: str, marker: str, length: int = 200) -> str:
        idx = text.find(marker)
        window = text[idx:] if idx != -1 else text
        if len(window) <= length:
            return window.strip() or "(no content available)"
        start = self._rng.randint(0, len(window) - length)
        return window[start : start + length].strip() or "(no content available)"

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
        props = schema.get("properties", {})
        usage = LLMUsage(input_tokens=max(len(user_content) // 4, 1), output_tokens=60, cost_usd=0.0)

        if "sub_questions" in props:
            data = {"sub_questions": [f"Background relevant to: {user_content[:120]}"]}
        elif "next_query" in props:
            n_done_notes = user_content.count("Query:")
            data = {
                "done": n_done_notes >= 2,
                "next_query": f"Follow-up relevant to: {user_content[:120]}",
                "rationale": "stub react step: bounded to 2 queries for comparability",
            }
        elif "claims" in props:
            snippet = self._snippet_after(user_content, "Sources:", length=150)
            data = {
                "claims": [{"text": snippet, "source_id": "src_1", "quote": snippet, "confidence": 0.5}],
                "open_gaps": [],
            }
        elif "coverage_score" in props:
            data = {
                "coverage_score": 0.9,
                "rationale": "stub judge: treating coverage as sufficient to keep the test run short",
                "should_replan": False,
                "new_sub_questions": [],
            }
        elif "cited_source_ids" in props:
            snippet = self._snippet_after(user_content, "Notes:", length=200)
            data = {"text": f"{snippet} [src_1]", "cited_source_ids": ["src_1"]}
        elif "correct" in props:
            data = {"correct": self._rng.random() < 0.7, "rationale": "stub judge — random verdict, not grounded"}
        elif "supported" in props:
            data = {"supported": self._rng.random() < 0.8, "rationale": "stub judge — random verdict, not grounded"}
        elif "short_answer" in props:
            snippet = self._snippet_after(user_content, "Report:", length=40)
            data = {"short_answer": snippet}
        else:
            raise ValueError(f"StubLLM doesn't recognize this schema shape: {schema}")

        return data, usage


@pytest.fixture
def make_stub_llm():
    """Factory fixture: make_stub_llm(seed=1) -> StubLLM(seed=1)."""
    return StubLLM
