"""Deterministic-ish stand-in for LLMClient, used automatically by
eval/run_eval.py when ANTHROPIC_API_KEY isn't set — this sandbox has no live
Anthropic key (same constraint noted in earlier sessions for Tavily/Redis/
Postgres). Lets the eval harness's *mechanics* — scoring, run-store
persistence, judge caching, reliability variance — be verified end-to-end
without a live key.

It does NOT attempt to answer questions correctly. Where it can, it echoes
real snippets of the source text it was given (so downstream "does the
report contain the gold answer" checks have a genuine, if random, chance of
matching) but there is no reasoning happening. Treat every accuracy/F1/
reliability number produced against this client as harness validation, not
a real model baseline — see docs/RESULTS.md.
"""

from __future__ import annotations

import random

from deepresearch.llm.client import LLMUsage


class FakeLLMClient:
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
            # react.py's next_action step: stop after 2 steps so a fake-LLM
            # react run terminates in bounded, comparable time to plan_first's
            # fixed 2-4 sub-question plan, rather than always hitting
            # max_react_steps.
            n_done_notes = user_content.count("Query:")
            data = {
                "done": n_done_notes >= 2,
                "next_query": f"Follow-up relevant to: {user_content[:120]}",
                "rationale": "fake react step: bounded to 2 queries for comparability",
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
                "rationale": "fake judge: treating coverage as sufficient to keep the demo run short",
                "should_replan": False,
                "new_sub_questions": [],
            }
        elif "cited_source_ids" in props:
            snippet = self._snippet_after(user_content, "Notes:", length=200)
            data = {"text": f"{snippet} [src_1]", "cited_source_ids": ["src_1"]}
        elif "correct" in props:
            data = {"correct": self._rng.random() < 0.7, "rationale": "fake judge — random verdict, not grounded"}
        elif "supported" in props:
            data = {"supported": self._rng.random() < 0.8, "rationale": "fake judge — random verdict, not grounded"}
        else:
            raise ValueError(f"FakeLLMClient doesn't recognize this schema shape: {schema}")

        return data, usage
