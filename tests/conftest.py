from __future__ import annotations

import random

import pytest

from deepresearch.agent.graph import GapCheck, HopVerdict
from deepresearch.agent.subagent import FindingDraft
from deepresearch.agent.synthesis import SynthesisDraft
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import Claim, Plan, SubQuestion


class StubLLM:
    """Deterministic in-process LLM double for unit tests only.

    Not used by eval.run_eval or any production/eval code path — those
    require a real LLMClient and a real provider key (no fake fallback).
    This stub exists purely so orchestrator control-flow tests (budget
    enforcement, persistence, event ordering) run fast, free, and offline,
    without exercising anything about model quality. Dispatches on
    `response_model` identity (matching agent/graph.py's/subagent.py's/
    synthesis.py's actual Pydantic response models), not a schema dict —
    this stub's job is to construct a valid instance of whatever model was
    requested, mirroring what Instructor's real validation would accept.
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

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        user_content: str,
        response_model: type,
        max_tokens: int = 4096,
    ):
        usage = LLMUsage(input_tokens=max(len(user_content) // 4, 1), output_tokens=60, cost_usd=0.0)

        if response_model is Plan:
            result = Plan(
                sub_questions=[
                    SubQuestion(id="n1", question=f"Background relevant to: {user_content[:120]}", depends_on=[])
                ]
            )
        elif response_model is FindingDraft:
            snippet = self._snippet_after(user_content, "Retrieved passages:", length=150)
            result = FindingDraft(
                answer=snippet,
                claims=[Claim(text=snippet, source_id="src_1", quote=snippet, confidence=0.5)],
                entities_extracted={},
                confidence=0.7,
                open_gaps=[],
            )
        elif response_model is HopVerdict:
            result = HopVerdict(grounded=True, reason="stub: always grounded")
        elif response_model is GapCheck:
            result = GapCheck(has_gaps=False, followup_questions=[], rationale="stub: no gaps")
        elif response_model is SynthesisDraft:
            snippet = self._snippet_after(user_content, "Notes:", length=200)
            result = SynthesisDraft(text=f"{snippet} [src_1]", cited_source_ids=["src_1"])
        else:
            raise ValueError(f"StubLLM doesn't recognize this response_model: {response_model}")

        return result, usage


@pytest.fixture
def make_stub_llm():
    """Factory fixture: make_stub_llm(seed=1) -> StubLLM(seed=1)."""
    return StubLLM
