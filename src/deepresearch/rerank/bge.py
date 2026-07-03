from __future__ import annotations

import asyncio
import os

from deepresearch.rerank.base import RankedChunk, RerankBackend

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderRerankBackend(RerankBackend):
    """Self-hosted cross-encoder reranker — default per docs/DESIGN.md row 7.

    Free and unlimited to run in CI/ablations (no rate limits, no per-call
    cost) at the price of CPU latency. Model name is overridable via
    DEEPRESEARCH_RERANK_MODEL so tests/ablations can swap in a smaller
    cross-encoder without touching the interface.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("DEEPRESEARCH_RERANK_MODEL", DEFAULT_MODEL)
        self._model = None  # lazy-loaded on first use, not at construction

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[str]) -> list[RankedChunk]:
        if not chunks:
            return []
        model = await asyncio.to_thread(self._load)
        pairs = [(query, chunk) for chunk in chunks]
        scores = await asyncio.to_thread(model.predict, pairs)
        return sorted(
            (RankedChunk(index=i, score=float(s)) for i, s in enumerate(scores)),
            key=lambda rc: rc.score,
            reverse=True,
        )
