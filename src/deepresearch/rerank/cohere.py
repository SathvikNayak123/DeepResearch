from __future__ import annotations

import os

import httpx

from deepresearch.rerank.base import RankedChunk, RerankBackend

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_MODEL = "rerank-v3.5"


class CohereRerankBackend(RerankBackend):
    """Hosted reranker option behind the same RerankBackend interface —
    proves the swap works, per docs/DESIGN.md row 7. Not the default: $2/1k
    searches, and Cohere's trial key isn't licensed for production use."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key or os.environ["COHERE_API_KEY"]
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, chunks: list[str]) -> list[RankedChunk]:
        if not chunks:
            return []
        resp = await self._client.post(
            COHERE_RERANK_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "query": query, "documents": chunks},
        )
        resp.raise_for_status()
        data = resp.json()
        return [RankedChunk(index=r["index"], score=r["relevance_score"]) for r in data["results"]]

    async def aclose(self) -> None:
        await self._client.aclose()
