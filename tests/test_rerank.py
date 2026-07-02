from __future__ import annotations

import pytest

from deepresearch.rerank.bge import CrossEncoderRerankBackend


class _FakeModel:
    """Stands in for the real cross-encoder — score = chunk length — so the
    test verifies rerank()'s sorting/indexing logic without downloading
    bge-reranker-v2-m3 (that's what scripts/rerank_ablation.py does, against
    real data)."""

    def predict(self, pairs):
        return [len(chunk) for _, chunk in pairs]


@pytest.mark.asyncio
async def test_rerank_sorts_by_score_descending(monkeypatch):
    backend = CrossEncoderRerankBackend(model_name="fake")
    monkeypatch.setattr(backend, "_load", lambda: _FakeModel())

    ranked = await backend.rerank("query", ["a", "abc", "ab"])

    assert [rc.index for rc in ranked] == [1, 2, 0]
    assert ranked[0].score == 3.0


@pytest.mark.asyncio
async def test_rerank_empty_chunks_returns_empty():
    backend = CrossEncoderRerankBackend(model_name="fake")
    assert await backend.rerank("query", []) == []
