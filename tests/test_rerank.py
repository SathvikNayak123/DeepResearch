from __future__ import annotations

import numpy as np
import pytest

from deepresearch.config import RunConfig
from deepresearch.rerank import build_rerank_backend
from deepresearch.rerank.bge import CrossEncoderRerankBackend
from deepresearch.rerank.bge_directml import DirectMLRerankBackend


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


def test_build_rerank_backend_is_cached_process_wide(monkeypatch):
    """build_rerank_backend must return the SAME instance across calls, not a
    fresh one each time -- a fresh instance per call means the ~600M-param
    model reloads from scratch every run_research() call (once per eval
    question, once per API request), and each instance's _inference_lock
    stops serializing CPU-bound rerank calls across concurrent runs, only
    within one run's own worker pool (see rerank/bge.py's docstring)."""
    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODEL", "test-cache-key-a")
    config = RunConfig(rerank_enabled=True, rerank_backend="bge")

    first = build_rerank_backend(config)
    second = build_rerank_backend(config)

    assert first is second


def test_build_rerank_backend_uses_separate_instance_per_model(monkeypatch):
    """A different DEEPRESEARCH_RERANK_MODEL must land in a different cache
    slot -- otherwise a test/ablation overriding the model between calls
    would silently get back a stale instance built for the wrong model."""
    config = RunConfig(rerank_enabled=True, rerank_backend="bge")

    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODEL", "test-cache-key-b")
    first = build_rerank_backend(config)

    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODEL", "test-cache-key-c")
    second = build_rerank_backend(config)

    assert first is not second


def test_build_rerank_backend_returns_none_when_disabled():
    config = RunConfig(rerank_enabled=False)
    assert build_rerank_backend(config) is None


class _FakeTokenizer:
    """Stands in for AutoTokenizer -- stashes the padded chunk list so
    _FakeSession can recover it (real tokenizer output loses the text,
    real inference recovers relevance from input_ids; here we just fake
    score = chunk length, same convention as CrossEncoderRerankBackend's
    tests above)."""

    def __init__(self) -> None:
        self.last_chunks: list[str] | None = None

    def __call__(self, queries, chunks, **kwargs):
        self.last_chunks = list(chunks)
        n = len(chunks)
        return {"input_ids": np.zeros((n, 1), dtype=np.int64), "attention_mask": np.ones((n, 1), dtype=np.int64)}


class _FakeOrtSession:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self._tokenizer = tokenizer

    def run(self, output_names, feed):
        chunks = self._tokenizer.last_chunks
        logits = np.array([float(len(c)) for c in chunks])
        return (logits,)


@pytest.mark.asyncio
async def test_directml_rerank_sorts_by_score_descending(monkeypatch):
    backend = DirectMLRerankBackend(model_name="fake", pad_to=5)
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(backend, "_load", lambda: (tokenizer, _FakeOrtSession(tokenizer)))

    ranked = await backend.rerank("query", ["a", "abc", "ab"])

    assert [rc.index for rc in ranked] == [1, 2, 0]


@pytest.mark.asyncio
async def test_directml_pads_short_batch_but_discards_padding_scores(monkeypatch):
    """pad_to=5 but only 3 real chunks -- the fixed-shape ONNX session must
    see a batch of 5 (padded), while rerank() only returns 3 RankedChunks:
    the 2 padding entries' scores must never leak into the result."""
    backend = DirectMLRerankBackend(model_name="fake", pad_to=5)
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(backend, "_load", lambda: (tokenizer, _FakeOrtSession(tokenizer)))

    ranked = await backend.rerank("query", ["a", "abc", "ab"])

    assert len(tokenizer.last_chunks) == 5  # session saw the padded batch
    assert len(ranked) == 3  # caller only sees the real chunks
    assert {rc.index for rc in ranked} == {0, 1, 2}


@pytest.mark.asyncio
async def test_directml_batches_chunks_over_pad_to(monkeypatch):
    """Real candidate counts routinely exceed pad_to (retrieve.py chunks each
    source into up to max_chunks_per_source pieces, so 20 sources can yield
    100+ candidate chunks -- confirmed live). rerank() must batch internally
    (multiple fixed-shape session.run calls) and return every chunk scored,
    correctly ordered across batch boundaries, not raise or drop any."""
    backend = DirectMLRerankBackend(model_name="fake", pad_to=2)
    tokenizer = _FakeTokenizer()
    session = _FakeOrtSession(tokenizer)
    run_call_count = 0
    real_run = session.run

    def counting_run(output_names, feed):
        nonlocal run_call_count
        run_call_count += 1
        return real_run(output_names, feed)

    session.run = counting_run
    monkeypatch.setattr(backend, "_load", lambda: (tokenizer, session))

    chunks = ["a", "abc", "ab", "abcde", "abcd"]  # 5 chunks, pad_to=2 -> 3 batches
    ranked = await backend.rerank("query", chunks)

    assert run_call_count == 3
    assert len(ranked) == 5
    assert [rc.index for rc in ranked] == [3, 4, 1, 2, 0]  # sorted by chunk length descending


@pytest.mark.asyncio
async def test_directml_rerank_empty_chunks_returns_empty():
    backend = DirectMLRerankBackend(model_name="fake")
    assert await backend.rerank("query", []) == []


def test_build_rerank_backend_selects_directml_backend(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODEL", "test-directml-cache-key")
    config = RunConfig(rerank_enabled=True, rerank_backend="bge_directml")

    backend = build_rerank_backend(config)

    assert isinstance(backend, DirectMLRerankBackend)
    assert build_rerank_backend(config) is backend  # cached, same instance
