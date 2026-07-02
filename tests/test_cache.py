from __future__ import annotations

import pytest
from fakeredis import aioredis as fakeredis_aio

from deepresearch.backends.cached import CachedSearchBackend
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.cache.redis_cache import RedisCache, normalize_query, normalize_url
from deepresearch.schemas import FetchResult, SearchResult


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  Who   Won\tthe World Cup? ") == "who won the world cup?"


def test_normalize_url_drops_fragment_trailing_slash_and_case():
    a = normalize_url("HTTPS://Example.com/Path/")
    b = normalize_url("https://example.com/Path#section")
    assert a == "https://example.com/Path"  # scheme/host lowercased, path casing preserved
    assert a == b  # trailing slash and fragment don't produce different keys
    assert "#" not in b


class _FakeInnerBackend:
    """Records how many times it's actually called, so tests can assert the
    cache — not the inner backend — served a repeat request."""

    def __init__(self) -> None:
        self.search_calls = 0
        self.fetch_calls = 0

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.search_calls += 1
        return [SearchResult(url="https://example.com/a", title="A", snippet="snippet", score=1.0)]

    async def fetch(self, url: str) -> FetchResult:
        self.fetch_calls += 1
        return FetchResult(url=url, content=f"content-for-{url}")


@pytest.fixture
def redis_client():
    return fakeredis_aio.FakeRedis(decode_responses=True)


@pytest.mark.asyncio
async def test_cache_hit_avoids_calling_inner_backend(redis_client):
    inner = _FakeInnerBackend()
    cache = RedisCache(redis_client)
    backend = CachedSearchBackend(
        inner, cache, search_ttl=60, fetch_ttl=60, search_cost_usd=0.01, fetch_cost_usd=0.002
    )

    results1 = await backend.search("what is the capital of France")
    results2 = await backend.search("  What IS the capital of France  ")  # normalizes the same

    assert inner.search_calls == 1  # second call was a cache hit, not a real search
    assert results1 == results2
    assert backend.stats.search_hits == 1
    assert backend.stats.search_misses == 1
    assert backend.stats.estimated_dollars_saved == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_fetch_cache_hit_avoids_refetching(redis_client):
    inner = _FakeInnerBackend()
    cache = RedisCache(redis_client)
    backend = CachedSearchBackend(
        inner, cache, search_ttl=60, fetch_ttl=60, search_cost_usd=0.01, fetch_cost_usd=0.002
    )

    r1 = await backend.fetch("https://example.com/page/")
    r2 = await backend.fetch("https://Example.com/page#anchor")  # same normalized URL

    assert inner.fetch_calls == 1
    assert r1.content == r2.content
    assert backend.stats.fetch_hits == 1
    assert backend.stats.fetch_misses == 1
    assert backend.stats.estimated_dollars_saved == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_distinct_queries_are_both_misses(redis_client):
    inner = _FakeInnerBackend()
    cache = RedisCache(redis_client)
    backend = CachedSearchBackend(
        inner, cache, search_ttl=60, fetch_ttl=60, search_cost_usd=0.01, fetch_cost_usd=0.002
    )

    await backend.search("question one")
    await backend.search("question two")

    assert inner.search_calls == 2
    assert backend.stats.search_misses == 2
    assert backend.stats.search_hits == 0


def test_cache_bypass_returns_uncached_backend_untouched(tmp_path):
    import json

    from deepresearch.backends import build_search_backend
    from deepresearch.config import RunConfig

    corpus_path = tmp_path / "q1.json"
    corpus_path.write_text(json.dumps([{"doc_id": "d1", "title": "T", "text": "some text"}]))

    cfg = RunConfig.from_overrides(
        {"cache_enabled": False, "search_backend": "local_corpus", "local_corpus_dir": str(corpus_path)}
    )
    backend = build_search_backend(cfg)

    assert isinstance(backend, LocalCorpusBackend)
    assert not hasattr(backend, "stats")  # no cache wrapper present at all
