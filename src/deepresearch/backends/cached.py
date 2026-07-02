from __future__ import annotations

import json

from deepresearch.backends.base import SearchBackend
from deepresearch.cache.redis_cache import RedisCache
from deepresearch.schemas import CacheStats, FetchResult, SearchResult


class CachedSearchBackend(SearchBackend):
    """Redis in front of another SearchBackend — same interface either side,
    worker.py never knows whether it's talking to a cached or raw backend.

    Bypass: don't build this wrapper at all (config.cache_enabled = False /
    DEEPRESEARCH_CACHE_ENABLED=false) — the one flag eval runs use to force
    cold, per this session's brief.
    """

    def __init__(
        self,
        inner: SearchBackend,
        cache: RedisCache,
        *,
        search_ttl: int,
        fetch_ttl: int,
        search_cost_usd: float,
        fetch_cost_usd: float,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._search_ttl = search_ttl
        self._fetch_ttl = fetch_ttl
        self._search_cost_usd = search_cost_usd
        self._fetch_cost_usd = fetch_cost_usd
        self.stats = CacheStats()

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        cached = await self._cache.get_search(query)
        if cached is not None:
            self.stats.search_hits += 1
            self.stats.estimated_dollars_saved += self._search_cost_usd
            return [SearchResult.model_validate(r) for r in json.loads(cached)]

        self.stats.search_misses += 1
        results = await self._inner.search(query, max_results=max_results)
        await self._cache.set_search(query, json.dumps([r.model_dump() for r in results]), self._search_ttl)
        return results

    async def fetch(self, url: str) -> FetchResult:
        cached = await self._cache.get_fetch(url)
        if cached is not None:
            self.stats.fetch_hits += 1
            self.stats.estimated_dollars_saved += self._fetch_cost_usd
            return FetchResult.model_validate_json(cached)

        self.stats.fetch_misses += 1
        result = await self._inner.fetch(url)
        await self._cache.set_fetch(url, result.model_dump_json(), self._fetch_ttl)
        return result
