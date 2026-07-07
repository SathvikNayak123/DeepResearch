from __future__ import annotations

from deepresearch.backends.base import SearchBackend
from deepresearch.config import RunConfig


def build_search_backend(config: RunConfig) -> SearchBackend:
    """Single place backend construction happens — API, CLI, and scripts all
    call this so the cache-bypass flag (config.cache_enabled) behaves the
    same everywhere instead of being reimplemented per call site."""
    if config.search_backend == "local_corpus":
        from deepresearch.backends.local_corpus import LocalCorpusBackend

        if not config.local_corpus_dir:
            raise ValueError("search_backend='local_corpus' needs config.local_corpus_dir set to a corpus JSON file")
        inner: SearchBackend = LocalCorpusBackend.from_json_file(config.local_corpus_dir)
    else:
        from deepresearch.backends.tavily import TavilyBackend

        inner = TavilyBackend()

    if not config.cache_enabled:
        return inner

    import redis.asyncio as redis_asyncio

    from deepresearch.backends.cached import CachedSearchBackend
    from deepresearch.cache.redis_cache import RedisCache

    # protocol=2 (RESP2): redis-py 8.x defaults to negotiating RESP3 via a
    # HELLO command on connect, which redis:7-alpine (docker-compose's
    # `redis` service) rejects with "unknown command 'HELLO'" - confirmed
    # live, first real-Redis exercise of this path (prior sessions had no
    # Docker daemon, so this default-cache_enabled=true path never actually
    # ran against a real Redis until now). RedisCache only does get/set/ex -
    # no RESP3-only feature is needed here.
    client = redis_asyncio.from_url(config.redis_url, decode_responses=True, protocol=2)
    return CachedSearchBackend(
        inner,
        RedisCache(client),
        search_ttl=config.search_cache_ttl_seconds,
        fetch_ttl=config.fetch_cache_ttl_seconds,
        search_cost_usd=config.search_cost_usd,
        fetch_cost_usd=config.fetch_cost_usd,
    )
