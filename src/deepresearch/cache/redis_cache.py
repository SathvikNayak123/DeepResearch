from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from prometheus_client import Counter

# Process-wide, scraped by Prometheus (docs/DESIGN.md decision row 8 + row 10).
# Per-run stats live separately on CachedSearchBackend.stats — these two are
# deliberately not the same counter: one is "since the process started" for
# Grafana, the other is "for this run" for the run record.
CACHE_HITS = Counter("deepresearch_cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("deepresearch_cache_misses_total", "Cache misses", ["cache_type"])


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))  # drop fragment


def _key(prefix: str, normalized: str) -> str:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"deepresearch:{prefix}:{digest}"


class RedisCache:
    """Raw key-value cache over a redis.asyncio-compatible client.

    Deliberately dumb: get/set on normalized-and-hashed keys, TTL'd. No
    reasoning state, no cross-run carryover — a cache, not agent memory
    (CLAUDE.md universal rules).
    """

    def __init__(self, client) -> None:
        self._client = client

    async def get_search(self, query: str) -> str | None:
        value = await self._client.get(_key("search", normalize_query(query)))
        CACHE_HITS.labels(cache_type="search").inc() if value is not None else CACHE_MISSES.labels(
            cache_type="search"
        ).inc()
        return value

    async def set_search(self, query: str, value: str, ttl: int) -> None:
        await self._client.set(_key("search", normalize_query(query)), value, ex=ttl)

    async def get_fetch(self, url: str) -> str | None:
        value = await self._client.get(_key("fetch", normalize_url(url)))
        CACHE_HITS.labels(cache_type="fetch").inc() if value is not None else CACHE_MISSES.labels(
            cache_type="fetch"
        ).inc()
        return value

    async def set_fetch(self, url: str, value: str, ttl: int) -> None:
        await self._client.set(_key("fetch", normalize_url(url)), value, ex=ttl)
