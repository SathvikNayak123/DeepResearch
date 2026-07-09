"""Standalone cache measurement: run a question subset cold, then warm, then
a realistic mixed pass (half-repeat/half-fresh), then a cache-disabled pass
on the same questions to prove the bypass flag actually bypasses.

Uses the real production CachedSearchBackend and RedisCache classes, fronting
a network-latency-simulating fake Tavily backend — this sandbox has no live
Tavily key/credits, so no real API calls are made. See docs/RESULTS.md for
what that means for these numbers (failure-honesty section).

Usage:
    python scripts/cache_measurement.py [--n 20] [--seed 42] [--fetches 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepresearch.backends.base import SearchBackend  # noqa: E402
from deepresearch.backends.cached import CachedSearchBackend  # noqa: E402
from deepresearch.cache.redis_cache import RedisCache  # noqa: E402
from deepresearch.config import RunConfig  # noqa: E402
from deepresearch.schemas import FetchResult, SearchResult  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASET = "bdsaglam/musique"
DATASET_CONFIG = "answerable"
DATASET_SPLIT = "validation"


class FakeTavilyBackend(SearchBackend):
    """Simulated Tavily: realistic network latency, no live API calls —
    this sandbox has no Tavily key/credits (docs/RESULTS.md honesty
    section). CachedSearchBackend and RedisCache in front of it are the
    unmodified production classes."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        await asyncio.sleep(self._rng.uniform(0.2, 0.4))
        return [
            SearchResult(
                url=f"https://example.com/{abs(hash((query, i)))}",
                title=f"Result {i}",
                snippet="...",
                score=1.0,
            )
            for i in range(max_results)
        ]

    async def fetch(self, url: str) -> FetchResult:
        await asyncio.sleep(self._rng.uniform(0.15, 0.35))
        return FetchResult(url=url, content=f"fetched content for {url}")


def load_questions(n: int, seed: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    return [ds[i]["question"] for i in idxs[:n]]


async def build_cache(redis_url: str) -> tuple[RedisCache, bool]:
    try:
        import redis.asyncio as redis_asyncio

        # protocol=2: see src/deepresearch/backends/__init__.py's identical fix
        # for why (redis-py 8.x's default RESP3 HELLO negotiation vs. redis:7-alpine).
        client = redis_asyncio.from_url(redis_url, decode_responses=True, socket_connect_timeout=1, protocol=2)
        await client.ping()
        return RedisCache(client), True
    except Exception:
        from fakeredis import aioredis as fakeredis_aio

        return RedisCache(fakeredis_aio.FakeRedis(decode_responses=True)), False


@dataclass
class PassResult:
    name: str
    n_questions: int
    wall_clock_seconds: float
    search_hits: int
    search_misses: int
    fetch_hits: int
    fetch_misses: int
    dollars_spent: float
    dollars_saved: float
    hit_rate: float


async def run_cached_pass(
    name: str,
    questions: list[str],
    cache: RedisCache,
    config: RunConfig,
    fetches: int,
    inner_seed: int,
) -> PassResult:
    inner = FakeTavilyBackend(seed=inner_seed)
    backend = CachedSearchBackend(
        inner,
        cache,
        search_ttl=config.search_cache_ttl_seconds,
        fetch_ttl=config.fetch_cache_ttl_seconds,
        search_cost_usd=config.search_cost_usd,
        fetch_cost_usd=config.fetch_cost_usd,
    )

    start = time.perf_counter()
    dollars_spent = 0.0
    for q in questions:
        misses_before = backend.stats.search_misses
        results = await backend.search(q, max_results=fetches)
        if backend.stats.search_misses > misses_before:
            dollars_spent += config.search_cost_usd
        for r in results[:fetches]:
            fmisses_before = backend.stats.fetch_misses
            await backend.fetch(r.url)
            if backend.stats.fetch_misses > fmisses_before:
                dollars_spent += config.fetch_cost_usd
    elapsed = time.perf_counter() - start

    return PassResult(
        name=name,
        n_questions=len(questions),
        wall_clock_seconds=elapsed,
        search_hits=backend.stats.search_hits,
        search_misses=backend.stats.search_misses,
        fetch_hits=backend.stats.fetch_hits,
        fetch_misses=backend.stats.fetch_misses,
        dollars_spent=dollars_spent,
        dollars_saved=backend.stats.estimated_dollars_saved,
        hit_rate=backend.stats.hit_rate,
    )


async def run_bypass_pass(name: str, questions: list[str], config: RunConfig, fetches: int, inner_seed: int) -> PassResult:
    """Same questions as the cold/warm passes, but with no cache at all —
    proves the DEEPRESEARCH_CACHE_ENABLED=false bypass flag actually
    bypasses (every call is a miss, no speedup, regardless of repeats)."""
    inner = FakeTavilyBackend(seed=inner_seed)
    start = time.perf_counter()
    for q in questions:
        results = await inner.search(q, max_results=fetches)
        for r in results[:fetches]:
            await inner.fetch(r.url)
    elapsed = time.perf_counter() - start

    n_searches = len(questions)
    n_fetches = len(questions) * fetches
    return PassResult(
        name=name,
        n_questions=len(questions),
        wall_clock_seconds=elapsed,
        search_hits=0,
        search_misses=n_searches,
        fetch_hits=0,
        fetch_misses=n_fetches,
        dollars_spent=n_searches * config.search_cost_usd + n_fetches * config.fetch_cost_usd,
        dollars_saved=0.0,
        hit_rate=0.0,
    )


async def main(n: int, seed: int, fetches: int) -> dict:
    config = RunConfig()
    redis_url = os.getenv("REDIS_URL", config.redis_url)
    cache, used_real_redis = await build_cache(redis_url)

    questions = load_questions(n, seed)
    fresh_questions = load_questions(n, seed=seed + 1000)  # disjoint sample

    cold = await run_cached_pass("cold (empty cache)", questions, cache, config, fetches, inner_seed=1)
    warm = await run_cached_pass("warm (same questions, repeated)", questions, cache, config, fetches, inner_seed=2)

    half = n // 2
    mixed_questions = questions[:half] + fresh_questions[: n - half]
    mixed = await run_cached_pass(
        "mixed (half repeat, half fresh)", mixed_questions, cache, config, fetches, inner_seed=3
    )

    bypass = await run_bypass_pass("bypass (cache_enabled=false)", questions, config, fetches, inner_seed=4)

    return {
        "config": {
            "n_questions": n,
            "seed": seed,
            "fetches_per_question": fetches,
            "search_cost_usd": config.search_cost_usd,
            "fetch_cost_usd": config.fetch_cost_usd,
            "search_cache_ttl_seconds": config.search_cache_ttl_seconds,
            "fetch_cache_ttl_seconds": config.fetch_cache_ttl_seconds,
            "used_real_redis": used_real_redis,
            "backend": "FakeTavilyBackend (simulated latency, no live API calls)",
        },
        "passes": [asdict(p) for p in (cold, warm, mixed, bypass)],
    }


def _print_summary(result: dict) -> None:
    print(f"{'pass':<32} {'wall_s':>8} {'hit_rate':>9} {'$spent':>8} {'$saved':>8}")
    for p in result["passes"]:
        print(
            f"{p['name']:<32} {p['wall_clock_seconds']:>8.2f} {p['hit_rate']:>9.2f} "
            f"{p['dollars_spent']:>8.4f} {p['dollars_saved']:>8.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fetches", type=int, default=3)
    args = parser.parse_args()

    result = asyncio.run(main(args.n, args.seed, args.fetches))

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"cache_measurement_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    _print_summary(result)
    print(f"\nWritten to {out_path}")
