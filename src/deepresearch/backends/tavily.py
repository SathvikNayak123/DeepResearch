from __future__ import annotations

import os

import httpx

from deepresearch.backends.base import SearchBackend
from deepresearch.schemas import FetchResult, SearchResult

TAVILY_BASE_URL = "https://api.tavily.com"


class TavilyBackend(SearchBackend):
    """Live web backend: Tavily's combined search + extract API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["TAVILY_API_KEY"]
        self._client = httpx.AsyncClient(base_url=TAVILY_BASE_URL, timeout=30.0)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.post(
            "/search",
            json={"api_key": self._api_key, "query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]

    async def fetch(self, url: str) -> FetchResult:
        resp = await self._client.post(
            "/extract",
            json={"api_key": self._api_key, "urls": [url]},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        content = results[0]["raw_content"] if results else ""
        return FetchResult(url=url, content=content)

    async def aclose(self) -> None:
        await self._client.aclose()
