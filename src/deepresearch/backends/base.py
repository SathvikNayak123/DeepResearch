from __future__ import annotations

from abc import ABC, abstractmethod

from deepresearch.schemas import FetchResult, SearchResult


class SearchBackend(ABC):
    """Tool interface the agent talks to — web now (Tavily), a fixed local
    corpus later (Session 4), same interface either way so benchmark/CI
    runs can swap backends without touching agent logic."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    @abstractmethod
    async def fetch(self, url: str) -> FetchResult:
        ...
