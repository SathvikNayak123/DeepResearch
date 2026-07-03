from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RankedChunk:
    index: int  # index into the `chunks` list passed to rerank()
    score: float


class RerankBackend(ABC):
    """Score candidate chunks against a query. Same interface for a
    self-hosted cross-encoder (default) and a hosted API (optional) —
    docs/DESIGN.md decision row 7."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[str]) -> list[RankedChunk]:
        """Return chunks ranked by relevance to query, best first."""
