from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from deepresearch.backends.base import SearchBackend
from deepresearch.schemas import FetchResult, SearchResult


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    text: str


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class LocalCorpusBackend(SearchBackend):
    """Fixed-corpus backend for reproducible benchmark/CI runs (docs/DESIGN.md
    decision row 5) — no live-web flakiness, no rate-limit exposure.

    Scoped to ONE benchmark question's candidate document pool (gold +
    distractors), not a global corpus: FRAMES and MuSiQue each ship their own
    per-question documents, and cross-question retrieval would leak the
    answer's location via corpus-membership alone. The eval harness
    constructs a fresh instance per question (eval/benchmarks/*.py).

    Search is real BM25 lexical retrieval over the provided documents — not a
    stub returning fixed results — so with/without-rerank and raw-vs-BM25
    orderings are both genuine retrieval, exercising the same worker.py code
    path as the live Tavily backend.
    """

    def __init__(self, documents: list[CorpusDocument]) -> None:
        if not documents:
            raise ValueError("LocalCorpusBackend needs at least one document")
        self._documents = {doc.doc_id: doc for doc in documents}
        self._ids = list(self._documents.keys())
        self._bm25 = BM25Okapi([_tokenize(self._documents[doc_id].text) for doc_id in self._ids])

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(url=f"corpus://{doc_id}", title=self._documents[doc_id].title, snippet=self._documents[doc_id].text[:500], score=float(score))
            for doc_id, score in ranked[:max_results]
        ]

    async def fetch(self, url: str) -> FetchResult:
        doc_id = url.removeprefix("corpus://")
        doc = self._documents.get(doc_id)
        if doc is None:
            raise KeyError(f"no such document in local corpus: {url}")
        return FetchResult(url=url, content=doc.text)

    @classmethod
    def from_dicts(cls, documents: list[dict]) -> "LocalCorpusBackend":
        return cls(
            [CorpusDocument(doc_id=d["doc_id"], title=d.get("title", ""), text=d["text"]) for d in documents]
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LocalCorpusBackend":
        """`config.local_corpus_dir` points at one question's corpus JSON
        file (a list of {doc_id, title, text} objects) — set per-question by
        the eval harness before each run_research() call."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dicts(data)
