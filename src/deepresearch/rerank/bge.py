from __future__ import annotations

import asyncio
import os

from deepresearch.rerank.base import RankedChunk, RerankBackend

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderRerankBackend(RerankBackend):
    """Self-hosted cross-encoder reranker — default per docs/DESIGN.md row 7.

    Free and unlimited to run in CI/ablations (no rate limits, no per-call
    cost) at the price of CPU latency. Model name is overridable via
    DEEPRESEARCH_RERANK_MODEL so tests/ablations can swap in a smaller
    cross-encoder without touching the interface.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("DEEPRESEARCH_RERANK_MODEL", DEFAULT_MODEL)
        self._model = None  # lazy-loaded on first use, not at construction
        # One backend instance is shared process-wide -- across every run's
        # own bounded worker pool AND across concurrent runs themselves
        # (rerank/__init__.py's build_rerank_backend caches by (kind, model)
        # instead of constructing fresh per call). Both locks below need that
        # same process-wide scope: a second concurrent run building its own
        # fresh instance would get its own locks, silently defeating the
        # protection they exist for.
        #
        # Concurrent first-use calls used to race into CrossEncoder(...)
        # construction simultaneously, corrupting transformers' meta-device
        # init state (NotImplementedError: "Cannot copy out of meta tensor")
        # — reproduced directly by firing 4 concurrent rerank() calls on a
        # fresh instance. This lock serializes the one-time load.
        self._load_lock = asyncio.Lock()
        # Real 4-way-concurrent measurement (a live FRAMES smoke run, 2026-07)
        # found mean rerank latency of 391s/call — ~28x the 13.8s mean from
        # the isolated single-call ablation (docs/RESULTS.md) — enough to
        # trip the 600s wall-clock budget. Root cause: each CrossEncoder
        # .predict() call internally spawns full-core-count BLAS/torch
        # threads; 4 concurrent workers each doing that thrashes the CPU
        # instead of scaling. Serializing actual inference (not just the
        # load) trades worker-level parallelism for CPU-level sanity — 4
        # sequential full-core calls measurably beats 4 oversubscribed
        # concurrent ones on a CPU-bound, limited-core box.
        self._inference_lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[str]) -> list[RankedChunk]:
        if not chunks:
            return []
        if self._model is None:
            async with self._load_lock:
                if self._model is None:  # double-checked: lost the race, already loaded
                    self._model = await asyncio.to_thread(self._load)
        model = self._model
        pairs = [(query, chunk) for chunk in chunks]
        async with self._inference_lock:
            scores = await asyncio.to_thread(model.predict, pairs)
        return sorted(
            (RankedChunk(index=i, score=float(s)) for i, s in enumerate(scores)),
            key=lambda rc: rc.score,
            reverse=True,
        )
