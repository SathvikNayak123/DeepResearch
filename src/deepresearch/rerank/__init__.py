from __future__ import annotations

from deepresearch.config import RunConfig
from deepresearch.rerank.base import RerankBackend


def build_rerank_backend(config: RunConfig) -> RerankBackend | None:
    """Factory: which RerankBackend to use, if any, per config.rerank_backend.

    Behind the same interface either way (docs/DESIGN.md decision row 7) —
    swapping "bge" <-> "cohere" never touches worker.py.
    """
    if not config.rerank_enabled:
        return None
    if config.rerank_backend == "cohere":
        from deepresearch.rerank.cohere import CohereRerankBackend

        return CohereRerankBackend()
    if config.rerank_backend == "bge":
        from deepresearch.rerank.bge import CrossEncoderRerankBackend

        return CrossEncoderRerankBackend()
    raise ValueError(f"unknown rerank_backend: {config.rerank_backend!r}")
