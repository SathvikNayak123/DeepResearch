from __future__ import annotations

import os

from deepresearch.config import RunConfig
from deepresearch.rerank.base import RerankBackend

# Process-wide cache, keyed by (backend kind, model name) so a test/ablation
# overriding DEEPRESEARCH_RERANK_MODEL between calls still gets its own
# instance rather than a stale one. Without this, every run_research() call
# (once per eval question, once per API request) built a brand-new
# CrossEncoderRerankBackend: the ~600M-param cross-encoder got reloaded from
# scratch every time (confirmed live: a 5-question eval run kept a ~2.5GB
# resident process the whole way through, reloading the model each question),
# and -- worse -- each fresh instance got its own _inference_lock, so the
# CPU-oversubscription protection that lock exists for (docs/RESULTS.md: 4
# concurrent .predict() calls measured at 391s/call vs. 13.8s serialized)
# never actually applied *across* concurrent requests/questions, only within
# one run's own worker pool. One process-wide instance fixes both: the model
# loads once, and its lock genuinely serializes CPU-bound inference across
# every concurrent run in this process, not just one run's own workers.
_rerank_backend_cache: dict[tuple[str, str | None], RerankBackend] = {}


def build_rerank_backend(config: RunConfig) -> RerankBackend | None:
    """Factory: which RerankBackend to use, if any, per config.rerank_backend.

    Behind the same interface either way (docs/DESIGN.md decision row 7) —
    swapping "bge" <-> "cohere" never touches worker.py. Cached process-wide
    (see module docstring above) -- callers get a shared instance, not a
    fresh one per call.
    """
    if not config.rerank_enabled:
        return None
    if config.rerank_backend == "cohere":
        cache_key = ("cohere", None)
        if cache_key not in _rerank_backend_cache:
            from deepresearch.rerank.cohere import CohereRerankBackend

            _rerank_backend_cache[cache_key] = CohereRerankBackend()
        return _rerank_backend_cache[cache_key]
    if config.rerank_backend == "bge":
        from deepresearch.rerank.bge import DEFAULT_MODEL

        model_name = os.getenv("DEEPRESEARCH_RERANK_MODEL", DEFAULT_MODEL)
        cache_key = ("bge", model_name)
        if cache_key not in _rerank_backend_cache:
            from deepresearch.rerank.bge import CrossEncoderRerankBackend

            _rerank_backend_cache[cache_key] = CrossEncoderRerankBackend(model_name)
        return _rerank_backend_cache[cache_key]
    if config.rerank_backend == "bge_directml":
        from deepresearch.rerank.bge_directml import DEFAULT_MODEL

        model_name = os.getenv("DEEPRESEARCH_RERANK_MODEL", DEFAULT_MODEL)
        cache_key = ("bge_directml", f"{model_name}_pad{config.candidate_pool_size}")
        if cache_key not in _rerank_backend_cache:
            from deepresearch.rerank.bge_directml import DirectMLRerankBackend

            _rerank_backend_cache[cache_key] = DirectMLRerankBackend(model_name, pad_to=config.candidate_pool_size)
        return _rerank_backend_cache[cache_key]
    raise ValueError(f"unknown rerank_backend: {config.rerank_backend!r}")
