"""DirectML (DirectX 12) GPU-accelerated variant of CrossEncoderRerankBackend.

Same model (BAAI/bge-reranker-v2-m3), same scores -- live-verified bit-identical
to the CPU CrossEncoder path (sigmoid(onnx_logits) == CrossEncoder.predict())
on a 20-pair batch -- but ~3.4x faster per-call inference, measured live
against an AMD RX 9060 XT (3.48s -> 1.02s for a 20-pair batch). Opt-in via
DEEPRESEARCH_RERANK_BACKEND=bge_directml; the plain "bge" CPU backend stays
the portable default since DirectML is Windows-only and this path won't run
in CI/Linux.

ONNX Runtime's DML execution provider only worked reliably here with a
STATIC input shape exported via the legacy TorchScript-based exporter
(torch.onnx.export(..., dynamo=False)). The newer torch.export/dynamo
exporter (torch's current default) hit two separate DML kernel failures on
this model architecture during live testing: a Reshape op failure with
dynamic axes, and (after switching to static shapes) a session-init failure.
Both went away with the legacy exporter + fixed shapes -- a real, reproduced
compatibility gap between DirectML's op coverage and the dynamo export path
for this transformer architecture, not a configuration mistake.

Static shape means the ONNX session only accepts exactly `pad_to` (query,
chunk) pairs per call. Real chunk counts vary a lot and routinely exceed
`pad_to` -- candidates aren't one-per-search-result, retrieve.py chunks each
source into up to `max_chunks_per_source` pieces first, so
candidate_pool_size=20 sources can easily produce 20-200+ candidate chunks
(confirmed live: 2 real MuSiQue questions produced 22 and 23 chunks against
a naive pad_to=20). rerank() handles this by batching internally -- chunks
are processed in pad_to-sized groups, with only the last (partial) group
padded, so real candidate counts of any size are scored in full, never
truncated or dropped.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from deepresearch.rerank.base import RankedChunk, RerankBackend

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_ONNX_CACHE_DIR = Path(os.getenv("DEEPRESEARCH_ONNX_CACHE_DIR", ".cache/onnx_rerank"))


class DirectMLRerankBackend(RerankBackend):
    def __init__(self, model_name: str | None = None, pad_to: int = 20, max_length: int = 512) -> None:
        self._model_name = model_name or os.getenv("DEEPRESEARCH_RERANK_MODEL", DEFAULT_MODEL)
        self._pad_to = pad_to
        self._max_length = max_length
        self._tokenizer = None
        self._session = None  # lazy-loaded on first use, not at construction
        # Same rationale as CrossEncoderRerankBackend (rerank/bge.py): one
        # instance is shared process-wide (rerank/__init__.py's cache), so
        # both locks need that same scope -- a concurrent first call racing
        # into export/session-init, or into GPU inference, is exactly the
        # failure mode these guard against.
        self._load_lock = asyncio.Lock()
        # Concurrent-DML-inference safety hasn't been separately measured the
        # way the CPU path's contention was (docs/RESULTS.md: 391s/call
        # oversubscribed vs 13.8s serialized) -- serializing here too is the
        # conservative default until that's actually tested under load.
        self._inference_lock = asyncio.Lock()

    def _onnx_path(self) -> Path:
        safe_name = self._model_name.replace("/", "__")
        return DEFAULT_ONNX_CACHE_DIR / f"{safe_name}_pad{self._pad_to}_len{self._max_length}.onnx"

    def _load(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        onnx_path = self._onnx_path()
        if not onnx_path.exists():
            self._export_onnx(onnx_path, tokenizer)
        session = ort.InferenceSession(str(onnx_path), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        return tokenizer, session

    def _export_onnx(self, onnx_path: Path, tokenizer) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        hf_model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        hf_model.eval()
        dummy = tokenizer(
            ["a"] * self._pad_to,
            ["b"] * self._pad_to,
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        torch.onnx.export(
            hf_model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )

    async def rerank(self, query: str, chunks: list[str]) -> list[RankedChunk]:
        if not chunks:
            return []
        if self._session is None:
            async with self._load_lock:
                if self._session is None:  # double-checked: lost the race, already loaded
                    self._tokenizer, self._session = await asyncio.to_thread(self._load)
        tokenizer, session = self._tokenizer, self._session

        all_scores: list[float] = []
        for start in range(0, len(chunks), self._pad_to):
            batch = chunks[start : start + self._pad_to]
            n_real = len(batch)
            padded_batch = batch + [""] * (self._pad_to - n_real)
            queries = [query] * self._pad_to

            def _run(padded_batch=padded_batch, queries=queries, n_real=n_real) -> list[float]:
                import numpy as np

                inputs = tokenizer(
                    queries,
                    padded_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="np",
                )
                (logits,) = session.run(
                    None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                )
                logits = logits.reshape(-1)[:n_real]
                return (1 / (1 + np.exp(-logits))).tolist()  # matches CrossEncoder.predict()'s sigmoid activation

            async with self._inference_lock:
                batch_scores = await asyncio.to_thread(_run)
            all_scores.extend(batch_scores)

        return sorted(
            (RankedChunk(index=i, score=all_scores[i]) for i in range(len(chunks))),
            key=lambda rc: rc.score,
            reverse=True,
        )
